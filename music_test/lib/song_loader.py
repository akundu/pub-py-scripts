"""
Song loader service for constraining chord recognition to known song chords.
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Set


class SongLoader:
    """Load and manage song data from the songs/ directory."""
    
    def __init__(self, songs_dir: str = "songs"):
        self.songs_dir = Path(songs_dir)
        self.manifest = self._load_manifest()
        self._song_cache = {}
        
    def _load_manifest(self) -> Dict:
        """Load the manifest.json file."""
        manifest_path = self.songs_dir / "manifest.json"
        if not manifest_path.exists():
            return {"songs": []}
        
        with open(manifest_path, 'r') as f:
            return json.load(f)
    
    def get_song_list(self) -> List[Dict]:
        """Get list of all available songs."""
        return self.manifest.get("songs", [])
    
    def load_song(self, song_id: str) -> Optional[Dict]:
        """Load a song's full data from its JSON file."""
        if song_id in self._song_cache:
            return self._song_cache[song_id]
        
        # Find song in manifest
        song_info = None
        for song in self.manifest.get("songs", []):
            if song.get("song_id") == song_id:
                song_info = song
                break
        
        if not song_info:
            return None
        
        # Load the full song file
        song_file = self.songs_dir / song_info["file"]
        if not song_file.exists():
            return None
        
        with open(song_file, 'r') as f:
            song_data = json.load(f)
        
        self._song_cache[song_id] = song_data
        return song_data
    
    def get_song_chords(self, song_id: str) -> List[str]:
        """Get the list of chords used in a song."""
        song_data = self.load_song(song_id)
        if not song_data:
            return []
        return song_data.get("chords", [])
    
    def get_song_info(self, song_id: str) -> Optional[Dict]:
        """Get basic song info (title, composer, etc.)."""
        for song in self.manifest.get("songs", []):
            if song.get("song_id") == song_id:
                return song
        return None
    
    @staticmethod
    def get_chord_pitch_classes(chord_name: str) -> Set[int]:
        """
        Return the pitch classes (0-11) for a chord.
        C=0, C#=1, D=2, ..., B=11
        """
        note_to_chroma = {
            'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
            'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11,
            'Db': 1, 'Eb': 3, 'Gb': 6, 'Ab': 8, 'Bb': 10
        }
        
        # Chord interval patterns
        chord_patterns = {
            '': [0, 4, 7],          # Major
            'm': [0, 3, 7],         # Minor
            'maj7': [0, 4, 7, 11],  # Major 7th
            'M7': [0, 4, 7, 11],    # Major 7th (alternate)
            'm7': [0, 3, 7, 10],    # Minor 7th
            '7': [0, 4, 7, 10],     # Dominant 7th
            'dim': [0, 3, 6],       # Diminished
            '°': [0, 3, 6],         # Diminished (symbol)
            'aug': [0, 4, 8],       # Augmented
            '+': [0, 4, 8],         # Augmented (symbol)
            'sus2': [0, 2, 7],      # Sus2
            'sus4': [0, 5, 7],      # Sus4
            '5': [0, 7],            # Power chord
            '6': [0, 4, 7, 9],      # Major 6th
            'm6': [0, 3, 7, 9],     # Minor 6th
            'add9': [0, 4, 7, 2],   # Add9
            '9': [0, 4, 7, 10, 2],  # 9th
        }
        
        # Parse chord name to extract root and type
        if not chord_name:
            return set()
        
        # Extract root note (1 or 2 characters)
        if len(chord_name) >= 2 and chord_name[1] in ['#', 'b']:
            root = chord_name[:2]
            chord_type = chord_name[2:]
        else:
            root = chord_name[0]
            chord_type = chord_name[1:]
        
        # Get root pitch class
        root_pc = note_to_chroma.get(root)
        if root_pc is None:
            return set()
        
        # Get chord pattern
        pattern = chord_patterns.get(chord_type, [0, 4, 7])  # Default to major
        
        # Calculate pitch classes
        pitch_classes = {(root_pc + interval) % 12 for interval in pattern}
        return pitch_classes
    
    @staticmethod
    def calculate_chord_similarity(detected_pcs: Set[int], song_chord_pcs: Set[int]) -> float:
        """
        Calculate similarity between two chords based on pitch class overlap.
        Returns value between 0.0 (no overlap) and 1.0 (identical).
        """
        if not detected_pcs or not song_chord_pcs:
            return 0.0
        
        intersection = detected_pcs & song_chord_pcs
        union = detected_pcs | song_chord_pcs
        
        if not union:
            return 0.0
        
        # Jaccard similarity
        return len(intersection) / len(union)
    
    def find_best_match_in_song(self, detected_chord: str, song_chords: List[str]) -> Optional[Dict]:
        """
        Find the closest chord from the song that matches the detected chord.
        
        Returns:
            Dict with 'chord', 'similarity', and 'match_type' keys, or None
        """
        if not detected_chord or not song_chords:
            return None
        
        detected_pcs = self.get_chord_pitch_classes(detected_chord)
        if not detected_pcs:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for song_chord in song_chords:
            song_pcs = self.get_chord_pitch_classes(song_chord)
            similarity = self.calculate_chord_similarity(detected_pcs, song_pcs)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = song_chord
        
        if best_match:
            # Classify match type
            if best_similarity >= 0.9:
                match_type = "exact"
            elif best_similarity >= 0.6:
                match_type = "close"
            else:
                match_type = "distant"
            
            return {
                "chord": best_match,
                "similarity": best_similarity,
                "match_type": match_type
            }
        
        return None


# Global instance for convenience
_song_loader = None

def get_song_loader(songs_dir: str = "songs") -> SongLoader:
    """Get or create the global SongLoader instance."""
    global _song_loader
    if _song_loader is None:
        _song_loader = SongLoader(songs_dir)
    return _song_loader
