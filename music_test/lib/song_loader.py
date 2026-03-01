"""
Song loader service for constraining chord recognition to known song chords.
"""
import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Set


# Standard guitar tuning: string number -> MIDI note (open string)
OPEN_STRING_MIDI = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

# Techniques that produce chord-relevant pitched notes
CHORD_TECHNIQUES = {'strum', 'pluck', 'fingerpick'}

# Events within this window (seconds) are grouped as simultaneous
SIMULTANEITY_THRESHOLD = 0.05


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
    
    @staticmethod
    def extract_chord_timeline(events: List[Dict], song_chords: List[str]) -> List[Dict]:
        """
        Build a chord timeline from song events.

        Groups near-simultaneous events into snapshots, identifies the chord
        for each snapshot by matching pitch classes against song_chords, and
        merges consecutive snapshots with the same chord into segments.

        Returns:
            List of dicts with 'start_time', 'end_time', 'chord' keys.
        """
        if not events or not song_chords:
            return []

        # Pre-compute pitch class sets for all song chords
        chord_pcs = {ch: SongLoader.get_chord_pitch_classes(ch) for ch in song_chords}

        # 1. Filter to chord-relevant techniques
        pitched = [e for e in events if e.get('technique') in CHORD_TECHNIQUES]
        if not pitched:
            return []

        # 2. Sort by time and group into simultaneous snapshots
        pitched.sort(key=lambda e: e['time'])
        snapshots = []  # list of (time, {pitch_classes}, max_duration)
        group_start = pitched[0]['time']
        group = [pitched[0]]

        for ev in pitched[1:]:
            if ev['time'] - group_start <= SIMULTANEITY_THRESHOLD:
                group.append(ev)
            else:
                # Finalise previous group
                pcs = set()
                max_dur = 0.0
                for g in group:
                    midi = OPEN_STRING_MIDI.get(g['string'], 0) + g.get('fret', 0)
                    pcs.add(midi % 12)
                    max_dur = max(max_dur, g.get('duration', 0.0))
                snapshots.append((group_start, pcs, max_dur))
                group_start = ev['time']
                group = [ev]

        # Don't forget the last group
        pcs = set()
        max_dur = 0.0
        for g in group:
            midi = OPEN_STRING_MIDI.get(g['string'], 0) + g.get('fret', 0)
            pcs.add(midi % 12)
            max_dur = max(max_dur, g.get('duration', 0.0))
        snapshots.append((group_start, pcs, max_dur))

        # 3. Match each snapshot to the best song chord
        segments = []  # list of (start_time, end_time, chord_name_or_None)
        for snap_time, snap_pcs, snap_dur in snapshots:
            best_chord = None
            best_sim = 0.0
            for ch_name, ch_pcs in chord_pcs.items():
                sim = SongLoader.calculate_chord_similarity(snap_pcs, ch_pcs)
                if sim > best_sim:
                    best_sim = sim
                    best_chord = ch_name
            # Require at least some similarity to call it a chord
            matched = best_chord if best_sim >= 0.3 else None
            end_time = snap_time + snap_dur
            segments.append((snap_time, end_time, matched))

        # 4. Merge consecutive segments with the same chord
        merged = []
        for start, end, chord in segments:
            if merged and merged[-1]['chord'] == chord:
                merged[-1]['end_time'] = max(merged[-1]['end_time'], end)
            else:
                merged.append({'start_time': start, 'end_time': end, 'chord': chord})

        # Extend each segment's end_time to the next segment's start_time (no gaps)
        for i in range(len(merged) - 1):
            merged[i]['end_time'] = merged[i + 1]['start_time']

        return merged

    def get_chord_timeline(self, song_id: str) -> List[Dict]:
        """
        Get the chord timeline for a song, computing and caching the result.
        """
        cache_key = f"_timeline_{song_id}"
        if cache_key in self._song_cache:
            return self._song_cache[cache_key]

        song_data = self.load_song(song_id)
        if not song_data:
            return []

        events = song_data.get('events', [])
        chords = song_data.get('chords', [])
        timeline = self.extract_chord_timeline(events, chords)
        self._song_cache[cache_key] = timeline
        return timeline

    @staticmethod
    def get_expected_chord_at_time(timeline: List[Dict], playback_time: float,
                                   tolerance: float = 0.0) -> Optional[str]:
        """
        Look up which chord should be playing at a given time.

        When tolerance > 0, the window for each segment is expanded by
        tolerance on both sides. If the time falls in an overlap between
        two expanded segments, the earlier segment is returned (the player
        is more likely finishing the previous chord than starting the next
        one early).
        """
        if not timeline:
            return None

        for seg in timeline:
            if seg['chord'] is None:
                continue
            if (seg['start_time'] - tolerance) <= playback_time <= (seg['end_time'] + tolerance):
                return seg['chord']

        return None

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
