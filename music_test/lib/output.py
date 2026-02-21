"""
Output handler classes for CLI and web interfaces.
"""
import time
import sys
import numpy as np
from abc import ABC, abstractmethod
from lib.common import clear_line
from lib.music_understanding import frequency_to_note


class OutputHandler(ABC):
    """
    Abstract base class for output handling.
    Defines the interface for outputting detection results.
    """
    
    @abstractmethod
    def chord_detected(self, chord_name, confidence, stability, detected_notes, 
                       detected_frequencies, chroma_vector, config):
        """Output a chord detection result."""
        pass
    
    @abstractmethod
    def notes_detected(self, detected_notes, detected_frequencies, chroma_vector, config):
        """Output a notes-only detection result."""
        pass
    
    @abstractmethod
    def frequencies_detected(self, detected_frequencies, chroma_vector, config):
        """Output a frequencies-only detection result."""
        pass
    
    @abstractmethod
    def no_detection(self, config):
        """Output when nothing is detected."""
        pass
    
    @abstractmethod
    def listening(self, config):
        """Output when actively listening but nothing detected yet."""
        pass


class ConsoleOutputHandler(OutputHandler):
    """
    Output handler for CLI - prints to stdout/stderr.
    """
    
    def __init__(self, config):
        self.config = config
        self.log_mode = config.get('log', False) if hasattr(config, 'get') else getattr(config, 'log', False)
        self.debug = config.get('debug', False) if hasattr(config, 'get') else getattr(config, 'debug', False)
    
    def _get_timestamp(self):
        return time.strftime("%Y-%m-%d %H:%M:%S")
    
    def _format_notes_with_frequencies(self, detected_notes, detected_frequencies):
        """Format notes with their frequencies."""
        note_freqs = {}
        for freq, amp in detected_frequencies:
            note = frequency_to_note(freq)
            if note in detected_notes:
                if note not in note_freqs:
                    note_freqs[note] = []
                note_freqs[note].append(freq)
        return ", ".join([f"{note}({np.mean(freqs):.0f}Hz)" for note, freqs in note_freqs.items()])
    
    def _format_chroma(self, chroma_vector):
        """Format chroma vector for display."""
        if chroma_vector is None:
            return ""
        chroma_values = [f"{val:.3f}" for val in chroma_vector]
        return f"[{', '.join(chroma_values)}]"
    
    def chord_detected(self, chord_name, confidence, stability, detected_notes,
                       detected_frequencies, chroma_vector, config):
        """Output a chord detection result."""
        show_chroma = config.get('show_chroma', False) if hasattr(config, 'get') else getattr(config, 'show_chroma', False)
        show_frequencies = config.get('show_frequencies', False) if hasattr(config, 'get') else getattr(config, 'show_frequencies', False)
        
        output_parts = []
        
        # Add frequency/note info if requested
        if show_frequencies and detected_frequencies:
            freq_str = self._format_notes_with_frequencies(detected_notes, detected_frequencies)
            output_parts.append(f"Notes: {freq_str}")
        
        # Add chroma vector if requested
        if show_chroma and chroma_vector is not None:
            output_parts.append(f"Chroma: {self._format_chroma(chroma_vector)}")
        
        # Build chord display
        chord_display = chord_name
        if self.debug:
            chord_display += f" ({confidence:.2f})"
        
        # Combine output
        if output_parts:
            full_output = " | ".join(output_parts) + f" -> {chord_display}"
        else:
            full_output = chord_display
        
        # Display with timestamp if logging
        stability_suffix = "" if stability >= 2 else " [unstable]"
        
        if self.log_mode:
            timestamp = self._get_timestamp()
            print(f"[{timestamp}] {full_output}{stability_suffix}")
        else:
            clear_line()
            stability_suffix = "" if stability >= 2 else " [?]"
            print(f"{full_output}{stability_suffix}", end='\r')
        
        return None  # Console handler doesn't return anything
    
    def notes_detected(self, detected_notes, detected_frequencies, chroma_vector, config):
        """Output a notes-only detection result."""
        show_chroma = config.get('show_chroma', False) if hasattr(config, 'get') else getattr(config, 'show_chroma', False)
        
        freq_str = self._format_notes_with_frequencies(detected_notes, detected_frequencies)
        chroma_str = ""
        if show_chroma and chroma_vector is not None:
            chroma_str = f" | Chroma: {self._format_chroma(chroma_vector)}"
        
        if self.log_mode:
            timestamp = self._get_timestamp()
            print(f"[{timestamp}] Notes: {freq_str}{chroma_str}")
        else:
            clear_line()
            print(f"Notes detected: {freq_str}{chroma_str}", end='\r')
        
        return None
    
    def frequencies_detected(self, detected_frequencies, chroma_vector, config):
        """Output a frequencies-only detection result."""
        show_chroma = config.get('show_chroma', False) if hasattr(config, 'get') else getattr(config, 'show_chroma', False)
        
        freq_str = ", ".join([f"{freq:.0f}Hz" for freq, _ in detected_frequencies])
        chroma_str = ""
        if show_chroma and chroma_vector is not None:
            chroma_str = f" | Chroma: {self._format_chroma(chroma_vector)}"
        
        if self.log_mode:
            timestamp = self._get_timestamp()
            print(f"[{timestamp}] Frequencies: {freq_str}{chroma_str}")
        else:
            clear_line()
            print(f"Frequencies detected: {freq_str}{chroma_str}", end='\r')
        
        return None
    
    def no_detection(self, config):
        """Output when nothing is detected."""
        if self.log_mode:
            if self.debug:
                timestamp = self._get_timestamp()
                print(f"[{timestamp}] No notes detected")
        else:
            if not self.debug:
                clear_line()
                print("🔇 Listening...", end='\r')
        return None
    
    def listening(self, config):
        """Output when actively listening."""
        if not self.log_mode:
            clear_line()
            print("🎵 Listening...", end='\r')
        return None


class DictOutputHandler(OutputHandler):
    """
    Output handler for web - returns dictionary for JSON serialization.
    """
    
    def __init__(self, config=None):
        self.config = config
    
    def chord_detected(self, chord_name, confidence, stability, detected_notes,
                       detected_frequencies, chroma_vector, config):
        """Return a chord detection result as dict."""
        result = {
            "type": "chord",
            "chord": chord_name,
            "confidence": confidence,
            "stability": stability,
            "timestamp": time.time()
        }
        
        show_frequencies = config.get('show_frequencies', False) if hasattr(config, 'get') else getattr(config, 'show_frequencies', False)
        show_chroma = config.get('show_chroma', False) if hasattr(config, 'get') else getattr(config, 'show_chroma', False)
        
        # Add notes with frequencies if requested
        if show_frequencies and detected_frequencies:
            note_freqs = {}
            for freq, amp in detected_frequencies:
                note = frequency_to_note(freq)
                if note in detected_notes:
                    if note not in note_freqs:
                        note_freqs[note] = []
                    note_freqs[note].append(freq)
            result["notes"] = [(note, np.mean(freqs)) for note, freqs in note_freqs.items()]
        
        # Add chroma vector if requested
        if show_chroma and chroma_vector is not None:
            result["chroma"] = chroma_vector.tolist() if hasattr(chroma_vector, 'tolist') else list(chroma_vector)
        
        return result
    
    def notes_detected(self, detected_notes, detected_frequencies, chroma_vector, config):
        """Return a notes-only detection result as dict."""
        note_freqs = {}
        for freq, amp in detected_frequencies:
            note = frequency_to_note(freq)
            if note in detected_notes:
                if note not in note_freqs:
                    note_freqs[note] = []
                note_freqs[note].append(freq)
        
        result = {
            "type": "notes",
            "notes": [(note, np.mean(freqs)) for note, freqs in note_freqs.items()],
            "timestamp": time.time()
        }
        
        show_chroma = config.get('show_chroma', False) if hasattr(config, 'get') else getattr(config, 'show_chroma', False)
        if show_chroma and chroma_vector is not None:
            result["chroma"] = chroma_vector.tolist() if hasattr(chroma_vector, 'tolist') else list(chroma_vector)
        
        return result
    
    def frequencies_detected(self, detected_frequencies, chroma_vector, config):
        """Return a frequencies-only detection result as dict."""
        result = {
            "type": "frequencies",
            "frequencies": [(freq, amp) for freq, amp in detected_frequencies],
            "timestamp": time.time()
        }
        
        show_chroma = config.get('show_chroma', False) if hasattr(config, 'get') else getattr(config, 'show_chroma', False)
        if show_chroma and chroma_vector is not None:
            result["chroma"] = chroma_vector.tolist() if hasattr(chroma_vector, 'tolist') else list(chroma_vector)
        
        return result
    
    def no_detection(self, config):
        """Return None when nothing is detected."""
        return None
    
    def listening(self, config):
        """Return listening status for web interface."""
        return {
            "type": "listening",
            "timestamp": time.time()
        }

