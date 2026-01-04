"""
Unified state management for audio processing.
"""
import numpy as np
import time
from collections import deque
from lib.common import get_buffer_size
from lib.music_understanding import INSTRUMENT_PRESETS


class AudioProcessingState:
    """
    Unified state container for audio processing.
    Used by both CLI and web interfaces.
    """
    
    def __init__(self, config):
        """
        Initialize processing state from config.
        
        Args:
            config: AudioConfig or dict-like object with configuration
        """
        # Audio buffer (circular)
        self.audio_buffer = np.zeros(get_buffer_size(), dtype=np.float32)
        self.buffer_index = 0
        
        # Detection history
        self.notes_history = deque(maxlen=5)
        self.frequencies_history = deque(maxlen=5)
        self.chroma_history = deque(maxlen=5)
        
        # Chord stability tracking
        self.last_chord = None
        self.chord_stability = 0
        
        # Timing
        self.last_log_time = time.time()
        
        # Store config reference
        self.config = config
        
        # Get frequency range from config or instrument preset
        if hasattr(config, 'get'):
            # Dict-like config
            instrument = config.get('instrument', 'guitar')
            preset = INSTRUMENT_PRESETS.get(instrument, INSTRUMENT_PRESETS['guitar'])
            self.low_freq = config.get('low_freq') or preset['low_freq']
            self.high_freq = config.get('high_freq') or preset['high_freq']
            self.instrument_name = config.get('instrument_name') or preset['name']
        else:
            # Argparse-like config (has attributes)
            instrument = getattr(config, 'instrument', 'guitar')
            preset = INSTRUMENT_PRESETS.get(instrument, INSTRUMENT_PRESETS['guitar'])
            self.low_freq = getattr(config, 'low_freq', None) or preset['low_freq']
            self.high_freq = getattr(config, 'high_freq', None) or preset['high_freq']
            self.instrument_name = getattr(config, 'instrument_name', None) or preset['name']
    
    def reset(self):
        """Reset state to initial values."""
        self.audio_buffer = np.zeros(get_buffer_size(), dtype=np.float32)
        self.buffer_index = 0
        self.notes_history.clear()
        self.frequencies_history.clear()
        self.chroma_history.clear()
        self.last_chord = None
        self.chord_stability = 0
        self.last_log_time = time.time()
    
    def update_buffer(self, new_samples, chunk_size):
        """
        Update the circular audio buffer with new samples.
        
        Args:
            new_samples: numpy array of new audio samples
            chunk_size: expected chunk size
        
        Returns:
            Updated buffer index
        """
        buffer_size = len(self.audio_buffer)
        actual_chunk_size = min(chunk_size, len(new_samples))
        end_index = min(self.buffer_index + actual_chunk_size, buffer_size)
        copy_size = end_index - self.buffer_index
        
        if copy_size > 0:
            self.audio_buffer[self.buffer_index:end_index] = new_samples[:copy_size]
        
        # Handle wrap-around
        if actual_chunk_size > copy_size:
            remaining = actual_chunk_size - copy_size
            self.audio_buffer[:remaining] = new_samples[copy_size:copy_size + remaining]
        
        self.buffer_index = (self.buffer_index + actual_chunk_size) % buffer_size
        return self.buffer_index
    
    def add_detection(self, notes, frequencies, chroma):
        """
        Add detection results to history.
        
        Args:
            notes: list of detected note names
            frequencies: list of (frequency, amplitude) tuples
            chroma: chroma vector (12-element array) or None
        """
        if notes:
            self.notes_history.append(notes)
        if frequencies:
            self.frequencies_history.append(frequencies)
        if chroma is not None:
            self.chroma_history.append(chroma)
    
    def update_chord_stability(self, chord_name):
        """
        Update chord stability tracking.
        
        Args:
            chord_name: detected chord name
        
        Returns:
            Current stability count
        """
        if chord_name == self.last_chord:
            self.chord_stability += 1
        else:
            self.chord_stability = 0
            self.last_chord = chord_name
        return self.chord_stability
    
    def reset_stability(self):
        """Reset chord stability when no chord detected."""
        self.chord_stability = 0
        self.last_chord = None
    
    def should_log(self, log_interval):
        """
        Check if enough time has passed for logging.
        
        Args:
            log_interval: minimum seconds between logs
        
        Returns:
            True if should log now
        """
        current_time = time.time()
        if current_time - self.last_log_time >= log_interval:
            self.last_log_time = current_time
            return True
        return False
    
    def has_enough_history(self, min_samples=3):
        """
        Check if we have enough history for progression analysis.
        
        Args:
            min_samples: minimum number of samples needed
        
        Returns:
            True if enough history
        """
        return len(self.notes_history) >= min_samples

