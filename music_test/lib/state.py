"""
Unified state management for audio processing.
"""
import math
import numpy as np
import time
from collections import deque, defaultdict
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

        # Detection history (8 frames gives ~1.5s of context at default settings)
        self.notes_history = deque(maxlen=8)
        self.frequencies_history = deque(maxlen=8)
        self.chroma_history = deque(maxlen=8)

        # Chord stability tracking
        self.last_chord = None
        self.chord_stability = 0

        # Timing
        self.last_log_time = time.time()

        # Store config reference
        self.config = config

        # Chord accumulation window for smoothing (reduces rapid chord changes)
        # Collects chord predictions over a time window and picks the best one
        self.chord_accumulator = []  # List of (chord_name, confidence, timestamp) tuples
        self.chord_window_start = time.time()

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
    
    def _config_get(self, key, default):
        """Get a config value, handling both dict and argparse-style configs."""
        if hasattr(self.config, 'get'):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

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
        self.chord_accumulator = []
        self.chord_window_start = time.time()
    
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

    def accumulate_chord(self, chord_name, confidence, notes=None, frequencies=None, chroma=None):
        """
        Add a chord detection to the accumulator for smoothing.

        Args:
            chord_name: detected chord name
            confidence: detection confidence (0.0 to 1.0)
            notes: optional list of detected notes
            frequencies: optional list of (frequency, amplitude) tuples
            chroma: optional chroma vector
        """
        self.chord_accumulator.append({
            'chord': chord_name,
            'confidence': confidence,
            'timestamp': time.time(),
            'notes': notes,
            'frequencies': frequencies,
            'chroma': chroma
        })

    def is_window_complete(self, window_duration):
        """
        Check if the chord accumulation window is complete.

        Args:
            window_duration: window duration in seconds

        Returns:
            True if window is complete and has accumulated chords
        """
        elapsed = time.time() - self.chord_window_start
        return elapsed >= window_duration and len(self.chord_accumulator) > 0

    def get_best_chord(self):
        """
        Get the best chord from the accumulated predictions using
        exponentially-weighted voting with temporal hysteresis.

        Enhancements over simple confidence-weighted voting:
        1. Exponential temporal weighting: recent detections count more than
           older ones, so the system responds faster to real chord changes
           while still smoothing brief glitches.
        2. Hysteresis bonus: the currently stable chord gets a small bonus,
           requiring new chords to win by a margin before switching. This
           prevents rapid oscillation between similar-scoring alternatives.

        Returns:
            dict with best chord info or None if no chords accumulated:
            {
                'chord': best chord name,
                'confidence': average confidence for this chord,
                'votes': number of times this chord was detected,
                'total_votes': total detections in window,
                'notes': notes from highest-confidence detection,
                'frequencies': frequencies from highest-confidence detection,
                'chroma': chroma from highest-confidence detection
            }
        """
        if not self.chord_accumulator:
            return None

        # --- Exponential temporal weighting ---
        # Decay factor: detections from 0.3s ago get ~50% weight
        # This makes recent detections much more influential
        DECAY_RATE = self._config_get('decay_rate', 2.3)
        now = time.time()

        chord_scores = defaultdict(lambda: {
            'weighted_confidence': 0.0, 'raw_confidence': 0.0,
            'count': 0, 'best_detection': None
        })

        for detection in self.chord_accumulator:
            chord = detection['chord']
            conf = detection['confidence']
            age = now - detection['timestamp']
            # Exponential decay: recent detections weighted much higher
            time_weight = math.exp(-DECAY_RATE * age)
            weighted_conf = conf * time_weight

            chord_scores[chord]['weighted_confidence'] += weighted_conf
            chord_scores[chord]['raw_confidence'] += conf
            chord_scores[chord]['count'] += 1

            # Track the highest-confidence detection for this chord
            if (chord_scores[chord]['best_detection'] is None or
                    conf > chord_scores[chord]['best_detection']['confidence']):
                chord_scores[chord]['best_detection'] = detection

        # --- Hysteresis bonus ---
        # If we've been showing a chord stably, give it a small bonus
        # to prevent unnecessary switching on close scores.
        # Only apply if stability >= 2 (chord has been shown for 2+ windows)
        HYSTERESIS_BONUS = self._config_get('hysteresis_bonus', 0.15)
        if self.last_chord and self.chord_stability >= 2:
            if self.last_chord in chord_scores:
                chord_scores[self.last_chord]['weighted_confidence'] *= (1.0 + HYSTERESIS_BONUS)

        # Find the chord with highest weighted confidence score
        best_chord = None
        best_score = 0.0

        for chord, data in chord_scores.items():
            if data['weighted_confidence'] > best_score:
                best_score = data['weighted_confidence']
                best_chord = chord

        if best_chord is None:
            return None

        best_data = chord_scores[best_chord]
        best_detection = best_data['best_detection']

        return {
            'chord': best_chord,
            'confidence': best_data['raw_confidence'] / best_data['count'],
            'votes': best_data['count'],
            'total_votes': len(self.chord_accumulator),
            'notes': best_detection.get('notes'),
            'frequencies': best_detection.get('frequencies'),
            'chroma': best_detection.get('chroma')
        }

    def reset_accumulator(self):
        """Clear the chord accumulator and start a new window."""
        self.chord_accumulator = []
        self.chord_window_start = time.time()

