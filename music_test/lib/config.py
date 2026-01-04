"""
Unified configuration interface for CLI and web.
"""
from lib.music_understanding import INSTRUMENT_PRESETS


class AudioConfig:
    """
    Unified configuration interface that wraps both argparse namespace (CLI) 
    and dictionary (web) configurations.
    """
    
    def __init__(self, source):
        """
        Initialize config from either argparse namespace or dictionary.
        
        Args:
            source: argparse.Namespace or dict
        """
        self._source = source
        self._is_dict = isinstance(source, dict)
    
    def get(self, key, default=None):
        """Get a configuration value."""
        if self._is_dict:
            return self._source.get(key, default)
        return getattr(self._source, key, default)
    
    def __getitem__(self, key):
        """Allow dict-like access."""
        return self.get(key)
    
    def __contains__(self, key):
        """Check if key exists."""
        if self._is_dict:
            return key in self._source
        return hasattr(self._source, key)
    
    # Common properties with consistent defaults (matching CLI)
    @property
    def instrument(self):
        return self.get('instrument', 'guitar')
    
    @property
    def sensitivity(self):
        return float(self.get('sensitivity', 1.0))
    
    @property
    def confidence_threshold(self):
        return float(self.get('confidence_threshold', 0.6))
    
    @property
    def silence_threshold(self):
        return float(self.get('silence_threshold', 0.005))
    
    @property
    def amplitude_threshold(self):
        return float(self.get('amplitude_threshold', 0.005))
    
    @property
    def overlap(self):
        return float(self.get('overlap', 0.75))
    
    @property
    def progression(self):
        return bool(self.get('progression', True))
    
    @property
    def multi_pitch(self):
        return bool(self.get('multi_pitch', True))
    
    @property
    def single_pitch(self):
        return bool(self.get('single_pitch', False))
    
    @property
    def show_frequencies(self):
        return bool(self.get('show_frequencies', False))
    
    @property
    def show_chroma(self):
        return bool(self.get('show_chroma', False))
    
    @property
    def show_fft(self):
        return bool(self.get('show_fft', False))
    
    @property
    def raw_frequencies(self):
        return bool(self.get('raw_frequencies', False))
    
    @property
    def frequencies_only(self):
        return bool(self.get('frequencies_only', False))
    
    @property
    def notes_only(self):
        return bool(self.get('notes_only', False))
    
    @property
    def debug(self):
        return bool(self.get('debug', False))
    
    @property
    def log(self):
        return bool(self.get('log', False))
    
    @property
    def log_interval(self):
        return float(self.get('log_interval', 0.5))
    
    @property
    def low_freq(self):
        """Get low frequency - from config or instrument preset."""
        freq = self.get('low_freq')
        if freq:
            return int(freq)
        preset = INSTRUMENT_PRESETS.get(self.instrument, INSTRUMENT_PRESETS['guitar'])
        return preset['low_freq']
    
    @property
    def high_freq(self):
        """Get high frequency - from config or instrument preset."""
        freq = self.get('high_freq')
        if freq:
            return int(freq)
        preset = INSTRUMENT_PRESETS.get(self.instrument, INSTRUMENT_PRESETS['guitar'])
        return preset['high_freq']
    
    @property
    def instrument_name(self):
        """Get instrument display name."""
        name = self.get('instrument_name')
        if name:
            return name
        preset = INSTRUMENT_PRESETS.get(self.instrument, INSTRUMENT_PRESETS['guitar'])
        return preset['name']
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            'instrument': self.instrument,
            'sensitivity': self.sensitivity,
            'confidence_threshold': self.confidence_threshold,
            'silence_threshold': self.silence_threshold,
            'amplitude_threshold': self.amplitude_threshold,
            'overlap': self.overlap,
            'progression': self.progression,
            'multi_pitch': self.multi_pitch,
            'single_pitch': self.single_pitch,
            'show_frequencies': self.show_frequencies,
            'show_chroma': self.show_chroma,
            'show_fft': self.show_fft,
            'raw_frequencies': self.raw_frequencies,
            'frequencies_only': self.frequencies_only,
            'notes_only': self.notes_only,
            'debug': self.debug,
            'log': self.log,
            'log_interval': self.log_interval,
            'low_freq': self.low_freq,
            'high_freq': self.high_freq,
            'instrument_name': self.instrument_name,
        }

