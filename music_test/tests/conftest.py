"""
Pytest fixtures for chord detector tests.
"""
import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.common import get_rate, get_chunk, get_buffer_size


@pytest.fixture
def sample_rate():
    """Standard sample rate."""
    return get_rate()


@pytest.fixture
def chunk_size():
    """Standard chunk size."""
    return get_chunk()


@pytest.fixture
def buffer_size():
    """Standard buffer size."""
    return get_buffer_size()


@pytest.fixture
def silence_audio(chunk_size):
    """Generate silent audio samples."""
    return np.zeros(chunk_size, dtype=np.float32)


@pytest.fixture
def sine_wave_440hz(sample_rate, chunk_size):
    """Generate a 440Hz sine wave (A4 note)."""
    t = np.linspace(0, chunk_size / sample_rate, chunk_size, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)


@pytest.fixture
def sine_wave_261hz(sample_rate, chunk_size):
    """Generate a 261.63Hz sine wave (C4 note)."""
    t = np.linspace(0, chunk_size / sample_rate, chunk_size, dtype=np.float32)
    return 0.5 * np.sin(2 * np.pi * 261.63 * t).astype(np.float32)


@pytest.fixture
def c_major_chord(sample_rate, chunk_size):
    """Generate a C major chord (C4, E4, G4)."""
    t = np.linspace(0, chunk_size / sample_rate, chunk_size, dtype=np.float32)
    c4 = 261.63  # C4
    e4 = 329.63  # E4
    g4 = 392.00  # G4
    
    signal = (
        0.33 * np.sin(2 * np.pi * c4 * t) +
        0.33 * np.sin(2 * np.pi * e4 * t) +
        0.33 * np.sin(2 * np.pi * g4 * t)
    )
    return signal.astype(np.float32)


@pytest.fixture
def a_minor_chord(sample_rate, chunk_size):
    """Generate an A minor chord (A4, C5, E5)."""
    t = np.linspace(0, chunk_size / sample_rate, chunk_size, dtype=np.float32)
    a4 = 440.00  # A4
    c5 = 523.25  # C5
    e5 = 659.25  # E5
    
    signal = (
        0.33 * np.sin(2 * np.pi * a4 * t) +
        0.33 * np.sin(2 * np.pi * c5 * t) +
        0.33 * np.sin(2 * np.pi * e5 * t)
    )
    return signal.astype(np.float32)


@pytest.fixture
def mock_args():
    """Create mock argparse namespace for CLI tests."""
    class MockArgs:
        def __init__(self):
            self.instrument = 'guitar'
            self.sensitivity = 1.0
            self.silence_threshold = 0.005
            self.amplitude_threshold = 0.005
            self.confidence_threshold = 0.6
            self.overlap = 0.75
            self.progression = True
            self.multi_pitch = True
            self.single_pitch = False
            self.show_frequencies = False
            self.show_chroma = False
            self.show_fft = False
            self.raw_frequencies = False
            self.frequencies_only = False
            self.notes_only = False
            self.debug = False
            self.log = False
            self.log_interval = 0.5
            self.wait_time = 0.0
            self.low_freq = None
            self.high_freq = None
    
    return MockArgs()


@pytest.fixture
def mock_config():
    """Create mock config dictionary for web tests."""
    return {
        'instrument': 'guitar',
        'sensitivity': 1.0,
        'silence_threshold': 0.005,
        'amplitude_threshold': 0.005,
        'confidence_threshold': 0.6,
        'overlap': 0.75,
        'progression': True,
        'multi_pitch': True,
        'single_pitch': False,
        'show_frequencies': False,
        'show_chroma': False,
        'show_fft': False,
        'raw_frequencies': False,
        'frequencies_only': False,
        'notes_only': False,
        'debug': False,
        'log': False,
        'log_interval': 0.5,
    }

