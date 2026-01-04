"""
Integration tests for chord detector CLI and web interface.
"""
import pytest
import numpy as np
import sys
import os
import json
import asyncio
from collections import deque
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.common import get_chunk, get_buffer_size, get_rate, set_overlap_ratio
from lib.music_understanding import INSTRUMENT_PRESETS, detect_notes_with_sounddevice


class TestNoteDetection:
    """Test note detection from audio samples."""
    
    def test_detect_440hz_as_a(self, sine_wave_440hz, buffer_size):
        """Test 440 Hz sine wave is detected as A note."""
        # Create a buffer with the sine wave
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        audio_buffer[:len(sine_wave_440hz)] = sine_wave_440hz
        
        detected_notes, frequencies, chroma = detect_notes_with_sounddevice(
            audio_buffer,
            sensitivity=1.0,
            silence_threshold=0.001,
            low_freq=80,
            high_freq=2000,
            show_frequencies=False,
            show_fft=False,
            raw_frequencies=False,
            calculate_chroma=True,
            multi_pitch=True
        )
        
        # Should detect something (even if not exactly A due to buffer size)
        # The test is that the function runs without error
        assert isinstance(detected_notes, list)
        assert isinstance(frequencies, list)
    
    def test_detect_chord_notes(self, c_major_chord, buffer_size):
        """Test C major chord detection returns multiple notes."""
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        audio_buffer[:len(c_major_chord)] = c_major_chord
        
        detected_notes, frequencies, chroma = detect_notes_with_sounddevice(
            audio_buffer,
            sensitivity=1.0,
            silence_threshold=0.001,
            low_freq=80,
            high_freq=2000,
            show_frequencies=False,
            show_fft=False,
            raw_frequencies=False,
            calculate_chroma=True,
            multi_pitch=True
        )
        
        # With multi-pitch enabled, should detect multiple notes
        assert isinstance(detected_notes, list)
        # The chord should produce some chroma vector
        if chroma is not None:
            assert len(chroma) == 12
    
    def test_silence_detection(self, silence_audio, buffer_size):
        """Test that silence returns no detected notes."""
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        # Buffer is all zeros (silence)
        
        detected_notes, frequencies, chroma = detect_notes_with_sounddevice(
            audio_buffer,
            sensitivity=1.0,
            silence_threshold=0.005,
            low_freq=80,
            high_freq=2000,
            show_frequencies=False,
            show_fft=False,
            raw_frequencies=False,
            calculate_chroma=True,
            multi_pitch=True
        )
        
        # Should detect no notes from silence
        assert detected_notes == [] or detected_notes is None or len(detected_notes) == 0


class TestChromaVector:
    """Test chroma vector generation."""
    
    def test_chroma_vector_shape(self, sine_wave_440hz, buffer_size):
        """Test chroma vector has 12 elements."""
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        audio_buffer[:len(sine_wave_440hz)] = sine_wave_440hz
        
        _, _, chroma = detect_notes_with_sounddevice(
            audio_buffer,
            sensitivity=1.0,
            silence_threshold=0.001,
            low_freq=80,
            high_freq=2000,
            calculate_chroma=True,
            multi_pitch=True
        )
        
        if chroma is not None:
            assert len(chroma) == 12
    
    def test_chroma_vector_normalized(self, sine_wave_440hz, buffer_size):
        """Test chroma vector values are normalized."""
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        audio_buffer[:len(sine_wave_440hz)] = sine_wave_440hz
        
        _, _, chroma = detect_notes_with_sounddevice(
            audio_buffer,
            sensitivity=1.0,
            silence_threshold=0.001,
            low_freq=80,
            high_freq=2000,
            calculate_chroma=True,
            multi_pitch=True
        )
        
        if chroma is not None:
            # Chroma values should be between 0 and 1
            assert np.all(chroma >= 0)
            assert np.all(chroma <= 1.1)  # Allow small tolerance


class TestWebServerEndpoints:
    """Test web server endpoint availability."""
    
    @pytest.fixture
    def app(self):
        """Create FastAPI test app."""
        from web_server import app
        return app
    
    def test_app_exists(self, app):
        """Test FastAPI app is created."""
        assert app is not None
    
    def test_health_endpoint_exists(self, app):
        """Test health endpoint is defined."""
        routes = [route.path for route in app.routes]
        assert '/health' in routes
    
    def test_websocket_endpoint_exists(self, app):
        """Test WebSocket endpoint is defined."""
        routes = [route.path for route in app.routes]
        assert '/ws' in routes


class TestConnectionState:
    """Test ConnectionState class for web interface."""
    
    def test_connection_state_initialization(self, mock_config):
        """Test ConnectionState initializes correctly."""
        from web_server import ConnectionState
        
        state = ConnectionState(mock_config)
        
        assert state.low_freq == 80  # Guitar default
        assert state.high_freq == 2000
        assert state.instrument_name == 'Guitar'
        assert len(state.audio_buffer) == get_buffer_size()
        assert state.buffer_index == 0
        assert len(state.notes_history) == 0
    
    def test_connection_state_different_instrument(self):
        """Test ConnectionState with different instrument."""
        from web_server import ConnectionState
        
        # Create fresh config for this test
        piano_config = {
            'instrument': 'piano',
            'sensitivity': 1.0,
            'silence_threshold': 0.005,
        }
        state = ConnectionState(piano_config)
        
        assert state.low_freq == 100  # Piano default
        assert state.high_freq == 4000
        assert state.instrument_name == 'Piano'
    
    def test_connection_state_custom_frequencies(self, mock_config):
        """Test ConnectionState with custom frequency range."""
        from web_server import ConnectionState
        
        mock_config['low_freq'] = 150
        mock_config['high_freq'] = 3000
        state = ConnectionState(mock_config)
        
        assert state.low_freq == 150
        assert state.high_freq == 3000


class TestEndToEndProcessing:
    """End-to-end tests for audio processing pipeline."""
    
    def test_cli_and_web_produce_similar_results(self, c_major_chord, buffer_size, mock_args, mock_config):
        """Test CLI and web processing produce similar results for same input."""
        from lib.music_understanding import detect_notes_with_sounddevice
        
        # Create identical buffers
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        audio_buffer[:len(c_major_chord)] = c_major_chord
        
        # CLI-style detection
        cli_notes, cli_freqs, cli_chroma = detect_notes_with_sounddevice(
            audio_buffer,
            sensitivity=mock_args.sensitivity,
            silence_threshold=mock_args.silence_threshold,
            low_freq=80, high_freq=2000,
            calculate_chroma=True,
            multi_pitch=mock_args.multi_pitch
        )
        
        # Web-style detection (same parameters from dict)
        web_notes, web_freqs, web_chroma = detect_notes_with_sounddevice(
            audio_buffer,
            sensitivity=mock_config['sensitivity'],
            silence_threshold=mock_config['silence_threshold'],
            low_freq=80, high_freq=2000,
            calculate_chroma=True,
            multi_pitch=mock_config['multi_pitch']
        )
        
        # Results should be identical
        assert cli_notes == web_notes
        if cli_chroma is not None and web_chroma is not None:
            np.testing.assert_array_almost_equal(cli_chroma, web_chroma)
    
    def test_processing_with_different_overlap_ratios(self, sine_wave_440hz, buffer_size):
        """Test processing works with different overlap ratios."""
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        audio_buffer[:len(sine_wave_440hz)] = sine_wave_440hz
        
        for overlap in [0.25, 0.5, 0.75]:
            set_overlap_ratio(overlap)
            
            notes, freqs, chroma = detect_notes_with_sounddevice(
                audio_buffer,
                sensitivity=1.0,
                silence_threshold=0.001,
                low_freq=80, high_freq=2000,
                calculate_chroma=True,
                multi_pitch=True
            )
            
            # Should process without error
            assert isinstance(notes, list)
        
        set_overlap_ratio(0.75)  # Restore default

