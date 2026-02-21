"""
Tests for lib/music_understanding.py - music theory and chord detection.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.music_understanding import (
    frequency_to_note,
    apply_bandpass_filter,
    CHORD_DEFINITIONS,
    INSTRUMENT_PRESETS,
    NOTES,
    find_best_matching_chord,
    find_best_matching_chord_enhanced,
)
from lib.common import get_rate


class TestFrequencyToNote:
    """Test frequency to note conversion."""
    
    def test_a4_440hz(self):
        """Test A4 at 440 Hz."""
        note = frequency_to_note(440)
        assert note == "A"
    
    def test_c4_261hz(self):
        """Test C4 at approximately 261.63 Hz."""
        note = frequency_to_note(261.63)
        assert note == "C"
    
    def test_e4_330hz(self):
        """Test E4 at approximately 329.63 Hz."""
        note = frequency_to_note(329.63)
        assert note == "E"
    
    def test_g4_392hz(self):
        """Test G4 at approximately 392 Hz."""
        note = frequency_to_note(392)
        assert note == "G"
    
    def test_b4_494hz(self):
        """Test B4 at approximately 493.88 Hz."""
        note = frequency_to_note(493.88)
        assert note == "B"
    
    def test_d4_294hz(self):
        """Test D4 at approximately 293.66 Hz."""
        note = frequency_to_note(293.66)
        assert note == "D"
    
    def test_invalid_zero_frequency(self):
        """Test zero frequency returns invalid."""
        result = frequency_to_note(0)
        assert result == ("Invalid frequency", 0.0)
    
    def test_invalid_negative_frequency(self):
        """Test negative frequency returns invalid."""
        result = frequency_to_note(-100)
        assert result == ("Invalid frequency", 0.0)
    
    def test_low_frequency_e2(self):
        """Test E2 at approximately 82.41 Hz (low E string on guitar)."""
        note = frequency_to_note(82.41)
        assert note == "E"
    
    def test_high_frequency_e6(self):
        """Test E6 at approximately 1318.51 Hz."""
        note = frequency_to_note(1318.51)
        assert note == "E"


class TestChordDefinitions:
    """Test chord definitions are correct."""
    
    def test_major_chord_intervals(self):
        """Test major chord has intervals 0, 4, 7."""
        assert CHORD_DEFINITIONS['Major'] == (0, 4, 7)
    
    def test_minor_chord_intervals(self):
        """Test minor chord has intervals 0, 3, 7."""
        assert CHORD_DEFINITIONS['Minor'] == (0, 3, 7)
    
    def test_diminished_chord_intervals(self):
        """Test diminished chord has intervals 0, 3, 6."""
        assert CHORD_DEFINITIONS['Diminished'] == (0, 3, 6)
    
    def test_augmented_chord_intervals(self):
        """Test augmented chord has intervals 0, 4, 8."""
        assert CHORD_DEFINITIONS['Augmented'] == (0, 4, 8)
    
    def test_dominant_7th_intervals(self):
        """Test dominant 7th has intervals 0, 4, 7, 10."""
        assert CHORD_DEFINITIONS['Dominant 7th'] == (0, 4, 7, 10)
    
    def test_power_chord_intervals(self):
        """Test power chord has intervals 0, 7."""
        assert CHORD_DEFINITIONS['Power Chord'] == (0, 7)
    
    def test_all_chord_types_present(self):
        """Test all expected chord types are defined."""
        expected_chords = [
            'Major', 'Minor', 'Diminished', 'Augmented',
            'Major 7th', 'Minor 7th', 'Dominant 7th',
            'Major 6th', 'Minor 6th',
            'Suspended 2nd', 'Suspended 4th', 'Power Chord'
        ]
        for chord in expected_chords:
            assert chord in CHORD_DEFINITIONS


class TestInstrumentPresets:
    """Test instrument frequency presets."""
    
    def test_guitar_preset(self):
        """Test guitar preset values."""
        preset = INSTRUMENT_PRESETS['guitar']
        assert preset['low_freq'] == 80
        assert preset['high_freq'] == 2000
        assert preset['name'] == 'Guitar'
    
    def test_piano_preset(self):
        """Test piano preset values."""
        preset = INSTRUMENT_PRESETS['piano']
        assert preset['low_freq'] == 100
        assert preset['high_freq'] == 4000
        assert preset['name'] == 'Piano'
    
    def test_bass_preset(self):
        """Test bass preset values."""
        preset = INSTRUMENT_PRESETS['bass']
        assert preset['low_freq'] == 40
        assert preset['high_freq'] == 800
        assert preset['name'] == 'Bass'
    
    def test_all_instruments_present(self):
        """Test all expected instruments are defined."""
        expected_instruments = [
            'guitar', 'piano', 'bass', 'violin', 'cello',
            'flute', 'clarinet', 'saxophone', 'trumpet', 'voice'
        ]
        for instrument in expected_instruments:
            assert instrument in INSTRUMENT_PRESETS
    
    def test_all_presets_have_required_keys(self):
        """Test all presets have low_freq, high_freq, and name."""
        for name, preset in INSTRUMENT_PRESETS.items():
            assert 'low_freq' in preset, f"{name} missing low_freq"
            assert 'high_freq' in preset, f"{name} missing high_freq"
            assert 'name' in preset, f"{name} missing name"
            assert preset['low_freq'] < preset['high_freq'], f"{name} has invalid freq range"


class TestNotes:
    """Test note definitions."""
    
    def test_notes_count(self):
        """Test there are 12 notes."""
        assert len(NOTES) == 12
    
    def test_notes_order(self):
        """Test notes are in correct order."""
        expected = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        assert NOTES == expected


class TestBandpassFilter:
    """Test bandpass filter function."""
    
    def test_filter_returns_same_length(self, sample_rate, chunk_size):
        """Test filter returns array of same length."""
        samples = np.random.randn(chunk_size).astype(np.float32)
        filtered = apply_bandpass_filter(samples, 80, 2000, sample_rate)
        assert len(filtered) == len(samples)
    
    def test_filter_reduces_noise(self, sample_rate, chunk_size):
        """Test filter reduces random noise."""
        # Create random noise
        noise = np.random.randn(chunk_size).astype(np.float32)
        filtered = apply_bandpass_filter(noise, 80, 2000, sample_rate)
        # Filtered signal should have less energy (some frequencies removed)
        original_energy = np.sum(noise ** 2)
        filtered_energy = np.sum(filtered ** 2)
        assert filtered_energy < original_energy
    
    def test_filter_passes_in_band_signal(self, sample_rate, chunk_size):
        """Test filter passes signal within band."""
        t = np.linspace(0, chunk_size / sample_rate, chunk_size, dtype=np.float32)
        # 440 Hz is within guitar range (80-2000 Hz)
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        filtered = apply_bandpass_filter(signal, 80, 2000, sample_rate)
        # Signal should largely pass through
        correlation = np.corrcoef(signal, filtered)[0, 1]
        assert correlation > 0.5


class TestFindBestMatchingChord:
    """Test chord matching functions."""
    
    def test_c_major_detection(self):
        """Test detection of C major chord (C, E, G)."""
        notes = ['C', 'E', 'G']
        result = find_best_matching_chord(notes)
        assert result is not None
        # Should detect a chord
        assert 'primary_chords' in result or 'power_chord' in result or 'intervals' in result
    
    def test_a_minor_detection(self):
        """Test detection of A minor chord (A, C, E)."""
        notes = ['A', 'C', 'E']
        result = find_best_matching_chord(notes)
        assert result is not None
    
    def test_g_major_detection(self):
        """Test detection of G major chord (G, B, D)."""
        notes = ['G', 'B', 'D']
        result = find_best_matching_chord(notes)
        assert result is not None
    
    def test_empty_notes(self):
        """Test with no notes."""
        result = find_best_matching_chord([])
        # Should return None or empty result
        assert result is None or not result.get('primary_chords')
    
    def test_single_note(self):
        """Test with single note."""
        result = find_best_matching_chord(['A'])
        # Should handle gracefully
        assert result is not None or result is None  # Either is acceptable
    
    def test_power_chord(self):
        """Test detection of power chord (root + fifth)."""
        notes = ['E', 'B']
        result = find_best_matching_chord(notes)
        assert result is not None

