"""
Comprehensive chord detection accuracy tests using synthetic audio.

Tests chord recognition across all major chord types, including
ambiguous cases (Am7/C6, Dm7/F6) that previously failed.
"""
import sys
import os
import numpy as np
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.music_understanding import (
    detect_chord_from_buffer,
    chroma_from_fft,
    _match_chroma_to_chord,
    NOTES,
)
from lib.common import get_rate, get_chunk, get_buffer_size


# --- Synthetic chord generator ---

# Standard note frequencies (octave 2-5)
NOTE_FREQS = {}
for octave in range(2, 6):
    for i, note in enumerate(NOTES):
        midi = (octave + 1) * 12 + i
        freq = 440.0 * (2 ** ((midi - 69) / 12.0))
        NOTE_FREQS[f"{note}{octave}"] = freq

# Chord voicings: (root_note, intervals_in_semitones, octave_for_root)
CHORD_VOICINGS = {
    # Major triads
    'C':    ('C', [0, 4, 7], 3),
    'D':    ('D', [0, 4, 7], 3),
    'E':    ('E', [0, 4, 7], 2),
    'F':    ('F', [0, 4, 7], 3),
    'G':    ('G', [0, 4, 7], 2),
    'A':    ('A', [0, 4, 7], 2),
    'B':    ('B', [0, 4, 7], 2),
    # Minor triads
    'Cm':   ('C', [0, 3, 7], 3),
    'Dm':   ('D', [0, 3, 7], 3),
    'Em':   ('E', [0, 3, 7], 2),
    'Am':   ('A', [0, 3, 7], 2),
    'Bm':   ('B', [0, 3, 7], 2),
    # Seventh chords
    'G7':   ('G', [0, 4, 7, 10], 2),
    'D7':   ('D', [0, 4, 7, 10], 3),
    'A7':   ('A', [0, 4, 7, 10], 2),
    'E7':   ('E', [0, 4, 7, 10], 2),
    'Cmaj7':('C', [0, 4, 7, 11], 3),
    'Dm7':  ('D', [0, 3, 7, 10], 3),
    'Em7':  ('E', [0, 3, 7, 10], 2),
    'Am7':  ('A', [0, 3, 7, 10], 2),
    # Sus chords
    'Dsus2':('D', [0, 2, 7], 3),
    'Dsus4':('D', [0, 5, 7], 3),
    'Asus2':('A', [0, 2, 7], 2),
    'Asus4':('A', [0, 5, 7], 2),
    # Power chords
    'E5':   ('E', [0, 7], 2),
    'A5':   ('A', [0, 7], 2),
    # Sixth chords (the ambiguous ones)
    'C6':   ('C', [0, 4, 7, 9], 3),   # C E G A — same pitch classes as Am7
    'F6':   ('F', [0, 4, 7, 9], 3),   # F A C D — same pitch classes as Dm7
}


def generate_guitar_chord(chord_name, duration=0.2, sample_rate=44100):
    """
    Generate a synthetic guitar chord with realistic harmonics and envelope.

    Each note gets harmonics (2x, 3x, 4x, 5x) with decreasing amplitude.
    The root note is made louder (1.5x) to simulate typical guitar voicing.
    An ADSR-like envelope is applied for realism.
    """
    if chord_name not in CHORD_VOICINGS:
        raise ValueError(f"Unknown chord: {chord_name}")

    root_name, intervals, base_octave = CHORD_VOICINGS[chord_name]

    # Get root MIDI note
    root_idx = NOTES.index(root_name)
    root_midi = (base_octave + 1) * 12 + root_idx

    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float64)
    signal = np.zeros_like(t)

    for i, interval in enumerate(intervals):
        midi = root_midi + interval
        freq = 440.0 * (2 ** ((midi - 69) / 12.0))

        # Root note is louder (simulates typical guitar voicing where bass note rings out)
        base_amp = 1.5 if i == 0 else 1.0

        # Add fundamental + harmonics (guitar has strong harmonics)
        for h in range(1, 6):  # harmonics 1-5
            harmonic_freq = freq * h
            if harmonic_freq > sample_rate / 2:
                break
            # Harmonic amplitude decreases with order
            harmonic_amp = base_amp / (h ** 1.2)
            signal += harmonic_amp * np.sin(2 * np.pi * harmonic_freq * t)

    # Apply guitar-like envelope (fast attack, slow decay)
    attack = int(0.005 * sample_rate)
    decay = int(0.05 * sample_rate)
    envelope = np.ones_like(t)
    if attack > 0:
        envelope[:attack] = np.linspace(0, 1, attack)
    if decay > 0 and len(envelope) > attack + decay:
        decay_start = attack
        decay_end = min(attack + decay, len(envelope))
        envelope[decay_start:decay_end] = np.linspace(1.0, 0.8, decay_end - decay_start)
        # Slow sustain decay
        if decay_end < len(envelope):
            envelope[decay_end:] = 0.8 * np.exp(-2.0 * np.linspace(0, duration, len(envelope) - decay_end))

    signal *= envelope

    # Normalize
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.7

    return signal.astype(np.float32)


def _detect_and_verify_chord(chord_name, expected_name=None):
    """
    Helper function to detect and verify a single chord.
    Returns (detected_chord, confidence, correct).
    """
    if expected_name is None:
        expected_name = chord_name

    audio = generate_guitar_chord(chord_name)

    # Pad to buffer size
    buffer_size = get_buffer_size()
    if len(audio) < buffer_size:
        audio_buffer = np.zeros(buffer_size, dtype=np.float32)
        audio_buffer[:len(audio)] = audio
    else:
        audio_buffer = audio[:buffer_size]

    chord, confidence, chroma, notes, freqs = detect_chord_from_buffer(
        audio_buffer,
        sample_rate=get_rate(),
        silence_threshold=0.001,
        low_freq=80,
        high_freq=2000
    )

    correct = chord == expected_name
    return chord, confidence, correct


def run_accuracy_suite():
    """Run all chord detection tests and report accuracy."""

    # Test groups with expected results
    test_cases = [
        # (chord_voicing_name, expected_detection)
        # --- Major triads ---
        ('C', 'C'),
        ('D', 'D'),
        ('E', 'E'),
        ('F', 'F'),
        ('G', 'G'),
        ('A', 'A'),
        ('B', 'B'),
        # --- Minor triads ---
        ('Cm', 'Cm'),
        ('Dm', 'Dm'),
        ('Em', 'Em'),
        ('Am', 'Am'),
        ('Bm', 'Bm'),
        # --- Dominant 7th ---
        ('G7', 'G7'),
        ('D7', 'D7'),
        ('A7', 'A7'),
        ('E7', 'E7'),
        # --- Minor 7th ---
        ('Dm7', 'Dm7'),
        ('Em7', 'Em7'),
        ('Am7', 'Am7'),   # KEY TEST: Am7 vs C6 disambiguation
        # --- Major 7th ---
        ('Cmaj7', 'Cmaj7'),
        # --- Sus chords ---
        ('Dsus2', 'Dsus2'),
        ('Dsus4', 'Dsus4'),
        ('Asus2', 'Asus2'),
        ('Asus4', 'Asus4'),
        # --- Power chords ---
        ('E5', 'E5'),
        ('A5', 'A5'),
        # --- Sixth chords (ambiguous cases) ---
        ('C6', 'C6'),     # KEY TEST: C6 vs Am7 disambiguation
        ('F6', 'F6'),     # KEY TEST: F6 vs Dm7 disambiguation
    ]

    results = []
    passed = 0
    failed = 0

    print("=" * 70)
    print("CHORD DETECTION ACCURACY TEST SUITE")
    print("=" * 70)
    print()

    for voicing, expected in test_cases:
        detected, confidence, correct = _detect_and_verify_chord(voicing, expected)
        status = "PASS" if correct else "FAIL"
        results.append((voicing, expected, detected, confidence, correct))

        if correct:
            passed += 1
        else:
            failed += 1

        marker = "  " if correct else ">>"
        print(f"  {marker} {voicing:8s} -> expected: {expected:8s}  detected: {str(detected):8s}  conf: {confidence:.3f}  [{status}]")

    print()
    print("-" * 70)
    total = len(test_cases)
    accuracy = passed / total * 100 if total > 0 else 0
    print(f"RESULTS: {passed}/{total} passed ({accuracy:.1f}% accuracy)")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print()

    # Report failures
    failures = [(v, e, d, c) for v, e, d, c, ok in results if not ok]
    if failures:
        print("FAILURES:")
        for voicing, expected, detected, conf in failures:
            print(f"  {voicing}: expected {expected}, got {detected} (conf={conf:.3f})")

    print("=" * 70)

    return passed, total, accuracy


def run_bass_disambiguation_test():
    """
    Specific test for Am7/C6 and Dm7/F6 bass note disambiguation.
    These chords have identical pitch classes but different roots.
    The bass note should disambiguate them.
    """
    print()
    print("=" * 70)
    print("BASS NOTE DISAMBIGUATION TEST")
    print("=" * 70)
    print()

    test_pairs = [
        ('Am7', 'C6'),    # {A, C, E, G} - bass A vs bass C
        ('Dm7', 'F6'),    # {D, F, A, C} - bass D vs bass F
    ]

    for chord_a, chord_b in test_pairs:
        print(f"  Testing: {chord_a} vs {chord_b}")

        # Test chord_a
        det_a, conf_a, correct_a = _detect_and_verify_chord(chord_a)
        marker_a = "PASS" if correct_a else "FAIL"
        print(f"    {chord_a} voicing -> detected: {det_a} (conf: {conf_a:.3f}) [{marker_a}]")

        # Test chord_b
        det_b, conf_b, correct_b = _detect_and_verify_chord(chord_b)
        marker_b = "PASS" if correct_b else "FAIL"
        print(f"    {chord_b} voicing -> detected: {det_b} (conf: {conf_b:.3f}) [{marker_b}]")

        both_correct = correct_a and correct_b
        print(f"    Disambiguation: {'SUCCESS' if both_correct else 'FAILED'}")
        print()

    print("=" * 70)


def run_chroma_direct_test():
    """
    Test chroma_from_fft directly to verify bass note detection.
    """
    print()
    print("=" * 70)
    print("CHROMA & BASS NOTE DETECTION TEST")
    print("=" * 70)
    print()

    for chord_name in ['Am7', 'C6', 'Em', 'C', 'G', 'E7']:
        audio = generate_guitar_chord(chord_name)

        # Use first chunk
        chunk_size = min(get_chunk(), len(audio))
        chunk = audio[:chunk_size]

        chroma, bass_pc = chroma_from_fft(chunk, get_rate(), 80, 2000)

        bass_note = NOTES[bass_pc] if bass_pc >= 0 else "None"

        # Show chroma vector
        active_notes = [(NOTES[i], f"{chroma[i]:.3f}") for i in range(12) if chroma[i] > 0.1]

        root_name = CHORD_VOICINGS[chord_name][0]
        bass_matches_root = bass_note == root_name

        print(f"  {chord_name:6s} | bass: {bass_note:3s} (root={root_name}) {'OK' if bass_matches_root else 'MISMATCH'} | active: {active_notes}")

    print()
    print("=" * 70)


def test_chord_accuracy_suite():
    """Pytest-compatible wrapper for accuracy test suite."""
    passed, total, accuracy = run_accuracy_suite()
    assert accuracy >= 85.0, f"Accuracy {accuracy:.1f}% below 85% threshold"
    assert passed >= 24, f"Only {passed}/{total} chords passed (expected ≥24)"


def test_bass_disambiguation():
    """Pytest-compatible test for Am7/C6 and Dm7/F6 disambiguation."""
    # Test Am7
    det_am7, conf_am7, ok_am7 = _detect_and_verify_chord('Am7', 'Am7')
    assert ok_am7, f"Am7 failed: detected {det_am7} with confidence {conf_am7:.3f}"

    # Test C6
    det_c6, conf_c6, ok_c6 = _detect_and_verify_chord('C6', 'C6')
    assert ok_c6, f"C6 failed: detected {det_c6} with confidence {conf_c6:.3f}"

    # Test Dm7
    det_dm7, conf_dm7, ok_dm7 = _detect_and_verify_chord('Dm7', 'Dm7')
    assert ok_dm7, f"Dm7 failed: detected {det_dm7} with confidence {conf_dm7:.3f}"

    # Test F6
    det_f6, conf_f6, ok_f6 = _detect_and_verify_chord('F6', 'F6')
    assert ok_f6, f"F6 failed: detected {det_f6} with confidence {conf_f6:.3f}"


if __name__ == '__main__':
    # Run direct chroma/bass test
    run_chroma_direct_test()

    # Run bass disambiguation test
    run_bass_disambiguation_test()

    # Run full accuracy suite
    passed, total, accuracy = run_accuracy_suite()

    # Exit with error if accuracy is below threshold
    if accuracy < 85:
        print(f"\nACCURACY BELOW THRESHOLD: {accuracy:.1f}% < 85%")
        sys.exit(1)
    else:
        print(f"\nAccuracy meets threshold: {accuracy:.1f}% >= 85%")
        sys.exit(0)
