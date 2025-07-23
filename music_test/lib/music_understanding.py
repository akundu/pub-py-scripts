import math
import numpy as np
import sys
from scipy.signal import butter, filtfilt, correlate
from common import get_chunk, get_hop_size, get_rate, get_channels, get_buffer_size

# Enhanced chord definitions with more comprehensive coverage
CHORD_DEFINITIONS = {
    'Major': (0, 4, 7),
    'Minor': (0, 3, 7),
    'Diminished': (0, 3, 6),
    'Augmented': (0, 4, 8),
    'Major 7th': (0, 4, 7, 11),
    'Minor 7th': (0, 3, 7, 10),
    'Dominant 7th': (0, 4, 7, 10),
    'Major 6th': (0, 4, 7, 9),
    'Minor 6th': (0, 3, 7, 9),
    'Suspended 2nd': (0, 2, 7),
    'Suspended 4th': (0, 5, 7),
    'Power Chord': (0, 7),
}

# Instrument frequency range presets
INSTRUMENT_PRESETS = {
    'guitar': {'low_freq': 80, 'high_freq': 2000, 'name': 'Guitar'},
    'piano': {'low_freq': 100, 'high_freq': 4000, 'name': 'Piano'},
    'bass': {'low_freq': 40, 'high_freq': 800, 'name': 'Bass'},
    'violin': {'low_freq': 200, 'high_freq': 3000, 'name': 'Violin'},
    'cello': {'low_freq': 65, 'high_freq': 1000, 'name': 'Cello'},
    'flute': {'low_freq': 250, 'high_freq': 2500, 'name': 'Flute'},
    'clarinet': {'low_freq': 150, 'high_freq': 1500, 'name': 'Clarinet'},
    'saxophone': {'low_freq': 100, 'high_freq': 800, 'name': 'Saxophone'},
    'trumpet': {'low_freq': 150, 'high_freq': 1000, 'name': 'Trumpet'},
    'voice': {'low_freq': 80, 'high_freq': 1000, 'name': 'Voice'},
}

# --- 2. Core Functions ---
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
def frequency_to_note(freq):
    global NOTES

    # Calculate the MIDI note number (A4 = 69, 440 Hz)
    if freq <= 0:
        return "Invalid frequency", 0.0

    midi_note = 12 * math.log2(freq / 440) + 69
    rounded_midi = round(midi_note)

    # Get octave and note index
    octave = (
        rounded_midi // 12
    ) - 1  # C0 is MIDI 12, but octaves start from -1 for very low notes
    note_index = rounded_midi % 12
    note_name = NOTES[note_index]

    # Calculate the exact frequency of this note for reference
    exact_freq = 440 * (2 ** ((rounded_midi - 69) / 12))

    # print ( f"{note_name}{octave}", round(exact_freq, 2), file=sys.stderr)
    #return f"{note_name}{octave}", round(exact_freq, 2)
    return note_name


def apply_bandpass_filter(samples, low_freq=80, high_freq=2000, sample_rate=get_rate()):
    """
    Apply bandpass filter to focus on instrument frequency range.
    """
    nyquist = sample_rate / 2
    low = low_freq / nyquist
    high = high_freq / nyquist
    
    b, a = butter(4, [low, high], btype='band')
    filtered_samples = filtfilt(b, a, samples)
    
    return filtered_samples

def get_pitch_and_amplitude(audio_data, sr):
    """
    Analyzes audio data to find the fundamental frequency (pitch)
    using autocorrelation and calculates the RMS amplitude.
    """
    if len(audio_data) == 0:
        return None, None, "No audio data to analyze."

    # Ensure data is mono and float32
    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0] # Take the first channel if stereo
    audio_data = audio_data.astype(np.float32)

    # --- Amplitude (RMS) Calculation ---
    # RMS is the square root of the mean of the squares of the values.
    # It's a good measure of the "power" or "loudness" of the signal.
    rms_amplitude = np.sqrt(np.mean(audio_data**2))

    # --- Pitch (Frequency) Detection using Autocorrelation ---
    # Autocorrelation measures the similarity between a signal and delayed copies of itself.
    # A peak in autocorrelation indicates a strong periodic component.

    # 1. Compute Autocorrelation
    # 'full' mode returns convolution at each point of overlap. We only care about positive lags.
    # The output length will be (len(audio_data) * 2) - 1. We'll take the relevant part.
    autocorr = correlate(audio_data, audio_data, mode='full')
    autocorr = autocorr[len(audio_data) - 1:] # Take only the positive lags (and lag 0)

    # Normalize autocorrelation for better peak detection
    autocorr = autocorr / np.max(np.abs(autocorr))

    # 2. Determine search range for lags corresponding to guitar frequencies
    # Lag = samplerate / frequency
    min_lag = int(sr / 1000) # Corresponds to highest frequency (1000 Hz)
    max_lag = int(sr / 50)   # Corresponds to lowest frequency (50 Hz)

    # Ensure lag range is valid for the signal length
    max_lag = min(max_lag, len(autocorr) - 1)
    if min_lag >= max_lag:
        return None, rms_amplitude, "Audio data too short or frequency range too broad for pitch detection."

    # 3. Find the peak in the autocorrelation within the relevant lag range
    # Exclude lag 0, as it's always the highest (signal perfectly correlates with itself).
    # We are looking for the next significant peak.
    search_window = autocorr[min_lag : max_lag + 1]
    
    if len(search_window) == 0:
        return None, rms_amplitude, "No valid search window for pitch detection."

    # Find the index of the maximum value within the search window
    peak_index_in_window = np.argmax(search_window)
    
    # Convert back to the original autocorrelation array index
    fundamental_lag = min_lag + peak_index_in_window

    if fundamental_lag == 0: # Should not happen if min_lag > 0, but a safeguard
        return None, rms_amplitude, "Could not find a significant fundamental frequency peak."

    # 4. Calculate fundamental frequency from the lag
    fundamental_frequency = sr / fundamental_lag

    return fundamental_frequency, rms_amplitude, None

def detect_notes_with_sounddevice(audio_buffer, sample_rate=get_rate(), sensitivity=1.0, 
                                 silence_threshold=200, low_freq=80, high_freq=2000, 
                                 show_frequencies=False, show_fft=False, raw_frequencies=False):
    """
    Enhanced note detection using sounddevice and autocorrelation.
    """
    detected_notes_all = []
    detected_frequencies_all = []

    # Process overlapping windows
    for i in range(0, len(audio_buffer) - get_chunk(), get_hop_size()):
        # Extract window
        window_samples = audio_buffer[i:i+get_chunk()]

        # Check for silence/low volume
        rms_energy = np.sqrt(np.mean(window_samples.astype(np.float64)**2))
        if rms_energy < silence_threshold:
            continue

        # Get pitch and amplitude using autocorrelation
        frequency, amplitude, error_message = get_pitch_and_amplitude(window_samples, sample_rate)
        if error_message or frequency is None:
            continue

        # Check if frequency is in the desired range (with a buffer) based on the instrument's frequency range
        BUFFER_pct = 0.25
        if not raw_frequencies and (frequency < low_freq*BUFFER_pct or frequency > high_freq*BUFFER_pct):
            continue
        if show_frequencies:
            note = frequency_to_note(frequency)
            print(f"frequency: {frequency:.1f} Hz -> {note}, amplitude: {amplitude:.4f}", file=sys.stderr)

        # Convert frequency to note
        note = frequency_to_note(frequency)
        if note:
            detected_notes_all.append(note)
            detected_frequencies_all.append((frequency, amplitude))

            # Show FFT data if requested
            if show_fft:
                print(f"\n🔍 Autocorrelation Analysis (Window {i//get_hop_size() + 1}):")
                print(f"  Fundamental Frequency: {frequency:.1f} Hz")
                print(f"  Amplitude (RMS): {amplitude:.4f}")
                print(f"  Note: {note}")

    # Return unique notes
    unique_notes = list(set(detected_notes_all))

    if show_frequencies and detected_frequencies_all:
        # Group frequencies by note
        note_frequencies = {}
        for freq, magnitude in detected_frequencies_all:
            note = frequency_to_note(freq)
            if note:
                if note not in note_frequencies:
                    note_frequencies[note] = []
                note_frequencies[note].append((freq, magnitude))

        # Print frequency information
        print("\n📊 Detected Frequencies:")
        for note in sorted(note_frequencies.keys()):
            freqs = note_frequencies[note]
            avg_freq = np.mean([f[0] for f in freqs])
            max_magnitude = max([f[1] for f in freqs])
            print(f"  {note}: {avg_freq:.1f} Hz (amplitude: {max_magnitude:.4f})")

    return unique_notes, detected_frequencies_all

def advanced_chord_identifier(notes, bass_note=None, key_context=None):
    """
    Advanced chord identification function for guitar and general music.

    Args:
        notes: List of note names (e.g., ['C4', 'E4', 'G4'] or ['C', 'E', 'G'])
        bass_note: Optional bass note for slash chords (e.g., 'E' for C/E)
        key_context: Optional key context to help with ambiguous chords (e.g., 'C major')

    Returns:
        Dictionary with chord analysis including primary matches, alternatives, intervals, etc.
    """

    # Normalize notes - remove octave numbers and convert to uppercase
    def normalize_note(note):
        clean_note = "".join([c for c in note if not c.isdigit()]).upper().strip()
        # Handle enharmonic equivalents
        enharmonic_map = {"DB": "C#", "EB": "D#", "GB": "F#", "AB": "G#", "BB": "A#"}
        return enharmonic_map.get(clean_note, clean_note)

    normalized_notes = [normalize_note(note) for note in notes]
    unique_notes = list(
        dict.fromkeys(normalized_notes)
    )  # Remove duplicates while preserving order

    # Note to semitone mapping
    note_map = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }

    # Convert to semitones
    try:
        semitones = [note_map[note] for note in unique_notes]
    except KeyError as e:
        return {"error": f"Invalid note: {e}"}

    semitones = sorted(list(set(semitones)))
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Comprehensive chord patterns
    chord_patterns = {
        # Basic triads
        "major": {"intervals": [0, 4, 7], "symbol": "", "required": [0, 4, 7]},
        "minor": {"intervals": [0, 3, 7], "symbol": "m", "required": [0, 3, 7]},
        "diminished": {"intervals": [0, 3, 6], "symbol": "°", "required": [0, 3, 6]},
        "augmented": {"intervals": [0, 4, 8], "symbol": "+", "required": [0, 4, 8]},
        "sus2": {"intervals": [0, 2, 7], "symbol": "sus2", "required": [0, 2, 7]},
        "sus4": {"intervals": [0, 5, 7], "symbol": "sus4", "required": [0, 5, 7]},
        # 7th chords
        "major7": {
            "intervals": [0, 4, 7, 11],
            "symbol": "maj7",
            "required": [0, 4, 11],
        },
        "minor7": {"intervals": [0, 3, 7, 10], "symbol": "m7", "required": [0, 3, 10]},
        "dominant7": {
            "intervals": [0, 4, 7, 10],
            "symbol": "7",
            "required": [0, 4, 10],
        },
        "diminished7": {
            "intervals": [0, 3, 6, 9],
            "symbol": "°7",
            "required": [0, 3, 6, 9],
        },
        "half_diminished7": {
            "intervals": [0, 3, 6, 10],
            "symbol": "ø7",
            "required": [0, 3, 6, 10],
        },
        "minor_major7": {
            "intervals": [0, 3, 7, 11],
            "symbol": "m(maj7)",
            "required": [0, 3, 11],
        },
        # Extended chords
        "add9": {"intervals": [0, 4, 7, 2], "symbol": "add9", "required": [0, 4, 2]},
        "minor_add9": {
            "intervals": [0, 3, 7, 2],
            "symbol": "madd9",
            "required": [0, 3, 2],
        },
        "6": {"intervals": [0, 4, 7, 9], "symbol": "6", "required": [0, 4, 9]},
        "minor6": {"intervals": [0, 3, 7, 9], "symbol": "m6", "required": [0, 3, 9]},
        "9": {"intervals": [0, 4, 7, 10, 2], "symbol": "9", "required": [0, 4, 10, 2]},
        "minor9": {
            "intervals": [0, 3, 7, 10, 2],
            "symbol": "m9",
            "required": [0, 3, 10, 2],
        },
        "major9": {
            "intervals": [0, 4, 7, 11, 2],
            "symbol": "maj9",
            "required": [0, 4, 11, 2],
        },
        "11": {
            "intervals": [0, 4, 7, 10, 2, 5],
            "symbol": "11",
            "required": [0, 4, 10, 5],
        },
        "13": {
            "intervals": [0, 4, 7, 10, 2, 9],
            "symbol": "13",
            "required": [0, 4, 10, 9],
        },
    }

    result = {
        "input_notes": unique_notes,
        "note_count": len(unique_notes),
        "primary_chords": [],
        "alternative_chords": [],
        "incomplete_chords": [],
        "intervals": [],
        "power_chord": None,
        "inversions": [],
        "slash_chords": [],
    }

    # Handle single note
    if len(unique_notes) == 1:
        result["intervals"] = [f"{unique_notes[0]} (single note)"]
        return result

    # Handle two notes (dyads/intervals)
    if len(unique_notes) == 2:
        interval = (semitones[1] - semitones[0]) % 12
        root_name = note_names[semitones[0]]

        interval_names = {
            1: "minor 2nd",
            2: "major 2nd",
            3: "minor 3rd",
            4: "major 3rd",
            5: "perfect 4th",
            6: "tritone",
            7: "perfect 5th",
            8: "minor 6th",
            9: "major 6th",
            10: "minor 7th",
            11: "major 7th",
        }

        result["intervals"] = [
            f"{root_name} - {interval_names.get(interval, 'unknown interval')}"
        ]

        # Power chord (root + fifth)
        if interval == 7:
            result["power_chord"] = f"{root_name}5"

        # Incomplete chords (suggest what they might be)
        if interval == 4:  # Major third
            result["incomplete_chords"] = [
                f"{root_name} (incomplete major - missing 5th)"
            ]
        elif interval == 3:  # Minor third
            result["incomplete_chords"] = [
                f"{root_name}m (incomplete minor - missing 5th)"
            ]

        return result

    # For 3+ notes, find chord matches
    def find_chord_matches(semitones, allow_missing=False):
        matches = []

        for root in range(12):
            root_name = note_names[root]

            for chord_type, pattern_info in chord_patterns.items():
                pattern = pattern_info["intervals"]
                required = pattern_info["required"]
                symbol = pattern_info["symbol"]

                # Transpose pattern to this root
                transposed_pattern = [(note + root) % 12 for note in pattern]
                transposed_required = [(note + root) % 12 for note in required]

                # Check for exact match
                if set(transposed_pattern) == set(semitones):
                    chord_name = f"{root_name}{symbol}"
                    matches.append(("exact", chord_name, chord_type))

                # Check if all required notes are present (allows extra notes)
                elif all(note in semitones for note in transposed_required):
                    chord_name = f"{root_name}{symbol}"
                    if len(set(transposed_pattern) - set(semitones)) == 0:
                        matches.append(("complete", chord_name, chord_type))
                    else:
                        matches.append(("extended", chord_name, chord_type))

                # Check for incomplete chords (missing non-essential notes)
                elif (
                    allow_missing
                    and len(set(transposed_required) & set(semitones)) >= 2
                ):
                    missing = set(transposed_required) - set(semitones)
                    if len(missing) <= 1:  # Only one note missing
                        chord_name = f"{root_name}{symbol}"
                        missing_note = note_names[list(missing)[0]] if missing else None
                        matches.append(
                            ("incomplete", chord_name, chord_type, missing_note)
                        )

        return matches

    # Find matches
    matches = find_chord_matches(semitones)
    incomplete_matches = find_chord_matches(semitones, allow_missing=True)

    # Categorize matches
    for match in matches:
        match_type = match[0]
        chord_name = match[1]

        if match_type == "exact":
            result["primary_chords"].append(chord_name)
        elif match_type in ["complete", "extended"]:
            if chord_name not in result["primary_chords"]:
                result["alternative_chords"].append(chord_name)

    # Add incomplete matches
    for match in incomplete_matches:
        if match[0] == "incomplete" and len(match) > 3:
            chord_name = match[1]
            missing_note = match[3]
            incomplete_desc = (
                f"{chord_name} (missing {missing_note})" if missing_note else chord_name
            )
            if incomplete_desc not in result["incomplete_chords"]:
                result["incomplete_chords"].append(incomplete_desc)

    # Check for inversions (bass note is not the root)
    if len(semitones) >= 3:
        bass_semitone = semitones[0]  # Lowest note
        for chord in result["primary_chords"] + result["alternative_chords"]:
            root_note = chord[0] if len(chord) > 0 else None
            if root_note and note_map.get(root_note, -1) != bass_semitone:
                bass_note_name = note_names[bass_semitone]
                result["inversions"].append(f"{chord}/{bass_note_name}")

    # Handle slash chords if bass_note is specified
    if bass_note:
        bass_normalized = normalize_note(bass_note)
        for chord in result["primary_chords"] + result["alternative_chords"]:
            result["slash_chords"].append(f"{chord}/{bass_normalized}")

    # Remove duplicates
    result["primary_chords"] = list(dict.fromkeys(result["primary_chords"]))
    result["alternative_chords"] = list(dict.fromkeys(result["alternative_chords"]))
    result["incomplete_chords"] = list(dict.fromkeys(result["incomplete_chords"]))
    result["inversions"] = list(dict.fromkeys(result["inversions"]))

    return result


def test_advanced_chord_identifier():
    # Test the function with various examples
    test_cases = [
        # Basic triads
        (["C", "E", "G"], "C major triad"),
        (["A", "C", "E"], "A minor triad"),
        (["F#", "A#", "C#"], "F# major triad"),
        # Power chord
        (["E", "B"], "E power chord"),
        # 7th chords
        (["G", "B", "D", "F"], "G7 chord"),
        (["C", "E", "G", "B"], "Cmaj7 chord"),
        # Incomplete chords
        (["C", "E"], "Incomplete C major"),
        (["A", "C"], "Incomplete A minor"),
        # Extended chords
        (["D", "F#", "A", "C", "E"], "D9 chord"),
        # With doubled notes (common in guitar)
        (["E", "B", "E", "G#", "B", "E"], "E major with doubled notes"),
        # Single note
        (["A"], "Single note"),
    ]

    print("=== Advanced Chord Identifier Test Results ===\n")

    for notes, description in test_cases:
        print(f"Input: {notes} ({description})")
        result = advanced_chord_identifier(notes)

        if "error" in result:
            print(f"Error: {result['error']}\n")
            continue

        print(f"Note count: {result['note_count']}")

        if result["primary_chords"]:
            print(f"Primary chords: {', '.join(result['primary_chords'])}")

        if result["alternative_chords"]:
            print(f"Alternative chords: {', '.join(result['alternative_chords'])}")

        if result["incomplete_chords"]:
            print(f"Incomplete chords: {', '.join(result['incomplete_chords'])}")

        if result["intervals"]:
            print(f"Intervals: {', '.join(result['intervals'])}")

        if result["power_chord"]:
            print(f"Power chord: {result['power_chord']}")

        if result["inversions"]:
            print(f"Inversions: {', '.join(result['inversions'])}")

        print("-" * 50)

    print("\n=== Function saved to advanced_chord_identifier.py ===")

def advanced_chord_identifier_with_instrument(notes, bass_note=None, key_context=None, instrument=None, note_positions=None):
    """
    Advanced chord identification function with automatic bass note detection for instruments.
    
    Args:
        notes: List of note names (e.g., ['C4', 'E4', 'G4'] or ['C', 'E', 'G'])
        bass_note: Optional bass note for slash chords (overrides auto-detection)
        key_context: Optional key context to help with ambiguous chords
        instrument: Instrument type ('guitar', 'bass', 'piano', 'ukulele', etc.)
        note_positions: Optional position info for auto bass detection
                       For guitar: [(string, fret), ...] e.g., [(6, 0), (5, 2), (4, 2)]
                       For piano: [midi_note_numbers] or ['low', 'mid', 'high'] indicators
    
    Returns:
        Dictionary with chord analysis including auto-detected bass note
    """
    
    # Standard tunings for different instruments
    instrument_tunings = {
        'guitar': {
            'standard': [40, 45, 50, 55, 59, 64],  # EADGBE in MIDI numbers (E2, A2, D3, G3, B3, E4)
            'drop_d': [38, 45, 50, 55, 59, 64],    # DADGBE
            'open_g': [38, 43, 50, 55, 59, 62],    # DGDGBD
        },
        'bass': {
            'standard': [28, 33, 38, 43],  # EADG (E1, A1, D2, G2)
            '5_string': [23, 28, 33, 38, 43],  # BEADG
        },
        'ukulele': {
            'standard': [67, 60, 64, 69],  # GCEA (G4, C4, E4, A4)
        },
        'mandolin': {
            'standard': [55, 62, 69, 76],  # GDAE (G3, D4, A4, E5)
        }
    }
    
    def note_to_midi(note_name):
        """Convert note name to MIDI number (C4 = 60)"""
        note_map = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                   'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        
        # Extract note and octave
        clean_note = ''.join([c for c in note_name if not c.isdigit()]).upper()
        octave_chars = ''.join([c for c in note_name if c.isdigit()])
        octave = int(octave_chars) if octave_chars else 4  # Default to octave 4
        
        # Handle enharmonics
        enharmonic_map = {'DB': 'C#', 'EB': 'D#', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#'}
        clean_note = enharmonic_map.get(clean_note, clean_note)
        
        return note_map[clean_note] + (octave + 1) * 12
    
    def midi_to_note(midi_num):
        """Convert MIDI number to note name"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_num // 12) - 1
        note = note_names[midi_num % 12]
        return f"{note}{octave}"
    
    def auto_detect_bass_note(notes, instrument, note_positions, tuning='standard'):
        """Automatically detect the bass note based on instrument and positions"""
        
        if not instrument or not note_positions:
            # Fallback: assume lowest note alphabetically/numerically is bass
            if any(char.isdigit() for note in notes for char in note):
                # Has octave info, find lowest MIDI number
                midi_notes = [note_to_midi(note) for note in notes]
                lowest_midi = min(midi_notes)
                return midi_to_note(lowest_midi).replace(str((lowest_midi // 12) - 1), '')
            else:
                # No octave info, can't determine bass reliably
                return None
        
        if instrument.lower() == 'guitar':
            if len(note_positions) != len(notes):
                return None
                
            tuning_notes = instrument_tunings['guitar'].get(tuning, instrument_tunings['guitar']['standard'])
            
            # Calculate actual MIDI note for each position
            actual_midi_notes = []
            for i, (string, fret) in enumerate(note_positions):
                if 1 <= string <= len(tuning_notes):
                    open_string_midi = tuning_notes[string - 1]  # Convert to 0-indexed
                    actual_midi = open_string_midi + fret
                    actual_midi_notes.append((actual_midi, notes[i]))
            
            if actual_midi_notes:
                # Find the lowest actual pitch
                lowest = min(actual_midi_notes, key=lambda x: x[0])
                return lowest[1].replace(''.join(filter(str.isdigit, lowest[1])), '')  # Remove octave
        
        elif instrument.lower() == 'bass':
            # For bass, typically the lowest string/fret combination
            if len(note_positions) != len(notes):
                return None
                
            tuning_notes = instrument_tunings['bass'].get(tuning, instrument_tunings['bass']['standard'])
            
            actual_midi_notes = []
            for i, (string, fret) in enumerate(note_positions):
                if 1 <= string <= len(tuning_notes):
                    open_string_midi = tuning_notes[string - 1]
                    actual_midi = open_string_midi + fret
                    actual_midi_notes.append((actual_midi, notes[i]))
            
            if actual_midi_notes:
                lowest = min(actual_midi_notes, key=lambda x: x[0])
                return lowest[1].replace(''.join(filter(str.isdigit, lowest[1])), '')
        
        elif instrument.lower() == 'piano':
            # For piano, use MIDI note numbers or position indicators
            if isinstance(note_positions[0], int):
                # MIDI numbers provided
                midi_notes = list(zip(note_positions, notes))
                lowest = min(midi_notes, key=lambda x: x[0])
                return lowest[1].replace(''.join(filter(str.isdigit, lowest[1])), '')
            elif isinstance(note_positions[0], str):
                # Position indicators like 'low', 'mid', 'high'
                position_priority = {'low': 0, 'mid': 1, 'high': 2}
                pos_notes = list(zip(note_positions, notes))
                lowest = min(pos_notes, key=lambda x: position_priority.get(x[0], 999))
                return lowest[1].replace(''.join(filter(str.isdigit, lowest[1])), '')
        
        return None
    
    # Auto-detect bass note if not provided
    detected_bass = None
    if bass_note is None and instrument:
        detected_bass = auto_detect_bass_note(notes, instrument, note_positions)
        bass_note = detected_bass
    
    # Use the original chord identification function
    def normalize_note(note):
        clean_note = ''.join([c for c in note if not c.isdigit()]).upper().strip()
        enharmonic_map = {'DB': 'C#', 'EB': 'D#', 'GB': 'F#', 'AB': 'G#', 'BB': 'A#'}
        return enharmonic_map.get(clean_note, clean_note)
    
    normalized_notes = [normalize_note(note) for note in notes]
    unique_notes = list(dict.fromkeys(normalized_notes))
    
    note_map = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    
    try:
        semitones = [note_map[note] for note in unique_notes]
    except KeyError as e:
        return {"error": f"Invalid note: {e}"}
    
    semitones = sorted(list(set(semitones)))
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Chord patterns (same as before)
    chord_patterns = {
        'major': {'intervals': [0, 4, 7], 'symbol': '', 'required': [0, 4, 7]},
        'minor': {'intervals': [0, 3, 7], 'symbol': 'm', 'required': [0, 3, 7]},
        'diminished': {'intervals': [0, 3, 6], 'symbol': '°', 'required': [0, 3, 6]},
        'augmented': {'intervals': [0, 4, 8], 'symbol': '+', 'required': [0, 4, 8]},
        'sus2': {'intervals': [0, 2, 7], 'symbol': 'sus2', 'required': [0, 2, 7]},
        'sus4': {'intervals': [0, 5, 7], 'symbol': 'sus4', 'required': [0, 5, 7]},
        'major7': {'intervals': [0, 4, 7, 11], 'symbol': 'maj7', 'required': [0, 4, 11]},
        'minor7': {'intervals': [0, 3, 7, 10], 'symbol': 'm7', 'required': [0, 3, 10]},
        'dominant7': {'intervals': [0, 4, 7, 10], 'symbol': '7', 'required': [0, 4, 10]},
        'diminished7': {'intervals': [0, 3, 6, 9], 'symbol': '°7', 'required': [0, 3, 6, 9]},
        'half_diminished7': {'intervals': [0, 3, 6, 10], 'symbol': 'ø7', 'required': [0, 3, 6, 10]},
        'add9': {'intervals': [0, 4, 7, 2], 'symbol': 'add9', 'required': [0, 4, 2]},
        '6': {'intervals': [0, 4, 7, 9], 'symbol': '6', 'required': [0, 4, 9]},
        '9': {'intervals': [0, 4, 7, 10, 2], 'symbol': '9', 'required': [0, 4, 10, 2]},
    }
    
    result = {
        'input_notes': unique_notes,
        'note_count': len(unique_notes),
        'instrument': instrument,
        'auto_detected_bass': detected_bass,
        'bass_note_used': bass_note,
        'primary_chords': [],
        'alternative_chords': [],
        'power_chord': None,
        'slash_chords': [],
        'intervals': []
    }
    
    # Handle single note
    if len(unique_notes) == 1:
        result['intervals'] = [f"{unique_notes[0]} (single note)"]
        return result
    
    # Handle two notes
    if len(unique_notes) == 2:
        interval = (semitones[1] - semitones[0]) % 12
        root_name = note_names[semitones[0]]
        
        interval_names = {
            1: 'minor 2nd', 2: 'major 2nd', 3: 'minor 3rd', 4: 'major 3rd',
            5: 'perfect 4th', 6: 'tritone', 7: 'perfect 5th', 8: 'minor 6th',
            9: 'major 6th', 10: 'minor 7th', 11: 'major 7th'
        }
        
        result['intervals'] = [f"{root_name} - {interval_names.get(interval, 'unknown interval')}"]
        
        if interval == 7:  # Power chord
            result['power_chord'] = f"{root_name}5"
        
        return result
    
    # Find chord matches for 3+ notes
    def find_chord_matches(semitones):
        matches = []
        for root in range(12):
            root_name = note_names[root]
            for chord_type, pattern_info in chord_patterns.items():
                pattern = pattern_info['intervals']
                required = pattern_info['required']
                symbol = pattern_info['symbol']
                
                transposed_pattern = [(note + root) % 12 for note in pattern]
                transposed_required = [(note + root) % 12 for note in required]
                
                if set(transposed_pattern) == set(semitones):
                    chord_name = f"{root_name}{symbol}"
                    matches.append(('exact', chord_name))
                elif all(note in semitones for note in transposed_required):
                    chord_name = f"{root_name}{symbol}"
                    matches.append(('partial', chord_name))
        
        return matches
    
    matches = find_chord_matches(semitones)
    
    for match_type, chord_name in matches:
        if match_type == 'exact':
            result['primary_chords'].append(chord_name)
        else:
            result['alternative_chords'].append(chord_name)
    
    # Handle slash chords
    if bass_note and len(semitones) >= 3:
        bass_normalized = normalize_note(bass_note)
        for chord in result['primary_chords'] + result['alternative_chords']:
            if chord and chord[0] != bass_normalized:  # Only if bass is different from root
                result['slash_chords'].append(f"{chord}/{bass_normalized}")
    
    # Remove duplicates
    result['primary_chords'] = list(dict.fromkeys(result['primary_chords']))
    result['alternative_chords'] = list(dict.fromkeys(result['alternative_chords']))
    result['slash_chords'] = list(dict.fromkeys(result['slash_chords']))
    
    return result

def test_advanced_chord_identifier_with_instrument():
    # Test the enhanced function
    print("=== Enhanced Chord Identifier with Instrument Support ===\n")

    test_cases = [
        # Guitar examples with string/fret positions
        {
            'notes': ['E', 'A', 'C#', 'E', 'A', 'E'],
            'instrument': 'guitar',
            'note_positions': [(6, 0), (5, 0), (4, 2), (3, 2), (2, 0), (1, 0)],  # A major open chord
            'description': 'A major open chord'
        },
        {
            'notes': ['E', 'B', 'E', 'G#', 'B', 'E'],
            'instrument': 'guitar', 
            'note_positions': [(6, 0), (5, 2), (4, 2), (3, 1), (2, 0), (1, 0)],  # E major open chord
            'description': 'E major open chord'
        },
        {
            'notes': ['G', 'B', 'D', 'G', 'B', 'G'],
            'instrument': 'guitar',
            'note_positions': [(6, 3), (5, 2), (4, 0), (3, 0), (2, 0), (1, 3)],  # G major open chord
            'description': 'G major open chord'
        },
        {
            'notes': ['F', 'A', 'C', 'F'],
            'instrument': 'guitar',
            'note_positions': [(4, 3), (3, 2), (2, 1), (1, 1)],  # F major barre chord (partial)
            'description': 'F major barre chord (4 strings)'
        },
        
        # Bass example
        {
            'notes': ['E', 'G#', 'B'],
            'instrument': 'bass',
            'note_positions': [(4, 0), (3, 1), (2, 0)],
            'description': 'E major on bass'
        },
        
        # Piano example with position indicators
        {
            'notes': ['C', 'E', 'G', 'C'],
            'instrument': 'piano',
            'note_positions': ['low', 'mid', 'mid', 'high'],
            'description': 'C major on piano'
        },
        
        # Manual bass note override
        {
            'notes': ['C', 'E', 'G'],
            'bass_note': 'E',
            'description': 'C major with manual bass note E'
        }
    ]

    for test in test_cases:
        print(f"Test: {test['description']}")
        print(f"Notes: {test['notes']}")
        
        result = advanced_chord_identifier_with_instrument(
            notes=test['notes'],
            bass_note=test.get('bass_note'),
            instrument=test.get('instrument'),
            note_positions=test.get('note_positions')
        )
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Instrument: {result['instrument']}")
            print(f"Auto-detected bass: {result['auto_detected_bass']}")
            print(f"Bass note used: {result['bass_note_used']}")
            print(f"Primary chords: {result['primary_chords']}")
            if result['slash_chords']:
                print(f"Slash chords: {result['slash_chords']}")
            if result['power_chord']:
                print(f"Power chord: {result['power_chord']}")
        
        print("-" * 60)

def find_best_matching_chord(notes, instrument="Guitar"):
    """
    Find the best matching chord for a list of notes.
    """
    return advanced_chord_identifier_with_instrument(notes, instrument=instrument.lower())

def analyze_chord_progression(notes_history, window_size=5):
    """
    Analyze chord progression over time for more stable detection.
    """
    if len(notes_history) < window_size:
        return None
    
    recent_notes = []
    notes_list = list(notes_history)
    for notes in notes_list[-window_size:]:
        recent_notes.extend(notes)
    
    note_counts = {}
    for note in recent_notes:
        note_counts[note] = note_counts.get(note, 0) + 1
    
    threshold = max(1, window_size // 2)
    stable_notes = [note for note, count in note_counts.items() if count >= threshold]
    
    if not stable_notes:
        return None
    
    return find_best_matching_chord(stable_notes)