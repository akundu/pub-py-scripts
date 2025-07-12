# file: chord_detector_sounddevice.py

import sounddevice as sd
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, correlate
from scipy.fft import fft, fftfreq
import time
import argparse
from collections import deque
import sys
import math

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


def frequency_to_note(freq):
    # List of note names in chromatic order starting from C
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

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
    note_name = notes[note_index]

    # Calculate the exact frequency of this note for reference
    exact_freq = 440 * (2 ** ((rounded_midi - 69) / 12))

    # print ( f"{note_name}{octave}", round(exact_freq, 2), file=sys.stderr)
    #return f"{note_name}{octave}", round(exact_freq, 2)
    return note_name

# Audio stream settings
CHUNK = 2048 * 2  # Number of audio samples per frame
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate in Hz

def apply_bandpass_filter(samples, low_freq=80, high_freq=2000, sample_rate=RATE):
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

def detect_notes_with_sounddevice(audio_buffer, sample_rate=RATE, sensitivity=1.0, 
                                 silence_threshold=200, low_freq=80, high_freq=2000, 
                                 show_frequencies=False, show_fft=False, raw_frequencies=False):
    """
    Enhanced note detection using sounddevice and autocorrelation.
    """
    detected_notes_all = []
    detected_frequencies_all = []
    
    # Process overlapping windows
    for i in range(0, len(audio_buffer) - CHUNK, HOP_SIZE):
        # Extract window
        window_samples = audio_buffer[i:i+CHUNK]
        
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
                print(f"\n🔍 Autocorrelation Analysis (Window {i//HOP_SIZE + 1}):")
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

def find_best_matching_chord(detected_notes):
    """
    Enhanced chord matching with confidence scoring.
    """
    if not detected_notes:
        return "No chord detected", 0.0

    detected_note_indices = sorted(list(set([NOTES.index(n) for n in detected_notes])))
    if not detected_note_indices:
        return "No chord detected", 0.0

    best_match = {'score': 0, 'chord': 'Unknown', 'confidence': 0.0}

    for i in range(12):
        root_note = NOTES[i]
        
        for chord_type, intervals in CHORD_DEFINITIONS.items():
            chord_notes_indices = set([(i + interval) % 12 for interval in intervals])
            
            matches = len(set(detected_note_indices) & chord_notes_indices)
            extra_notes = len(chord_notes_indices) - matches
            missing_notes = len(detected_note_indices) - matches
            
            score = matches * 2 - extra_notes - missing_notes * 0.5
            
            if len(chord_notes_indices) > 0:
                confidence = matches / len(chord_notes_indices)
            else:
                confidence = 0.0

            if score > best_match['score']:
                best_match['score'] = score
                best_match['chord'] = f"{root_note} {chord_type}"
                best_match['confidence'] = confidence

    if best_match['score'] < 1.5 or best_match['confidence'] < 0.6:
        return "Unknown or complex chord", best_match['confidence']
        
    return best_match['chord'], best_match['confidence']

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

def clear_line():
    """Clear the current line by printing spaces and returning to start"""
    print(" " * 80, end='\r')

# Overlapping window settings
OVERLAP_RATIO = 0.75  # 75% overlap
HOP_SIZE = int(CHUNK * (1 - OVERLAP_RATIO))  # 512 samples
BUFFER_SIZE = CHUNK * 2  # 4096 samples for buffer

def grab_audio_chunk(CHUNK, RATE, CHANNELS):
    """Grab a chunk of audio from the microphone"""
    myrecording = sd.rec(CHUNK, samplerate=RATE, channels=CHANNELS, dtype='float32')
    sd.wait()  # Wait until recording is finished
    #return myrecording.flatten()
    return myrecording

def recognize_audio(args, audio_buffer, buffer_index, low_freq, high_freq, notes_history, last_chord, chord_stability, last_log_time):
    myrecording = grab_audio_chunk(CHUNK, RATE, CHANNELS)
    
    # Convert to numpy array and ensure it's the right shape
    new_samples = myrecording.flatten()
    
    # Update circular buffer
    audio_buffer[buffer_index:buffer_index+CHUNK] = new_samples
    buffer_index = (buffer_index + CHUNK) % BUFFER_SIZE
    
    # Detect notes using sounddevice and autocorrelation
    detected_notes, detected_frequencies_all = detect_notes_with_sounddevice(
        audio_buffer, sensitivity=args.sensitivity, 
        silence_threshold=args.silence_threshold,
        low_freq=low_freq, high_freq=high_freq,
        show_frequencies=args.show_frequencies or args.frequencies_only,
        show_fft=args.show_fft,
        raw_frequencies=args.raw_frequencies
    )
    
    # Debug mode
    if args.debug:
        rms_energy = np.sqrt(np.mean(new_samples.astype(np.float64)**2))
        print(f"Audio Level: {rms_energy:.4f} (threshold: {args.silence_threshold})", end='\r', file=sys.stderr)
        if rms_energy < args.silence_threshold:
            raise ValueError("Too low audio level")
    
    # Frequencies-only mode - skip chord processing
    if args.frequencies_only:
        current_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if detected_notes:
            # Get frequencies for the detected notes
            note_freqs = {}
            for freq, amp in detected_frequencies_all:
                note = frequency_to_note(freq)
                if note in detected_notes:
                    if note not in note_freqs:
                        note_freqs[note] = []
                    note_freqs[note].append(freq)
            
            freq_str = ", ".join([f"{note}({np.mean(freqs):.0f}Hz)" for note, freqs in note_freqs.items()])
            if args.log:
                print(f"[{timestamp}] Frequencies: {freq_str}")
            else:
                clear_line()
                print(f"Frequencies detected: {freq_str}", end='\r')
        else:
            if args.log and current_time - last_log_time >= args.log_interval:
                print(f"[{timestamp}] No frequencies detected")
            else:
                clear_line()
                print("🔇 No frequencies detected...", end='\r')
        last_log_time = current_time
    # Notes-only mode - show notes without chord processing
    elif args.notes_only:
        current_time = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if detected_notes:
            # Get frequencies for the detected notes
            note_freqs = {}
            for freq, amp in detected_frequencies_all:
                note = frequency_to_note(freq)
                if note in detected_notes:
                    if note not in note_freqs:
                        note_freqs[note] = []
                    note_freqs[note].append(freq)
            
            freq_str = ", ".join([f"{note}({np.mean(freqs):.0f}Hz)" for note, freqs in note_freqs.items()])
            if args.log and current_time - last_log_time >= args.log_interval:
                print(f"[{timestamp}] Notes: {freq_str}")
            else:
                clear_line()
                print(f"Notes detected: {freq_str}", end='\r')
        else:
            if args.log and current_time - last_log_time >= args.log_interval:
                print(f"[{timestamp}] No notes detected")
            else:
                clear_line()
                print("🔇 No notes detected...", end='\r')
        last_log_time = current_time
    else:
        # Normal chord processing mode
        if detected_notes:
            print(f"detected_notes: {detected_notes}", file=sys.stderr) if args.debug else None
            notes_history.append(detected_notes)
            print(f"notes_history: {notes_history}", file=sys.stderr) if args.debug else None
            
            if args.progression:
                result = analyze_chord_progression(notes_history)
                if result is not None:
                    chord, confidence = result
                else:
                    chord, confidence = "No chord detected", 0.0
            else:
                chord, confidence = find_best_matching_chord(detected_notes)
                print(f"chord: {chord}, confidence: {confidence}", file=sys.stderr) if args.debug else None
            
            # Stability check
            if chord == last_chord:
                chord_stability += 1
            else:
                chord_stability = 0
                last_chord = chord
            
            # Display results
            current_time = time.time()
            
            # Get frequencies for the detected notes
            note_freqs = {}
            for freq, amp in detected_frequencies_all:
                note = frequency_to_note(freq)
                if note in detected_notes:
                    if note not in note_freqs:
                        note_freqs[note] = []
                    note_freqs[note].append(freq)
            
            freq_str = ", ".join([f"{note}({np.mean(freqs):.0f}Hz)" for note, freqs in note_freqs.items()])
            
            if args.log:
                if current_time - last_log_time >= args.log_interval:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    if chord_stability >= 2 or confidence > 0.8:
                        print(f"[{timestamp}] Notes: {freq_str} -> {chord} (conf: {confidence:.2f})")
                    else:
                        print(f"[{timestamp}] Notes: {freq_str} -> {chord} (conf: {confidence:.2f}) [unstable]")
                    last_log_time = current_time
            else:
                clear_line()
                if chord_stability >= 2 or confidence > 0.8:
                    print(f"Notes: {freq_str} -> {chord.ljust(20)} (conf: {confidence:.2f})", end='\r')
                else:
                    print(f"Notes: {freq_str} -> {chord.ljust(20)} (conf: {confidence:.2f}) [unstable]", end='\r')
        else:
            current_time = time.time()
            if args.log:
                if current_time - last_log_time >= args.log_interval:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{timestamp}] No notes detected")
                    last_log_time = current_time
            else:
                clear_line()
                print("🔇 Listening for audio...", end='\r')

# --- 3. Main Application Logic ---
def main():
    """
    Main function to run the sounddevice-based chord detector.
    """
    parser = argparse.ArgumentParser(description='Sounddevice-based chord detector')
    parser.add_argument('--log', action='store_true', help='Enable logging mode with timestamps')
    parser.add_argument('--log-interval', type=float, default=0.5, help='Logging interval in seconds (default: 0.5)')
    parser.add_argument('--progression', action='store_true', default=True, help='Enable chord progression analysis (default: enabled)')
    parser.add_argument('--sensitivity', type=float, default=1.0, help='Detection sensitivity (0.1-2.0, default: 1.0)')
    parser.add_argument('--silence-threshold', type=int, default=0.005, help='Silence threshold (0-1, default: 0.005)')
    parser.add_argument('--debug', action='store_true', help='Show audio levels for threshold tuning')
    parser.add_argument('--instrument', choices=list(INSTRUMENT_PRESETS.keys()), default='guitar', 
                       help='Instrument preset for frequency filtering (default: guitar)')
    parser.add_argument('--low-freq', type=int, help='Custom low frequency cutoff (Hz)')
    parser.add_argument('--high-freq', type=int, help='Custom high frequency cutoff (Hz)')
    parser.add_argument('--overlap', type=float, default=0.75, help='Overlap ratio (0.0-0.9, default: 0.75)')
    parser.add_argument('--show-frequencies', action='store_true', help='Show detected frequencies')
    parser.add_argument('--show-fft', action='store_true', help='Show autocorrelation analysis data')
    parser.add_argument('--raw-frequencies', action='store_true', help='Use raw frequencies without filtering or processing')
    parser.add_argument('--frequencies-only', action='store_true', help='Show only frequencies, skip chord processing')
    parser.add_argument('--notes-only', action='store_true', help='Show only detected notes, skip chord processing')
    parser.add_argument('--wait-time', type=float, default=0.0, help='Wait time between iterations in seconds (default: 0.0)')
    parser.add_argument('--amplitude-threshold', type=float, default=0.005, help='Minimum amplitude threshold for processing (default: 0.005)')
    args = parser.parse_args()

    # Update overlap settings
    global OVERLAP_RATIO, HOP_SIZE
    OVERLAP_RATIO = args.overlap
    HOP_SIZE = int(CHUNK * (1 - OVERLAP_RATIO))

    # Get instrument settings
    if args.low_freq and args.high_freq:
        low_freq = args.low_freq
        high_freq = args.high_freq
        instrument_name = f"Custom ({low_freq}-{high_freq} Hz)"
    else:
        preset = INSTRUMENT_PRESETS[args.instrument]
        low_freq = preset['low_freq']
        high_freq = preset['high_freq']
        instrument_name = preset['name']

    print(f"🎸 Sounddevice-based chord detector listening for {instrument_name}... Press Ctrl+C to stop.")
    if args.raw_frequencies:
        print("🔓 RAW MODE: Using unfiltered frequencies (no frequency range limits)")
    else:
        print(f"📊 Frequency range: {low_freq}-{high_freq} Hz")
    print(f"🔄 Overlap: {OVERLAP_RATIO*100:.0f}% (hop size: {HOP_SIZE} samples)")
    print(f"📊 Amplitude threshold: {args.amplitude_threshold}")
    if args.frequencies_only:
        print("📊 FREQUENCIES ONLY MODE: Showing frequencies, skipping chord processing")
    if args.notes_only:
        print("🎵 NOTES ONLY MODE: Showing detected notes, skipping chord processing")
    if args.progression and not args.frequencies_only and not args.notes_only:
        print("📈 Chord progression analysis enabled")
    if args.wait_time > 0:
        print(f"⏱️  Wait time between iterations: {args.wait_time} seconds")

    # Audio buffer for overlapping windows
    audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)
    buffer_index = 0
    
    notes_history = deque(maxlen=10)
    last_chord = None
    chord_stability = 0
    last_log_time = time.time()

    try:
        while True:
            try:
                recognize_audio(args, audio_buffer, buffer_index, low_freq, high_freq, notes_history, last_chord, chord_stability, last_log_time)
            except ValueError as e:
                pass
                # print(f"Caught exception: {e}", file=sys.stderr)
            except Exception as e:
                print(f"Caught exception: {e}", file=sys.stderr)
            finally:
                # Wait between iterations if specified
                if args.wait_time > 0:
                    time.sleep(args.wait_time)


    except KeyboardInterrupt:
        print("\n🛑 Stopping sounddevice-based chord detector.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Common reasons for errors:")
        print("1. Microphone not detected or properly configured.")
        print("2. Insufficient permissions (check your OS privacy settings for microphone access).")
        print("3. Another application is already using the microphone.")
        print("4. Missing libraries: Ensure 'sounddevice' and 'scipy' are installed.")

if __name__ == '__main__':
    main() 