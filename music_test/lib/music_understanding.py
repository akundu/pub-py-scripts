import math
import numpy as np
import sys
from scipy.signal import butter, filtfilt, correlate
from lib.common import get_chunk, get_hop_size, get_rate, get_channels, get_buffer_size

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

NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Chord templates: each maps a chord suffix to semitone intervals from root
CHORD_TEMPLATES = {
    '':      [0, 4, 7],            # Major
    'm':     [0, 3, 7],            # Minor
    '7':     [0, 4, 7, 10],        # Dominant 7th
    'maj7':  [0, 4, 7, 11],        # Major 7th
    'm7':    [0, 3, 7, 10],        # Minor 7th
    '5':     [0, 7],               # Power chord
    'sus2':  [0, 2, 7],            # Sus2
    'sus4':  [0, 5, 7],            # Sus4
    '\u00b0':     [0, 3, 6],            # Diminished
    '+':     [0, 4, 8],            # Augmented
    '6':     [0, 4, 7, 9],         # Major 6th
    'm6':    [0, 3, 7, 9],         # Minor 6th
    'add9':  [0, 2, 4, 7],         # Add 9
    '9':     [0, 2, 4, 7, 10],     # Dominant 9th
}


def chroma_from_fft(audio_data, sample_rate, low_freq=80, high_freq=2000):
    """
    Compute a 12-dimensional chroma vector using peak-based harmonic grouping.

    1. Find prominent peaks in the FFT magnitude spectrum
    2. Group peaks by harmonic series: if a peak is at an integer multiple
       of a stronger lower-frequency peak, assign it as a harmonic
    3. Build chroma from identified fundamentals only
    4. Identify bass note (lowest significant fundamental) for root disambiguation

    Enhanced with:
    - Blackman window for better sidelobe rejection (-58 dB vs -43 dB for Hann)
    - SNR-based adaptive peak threshold
    - Adaptive harmonic tolerance (higher harmonics get more tolerance)
    - Inharmonicity compensation for real instrument strings
    - Bass pitch class detection for Am7/C6-type disambiguation

    Returns:
        tuple: (chroma_vector, bass_pitch_class)
            - chroma_vector: 12-element numpy array (L2 normalized)
            - bass_pitch_class: int 0-11 for the lowest detected fundamental,
              or -1 if no bass detected
    """
    # Blackman window: -58 dB sidelobe rejection vs Hann's -43 dB
    # Reduces spectral leakage that can pollute neighboring pitch classes
    window = np.blackman(len(audio_data))
    windowed = audio_data.astype(np.float64) * window

    # 4x zero-padding for finer frequency grid (~2.7 Hz resolution at 44.1kHz/4096)
    # Critical for low-frequency notes where semitones are only 5 Hz apart
    padded_len = len(windowed) * 4
    fft_result = np.fft.rfft(windowed, n=padded_len)
    mag = np.abs(fft_result)
    freqs = np.fft.rfftfreq(padded_len, 1.0 / sample_rate)

    # --- Step 1: Find peaks with adaptive threshold ---
    max_mag = np.max(mag)
    if max_mag == 0:
        return np.zeros(12), -1

    # Compute noise floor for SNR-based threshold
    freq_mask = (freqs >= low_freq * 0.8) & (freqs <= high_freq * 1.5)
    in_range_mag = mag[freq_mask]
    if len(in_range_mag) > 0:
        noise_floor = np.median(in_range_mag)
        snr = max_mag / (noise_floor + 1e-10)
    else:
        snr = 30.0  # Assume clean if no range data

    # Adaptive threshold: clean signals get lower threshold (catch subtle notes),
    # noisy signals get higher threshold (reject noise peaks)
    if snr > 30:      # Very clean signal
        peak_threshold = max_mag * 0.04
    elif snr > 15:    # Moderate SNR
        peak_threshold = max_mag * 0.06
    else:             # Noisy
        peak_threshold = max_mag * 0.10

    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    peaks = []  # (freq, amplitude)
    for i in range(2, len(mag) - 2):
        f = freqs[i]
        if f < low_freq * 0.8 or f > high_freq * 1.5:
            continue
        if (mag[i] > mag[i - 1] and mag[i] > mag[i + 1] and
                mag[i] > mag[i - 2] and mag[i] > mag[i + 2] and
                mag[i] > peak_threshold):
            # Parabolic interpolation for sub-bin frequency accuracy
            alpha = float(mag[i - 1])
            beta = float(mag[i])
            gamma = float(mag[i + 1])
            denom = alpha - 2 * beta + gamma
            if abs(denom) > 1e-10:
                p = 0.5 * (alpha - gamma) / denom
                interp_freq = f + p * freq_res
            else:
                interp_freq = f
            peaks.append((interp_freq, beta))

    if not peaks:
        return np.zeros(12), -1

    # Sort by amplitude descending (process strongest first)
    peaks.sort(key=lambda x: x[1], reverse=True)

    # --- Step 2: Group peaks by harmonic series ---
    # Enhanced with adaptive tolerance and inharmonicity compensation
    #
    # Real instrument strings have inharmonicity: harmonics are slightly sharp.
    # For a typical guitar string, B ≈ 0.0003-0.0005, giving:
    #   f_n = n * f0 * sqrt(1 + B * n^2) ≈ n * f0 * (1 + B * n^2 / 2)
    INHARMONICITY_B = 0.0004  # Typical guitar string stiffness factor

    fundamentals = {}  # freq -> total_weighted_energy

    for freq, amp in peaks:
        assigned = False
        for f0 in sorted(fundamentals.keys()):
            for h in range(2, 8):
                # Expected harmonic with inharmonicity stretch
                expected = f0 * h * (1.0 + INHARMONICITY_B * h * h * 0.5)
                if expected <= 0:
                    continue
                # Adaptive tolerance: higher harmonics need more tolerance
                # because absolute frequency errors accumulate
                tolerance = 0.02 + (h - 2) * 0.005  # 2% base, +0.5% per harmonic
                if abs(freq - expected) / expected < tolerance:
                    # This peak is a harmonic of f0 — reinforce fundamental
                    fundamentals[f0] += amp / h
                    assigned = True
                    break
            if assigned:
                break

        if not assigned:
            # New fundamental
            fundamentals[freq] = amp

    # --- Step 3: Build chroma from fundamentals and detect bass ---
    chroma = np.zeros(12)
    bass_freq = float('inf')
    bass_pitch_class = -1
    # Threshold: bass must have at least 15% of strongest fundamental's energy
    max_fundamental_energy = max(fundamentals.values()) if fundamentals else 0

    for freq, energy in fundamentals.items():
        if freq <= 0:
            continue
        midi = 12 * math.log2(freq / 440.0) + 69
        pitch_class = int(round(midi)) % 12
        chroma[pitch_class] += energy
        # Track bass: lowest fundamental with significant energy
        if freq < bass_freq and energy >= max_fundamental_energy * 0.15:
            bass_freq = freq
            bass_pitch_class = pitch_class

    norm = np.linalg.norm(chroma)
    if norm > 0:
        chroma = chroma / norm
    return chroma, bass_pitch_class


def _match_chroma_to_chord(chroma, bass_pitch_class=-1):
    """
    Match a chroma vector against chord templates using precision-weighted scoring
    with bass note disambiguation.

    Score = (energy in chord bins with root weighting) / (total energy)
            - complexity penalty
            + bass bonus (if detected bass matches chord root)

    The bass bonus resolves ambiguous cases like Am7 vs C6 where both chords
    have identical pitch classes {A, C, E, G} but different roots. The lowest
    sounding note strongly suggests the root.

    Args:
        chroma: 12-element numpy array (L2 normalized chroma vector)
        bass_pitch_class: int 0-11 for the detected lowest fundamental,
                         or -1 if no bass detected

    Returns (chord_name, confidence, detected_notes).
    """
    if np.max(chroma) < 0.10:
        return None, 0.0, []

    total_energy = float(np.sum(chroma))
    if total_energy <= 0:
        return None, 0.0, []

    best_name = None
    best_score = -1.0

    for root_idx in range(12):
        root_name = NOTES[root_idx]
        for suffix, intervals in CHORD_TEMPLATES.items():
            chord_bins = set((root_idx + iv) % 12 for iv in intervals)

            # Energy in chord bins (root gets dynamic weight based on actual strength)
            root_energy_ratio = chroma[root_idx] / total_energy if total_energy > 0 else 0
            # Strong root: 2x weight. Weak root: 1.5x (still preferred, but less so)
            root_weight = 2.0 if root_energy_ratio >= 0.15 else 1.5

            chord_energy = 0.0
            for b in chord_bins:
                weight = root_weight if b == root_idx else 1.0
                chord_energy += chroma[b] * weight

            # Precision: fraction of total energy explained by chord
            weighted_total = total_energy + chroma[root_idx] * (root_weight - 1.0)
            score = chord_energy / weighted_total if weighted_total > 0 else 0.0

            # Penalty for chroma energy NOT in chord bins
            non_chord_energy = sum(chroma[i] for i in range(12) if i not in chord_bins)
            score -= (non_chord_energy / total_energy) * 0.15

            # Complexity penalty: prefer triads over 7ths/9ths
            score -= len(intervals) * 0.02

            # --- Bass note disambiguation ---
            # This is the key innovation for resolving Am7/C6, Dm7/F6, etc.
            # The lowest sounding note in a chord is almost always the root
            # (or at least strongly suggests it).
            if bass_pitch_class >= 0:
                if bass_pitch_class == root_idx:
                    # Bass matches root: strong bonus
                    score += 0.12
                elif bass_pitch_class in chord_bins:
                    # Bass is a chord tone but not root: suggests inversion,
                    # which is less common than root position
                    score -= 0.04

            if score > best_score:
                best_score = score
                best_name = f"{root_name}{suffix}"

    # Extract detected notes from chroma peaks
    detected_notes = []
    chroma_threshold = np.max(chroma) * 0.3
    for i in range(12):
        if chroma[i] >= chroma_threshold:
            detected_notes.append(NOTES[i])

    return best_name, max(0.0, best_score), detected_notes


def detect_chord_from_chroma(audio_data, sample_rate, low_freq=80, high_freq=2000):
    """
    Identify a chord by correlating the observed chroma vector with pre-built
    chord templates. Returns (chord_name, confidence, chroma_vector, detected_notes).
    """
    chroma, bass_pc = chroma_from_fft(audio_data, sample_rate, low_freq, high_freq)
    chord_name, confidence, detected_notes = _match_chroma_to_chord(chroma, bass_pitch_class=bass_pc)
    return chord_name, confidence, chroma, detected_notes


def detect_chord_from_buffer(audio_buffer, sample_rate, silence_threshold=0.005,
                              low_freq=80, high_freq=2000):
    """
    Process an audio buffer through overlapping windows, compute per-window
    chroma vectors, average them, and identify the chord.

    Enhanced with:
    - Multi-resolution: blends short-window chromas with a full-buffer FFT
      for better low-frequency resolution (1.35 Hz vs 2.7 Hz)
    - Bass voting: collects bass pitch class votes across all windows and
      the full-buffer analysis for robust root disambiguation
    - RMS-weighted accumulation: louder windows contribute more

    Returns (chord_name, confidence, chroma, detected_notes, frequencies).
    """
    from lib.common import get_chunk, get_hop_size

    chunk = get_chunk()
    hop = get_hop_size()
    chroma_accum = np.zeros(12)
    bass_votes = {}  # pitch_class -> total_weight
    n_active = 0

    # --- Pass 1: Short overlapping windows (good time resolution) ---
    for i in range(0, len(audio_buffer) - chunk, hop):
        win = audio_buffer[i:i + chunk]
        rms = np.sqrt(np.mean(win.astype(np.float64) ** 2))
        if rms < silence_threshold:
            continue
        c, bass_pc = chroma_from_fft(win, sample_rate, low_freq, high_freq)
        chroma_accum += c * rms   # weight by energy so louder windows count more
        if bass_pc >= 0:
            bass_votes[bass_pc] = bass_votes.get(bass_pc, 0) + rms
        n_active += 1

    if n_active == 0:
        return None, 0.0, np.zeros(12), [], []

    # --- Pass 2: Full-buffer FFT (better frequency resolution) ---
    # The full buffer is 2x the chunk size, giving ~1.35 Hz resolution
    # vs ~2.7 Hz for individual windows. This helps with:
    # - Low-frequency notes where semitones are only 5 Hz apart
    # - Distinguishing close fundamentals (e.g., E2=82.4 Hz vs F2=87.3 Hz)
    full_rms = np.sqrt(np.mean(audio_buffer.astype(np.float64) ** 2))
    if full_rms >= silence_threshold:
        full_c, full_bass = chroma_from_fft(audio_buffer, sample_rate, low_freq, high_freq)
        # Blend: 70% short windows (better averaging), 30% full buffer (better resolution)
        avg_rms = np.sum(chroma_accum) / n_active if n_active > 0 else full_rms
        chroma_accum = chroma_accum + full_c * avg_rms * n_active * 0.4
        if full_bass >= 0:
            # Full buffer gets double vote weight for bass (longer window = more reliable bass)
            bass_votes[full_bass] = bass_votes.get(full_bass, 0) + full_rms * 2.0

    # Normalize accumulated chroma
    norm = np.linalg.norm(chroma_accum)
    if norm > 0:
        chroma_accum = chroma_accum / norm

    # Determine consensus bass pitch class from votes
    bass_pitch_class = max(bass_votes, key=bass_votes.get) if bass_votes else -1

    # Use shared matching function with bass disambiguation
    chord_name, confidence, detected_notes = _match_chroma_to_chord(
        chroma_accum, bass_pitch_class=bass_pitch_class
    )

    # Build approximate frequencies from chroma for legacy compatibility
    detected_freqs = []
    for i in range(12):
        if chroma_accum[i] >= np.max(chroma_accum) * 0.3:
            representative_freq = 440.0 * (2 ** ((i - 9) / 12.0))
            detected_freqs.append((representative_freq, float(chroma_accum[i])))

    return chord_name, confidence, chroma_accum, detected_notes, detected_freqs


def calculate_chroma_vector(detected_notes, detected_frequencies_all):
    """
    Calculate a 12-dimensional chroma vector from detected notes and their frequencies.
    Returns a normalized chroma vector where each element represents the strength of each pitch class.
    """
    # Initialize chroma vector (12 pitch classes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
    chroma = np.zeros(12)
    
    # Note mapping to chroma indices
    note_to_chroma = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    
    # Accumulate energy for each pitch class from all detected frequencies
    if detected_frequencies_all:
        # Group by note and sum amplitudes to handle multiple detections of same note
        note_amplitudes = {}
        for freq, amplitude in detected_frequencies_all:
            note = frequency_to_note(freq)
            if note and note in note_to_chroma:
                if note not in note_amplitudes:
                    note_amplitudes[note] = 0
                note_amplitudes[note] += amplitude
        
        # Populate chroma vector with accumulated amplitudes
        for note, total_amplitude in note_amplitudes.items():
            chroma_idx = note_to_chroma[note]
            chroma[chroma_idx] = total_amplitude
    
    # Normalize the chroma vector (L2 normalization)
    magnitude = np.linalg.norm(chroma)
    if magnitude > 0:
        chroma = chroma / magnitude
    
    return chroma

def detect_multiple_pitches_fft(audio_data, sample_rate, low_freq=80, high_freq=2000, num_peaks=4):
    """
    Detect multiple pitches in audio using FFT peak detection with
    iterative harmonic sieving: find strongest peak, suppress its
    harmonics, repeat. This correctly handles polyphonic signals.
    """
    # Apply bandpass filter to remove noise outside instrument range
    filtered_audio = apply_bandpass_filter(audio_data, low_freq, high_freq, sample_rate)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(len(filtered_audio))
    windowed = filtered_audio * window

    # Zero-pad to double length for better frequency interpolation
    # without increasing the noise integration time
    padded_len = len(windowed) * 2
    fft = np.fft.rfft(windowed, n=padded_len)
    magnitude = np.abs(fft)

    # Create frequency bins (using padded length for resolution)
    freqs = np.fft.rfftfreq(padded_len, 1 / sample_rate)
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    # Apply frequency range mask
    freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
    working_mag = magnitude * freq_mask.astype(np.float64)

    # Spectral flatness check: if the spectrum is noise-like (flat), reject it.
    # Flatness = geometric mean / arithmetic mean. Close to 1.0 = noise, close to 0 = tonal.
    in_range = working_mag[freq_mask]
    if len(in_range) > 0 and np.mean(in_range) > 0:
        log_mean = np.mean(np.log(in_range + 1e-10))
        geo_mean = np.exp(log_mean)
        arith_mean = np.mean(in_range)
        flatness = geo_mean / arith_mean if arith_mean > 0 else 1.0
        if flatness > 0.85:  # Too flat = noise, not tonal content
            return []

    # Peaks must stand out clearly above the noise floor.
    # Use both: 20% of max AND 3x the median magnitude.
    max_mag = np.max(working_mag)
    median_mag = np.median(in_range) if len(in_range) > 0 else 0
    min_peak_height = max(max_mag * 0.20, median_mag * 4.0)

    # Width (in bins) to suppress around each harmonic
    suppress_width = max(3, int(15 / freq_resolution))

    selected = []
    for _ in range(num_peaks):
        # Find all local maxima above threshold
        best_idx = -1
        best_val = 0.0
        for i in range(1, len(working_mag) - 1):
            if (working_mag[i] > working_mag[i - 1] and
                working_mag[i] > working_mag[i + 1] and
                working_mag[i] > min_peak_height and
                working_mag[i] > best_val):
                best_idx = i
                best_val = working_mag[i]

        if best_idx < 0:
            break

        peak_freq = freqs[best_idx]

        # Check minimum distance from already-selected peaks
        too_close = False
        for sel_freq, _ in selected:
            if abs(peak_freq - sel_freq) < 25:
                too_close = True
                break
        if too_close:
            # Suppress this bin and try again
            lo = max(0, best_idx - suppress_width)
            hi = min(len(working_mag), best_idx + suppress_width + 1)
            working_mag[lo:hi] = 0
            continue

        # Record this fundamental with its original magnitude
        selected.append((peak_freq, magnitude[best_idx]))

        # Suppress this peak and all its harmonics (2f, 3f, 4f, 5f, 6f)
        for harmonic in range(1, 7):
            harmonic_freq = peak_freq * harmonic
            if harmonic_freq > high_freq * 1.5:
                break
            harmonic_bin = int(round(harmonic_freq / freq_resolution))
            lo = max(0, harmonic_bin - suppress_width)
            hi = min(len(working_mag), harmonic_bin + suppress_width + 1)
            working_mag[lo:hi] = 0

    return selected

def detect_notes_with_sounddevice(audio_buffer, sample_rate=get_rate(), sensitivity=1.0, 
                                 silence_threshold=200, low_freq=80, high_freq=2000, 
                                 show_frequencies=False, show_fft=False, raw_frequencies=False, 
                                 calculate_chroma=False, multi_pitch=False):
    """
    Enhanced note detection using sounddevice and autocorrelation.
    """
    detected_notes_all = []
    detected_frequencies_all = []

    # For multi-pitch: track notes per window so we can vote on consistency
    per_window_notes = []  # list of sets, one per active window
    per_window_freqs = []  # list of lists of (freq, amp), one per active window

    # Process overlapping windows
    for i in range(0, len(audio_buffer) - get_chunk(), get_hop_size()):
        # Extract window
        window_samples = audio_buffer[i:i+get_chunk()]

        # Check for silence/low volume
        rms_energy = np.sqrt(np.mean(window_samples.astype(np.float64)**2))
        if rms_energy < silence_threshold:
            continue

        # Use multi-pitch detection if enabled, otherwise use single-pitch autocorrelation
        if multi_pitch:
            # FFT-based multi-pitch detection (bandpass + window + HPS applied inside)
            fft_peaks = detect_multiple_pitches_fft(window_samples, sample_rate, low_freq, high_freq)
            if fft_peaks:
                max_mag = max(p[1] for p in fft_peaks)
                window_notes = set()
                window_freqs = []
                for freq, magnitude in fft_peaks:
                    amplitude = (magnitude / max_mag) * rms_energy if max_mag > 0 else 0
                    note = frequency_to_note(freq)
                    if note:
                        window_notes.add(note)
                        window_freqs.append((freq, amplitude))
                if window_notes:
                    per_window_notes.append(window_notes)
                    per_window_freqs.append(window_freqs)
        else:
            # Single-pitch autocorrelation detection (original method)
            frequency, amplitude, error_message = get_pitch_and_amplitude(window_samples, sample_rate)
            if error_message or frequency is None:
                continue

            # Check if frequency is in the desired range (with a buffer) based on the instrument's frequency range
            BUFFER_pct = 0.25
            if not raw_frequencies and (frequency < low_freq*BUFFER_pct or frequency > high_freq*BUFFER_pct):
                continue

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

    # For multi-pitch: vote on notes across windows.
    # Only keep notes that appear in at least half the active windows.
    if multi_pitch and per_window_notes:
        num_windows = len(per_window_notes)
        note_counts = {}
        for window_notes in per_window_notes:
            for note in window_notes:
                note_counts[note] = note_counts.get(note, 0) + 1

        vote_threshold = max(1, num_windows // 2)
        consistent_notes = {note for note, count in note_counts.items() if count >= vote_threshold}

        # Collect frequencies only for consistent notes
        for window_freqs in per_window_freqs:
            for freq, amplitude in window_freqs:
                note = frequency_to_note(freq)
                if note in consistent_notes:
                    detected_notes_all.append(note)
                    detected_frequencies_all.append((freq, amplitude))

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

    # Calculate chroma vector if requested
    chroma_vector = None
    if calculate_chroma:
        chroma_vector = calculate_chroma_vector(unique_notes, detected_frequencies_all)
    
    return unique_notes, detected_frequencies_all, chroma_vector

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

def find_best_matching_chord_enhanced(notes, detected_frequencies_all=None, chroma_vector=None, instrument="Guitar", verbose=False):
    """
    Enhanced chord matching that uses frequency amplitudes and chroma vectors for better accuracy.
    
    Args:
        notes: List of detected note names
        detected_frequencies_all: List of (frequency, amplitude) tuples  
        chroma_vector: 12-dimensional chroma vector
        instrument: Instrument type for context
        verbose: If True, print detailed analysis information
        
    Returns:
        Dictionary with chord analysis including confidence scores
    """
    if not notes:
        return {"error": "No notes detected"}
    
    if verbose:
        print(f"\n🔍 DETAILED CHORD ANALYSIS", file=sys.stderr)
        print(f"📝 Input Notes: {notes}", file=sys.stderr)
        if detected_frequencies_all:
            print(f"🎵 Frequencies: {[(f'{freq:.1f}Hz', f'{amp:.3f}') for freq, amp in detected_frequencies_all]}", file=sys.stderr)
        if chroma_vector is not None:
            chroma_str = ', '.join([f'{val:.3f}' for val in chroma_vector])
            print(f"🌈 Chroma Vector: [{chroma_str}]", file=sys.stderr)
            # Show which pitch classes are active
            active_notes = []
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            for i, val in enumerate(chroma_vector):
                if val > 0.1:  # Threshold for "active"
                    active_notes.append(f"{note_names[i]}({val:.3f})")
            print(f"🎯 Active Pitch Classes: {active_notes}", file=sys.stderr)
    
    # Get base chord analysis
    base_result = advanced_chord_identifier_with_instrument(notes, instrument=instrument.lower())
    
    if verbose:
        print(f"🔤 Base Analysis: Primary={base_result.get('primary_chords', [])}, Alt={base_result.get('alternative_chords', [])}", file=sys.stderr)
    
    # If we have additional data, enhance the analysis
    if detected_frequencies_all and chroma_vector is not None:
        # Calculate amplitude-weighted note strengths
        note_amplitudes = {}
        for freq, amplitude in detected_frequencies_all:
            note = frequency_to_note(freq)
            if note:
                if note not in note_amplitudes:
                    note_amplitudes[note] = 0
                note_amplitudes[note] += amplitude
        
        if verbose:
            print(f"📊 Note Amplitudes: {note_amplitudes}", file=sys.stderr)
        
        # Use chroma vector to refine chord selection
        enhanced_result = enhance_chord_with_chroma(base_result, chroma_vector, note_amplitudes, verbose=verbose)
        
        if verbose:
            print(f"✨ Enhanced Result: {enhanced_result.get('primary_chords', [])} (confidence: {enhanced_result.get('chord_confidence', 'N/A')})", file=sys.stderr)
            print(f"=" * 50, file=sys.stderr)
        
        return enhanced_result
    
    return base_result

def enhance_chord_with_chroma(base_result, chroma_vector, note_amplitudes, verbose=False):
    """
    Use chroma vector and amplitude data to refine chord identification.
    """
    note_to_chroma = {
        'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
        'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11
    }
    
    # Score each potential chord based on chroma vector alignment
    scored_chords = []
    
    all_candidates = (base_result.get('primary_chords', []) + 
                     base_result.get('alternative_chords', []))
    
    for chord_name in all_candidates:
        score = calculate_chord_chroma_score(chord_name, chroma_vector, note_amplitudes, note_to_chroma)
        scored_chords.append((chord_name, score))
        # if verbose:
        #     print(f"🎯 Chord Score: {chord_name} = {score:.3f}", file=sys.stderr)
    
    # Sort by score (highest first)
    scored_chords.sort(key=lambda x: x[1], reverse=True)
    
    if verbose and scored_chords:
        print(f"🏆 Top Scored Chords: {[(name, f'{score:.3f}') for name, score in scored_chords[:3]]}", file=sys.stderr)
    
    # Reorganize results based on chroma-enhanced scoring
    enhanced_result = base_result.copy()
    if scored_chords:
        best_chord, best_score = scored_chords[0]
        
        # Move best scoring chord to primary
        enhanced_result['primary_chords'] = [best_chord]
        enhanced_result['alternative_chords'] = [chord for chord, _ in scored_chords[1:3]]  # Top 2 alternatives
        enhanced_result['chord_confidence'] = best_score
        enhanced_result['chroma_enhanced'] = True
    
    return enhanced_result

def calculate_chord_chroma_score(chord_name, chroma_vector, note_amplitudes, note_to_chroma):
    """
    Calculate how well a chord matches the chroma vector and amplitude data.
    Root note is weighted more heavily since it is typically loudest.
    Penalizes energy in non-chord pitch classes.
    """
    if len(chord_name) == 0:
        return 0.0

    # Extract root note — handle multi-character roots like C#, F#, etc.
    if len(chord_name) >= 2 and chord_name[1] == '#':
        root_note = chord_name[:2]
        chord_type = chord_name[2:]
    else:
        root_note = chord_name[0]
        chord_type = chord_name[1:] if len(chord_name) > 1 else ""

    # Define expected chord patterns
    chord_patterns = {
        "": [0, 4, 7],          # Major
        "m": [0, 3, 7],         # Minor
        "7": [0, 4, 7, 10],     # Dominant 7th
        "maj7": [0, 4, 7, 11],  # Major 7th
        "m7": [0, 3, 7, 10],    # Minor 7th
        "5": [0, 7],            # Power chord
        "sus2": [0, 2, 7],      # Sus2
        "sus4": [0, 5, 7],      # Sus4
        "\u00b0": [0, 3, 6],         # Diminished
        "+": [0, 4, 8],         # Augmented
        "\u00b07": [0, 3, 6, 9],     # Diminished 7th
        "\u00f87": [0, 3, 6, 10],    # Half-diminished 7th
        "6": [0, 4, 7, 9],      # Major 6th
        "m6": [0, 3, 7, 9],     # Minor 6th
        "add9": [0, 4, 7, 2],   # Add 9
        "9": [0, 4, 7, 10, 2],  # Dominant 9th
    }

    pattern = chord_patterns.get(chord_type, [0, 4, 7])

    if root_note not in note_to_chroma:
        return 0.0
    root_chroma_idx = note_to_chroma[root_note]

    expected_positions = [(root_chroma_idx + interval) % 12 for interval in pattern]
    non_chord_positions = [i for i in range(12) if i not in expected_positions]

    note_names_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # --- Chroma score with root weighting ---
    # Root gets 2x weight; other chord tones get 1x
    chroma_score = 0.0
    total_weight = 0.0
    for i, pos in enumerate(expected_positions):
        weight = 2.0 if i == 0 else 1.0  # index 0 is root (interval 0)
        chroma_score += chroma_vector[pos] * weight
        total_weight += weight
    if total_weight > 0:
        chroma_score /= total_weight

    # --- Penalty for energy in non-chord pitch classes ---
    non_chord_energy = sum(chroma_vector[pos] for pos in non_chord_positions)
    chord_energy = sum(chroma_vector[pos] for pos in expected_positions)
    total_energy = chord_energy + non_chord_energy
    spill_penalty = (non_chord_energy / total_energy) if total_energy > 0 else 0.0

    # --- Amplitude score with root weighting ---
    amplitude_score = 0.0
    if note_amplitudes:
        chord_note_names = [note_names_list[(root_chroma_idx + interval) % 12] for interval in pattern]
        total_amplitude = sum(note_amplitudes.values())
        if total_amplitude > 0:
            root_name = note_names_list[root_chroma_idx]
            root_amp = note_amplitudes.get(root_name, 0)
            chord_amp = sum(note_amplitudes.get(n, 0) for n in chord_note_names)
            # Root should be strongest — bonus if it is
            root_ratio = root_amp / total_amplitude
            chord_ratio = chord_amp / total_amplitude
            amplitude_score = chord_ratio * 0.6 + root_ratio * 0.4

    # Combine: chroma 50%, amplitude 30%, spill penalty 20%
    final_score = chroma_score * 0.50 + amplitude_score * 0.30 - spill_penalty * 0.20

    return max(0.0, final_score)

def find_best_matching_chord(notes, instrument="Guitar"):
    """
    Legacy function for backward compatibility.
    Find the best matching chord for a list of notes.
    """
    return advanced_chord_identifier_with_instrument(notes, instrument=instrument.lower())

def analyze_chord_progression_enhanced(notes_history, frequencies_history=None, chroma_history=None, window_size=6, verbose=False):
    """
    Enhanced chord progression analysis using chroma vectors and frequency data.
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

    # Note must appear in at least 60% of windows to be considered stable
    threshold = max(2, int(window_size * 0.6))
    stable_notes = [note for note, count in note_counts.items() if count >= threshold]
    
    if not stable_notes:
        return None
    
    # Use enhanced analysis if we have frequency and chroma data
    if frequencies_history and chroma_history and len(frequencies_history) > 0 and len(chroma_history) > 0:
        # Aggregate recent frequency and chroma data
        recent_frequencies = frequencies_history[-1] if frequencies_history else None
        recent_chroma = chroma_history[-1] if chroma_history else None
        
        return find_best_matching_chord_enhanced(
            stable_notes, 
            detected_frequencies_all=recent_frequencies,
            chroma_vector=recent_chroma,
            verbose=verbose
        )
    
    # Fallback to basic analysis
    return find_best_matching_chord(stable_notes)

def analyze_chord_progression(notes_history, window_size=6):
    """
    Legacy function for backward compatibility.
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

    threshold = max(2, int(window_size * 0.6))
    stable_notes = [note for note, count in note_counts.items() if count >= threshold]
    
    if not stable_notes:
        return None
    
    return find_best_matching_chord(stable_notes)


def normalize_chord_variant(chord_name: str) -> str:
    """
    Map similar chord name variants to a single canonical form for stable display.
    Works with or without a song. E.g. Em7, Emin7, Emin -> Em so the UI doesn't jump.
    """
    if not chord_name or not isinstance(chord_name, str):
        return chord_name
    s = chord_name.strip()
    if not s:
        return chord_name
    # Minor: Emin7, Em7, Emin -> Em (and similar for other roots)
    if s.endswith("min7"):
        return s[:-4] + "m"
    if s.endswith("m7"):
        return s[:-2] + "m"
    if s.endswith("min"):
        return s[:-3] + "m"
    # Major 7: Cmaj7 -> C (optional; keeps display simpler)
    if s.endswith("maj7"):
        return s[:-4]
    return s


def constrain_chord_to_song(
    raw_result: dict,
    song_chords: list,
    in_song_weight: float = 1.0,
    out_of_song_penalty: float = 0.4,
    related_chord_weight: float = 0.7,
    song_influence: float = None,
    verbose: bool = False
) -> dict:
    """
    Re-weight chord candidates based on song context.
    
    Args:
        raw_result: Raw chord detection result dict with 'primary_chords', 'confidence', etc.
        song_chords: List of chords known to be in the song
        in_song_weight: Multiplier for chords that appear in the song (default: 1.0)
        out_of_song_penalty: Multiplier for chords NOT in the song (default: 0.4)
        related_chord_weight: Multiplier for harmonically related chords (default: 0.7)
        song_influence: If set (0.0-1.0), overrides weights: 0=no constraint, 1=max constraint
        verbose: If True, print debug info
    
    Returns:
        Enhanced result dict with both raw and song-constrained results
    """
    from lib.song_loader import SongLoader
    
    # Single "song influence" knob: 0 = raw wins, 1 = song context dominates
    if song_influence is not None:
        song_influence = max(0.0, min(1.0, float(song_influence)))
        # out_of_song: 1.0 (no penalty) -> 0.05 (very heavy penalty at max)
        out_of_song_penalty = 1.0 - 0.95 * song_influence
        # related: 1.0 -> 0.6 at max (related chords still slightly down so song chord wins)
        related_chord_weight = 1.0 - 0.4 * song_influence
        # in_song: stronger boost at high influence so song chords clearly win
        in_song_weight = 1.0 + 0.35 * song_influence
    
    if not song_chords:
        # No song context, return raw result with metadata
        return {
            **raw_result,
            "song_constrained": False,
            "raw_chord": raw_result.get("primary_chords", [None])[0] if raw_result.get("primary_chords") else None,
            "raw_confidence": raw_result.get("chord_confidence", 0.0)
        }
    
    loader = SongLoader()
    
    # Get all candidate chords from raw result
    all_candidates = []
    
    # Primary chords with their confidence
    if raw_result.get("primary_chords"):
        for chord in raw_result["primary_chords"]:
            conf = raw_result.get("chord_confidence", 0.5)
            all_candidates.append((chord, conf, "primary"))
    
    # Alternative chords with lower confidence
    if raw_result.get("alternative_chords"):
        for chord in raw_result["alternative_chords"]:
            conf = raw_result.get("chord_confidence", 0.3)
            all_candidates.append((chord, conf * 0.8, "alternative"))
    
    if not all_candidates:
        return {
            **raw_result,
            "song_constrained": True,
            "song_chords": song_chords,
            "raw_chord": None,
            "raw_confidence": 0.0,
            "final_chord": None,
            "final_confidence": 0.0,
            "song_match": False
        }
    
    # Influence for "suggested chord" injection (when detected is out-of-song)
    influence_for_boost = song_influence if song_influence is not None else 0.5
    suggested_boost = 0.4 + 0.95 * influence_for_boost  # 0.875 at default, 1.35 at max so song chord wins more
    
    # Score each candidate against song chords
    scored_candidates = []
    
    for chord_name, raw_conf, source in all_candidates:
        # Check if chord is directly in the song
        if chord_name in song_chords:
            weighted_conf = raw_conf * in_song_weight
            match_type = "exact"
            suggested_chord = chord_name
            best_similarity = 1.0
            
        else:
            # Find best matching chord from song
            best_match = loader.find_best_match_in_song(chord_name, song_chords)
            
            if best_match:
                similarity = best_match["similarity"]
                suggested_chord = best_match["chord"]
                best_similarity = similarity
                
                if similarity >= 0.9:
                    # Very similar (e.g., Em vs Em7)
                    weighted_conf = raw_conf * related_chord_weight
                    match_type = "related"
                elif similarity >= 0.6:
                    # Somewhat similar (shares many notes)
                    weighted_conf = raw_conf * (related_chord_weight * 0.8)
                    match_type = "partial"
                else:
                    # Not very similar
                    weighted_conf = raw_conf * out_of_song_penalty
                    match_type = "none"
            else:
                # No match found
                weighted_conf = raw_conf * out_of_song_penalty
                match_type = "none"
                suggested_chord = song_chords[0] if song_chords else chord_name
                best_similarity = 0.0
        
        scored_candidates.append({
            "detected": chord_name,
            "raw_confidence": raw_conf,
            "weighted_confidence": weighted_conf,
            "suggested": suggested_chord if chord_name not in song_chords else None,
            "match_type": match_type,
            "source": source
        })
        
        # At higher influence, add the suggested SONG chord as a competing candidate
        # so it can beat the penalized detected chord
        if chord_name not in song_chords and suggested_chord and best_similarity > 0:
            suggested_score = raw_conf * best_similarity * suggested_boost
            scored_candidates.append({
                "detected": suggested_chord,
                "raw_confidence": raw_conf,
                "weighted_confidence": suggested_score,
                "suggested": None,
                "match_type": "related" if best_similarity >= 0.9 else "partial",
                "source": "song_suggested"
            })
    
    # Collapse similar chords to a single song chord: Em, Em7, Emin → one "Em" with best score
    # so the result doesn't jump between variants when the song has only "Em"
    canonical_to_best = {}
    for c in scored_candidates:
        if c["match_type"] in ("related", "partial") and c.get("suggested"):
            canonical = c["suggested"]
        else:
            canonical = c["detected"]
        if canonical not in canonical_to_best or c["weighted_confidence"] > canonical_to_best[canonical]["weighted_confidence"]:
            canonical_to_best[canonical] = {
                "detected": canonical,
                "raw_confidence": c["raw_confidence"],
                "weighted_confidence": c["weighted_confidence"],
                "suggested": None,
                "match_type": "exact" if canonical in song_chords else c["match_type"],
                "source": c["source"]
            }
    scored_candidates = list(canonical_to_best.values())
    
    # Sort by weighted confidence
    scored_candidates.sort(key=lambda x: x["weighted_confidence"], reverse=True)
    
    if verbose and scored_candidates:
        print(f"🎵 Song-Constrained Analysis:", file=sys.stderr)
        print(f"   Song Chords: {song_chords}", file=sys.stderr)
        for i, candidate in enumerate(scored_candidates[:3]):
            print(f"   #{i+1}: {candidate['detected']} (raw: {candidate['raw_confidence']:.2f}, "
                  f"weighted: {candidate['weighted_confidence']:.2f}, match: {candidate['match_type']})",
                  file=sys.stderr)
    
    # Build enhanced result
    best = scored_candidates[0] if scored_candidates else None
    raw_best = all_candidates[0] if all_candidates else (None, 0.0, None)
    
    result = {
        **raw_result,
        "song_constrained": True,
        "song_chords": song_chords,
        "raw_chord": raw_best[0],
        "raw_confidence": raw_best[1],
        "final_chord": best["detected"] if best else None,
        "final_confidence": best["weighted_confidence"] if best else 0.0,
        "suggested_chord": best.get("suggested") if best else None,
        "match_type": best["match_type"] if best else "none",
        "song_match": best["match_type"] in ["exact", "related", "partial"] if best else False,
        "all_candidates": scored_candidates[:5]  # Top 5 for debugging
    }
    
    # Always output the song chord when we matched (exact/related/partial) so similar
    # variants (Em, Em7, Emin) map to the single chord in the song
    if best and best["match_type"] in ["exact", "related", "partial"]:
        # After collapse, best["detected"] is already the canonical chord (song chord when matched)
        result["final_chord"] = best["detected"]
        if best["match_type"] != "exact":
            result["promotion_reason"] = f"Mapped to song chord {best['detected']}"
    
    return result
