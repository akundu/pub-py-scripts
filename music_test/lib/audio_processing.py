"""
Core audio processing function - shared between CLI and web.
"""
import numpy as np
import time
import sys
from lib.common import get_chunk, get_rate
from lib.music_understanding import (
    detect_notes_with_sounddevice, frequency_to_note,
    detect_chord_from_buffer
)


def process_audio_chunk(new_samples, state, config, output_handler=None, debug_log_level='INFO'):
    """
    Core audio processing - shared between CLI and web.

    Pipeline:
    1. Update circular buffer with new samples
    2. For frequencies/notes modes: run FFT note detection
    3. For chord mode: run chroma-template matching on the buffer
    4. Apply temporal smoothing via chord accumulation window
    5. Format and output results

    Args:
        new_samples: numpy array of audio samples (Float32)
        state: AudioProcessingState object with buffers and history
        config: AudioConfig object or dict-like configuration
        output_handler: OutputHandler for result formatting (optional)
                       If None, returns dict result (web mode)
        debug_log_level: server log level for debug output

    Returns:
        dict with detection results (web mode) or None (CLI mode with output_handler)

    Raises:
        ValueError: if audio level is too low (in debug mode)
    """
    # Ensure audio chunk is the right type
    new_samples = new_samples.flatten().astype(np.float32)

    # Get config values (handle both dict and object)
    def get_config(key, default=None):
        if hasattr(config, 'get'):
            return config.get(key, default)
        return getattr(config, key, default)

    # Debug mode - check audio level
    debug = get_config('debug', False)
    silence_threshold = get_config('silence_threshold', 0.005)

    # Always calculate RMS for diagnostics
    rms_energy = np.sqrt(np.mean(new_samples.astype(np.float64)**2))
    max_amplitude = np.max(np.abs(new_samples))

    if debug:
        if debug_log_level == 'DEBUG':
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Audio Level: RMS={rms_energy:.6f}, Max={max_amplitude:.6f}, Threshold={silence_threshold:.6f}",
                  end='\r', file=sys.stderr)
        if rms_energy < silence_threshold:
            raise ValueError(f"Too low audio level: RMS={rms_energy:.6f} < threshold={silence_threshold:.6f}")

    # Update circular buffer
    chunk_size = get_chunk()
    state.update_buffer(new_samples, chunk_size)

    # Handle different output modes
    frequencies_only = get_config('frequencies_only', False)
    notes_only = get_config('notes_only', False)
    log_mode = get_config('log', False)
    log_interval = get_config('log_interval', 0.5)

    # --- Frequencies-only and notes-only modes use FFT note detection ---
    if frequencies_only or notes_only:
        detected_notes, detected_frequencies_all, chroma_vector = detect_notes_with_sounddevice(
            state.audio_buffer,
            sample_rate=get_rate(),
            sensitivity=get_config('sensitivity', 1.0),
            silence_threshold=silence_threshold,
            low_freq=state.low_freq,
            high_freq=state.high_freq,
            show_frequencies=get_config('show_frequencies', False) or frequencies_only,
            show_fft=get_config('show_fft', False),
            raw_frequencies=get_config('raw_frequencies', False),
            calculate_chroma=True,
            multi_pitch=get_config('multi_pitch', True)
        )

        if frequencies_only:
            if detected_notes and detected_frequencies_all:
                if output_handler:
                    return output_handler.frequencies_detected(detected_frequencies_all, chroma_vector, config)
                return {
                    "type": "frequencies",
                    "frequencies": [(freq, amp) for freq, amp in detected_frequencies_all],
                    "chroma": chroma_vector.tolist() if chroma_vector is not None else None,
                    "timestamp": time.time()
                }
            else:
                if output_handler:
                    return output_handler.no_detection(config)
                return None

        if notes_only:
            if detected_notes:
                if output_handler:
                    return output_handler.notes_detected(detected_notes, detected_frequencies_all, chroma_vector, config)
                else:
                    note_freqs = {}
                    for freq, amp in detected_frequencies_all:
                        note = frequency_to_note(freq)
                        if note in detected_notes:
                            if note not in note_freqs:
                                note_freqs[note] = []
                            note_freqs[note].append(freq)
                    return {
                        "type": "notes",
                        "notes": [(note, float(np.mean(freqs))) for note, freqs in note_freqs.items()],
                        "chroma": chroma_vector.tolist() if chroma_vector is not None else None,
                        "timestamp": time.time()
                    }
            else:
                if output_handler:
                    return output_handler.no_detection(config)
                return None

    # --- Default chord processing mode: chroma-template matching ---
    chord_name, confidence, chroma_vector, detected_notes, detected_freqs = detect_chord_from_buffer(
        state.audio_buffer,
        sample_rate=get_rate(),
        silence_threshold=silence_threshold,
        low_freq=state.low_freq,
        high_freq=state.high_freq
    )

    if debug and debug_log_level == 'DEBUG':
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] chroma chord: {chord_name}, conf={confidence:.3f}, notes={detected_notes}",
              file=sys.stderr)

    if not chord_name:
        state.reset_stability()
        if output_handler:
            return output_handler.no_detection(config)
        return None

    # Note: Temporal smoothing (chord accumulation window) is handled by callers:
    # - chord_detector.py uses state.accumulate_chord / get_best_chord
    # - web_server.py can optionally smooth or use per-buffer results directly

    # Check confidence threshold
    confidence_threshold = get_config('confidence_threshold', 0.4)

    if debug and debug_log_level == 'DEBUG':
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] chord result: {chord_name}, confidence: {confidence:.3f}, threshold: {confidence_threshold}",
              file=sys.stderr)

    # Determine if we should output the chord
    if log_mode and chord_name:
        should_output = True
    elif chord_name and confidence >= confidence_threshold:
        should_output = True
    else:
        should_output = False

    if should_output:
        stability = state.update_chord_stability(chord_name)

        # Check log timing
        if log_mode and not state.should_log(log_interval):
            if not hasattr(state, 'pending_chord'):
                state.pending_chord = None
            state.pending_chord = {
                "chord_name": chord_name,
                "confidence": confidence,
                "stability": stability,
                "detected_notes": detected_notes,
                "detected_frequencies": detected_freqs,
                "chroma_vector": chroma_vector
            }
            return None

        # Output result
        if output_handler:
            return output_handler.chord_detected(
                chord_name, confidence, stability,
                detected_notes, detected_freqs, chroma_vector,
                config
            )
        else:
            result_dict = {
                "type": "chord",
                "chord": chord_name,
                "confidence": confidence,
                "stability": stability,
                "timestamp": time.time()
            }
            if get_config('show_frequencies') and detected_freqs:
                result_dict["notes"] = [(frequency_to_note(f), f) for f, _ in detected_freqs]
            if get_config('show_chroma') and chroma_vector is not None:
                result_dict["chroma"] = chroma_vector.tolist()
            return result_dict
    else:
        if debug and chord_name:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Chord {chord_name} below confidence threshold ({confidence:.3f} < {confidence_threshold})",
                  file=sys.stderr)

        state.reset_stability()

        if output_handler:
            return output_handler.listening(config)
        return None

