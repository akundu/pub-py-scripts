"""
Core audio processing function - shared between CLI and web.
"""
import numpy as np
import time
import sys
from lib.common import get_chunk, get_rate
from lib.music_understanding import (
    detect_notes_with_sounddevice, frequency_to_note,
    analyze_chord_progression, find_best_matching_chord,
    analyze_chord_progression_enhanced, find_best_matching_chord_enhanced
)


def process_audio_chunk(new_samples, state, config, output_handler=None, debug_log_level='INFO'):
    """
    Core audio processing - shared between CLI and web.
    
    This function processes an audio chunk through the detection pipeline:
    1. Update circular buffer
    2. Detect notes using FFT/autocorrelation
    3. Build detection history
    4. Perform chord analysis
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
    
    if debug:
        rms_energy = np.sqrt(np.mean(new_samples.astype(np.float64)**2))
        if debug_log_level == 'DEBUG':
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] Audio Level: {rms_energy:.4f} (threshold: {silence_threshold})", 
                  end='\r', file=sys.stderr)
        if rms_energy < silence_threshold:
            raise ValueError("Too low audio level")
    
    # Update circular buffer
    chunk_size = get_chunk()
    state.update_buffer(new_samples, chunk_size)
    
    # Detect notes
    detected_notes, detected_frequencies_all, chroma_vector = detect_notes_with_sounddevice(
        state.audio_buffer,
        sample_rate=get_rate(),
        sensitivity=get_config('sensitivity', 1.0),
        silence_threshold=silence_threshold,
        low_freq=state.low_freq,
        high_freq=state.high_freq,
        show_frequencies=get_config('show_frequencies', False) or get_config('frequencies_only', False),
        show_fft=get_config('show_fft', False),
        raw_frequencies=get_config('raw_frequencies', False),
        calculate_chroma=True,  # Always calculate for enhanced chord analysis
        multi_pitch=get_config('multi_pitch', True)
    )
    
    # Debug log detection
    if debug and debug_log_level == 'DEBUG' and detected_notes:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] detected_notes: {detected_notes}", file=sys.stderr)
    
    # Handle different output modes
    frequencies_only = get_config('frequencies_only', False)
    notes_only = get_config('notes_only', False)
    log_mode = get_config('log', False)
    log_interval = get_config('log_interval', 0.5)
    
    # Frequencies-only mode
    if frequencies_only:
        if detected_notes and detected_frequencies_all:
            if output_handler:
                return output_handler.frequencies_detected(detected_frequencies_all, chroma_vector, config)
            else:
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
    
    # Notes-only mode
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
    
    # Default chord processing mode
    if detected_notes:
        # Add to history
        state.add_detection(detected_notes, detected_frequencies_all, chroma_vector)
        
        if debug and debug_log_level == 'DEBUG':
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] notes_history length: {len(state.notes_history)}, contents: {list(state.notes_history)}", 
                  file=sys.stderr)
        
        # Check if we have enough history for progression analysis
        progression = get_config('progression', True)
        if progression and not state.has_enough_history(3):
            if debug and debug_log_level == 'DEBUG':
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Building history: {len(state.notes_history)}/3 samples", file=sys.stderr)
            return None
        
        # Use enhanced chord analysis when multi-pitch is enabled
        use_enhanced = get_config('multi_pitch', True) and chroma_vector is not None
        
        # Perform chord detection
        if progression:
            if use_enhanced and len(state.frequencies_history) >= 3 and len(state.chroma_history) >= 3:
                result = analyze_chord_progression_enhanced(
                    state.notes_history, state.frequencies_history, state.chroma_history,
                    verbose=debug
                )
            else:
                result = analyze_chord_progression(state.notes_history)
        else:
            if use_enhanced:
                result = find_best_matching_chord_enhanced(
                    detected_notes, detected_frequencies_all, chroma_vector,
                    verbose=debug
                )
            else:
                result = find_best_matching_chord(detected_notes)
        
        # Extract chord information
        chord_name = None
        confidence = result.get('chord_confidence', 0.0) if result else 0.0
        
        if result:
            if result.get('primary_chords'):
                chord_name = result['primary_chords'][0]
            elif result.get('power_chord'):
                chord_name = result['power_chord']
            elif result.get('intervals'):
                chord_name = result['intervals'][0]
        
        # Check confidence threshold
        confidence_threshold = get_config('confidence_threshold', 0.6)
        
        if debug and debug_log_level == 'DEBUG':
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{timestamp}] chord analysis: {chord_name}, confidence: {confidence:.3f}, threshold: {confidence_threshold}", 
                  file=sys.stderr)
        
        # Determine if we should output the chord
        # In log mode: always show detected chords (even if below threshold)
        # In non-log mode: only show chords above threshold
        if log_mode and chord_name:
            should_output = True
        elif chord_name and confidence >= confidence_threshold:
            should_output = True
        else:
            should_output = False
        
        if should_output:
            # Update stability
            stability = state.update_chord_stability(chord_name)
            
            # Check log timing
            if log_mode and not state.should_log(log_interval):
                # Store pending chord for later output
                if not hasattr(state, 'pending_chord'):
                    state.pending_chord = None
                state.pending_chord = {
                    "chord_name": chord_name,
                    "confidence": confidence,
                    "stability": stability,
                    "detected_notes": detected_notes,
                    "detected_frequencies": detected_frequencies_all,
                    "chroma_vector": chroma_vector
                }
                return None
            
            # Output result
            if output_handler:
                return output_handler.chord_detected(
                    chord_name, confidence, stability,
                    detected_notes, detected_frequencies_all, chroma_vector,
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
                if get_config('show_frequencies'):
                    note_freqs = {}
                    for freq, amp in detected_frequencies_all:
                        note = frequency_to_note(freq)
                        if note in detected_notes:
                            if note not in note_freqs:
                                note_freqs[note] = []
                            note_freqs[note].append(freq)
                    result_dict["notes"] = [(note, float(np.mean(freqs))) for note, freqs in note_freqs.items()]
                if get_config('show_chroma') and chroma_vector is not None:
                    result_dict["chroma"] = chroma_vector.tolist()
                return result_dict
        else:
            # Below confidence threshold
            if debug and chord_name:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Chord {chord_name} below confidence threshold ({confidence:.3f} < {confidence_threshold})", 
                      file=sys.stderr)
            
            state.reset_stability()
            
            if output_handler:
                return output_handler.listening(config)
            return None
    else:
        # No notes detected
        state.reset_stability()
        
        if output_handler:
            return output_handler.no_detection(config)
        return None

