"""
Web audio processing functions - refactored version of recognize_audio that accepts audio data directly.
"""
import numpy as np
import time
import sys
from lib.common import get_chunk, get_rate, get_buffer_size
from lib.music_understanding import (
    detect_notes_with_sounddevice, frequency_to_note, 
    analyze_chord_progression, find_best_matching_chord,
    analyze_chord_progression_enhanced, find_best_matching_chord_enhanced
)

# Try to import server log level (may not be available if imported before web_server is initialized)
try:
    from web_server import SERVER_LOG_LEVEL
except ImportError:
    SERVER_LOG_LEVEL = "INFO"

def get_timestamp():
    """Get formatted timestamp matching CLI format."""
    return time.strftime("%Y-%m-%d %H:%M:%S")

def recognize_audio_web(audio_chunk, state, config):
    """
    Process audio chunk from web stream and return detection results.
    
    Args:
        audio_chunk: numpy array of audio samples (Float32)
        state: ConnectionState object with buffers and history
        config: configuration dictionary
    
    Returns:
        dict with detection results or None if no valid detection
    """
    # Ensure audio chunk is the right shape and type
    new_samples = audio_chunk.flatten().astype(np.float32)
    
    # CLI processes every chunk continuously - we should do the same
    # The difference is CLI waits for sounddevice to record, web gets chunks faster
    # But we should still process every chunk to build history properly
    current_time = time.time()
    
    # Debug mode - check amplitude (match CLI behavior exactly)
    # CLI only checks in debug mode, and uses silence_threshold for the check
    # Only check and raise in debug mode, but don't log unless SERVER_LOG_LEVEL is DEBUG
    if config.get('debug'):
        rms_energy = np.sqrt(np.mean(new_samples.astype(np.float64)**2))
        silence_threshold = config.get('silence_threshold', 0.005)
        if SERVER_LOG_LEVEL == 'DEBUG':
            print(f"[{get_timestamp()}] Audio Level: {rms_energy:.4f} (threshold: {silence_threshold})", end='\r', file=sys.stderr)
        if rms_energy < silence_threshold:
            raise ValueError("Too low audio level")
    
    # Update circular buffer
    chunk_size = get_chunk()
    buffer_size = get_buffer_size()
    
    # Update buffer (circular)
    # Handle case where chunk might be smaller than expected
    actual_chunk_size = min(chunk_size, len(new_samples))
    end_index = min(state.buffer_index + actual_chunk_size, buffer_size)
    copy_size = end_index - state.buffer_index
    
    if copy_size > 0:
        state.audio_buffer[state.buffer_index:end_index] = new_samples[:copy_size]
    
    # If chunk is larger than remaining buffer space, wrap around
    if actual_chunk_size > copy_size:
        remaining = actual_chunk_size - copy_size
        state.audio_buffer[:remaining] = new_samples[copy_size:copy_size+remaining]
    
    state.buffer_index = (state.buffer_index + actual_chunk_size) % buffer_size
    
    # Always detect notes to build up history (like CLI does)
    # But throttle chord detection to avoid spam
    # Detect notes (match CLI defaults exactly)
    detected_notes, detected_frequencies_all, chroma_vector = detect_notes_with_sounddevice(
        state.audio_buffer, 
        sample_rate=get_rate(),
        sensitivity=config.get('sensitivity', 1.0), 
        silence_threshold=config.get('silence_threshold', 0.005),  # Match CLI default
        low_freq=state.low_freq, 
        high_freq=state.high_freq,
        show_frequencies=config.get('show_frequencies', False),
        show_fft=config.get('show_fft', False),
        raw_frequencies=config.get('raw_frequencies', False),
        calculate_chroma=True,  # Always calculate for enhanced chord analysis
        multi_pitch=config.get('multi_pitch', True)
    )
    
    # Debug: log detection results (match CLI behavior)
    if config.get('debug'):
        if detected_notes:
            print(f"detected_notes: {detected_notes}", file=sys.stderr)
    
    # Debug mode
    if config.get('debug'):
        print(f"Audio Level: {rms_energy:.4f} (threshold: {config.get('silence_threshold', 0.005)})", 
              file=sys.stderr)
        if detected_notes:
            print(f"detected_notes: {detected_notes}", file=sys.stderr)
    
    try:
        # Frequencies-only mode
        if config.get('frequencies_only', False):
            current_time = time.time()
            if detected_notes:
                # Get frequencies for the detected notes
                note_freqs = {}
                for freq, amp in detected_frequencies_all:
                    note = frequency_to_note(freq)
                    if note in detected_notes:
                        if note not in note_freqs:
                            note_freqs[note] = []
                        note_freqs[note].append(freq)
                
                frequencies = [(note, np.mean(freqs)) for note, freqs in note_freqs.items()]
                
                result = {
                    "type": "frequencies",
                    "frequencies": frequencies,
                    "timestamp": current_time
                }
                
                if config.get('show_chroma') and chroma_vector is not None:
                    result["chroma"] = chroma_vector.tolist()
                
                state.last_log_time = current_time
                return result
            else:
                if config.get('log') and current_time - state.last_log_time >= config.get('log_interval', 0.5):
                    state.last_log_time = current_time
                    return {
                        "type": "frequencies",
                        "frequencies": [],
                        "timestamp": current_time
                    }
                return None
        
        # Notes-only mode
        elif config.get('notes_only', False):
            current_time = time.time()
            if detected_notes:
                # Get frequencies for the detected notes
                note_freqs = {}
                for freq, amp in detected_frequencies_all:
                    note = frequency_to_note(freq)
                    if note in detected_notes:
                        if note not in note_freqs:
                            note_freqs[note] = []
                        note_freqs[note].append(freq)
                
                notes = [(note, np.mean(freqs)) for note, freqs in note_freqs.items()]
                
                result = {
                    "type": "notes",
                    "notes": notes,
                    "timestamp": current_time
                }
                
                if config.get('show_chroma') and chroma_vector is not None:
                    result["chroma"] = chroma_vector.tolist()
                
                if config.get('log') and current_time - state.last_log_time >= config.get('log_interval', 0.5):
                    state.last_log_time = current_time
                    return result
                return result
            else:
                if config.get('log') and current_time - state.last_log_time >= config.get('log_interval', 0.5):
                    state.last_log_time = current_time
                    return {
                        "type": "notes",
                        "notes": [],
                        "timestamp": current_time
                    }
                return None
        
        # Default chord processing mode
        else:
            # In log mode, check if we should return a pending chord from previous processing
            if config.get('log') and hasattr(state, 'pending_chord'):
                current_time = time.time()
                if current_time - state.last_log_time >= config.get('log_interval', 0.5):
                    # Return the pending chord
                    pending = state.pending_chord
                    delattr(state, 'pending_chord')
                    state.last_log_time = current_time
                    return pending
            
            if detected_notes:
                # Only print detected_notes in DEBUG log level (not ERROR/INFO/WARNING)
                if SERVER_LOG_LEVEL == 'DEBUG':
                    print(f"[{get_timestamp()}] detected_notes: {detected_notes}", file=sys.stderr)
                
                state.notes_history.append(detected_notes)
                
                # Store frequency and chroma history for enhanced analysis
                if state.frequencies_history is not None:
                    state.frequencies_history.append(detected_frequencies_all)
                if state.chroma_history is not None and chroma_vector is not None:
                    state.chroma_history.append(chroma_vector)
                
                # Only print notes_history in DEBUG log level (not ERROR/INFO/WARNING)
                if SERVER_LOG_LEVEL == 'DEBUG':
                    print(f"[{get_timestamp()}] notes_history length: {len(state.notes_history)}, contents: {list(state.notes_history)}", file=sys.stderr)
                
                # Use enhanced chord analysis when multi-pitch is enabled
                use_enhanced = config.get('multi_pitch', True) and chroma_vector is not None
                
                # Perform chord detection
                # Process every chunk like CLI does - the log interval check happens later
                
                # For progression analysis, we need at least some history
                if config.get('progression', True):
                    # Wait for history to build up (progression needs multiple samples)
                    if len(state.notes_history) < 3:
                        if SERVER_LOG_LEVEL == 'DEBUG':
                            print(f"[{get_timestamp()}] Building history: {len(state.notes_history)}/3 samples", file=sys.stderr)
                        return None  # Not enough history yet
                    
                    if use_enhanced and state.frequencies_history and state.chroma_history and len(state.frequencies_history) >= 3 and len(state.chroma_history) >= 3:
                        result = analyze_chord_progression_enhanced(
                            state.notes_history, state.frequencies_history, state.chroma_history,
                            verbose=config.get('debug', False)
                        )
                    else:
                        result = analyze_chord_progression(state.notes_history)
                else:
                    if use_enhanced:
                        result = find_best_matching_chord_enhanced(
                            detected_notes, detected_frequencies_all, chroma_vector,
                            verbose=config.get('debug', False)
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
                
                # Check confidence threshold (define before using in debug/log)
                confidence_threshold = config.get('confidence_threshold', 0.6)
                log_mode = config.get('log', False)
                
                # Only print chord analysis in DEBUG log level (not ERROR/INFO/WARNING)
                if SERVER_LOG_LEVEL == 'DEBUG':
                    print(f"[{get_timestamp()}] chord analysis: {chord_name}, confidence: {confidence:.3f}, threshold: {confidence_threshold}", file=sys.stderr)
                
                # In log mode, always show chords (even if below threshold) - matches CLI behavior
                # In non-log mode, only show chords above threshold
                if log_mode and chord_name:
                    # Log mode: show all detected chords
                    should_show_chord = True
                elif chord_name and confidence >= confidence_threshold:
                    # Non-log mode: only show chords above threshold
                    should_show_chord = True
                else:
                    should_show_chord = False
                
                if should_show_chord:
                    # Stability check
                    if chord_name == state.last_chord:
                        state.chord_stability += 1
                    else:
                        state.chord_stability = 0
                        state.last_chord = chord_name
                    
                    current_time = time.time()
                    
                    # Build result dictionary
                    result_dict = {
                        "type": "chord",
                        "chord": chord_name,
                        "confidence": confidence,
                        "stability": state.chord_stability,
                        "timestamp": current_time
                    }
                    
                    # Add frequency/note info if requested
                    if config.get('show_frequencies'):
                        note_freqs = {}
                        for freq, amp in detected_frequencies_all:
                            note = frequency_to_note(freq)
                            if note in detected_notes:
                                if note not in note_freqs:
                                    note_freqs[note] = []
                                note_freqs[note].append(freq)
                        result_dict["notes"] = [(note, np.mean(freqs)) for note, freqs in note_freqs.items()]
                    
                    # Add chroma vector if requested
                    if config.get('show_chroma') and chroma_vector is not None:
                        result_dict["chroma"] = chroma_vector.tolist()
                    
                    # Log timing - in log mode, only return when interval has passed (matches CLI)
                    if log_mode:
                        # Check if log interval has passed
                        if current_time - state.last_log_time >= config.get('log_interval', 0.5):
                            state.last_log_time = current_time
                            return result_dict
                        # Store the most recent chord for next interval (CLI behavior)
                        # Don't return yet, but keep processing
                        if not hasattr(state, 'pending_chord') or current_time > state.pending_chord.get('timestamp', 0):
                            state.pending_chord = result_dict
                        return None
                    else:
                        # Non-log mode: return immediately
                        return result_dict
                else:
                    # No chord detected or below threshold (and not in log mode)
                    if (config.get('debug') or SERVER_LOG_LEVEL == 'DEBUG') and chord_name:
                        print(f"[{get_timestamp()}] Chord {chord_name} below confidence threshold ({confidence:.3f} < {confidence_threshold})", 
                              file=sys.stderr)
                    
                    # Reset stability when no confident chord
                    state.chord_stability = 0
                    state.last_chord = None
                    
                    # In log mode, still return chord candidate even if below threshold (match CLI behavior)
                    # CLI shows chords even when unstable, so we should too in log mode
                    if log_mode and chord_name:
                        current_time = time.time()
                        result_dict = {
                            "type": "chord",
                            "chord": chord_name,
                            "confidence": confidence,
                            "stability": 0,  # Always unstable if below threshold
                            "timestamp": current_time
                        }
                        
                        # Add chroma if requested
                        if config.get('show_chroma') and chroma_vector is not None:
                            result_dict["chroma"] = chroma_vector.tolist()
                        
                        # Add notes if requested
                        if config.get('show_frequencies') and detected_notes:
                            note_freqs = {}
                            for freq, amp in detected_frequencies_all:
                                note = frequency_to_note(freq)
                                if note in detected_notes:
                                    if note not in note_freqs:
                                        note_freqs[note] = []
                                    note_freqs[note].append(freq)
                            result_dict["notes"] = [(note, np.mean(freqs)) for note, freqs in note_freqs.items()]
                        
                        # Only log if interval has passed
                        if current_time - state.last_log_time >= config.get('log_interval', 0.5):
                            state.last_log_time = current_time
                            return result_dict
                        return None
                    else:
                        # Not in log mode or no chord candidate - return notes
                        result_dict = {
                            "type": "notes",
                            "notes": [],
                            "timestamp": current_time
                        }
                        
                        # Add detected notes
                        if detected_notes:
                            note_freqs = {}
                            for freq, amp in detected_frequencies_all:
                                note = frequency_to_note(freq)
                                if note in detected_notes:
                                    if note not in note_freqs:
                                        note_freqs[note] = []
                                    note_freqs[note].append(freq)
                            result_dict["notes"] = [(note, np.mean(freqs)) for note, freqs in note_freqs.items()]
                            result_dict["chord_candidate"] = chord_name if chord_name else None
                            result_dict["confidence"] = confidence
                        
                        return result_dict
            else:
                # No notes detected
                current_time = time.time()
                state.chord_stability = 0
                state.last_chord = None
                
                if config.get('log'):
                    if current_time - state.last_log_time >= config.get('log_interval', 0.5):
                        state.last_log_time = current_time
                        return {
                            "type": "listening",
                            "timestamp": current_time
                        }
                    return None
                else:
                    return {
                        "type": "listening",
                        "timestamp": current_time
                    }
                    
    except Exception as e:
        import traceback
        print(f"[{get_timestamp()}] ERROR in recognize_audio_web: {type(e).__name__}: {str(e)}", file=sys.stderr)
        if config.get('debug'):
            traceback.print_exc(file=sys.stderr)
        raise

