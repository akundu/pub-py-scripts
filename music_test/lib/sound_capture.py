from lib.common import clear_line, get_chunk, get_rate, get_channels, get_buffer_size
from lib.music_understanding import (
    detect_notes_with_sounddevice, frequency_to_note, 
    analyze_chord_progression, find_best_matching_chord,
    analyze_chord_progression_enhanced, find_best_matching_chord_enhanced
)
import sounddevice as sd
import numpy as np
import time
import sys

def grab_audio_chunk(chunk, rate, channels):
    """Grab a chunk of audio from the microphone"""
    myrecording = sd.rec(chunk, samplerate=rate, channels=channels, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return myrecording

def recognize_audio(args, audio_buffer, buffer_index, low_freq, high_freq, notes_history, last_chord, chord_stability, last_log_time, frequencies_history=None, chroma_history=None, debug=False, frequencies_only=False, notes_only=False, log=False):
    myrecording = grab_audio_chunk(get_chunk(), get_rate(), get_channels())
    
    # Convert to numpy array and ensure it's the right shape
    new_samples = myrecording.flatten()
    
    # Update circular buffer
    audio_buffer[buffer_index:buffer_index+get_chunk()] = new_samples
    buffer_index = (buffer_index + get_chunk()) % get_buffer_size()
    
    # Detect notes using sounddevice and autocorrelation
    detected_notes, detected_frequencies_all, chroma_vector = detect_notes_with_sounddevice(
        audio_buffer, sensitivity=args.sensitivity, 
        silence_threshold=args.silence_threshold,
        low_freq=low_freq, high_freq=high_freq,
        show_frequencies=args.show_frequencies or frequencies_only,
        show_fft=args.show_fft,
        raw_frequencies=args.raw_frequencies,
        calculate_chroma=True,  # Always calculate for enhanced chord analysis
        multi_pitch=getattr(args, 'multi_pitch', True)
    )
    
    # Debug mode
    if debug:
        rms_energy = np.sqrt(np.mean(new_samples.astype(np.float64)**2))
        print(f"Audio Level: {rms_energy:.4f} (threshold: {args.silence_threshold})", end='\r', file=sys.stderr)
        if rms_energy < args.silence_threshold:
            raise ValueError("Too low audio level")
        
    try:
        # Frequencies-only mode - skip chord processing
        if frequencies_only:
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
                
                # Add chroma vector display if requested
                chroma_str = ""
                if getattr(args, 'show_chroma', False) and chroma_vector is not None:
                    chroma_values = [f"{val:.3f}" for val in chroma_vector]
                    chroma_str = f" | Chroma: [{', '.join(chroma_values)}]"
                
                if log:
                    print(f"[{timestamp}] Frequencies: {freq_str}{chroma_str}")
                else:
                    clear_line()
                    print(f"Frequencies detected: {freq_str}{chroma_str}", end='\r')
            else:
                if log and current_time - last_log_time >= args.log_interval:
                    print(f"[{timestamp}] No frequencies detected")
                else:
                    clear_line()
                    print("🔇 No frequencies detected...", end='\r')
            last_log_time = current_time
        # Notes-only mode - show notes without chord processing
        elif notes_only:
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
                
                # Add chroma vector display if requested
                chroma_str = ""
                if getattr(args, 'show_chroma', False) and chroma_vector is not None:
                    chroma_values = [f"{val:.3f}" for val in chroma_vector]
                    chroma_str = f" | Chroma: [{', '.join(chroma_values)}]"
                
                if log and current_time - last_log_time >= args.log_interval:
                    print(f"[{timestamp}] Notes: {freq_str}{chroma_str}")
                else:
                    clear_line()
                    print(f"Notes detected: {freq_str}{chroma_str}", end='\r')
            else:
                if log and current_time - last_log_time >= args.log_interval:
                    print(f"[{timestamp}] No notes detected")
                else:
                    clear_line()
                    print("🔇 No notes detected...", end='\r')
            last_log_time = current_time
        else:
            # Default chord processing mode - NEW BEHAVIOR
            if detected_notes:
                print(f"detected_notes: {detected_notes}", file=sys.stderr) if debug else None
                notes_history.append(detected_notes)
                
                # Store frequency and chroma history for enhanced analysis
                if frequencies_history is not None:
                    frequencies_history.append(detected_frequencies_all)
                if chroma_history is not None and chroma_vector is not None:
                    chroma_history.append(chroma_vector)
                
                print(f"notes_history: {notes_history}", file=sys.stderr) if debug else None
                
                # Use enhanced chord analysis when multi-pitch is enabled
                use_enhanced = getattr(args, 'multi_pitch', True) and chroma_vector is not None
                
                # Perform chord detection
                if args.progression:
                    if use_enhanced and frequencies_history and chroma_history:
                        result = analyze_chord_progression_enhanced(
                            notes_history, frequencies_history, chroma_history,
                            verbose=debug
                        )
                    else:
                        result = analyze_chord_progression(notes_history)
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
                
                # Check confidence threshold (define before using in debug print)
                confidence_threshold = getattr(args, 'confidence_threshold', 0.6)
                
                if result:
                    if result.get('primary_chords'):
                        chord_name = result['primary_chords'][0]
                    elif result.get('power_chord'):
                        chord_name = result['power_chord']
                    elif result.get('intervals'):
                        chord_name = result['intervals'][0]
                
                if debug:
                    print(f"chord analysis: {chord_name}, confidence: {confidence:.3f}, threshold: {confidence_threshold}", file=sys.stderr)
                
                if chord_name and confidence >= confidence_threshold:
                    # Stability check for clean output
                    if chord_name == last_chord:
                        chord_stability += 1
                    else:
                        chord_stability = 0
                        last_chord = chord_name
                    
                    # Display chord (default behavior)
                    current_time = time.time()
                    
                    # Build output string based on flags
                    output_parts = []
                    
                    # Add frequency/note info if requested
                    if getattr(args, 'show_frequencies', False):
                        note_freqs = {}
                        for freq, amp in detected_frequencies_all:
                            note = frequency_to_note(freq)
                            if note in detected_notes:
                                if note not in note_freqs:
                                    note_freqs[note] = []
                                note_freqs[note].append(freq)
                        freq_str = ", ".join([f"{note}({np.mean(freqs):.0f}Hz)" for note, freqs in note_freqs.items()])
                        output_parts.append(f"Notes: {freq_str}")
                    
                    # Add chroma vector if requested
                    if getattr(args, 'show_chroma', False) and chroma_vector is not None:
                        chroma_values = [f"{val:.3f}" for val in chroma_vector]
                        output_parts.append(f"Chroma: [{', '.join(chroma_values)}]")
                    
                    # Add chord result
                    chord_display = chord_name
                    if debug:
                        chord_display += f" ({confidence:.2f})"
                    
                    # Combine output
                    if output_parts:
                        full_output = " | ".join(output_parts) + f" -> {chord_display}"
                    else:
                        # Default simple chord display
                        full_output = chord_display
                    
                    # Display with timestamp if logging
                    if log:
                        if current_time - last_log_time >= args.log_interval:
                            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                            stability_suffix = "" if chord_stability >= 2 else " [unstable]"
                            print(f"[{timestamp}] {full_output}{stability_suffix}")
                            last_log_time = current_time
                    else:
                        clear_line()
                        stability_suffix = "" if chord_stability >= 2 else " [?]"
                        print(f"{full_output}{stability_suffix}", end='\r')
                else:
                    # Below confidence threshold or no chord detected
                    if debug and chord_name:
                        print(f"Chord {chord_name} below confidence threshold ({confidence:.3f} < {confidence_threshold})", file=sys.stderr)
                    
                    # Reset stability when no confident chord
                    chord_stability = 0
                    last_chord = None
                    
                    # Only show "listening" message in non-log mode
                    if not log:
                        clear_line()
                        print("🎵 Listening...", end='\r')
            else:
                # No notes detected
                current_time = time.time()
                chord_stability = 0
                last_chord = None
                
                if log:
                    if current_time - last_log_time >= args.log_interval:
                        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                        if debug:
                            print(f"[{timestamp}] No notes detected")
                        last_log_time = current_time
                else:
                    if not debug:  # Only show in non-debug mode to avoid clutter
                        clear_line()
                        print("🔇 Listening...", end='\r')
    except Exception as e:
        import traceback
        print(f"ERROR in recognize_audio function: {type(e).__name__}: {str(e)}", file=sys.stderr)
        print(f"Exception occurred at line {traceback.extract_tb(e.__traceback__)[-1].lineno}", file=sys.stderr)
        print(f"Full traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise  # Re-raise the exception to continue the error flow


# def find_best_matching_chord(detected_notes):
#     """
#     Enhanced chord matching with confidence scoring.
#     """
#     if not detected_notes:
#         return "No chord detected", 0.0

#     detected_note_indices = sorted(list(set([NOTES.index(n) for n in detected_notes])))
#     if not detected_note_indices:
#         return "No chord detected", 0.0

#     best_match = {'score': 0, 'chord': 'Unknown', 'confidence': 0.0}

#     for i in range(12):
#         root_note = NOTES[i]

#         for chord_type, intervals in CHORD_DEFINITIONS.items():
#             chord_notes_indices = set([(i + interval) % 12 for interval in intervals])

#             matches = len(set(detected_note_indices) & chord_notes_indices)
#             extra_notes = len(chord_notes_indices) - matches
#             missing_notes = len(detected_note_indices) - matches

#             score = matches * 2 - extra_notes - missing_notes * 0.5

#             if len(chord_notes_indices) > 0:
#                 confidence = matches / len(chord_notes_indices)
#             else:
#                 confidence = 0.0

#             if score > best_match['score']:
#                 best_match['score'] = score
#                 best_match['chord'] = f"{root_note} {chord_type}"
#                 best_match['confidence'] = confidence

#     if best_match['score'] < 1.5 or best_match['confidence'] < 0.6:
#         return "Unknown or complex chord", best_match['confidence']

#     return best_match['chord'], best_match['confidence']
