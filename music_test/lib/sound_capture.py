"""
Sound capture and audio processing for CLI.
Uses the core audio processing function for chord detection.
"""
from lib.common import clear_line, get_chunk, get_rate, get_channels, get_buffer_size
from lib.audio_processing import process_audio_chunk
from lib.state import AudioProcessingState
from lib.output import ConsoleOutputHandler
import sounddevice as sd
import numpy as np
import time
import sys


def grab_audio_chunk(chunk, rate, channels, device=None):
    """Grab a chunk of audio from the microphone"""
    try:
        # Try to explicitly use the default input device if not specified
        if device is None:
            # Get the default input device ID
            default_input = sd.default.device[0]
            if default_input is not None:
                device = default_input
        
        # Record with explicit device selection
        myrecording = sd.rec(chunk, samplerate=rate, channels=channels, dtype='float32', device=device)
        sd.wait()  # Wait until recording is finished
        
        # Check if we got actual audio data
        if np.all(myrecording == 0):
            import sys
            print("⚠️  WARNING: Recording returned all zeros", file=sys.stderr)
            print("   Possible causes:", file=sys.stderr)
            print("   1. macOS: Check System Settings → Privacy & Security → Microphone", file=sys.stderr)
            print("   2. Grant microphone access to Terminal or your Python interpreter", file=sys.stderr)
            print("   3. Verify the microphone is working in another app", file=sys.stderr)
            print("   4. Try selecting a different input device", file=sys.stderr)
        
        return myrecording
    except PermissionError as e:
        import sys
        print(f"❌ PERMISSION ERROR: {e}", file=sys.stderr)
        print("   On macOS: System Settings → Privacy & Security → Microphone", file=sys.stderr)
        print("   Grant access to Terminal or your Python interpreter", file=sys.stderr)
        raise
    except Exception as e:
        import sys
        print(f"❌ ERROR capturing audio: {e}", file=sys.stderr)
        print("   Check microphone permissions and that a microphone is connected", file=sys.stderr)
        raise


def recognize_audio(args, audio_buffer, buffer_index, low_freq, high_freq, notes_history, 
                    last_chord, chord_stability, last_log_time, frequencies_history=None, 
                    chroma_history=None, debug=False, frequencies_only=False, notes_only=False, log=False):
    """
    CLI audio recognition function.
    
    This function maintains backward compatibility with the original signature while
    using the shared core processing function internally.
    
    Args:
        args: argparse namespace with configuration
        audio_buffer: numpy array for circular buffer (modified in place)
        buffer_index: current position in circular buffer
        low_freq: minimum frequency to detect
        high_freq: maximum frequency to detect
        notes_history: deque of detected notes history
        last_chord: last detected chord name
        chord_stability: count of consecutive same-chord detections
        last_log_time: timestamp of last log output
        frequencies_history: optional deque of frequency history
        chroma_history: optional deque of chroma vector history
        debug: enable debug output
        frequencies_only: only output frequencies
        notes_only: only output notes
        log: enable timestamped logging
    
    Note: This function modifies audio_buffer and notes_history in place.
          Returns are used by caller to update chord_stability and last_log_time.
    """
    # Capture audio from microphone
    myrecording = grab_audio_chunk(get_chunk(), get_rate(), get_channels())
    new_samples = myrecording.flatten()
    
    # Create a temporary state object that mirrors the passed-in state
    # This allows us to use the core function while maintaining compatibility
    class TempState:
        def __init__(self):
            self.audio_buffer = audio_buffer
            self.buffer_index = buffer_index
            self.notes_history = notes_history
            self.frequencies_history = frequencies_history if frequencies_history is not None else []
            self.chroma_history = chroma_history if chroma_history is not None else []
            self.last_chord = last_chord
            self.chord_stability = chord_stability
            self.last_log_time = last_log_time
            self.low_freq = low_freq
            self.high_freq = high_freq
            self.instrument_name = getattr(args, 'instrument_name', 'Guitar')
        
        def update_buffer(self, samples, chunk_size):
            """Update circular buffer."""
            buffer_size = len(self.audio_buffer)
            actual_chunk_size = min(chunk_size, len(samples))
            end_index = min(self.buffer_index + actual_chunk_size, buffer_size)
            copy_size = end_index - self.buffer_index
            
            if copy_size > 0:
                self.audio_buffer[self.buffer_index:end_index] = samples[:copy_size]
            
            if actual_chunk_size > copy_size:
                remaining = actual_chunk_size - copy_size
                self.audio_buffer[:remaining] = samples[copy_size:copy_size + remaining]
            
            self.buffer_index = (self.buffer_index + actual_chunk_size) % buffer_size
            return self.buffer_index
        
        def add_detection(self, notes, frequencies, chroma):
            """Add detection to history."""
            if notes:
                self.notes_history.append(notes)
            if frequencies and hasattr(self.frequencies_history, 'append'):
                self.frequencies_history.append(frequencies)
            if chroma is not None and hasattr(self.chroma_history, 'append'):
                self.chroma_history.append(chroma)
        
        def update_chord_stability(self, chord_name):
            """Update chord stability."""
            if chord_name == self.last_chord:
                self.chord_stability += 1
            else:
                self.chord_stability = 0
                self.last_chord = chord_name
            return self.chord_stability
        
        def reset_stability(self):
            """Reset chord stability."""
            self.chord_stability = 0
            self.last_chord = None
        
        def should_log(self, log_interval):
            """Check if should log now."""
            current_time = time.time()
            if current_time - self.last_log_time >= log_interval:
                self.last_log_time = current_time
                return True
            return False
        
        def has_enough_history(self, min_samples=3):
            """Check if enough history for progression."""
            return len(self.notes_history) >= min_samples
    
    state = TempState()
    
    # Create config dict from args
    config = {
        'instrument': getattr(args, 'instrument', 'guitar'),
        'silence_threshold': getattr(args, 'silence_threshold', 0.005),
        'confidence_threshold': getattr(args, 'confidence_threshold', 0.45),
        'overlap': getattr(args, 'overlap', 0.75),
        'show_frequencies': getattr(args, 'show_frequencies', False),
        'show_chroma': getattr(args, 'show_chroma', False),
        'frequencies_only': frequencies_only,
        'notes_only': notes_only,
        'debug': debug,
        'log': log,
        'log_interval': getattr(args, 'log_interval', 0.5),
        # Frequencies/notes-only mode params
        'sensitivity': getattr(args, 'sensitivity', 1.0),
        'multi_pitch': getattr(args, 'multi_pitch', True),
        'show_fft': getattr(args, 'show_fft', False),
        'raw_frequencies': getattr(args, 'raw_frequencies', False),
    }
    
    # Create console output handler
    output_handler = ConsoleOutputHandler(config)
    
    try:
        # Use core processing function
        process_audio_chunk(
            new_samples, state, config,
            output_handler=output_handler,
            debug_log_level='DEBUG' if debug else 'INFO'
        )
    except ValueError as e:
        # Handle "Too low audio level" error
        if debug:
            print(f"Audio level error: {e}", file=sys.stderr)
        raise
    except Exception as e:
        import traceback
        print(f"ERROR in recognize_audio function: {type(e).__name__}: {str(e)}", file=sys.stderr)
        print(f"Exception occurred at line {traceback.extract_tb(e.__traceback__)[-1].lineno}", file=sys.stderr)
        print(f"Full traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
    
    # Note: The caller expects to update their local variables based on state changes.
    # Since Python passes mutable objects by reference, audio_buffer and notes_history
    # are already updated. However, the caller's buffer_index, last_chord, chord_stability,
    # and last_log_time need to be returned or the caller needs to be modified.
    #
    # For backward compatibility, we return nothing here and the caller must track
    # buffer_index separately using buffer_index = (buffer_index + get_chunk()) % get_buffer_size()
    #
    # TODO: Consider returning state changes to the caller in a future refactor.


def recognize_audio_with_result(state, config):
    """
    CLI audio recognition function that returns the result instead of printing.

    Used for chord_window accumulation mode where we need to collect results
    before deciding what to output.

    Args:
        state: AudioProcessingState object
        config: dict with configuration

    Returns:
        dict with detection result or None
    """
    # Capture audio from microphone
    myrecording = grab_audio_chunk(get_chunk(), get_rate(), get_channels())
    new_samples = myrecording.flatten()

    debug = config.get('debug', False)

    try:
        # Use core processing function without output handler to get result
        result = process_audio_chunk(
            new_samples, state, config,
            output_handler=None,  # Return dict instead of printing
            debug_log_level='DEBUG' if debug else 'INFO'
        )
        return result
    except ValueError as e:
        # Handle "Too low audio level" error
        if debug:
            print(f"Audio level error: {e}", file=sys.stderr)
        raise
    except Exception as e:
        import traceback
        print(f"ERROR in recognize_audio_with_result: {type(e).__name__}: {str(e)}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        raise
