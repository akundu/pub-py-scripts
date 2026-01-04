"""
Web audio processing functions - uses the shared core audio processing function.
"""
import numpy as np
import time
import sys
from lib.common import get_chunk, get_rate, get_buffer_size
from lib.audio_processing import process_audio_chunk
from lib.music_understanding import frequency_to_note

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
    
    This function uses the shared core audio processing function to ensure
    consistent behavior between CLI and web interfaces.
    
    Args:
        audio_chunk: numpy array of audio samples (Float32)
        state: ConnectionState object with buffers and history
        config: configuration dictionary
    
    Returns:
        dict with detection results or None if no valid detection
    """
    # Get debug log level
    debug_log_level = SERVER_LOG_LEVEL
    
    try:
        # Use the shared core processing function
        result = process_audio_chunk(
            audio_chunk, state, config,
            output_handler=None,  # Return dict instead of printing
            debug_log_level=debug_log_level
        )
        
        return result
        
    except ValueError as e:
        # Handle "Too low audio level" error - this is expected in debug mode
        if "Too low audio level" in str(e):
            # Don't log this unless in DEBUG mode
            if debug_log_level == 'DEBUG':
                print(f"[{get_timestamp()}] ⚠️ Audio level too low", file=sys.stderr)
            raise
        else:
            raise
    except Exception as e:
        # Log unexpected errors
        if config.get('debug') or debug_log_level == 'DEBUG':
            import traceback
            print(f"[{get_timestamp()}] Error in recognize_audio_web: {type(e).__name__}: {str(e)}", 
                  file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        raise


def log_result(result, config, connection_id=None):
    """
    Log detection result in CLI-compatible format.
    
    Args:
        result: dict from recognize_audio_web
        config: configuration dictionary
        connection_id: optional connection identifier for web logs
    """
    if result is None:
        return
    
    # Only log if log or debug mode is enabled
    if not (config.get('log') or config.get('debug')):
        return
    
    timestamp = get_timestamp()
    result_type = result.get('type')
    
    if result_type == 'chord':
        chord = result.get('chord')
        confidence = result.get('confidence', 0)
        stability = result.get('stability', 0)
        
        # Build output parts
        output_parts = []
        
        # Add chroma if present and requested
        if config.get('show_chroma') and 'chroma' in result:
            chroma_values = [f"{val:.3f}" for val in result['chroma']]
            output_parts.append(f"Chroma: [{', '.join(chroma_values)}]")
        
        # Add notes with frequencies if present
        if config.get('show_frequencies') and 'notes' in result:
            notes_str = ", ".join([f"{note}({freq:.0f}Hz)" for note, freq in result['notes']])
            output_parts.append(f"Notes: {notes_str}")
        
        # Add chord
        stability_suffix = "" if stability >= 2 else " [unstable]"
        
        if output_parts:
            full_output = " | ".join(output_parts) + f" -> {chord}{stability_suffix}"
        else:
            full_output = f"{chord}{stability_suffix}"
        
        # Print in CLI format (without connection_id to match CLI exactly)
        print(f"[{timestamp}] {full_output}")
        
    elif result_type == 'notes':
        notes = result.get('notes', [])
        if notes:
            notes_str = ", ".join([f"{note}({freq:.0f}Hz)" for note, freq in notes])
            
            # Add connection_id for web logging
            if connection_id:
                print(f"[{timestamp}] [{connection_id}] Notes: {notes_str}")
            else:
                print(f"[{timestamp}] Notes: {notes_str}")
    
    elif result_type == 'frequencies':
        frequencies = result.get('frequencies', [])
        if frequencies:
            freq_str = ", ".join([f"{freq:.0f}Hz" for freq, _ in frequencies])
            
            if connection_id:
                print(f"[{timestamp}] [{connection_id}] Frequencies: {freq_str}")
            else:
                print(f"[{timestamp}] Frequencies: {freq_str}")
    
    elif result_type == 'listening':
        # Don't log listening status by default
        pass


def format_result_for_client(result, config):
    """
    Format result for sending to web client.
    
    Args:
        result: dict from recognize_audio_web
        config: configuration dictionary
    
    Returns:
        dict suitable for JSON serialization to client
    """
    if result is None:
        return None
    
    # The result from process_audio_chunk is already in the right format
    # Just ensure all numpy arrays are converted to lists
    formatted = {}
    
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            formatted[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            # Handle lists of tuples (like notes)
            formatted[key] = [
                tuple(v) if isinstance(v, (list, tuple)) else v 
                for v in value
            ]
        else:
            formatted[key] = value
    
    return formatted
