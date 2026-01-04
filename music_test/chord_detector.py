# file: chord_detector_sounddevice.py
import numpy as np
import time
import argparse
from collections import deque
import sys
from lib.sound_capture import recognize_audio
from lib.common import get_hop_size, get_overlap_ratio, set_overlap_ratio, get_buffer_size
from lib.music_understanding import INSTRUMENT_PRESETS

def parse_args():
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
    parser.add_argument('--show-chroma', action='store_true', help='Show chroma vector along with frequencies')
    parser.add_argument('--single-pitch', action='store_true', help='Use single-pitch detection (autocorrelation only)')
    parser.add_argument('--multi-pitch', action='store_true', default=True, help='Enable multi-pitch detection using FFT (default, better for chords)')
    parser.add_argument('--confidence-threshold', type=float, default=0.6, help='Minimum confidence score to show chords (default: 0.6)')
    args = parser.parse_args()
    
    # Handle single-pitch override
    if args.single_pitch:
        args.multi_pitch = False
    
    return args

# --- 3. Main Application Logic ---
def main():
    """
    Main function to run the sounddevice-based chord detector.
    """
    args = parse_args()

    # Update overlap settings
    set_overlap_ratio(args.overlap)

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

    print(f"🎸 Chord detector listening for {instrument_name}... Press Ctrl+C to stop.")
    
    # Always print all configuration parameters
    print("📊 Configuration Parameters:")
    print(f"  Instrument: {args.instrument} ({instrument_name})")
    if args.low_freq and args.high_freq:
        print(f"  Frequency Range: {low_freq}-{high_freq} Hz (Custom)")
    else:
        print(f"  Frequency Range: {low_freq}-{high_freq} Hz")
    print(f"  Sensitivity: {args.sensitivity}")
    print(f"  Confidence Threshold: {args.confidence_threshold}")
    print(f"  Silence Threshold: {args.silence_threshold}")
    print(f"  Amplitude Threshold: {args.amplitude_threshold}")
    print(f"  Overlap: {args.overlap} ({get_overlap_ratio()*100:.0f}%, hop size: {get_hop_size()} samples)")
    print(f"  Progression: {args.progression}")
    print(f"  Multi-pitch: {getattr(args, 'multi_pitch', True)}")
    print(f"  Single-pitch: {args.single_pitch}")
    print(f"  Show Frequencies: {args.show_frequencies}")
    print(f"  Show Chroma: {getattr(args, 'show_chroma', False)}")
    print(f"  Show FFT: {args.show_fft}")
    print(f"  Raw Frequencies: {args.raw_frequencies}")
    print(f"  Frequencies Only: {args.frequencies_only}")
    print(f"  Notes Only: {args.notes_only}")
    print(f"  Debug: {args.debug}")
    print(f"  Log: {args.log}")
    print(f"  Log Interval: {args.log_interval}s")
    print(f"  Wait Time: {args.wait_time}s")
    
    # Show mode information
    if args.frequencies_only:
        print("📊 FREQUENCIES ONLY MODE: Showing frequencies, skipping chord processing")
    elif args.notes_only:
        print("🎵 NOTES ONLY MODE: Showing detected notes, skipping chord processing")
    else:
        # Default chord detection mode
        print(f"🎯 Confidence threshold: {args.confidence_threshold:.1f} (only showing chords above this score)")
        if args.log:
            print("📝 Logging mode: Showing timestamped chord detections")
    
    if args.debug:
        if getattr(args, 'multi_pitch', True):
            print("🎼 Multi-pitch detection enabled (default, better for chords)")
        else:
            print("🎵 Single-pitch detection enabled (autocorrelation only)")
        if getattr(args, 'show_chroma', False):
            print("🌈 Chroma vector output enabled")
        if args.progression and not args.frequencies_only and not args.notes_only:
            print("📈 Chord progression analysis enabled")

    # Audio buffer for overlapping windows
    audio_buffer = np.zeros(get_buffer_size(), dtype=np.float32)
    buffer_index = 0
    
    notes_history = deque(maxlen=5)
    frequencies_history = deque(maxlen=5)  # For enhanced chord analysis
    chroma_history = deque(maxlen=5)       # For enhanced chord analysis
    last_chord = None
    chord_stability = 0
    last_log_time = time.time()

    try:
        while True:
            try:
                recognize_audio(args, audio_buffer, buffer_index, low_freq, high_freq, notes_history, last_chord, chord_stability, last_log_time, frequencies_history, chroma_history, debug=args.debug, frequencies_only=args.frequencies_only, notes_only=args.notes_only, log=args.log)
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