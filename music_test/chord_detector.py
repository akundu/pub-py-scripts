# file: chord_detector_sounddevice.py
import numpy as np
import time
import argparse
from collections import deque
import sys
from common_sound_capture import recognize_audio
from common import get_hop_size, get_overlap_ratio, set_overlap_ratio, get_buffer_size
from common_music_understanding import INSTRUMENT_PRESETS

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
    args = parser.parse_args()
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

    print(f"🎸 Sounddevice-based chord detector listening for {instrument_name}... Press Ctrl+C to stop.")
    if args.raw_frequencies:
        print("🔓 RAW MODE: Using unfiltered frequencies (no frequency range limits)")
    else:
        print(f"📊 Frequency range: {low_freq}-{high_freq} Hz")
    print(f"🔄 Overlap: {get_overlap_ratio()*100:.0f}% (hop size: {get_hop_size()} samples)")
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
    audio_buffer = np.zeros(get_buffer_size(), dtype=np.float32)
    buffer_index = 0
    
    notes_history = deque(maxlen=5)
    last_chord = None
    chord_stability = 0
    last_log_time = time.time()

    try:
        while True:
            try:
                recognize_audio(args, audio_buffer, buffer_index, low_freq, high_freq, notes_history, last_chord, chord_stability, last_log_time, debug=args.debug, frequencies_only=args.frequencies_only, notes_only=args.notes_only, log=args.log)
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