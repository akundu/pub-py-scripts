#!/usr/bin/env python3
"""
Play a song from the database with real-time chord display.

Usage:
    python3 play_song.py "smoke on the water"
    python3 play_song.py --list
    python3 play_song.py --list rock
"""
import sys
import os
import time
import json
import argparse
import numpy as np

# Add project root to path for lib imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.song_loader import SongLoader

SAMPLE_RATE = 44100

# Standard guitar tuning: string number -> open string frequency (Hz)
OPEN_STRING_FREQ = {
    1: 329.63,  # E4
    2: 246.94,  # B3
    3: 196.00,  # G3
    4: 146.83,  # D3
    5: 110.00,  # A2
    6: 82.41,   # E2
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# MIDI note number of each open string (for note name calculation)
OPEN_STRING_MIDI = {
    1: 64,  # E4
    2: 59,  # B3
    3: 55,  # G3
    4: 50,  # D3
    5: 45,  # A2
    6: 40,  # E2
}


def fret_to_freq(string_num, fret):
    """Convert string/fret to frequency in Hz."""
    return OPEN_STRING_FREQ[string_num] * (2 ** (fret / 12.0))


def fret_to_note_name(string_num, fret):
    """Convert string/fret to note name (e.g., 'E', 'G#')."""
    midi = OPEN_STRING_MIDI[string_num] + fret
    return NOTE_NAMES[midi % 12]


def fret_to_pitch_class(string_num, fret):
    """Convert string/fret to pitch class (0-11)."""
    midi = OPEN_STRING_MIDI[string_num] + fret
    return midi % 12


def synthesize_note(freq, duration, velocity, technique, sample_rate=SAMPLE_RATE):
    """Synthesize a single guitar note using additive synthesis with decay."""
    n_samples = int(duration * sample_rate)
    if n_samples == 0:
        return np.array([], dtype=np.float32)

    t = np.arange(n_samples, dtype=np.float64) / sample_rate

    # Muted notes: very short, percussive
    if technique == 'mute':
        decay_rate = 20.0
    else:
        # Gentle decay — guitar strings ring for a while
        decay_rate = 2.5 / max(duration, 0.3)

    # ADSR-like envelope: quick attack + exponential decay
    attack_samples = min(int(0.005 * sample_rate), n_samples)
    envelope = np.exp(-decay_rate * t)
    if attack_samples > 0:
        envelope[:attack_samples] *= np.linspace(0, 1, attack_samples)

    # Additive synthesis: fundamental + harmonics with decreasing amplitude
    harmonic_amps = [1.0, 0.5, 0.3, 0.15, 0.08]
    signal = np.zeros(n_samples, dtype=np.float64)
    for i, amp in enumerate(harmonic_amps):
        h = i + 1
        harmonic_freq = freq * h
        if harmonic_freq > sample_rate / 2:
            break
        # Slight inharmonicity for realism
        detune = 1.0 + 0.0003 * h * h
        # Higher harmonics decay faster
        h_envelope = np.exp(-(decay_rate + h * 1.5) * t)
        signal += amp * np.sin(2 * np.pi * harmonic_freq * detune * t) * h_envelope

    signal *= envelope * velocity * 0.3
    return signal.astype(np.float32)


def group_events(events, threshold=0.1):
    """
    Group events that occur within `threshold` seconds of each other.
    Returns list of (group_time, [events]).
    """
    if not events:
        return []

    sorted_events = sorted(events, key=lambda e: e['time'])
    groups = []
    current_group = [sorted_events[0]]
    group_start = sorted_events[0]['time']

    for event in sorted_events[1:]:
        if event['time'] - group_start <= threshold:
            current_group.append(event)
        else:
            groups.append((group_start, current_group))
            current_group = [event]
            group_start = event['time']

    groups.append((group_start, current_group))
    return groups


def identify_chord(event_group, song_chords, chord_pc_map):
    """
    Identify what chord is being played from a group of simultaneous events.
    Returns the best matching chord name from the song's chord list.
    """
    pitch_classes = set()
    for event in event_group:
        pc = fret_to_pitch_class(event['string'], event['fret'])
        pitch_classes.add(pc)

    if not pitch_classes:
        return None

    best_chord = None
    best_score = -1

    for chord_name, chord_pcs in chord_pc_map.items():
        if not chord_pcs:
            continue
        intersection = pitch_classes & chord_pcs
        union = pitch_classes | chord_pcs
        score = len(intersection) / len(union) if union else 0
        if score > best_score:
            best_score = score
            best_chord = chord_name

    if best_score >= 0.3:
        return best_chord
    return None


def compress_events(events, song_chords, tempo, bars_per_chord=4):
    """
    Compress practice arrangements by trimming each chord section to N bars.

    The song database uses practice arrangements where each chord repeats for
    20-40 seconds. This function keeps only `bars_per_chord` bars of each
    chord section and re-times them back-to-back for musical playback.
    """
    if not events or not song_chords:
        return events

    chord_pc_map = {c: SongLoader.get_chord_pitch_classes(c) for c in song_chords}
    beat_dur = 60.0 / tempo
    bar_dur = beat_dur * 4  # assume 4/4
    section_dur = bar_dur * bars_per_chord

    # Group events and detect chord sections
    groups = group_events(events)
    sections = []  # [(chord, start_time, end_time, [events])]
    current_chord = None
    section_events = []
    section_start = 0.0

    for group_time, group_evts in groups:
        non_muted = [e for e in group_evts if e.get('technique') != 'mute']
        if non_muted:
            chord = identify_chord(non_muted, song_chords, chord_pc_map)
        else:
            chord = current_chord  # muted groups belong to current section

        if chord != current_chord and current_chord is not None:
            sections.append((current_chord, section_start, group_time, section_events))
            section_events = []
            section_start = group_time

        current_chord = chord
        section_events.extend(group_evts)

    if section_events:
        last_t = max(e['time'] for e in section_events)
        sections.append((current_chord, section_start, last_t + 0.5, section_events))

    if not sections:
        return events

    # Check if compression is actually needed
    avg_section = sum(s[2] - s[1] for s in sections) / len(sections)
    if avg_section <= section_dur * 1.5:
        return events  # already short enough

    # Trim each section and re-time
    compressed = []
    current_time = 0.0

    for chord, sec_start, sec_end, sec_events in sections:
        # Keep events within the first `section_dur` of this section
        cutoff = sec_start + section_dur
        kept = []
        for e in sec_events:
            if e['time'] < cutoff:
                new_e = dict(e)
                new_e['time'] = current_time + (e['time'] - sec_start)
                kept.append(new_e)

        if kept:
            compressed.extend(kept)
            current_time += min(section_dur, sec_end - sec_start)

    return compressed


def extend_note_durations(events):
    """Extend note durations so each note rings until the next event on the same string."""
    from collections import defaultdict

    sorted_events = sorted(events, key=lambda e: (e['string'], e['time']))

    # Group by string
    by_string = defaultdict(list)
    for e in sorted_events:
        by_string[e['string']].append(e)

    extended = []
    for string_num, string_events in by_string.items():
        for i, event in enumerate(string_events):
            e = dict(event)  # copy
            if e.get('technique') == 'mute':
                # Muted notes stay short
                extended.append(e)
                continue
            # Extend to next event on same string, or use a reasonable sustain
            if i + 1 < len(string_events):
                gap = string_events[i + 1]['time'] - e['time']
                e['duration'] = max(e['duration'], min(gap, 2.0))
            else:
                # Last note on this string — let it ring
                e['duration'] = max(e['duration'], 1.0)
            extended.append(e)

    return extended


def synthesize_song(events, sample_rate=SAMPLE_RATE):
    """Synthesize full song audio from events. Returns numpy array."""
    if not events:
        return np.array([], dtype=np.float32)

    # Extend durations so notes ring naturally
    events = extend_note_durations(events)

    # Find total duration
    max_end = max(e['time'] + e['duration'] for e in events)
    total_samples = int((max_end + 0.5) * sample_rate)
    audio = np.zeros(total_samples, dtype=np.float64)

    for event in events:
        freq = fret_to_freq(event['string'], event['fret'])
        note_audio = synthesize_note(
            freq, event['duration'], event['velocity'],
            event.get('technique', 'strum'), sample_rate
        )
        start_sample = int(event['time'] * sample_rate)
        end_sample = start_sample + len(note_audio)
        if end_sample > total_samples:
            note_audio = note_audio[:total_samples - start_sample]
            end_sample = total_samples
        if len(note_audio) > 0:
            audio[start_sample:end_sample] += note_audio

    # Normalize to prevent clipping
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85

    return audio.astype(np.float32)


def build_chord_timeline(events, song_chords):
    """Build a timeline of chord changes: [(time, chord_name), ...]."""
    # Pre-compute pitch classes for each song chord
    chord_pc_map = {}
    for chord_name in song_chords:
        chord_pc_map[chord_name] = SongLoader.get_chord_pitch_classes(chord_name)

    groups = group_events(events)
    timeline = []
    prev_chord = None

    for group_time, group_events_list in groups:
        # Skip muted-only groups for chord identification
        non_muted = [e for e in group_events_list if e.get('technique') != 'mute']
        if not non_muted:
            continue

        chord = identify_chord(non_muted, song_chords, chord_pc_map)
        if chord and chord != prev_chord:
            timeline.append((group_time, chord))
            prev_chord = chord

    return timeline


def fuzzy_search(query, songs):
    """Search songs by fuzzy title matching. Returns list of matching song entries."""
    query_lower = query.lower().strip()
    query_words = set(query_lower.split())

    scored = []
    for song in songs:
        title_lower = song['title'].lower()

        # Exact substring match
        if query_lower in title_lower:
            scored.append((song, 100))
            continue

        # Title contains the query as substring
        if title_lower in query_lower:
            scored.append((song, 90))
            continue

        # Word overlap
        title_words = set(title_lower.split())
        common = query_words & title_words
        if common:
            score = len(common) / max(len(query_words), len(title_words)) * 80
            scored.append((song, score))
            continue

        # Composer match
        composer_lower = song.get('composer', '').lower()
        if query_lower in composer_lower:
            scored.append((song, 50))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in scored]


def format_time(seconds):
    """Format seconds as M:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def list_songs(loader, genre_filter=None):
    """Print all available songs."""
    songs = loader.get_song_list()
    if genre_filter:
        songs = [s for s in songs if s.get('genre', '').lower() == genre_filter.lower()]

    if not songs:
        print("No songs found.")
        return

    # Group by genre
    by_genre = {}
    for song in songs:
        genre = song.get('genre', 'unknown')
        by_genre.setdefault(genre, []).append(song)

    print(f"\n  Available Songs ({len(songs)} total)\n")
    for genre in sorted(by_genre.keys()):
        genre_songs = sorted(by_genre[genre], key=lambda s: s['title'])
        print(f"  [{genre.upper()}]")
        for s in genre_songs:
            diff = s.get('difficulty', '?')
            tempo = s.get('tempo', '?')
            print(f"    {s['title']:<40} {s.get('composer', ''):<25} {diff:<14} {tempo} bpm")
        print()


def play_song(song_data, loader, bars_per_chord=4, repeat=False):
    """Synthesize and play a song with real-time chord display."""
    import sounddevice as sd

    title = song_data['title']
    composer = song_data.get('composer', 'Unknown')
    tempo = song_data.get('tempo', 120)
    time_sig = song_data.get('time_signature', '4/4')
    difficulty = song_data.get('difficulty', '?')
    genre = song_data.get('genre', '?')
    chords = song_data.get('chords', [])
    events = song_data.get('events', [])

    if not events:
        print("No events in this song.")
        return

    # Compress chord sections for musical playback
    events = compress_events(events, chords, tempo, bars_per_chord)

    print(f"\n  Song: {title}")
    print(f"  Artist: {composer}")
    print(f"  Genre: {genre} | Tempo: {tempo} BPM | Time: {time_sig} | Difficulty: {difficulty}")
    print(f"  Chords: {', '.join(chords)}")
    print(f"  Bars per chord: {bars_per_chord}")

    # Build chord timeline
    print("\n  Analyzing chords...")
    chord_timeline = build_chord_timeline(events, chords)

    # Synthesize audio
    print("  Synthesizing audio...")
    audio = synthesize_song(events)
    duration = len(audio) / SAMPLE_RATE
    print(f"  Duration: {format_time(duration)}")
    if repeat:
        print(f"  Mode: repeat (Ctrl+C to stop)")

    if len(chord_timeline) == 0:
        print("  Warning: No chord changes detected.")

    # Display chord map
    print(f"\n  Chord progression ({len(chord_timeline)} changes):")
    for t, chord in chord_timeline[:20]:
        print(f"    {format_time(t)}  {chord}")
    if len(chord_timeline) > 20:
        print(f"    ... and {len(chord_timeline) - 20} more")

    print(f"\n  {'=' * 50}")
    print(f"  Playing{'  [REPEAT]' if repeat else ''}... (Ctrl+C to stop)")
    print(f"  {'=' * 50}\n")

    loop_count = 0

    try:
        while True:
            loop_count += 1
            if loop_count > 1:
                print(f"\n  --- Loop {loop_count} ---\n")

            # Start playback
            sd.play(audio, SAMPLE_RATE)
            start_time = time.time()
            chord_idx = 0

            while sd.get_stream().active:
                elapsed = time.time() - start_time

                # Check for chord changes
                while chord_idx < len(chord_timeline) and elapsed >= chord_timeline[chord_idx][0]:
                    current_chord = chord_timeline[chord_idx][1]
                    chord_time = chord_timeline[chord_idx][0]

                    # Show upcoming chord
                    upcoming = ""
                    if chord_idx + 1 < len(chord_timeline):
                        next_chord = chord_timeline[chord_idx + 1][1]
                        next_time = chord_timeline[chord_idx + 1][0]
                        upcoming = f"  (next: {next_chord} at {format_time(next_time)})"

                    print(f"  {format_time(chord_time)}  [ {current_chord:>6} ]{upcoming}")
                    chord_idx += 1

                time.sleep(0.02)

            if not repeat:
                break

    except KeyboardInterrupt:
        sd.stop()
        total = time.time() - start_time
        print(f"\n\n  Stopped at {format_time(total)}" +
              (f" (loop {loop_count})" if repeat else ""))
        return

    print(f"\n  Finished playing {title}.")


def main():
    parser = argparse.ArgumentParser(description='Play a song with real-time chord display')
    parser.add_argument('query', nargs='?', help='Song name to search for')
    parser.add_argument('--list', '-l', nargs='?', const='', default=None,
                        metavar='GENRE', help='List available songs (optionally filter by genre)')
    parser.add_argument('--bars', '-b', type=float, default=4,
                        help='Bars per chord section, supports fractions like 0.5 (default: 4)')
    parser.add_argument('--repeat', '-r', action='store_true',
                        help='Loop playback until Ctrl+C')
    args = parser.parse_args()

    # Determine songs directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    songs_dir = os.path.join(script_dir, 'songs')
    loader = SongLoader(songs_dir)

    if args.list is not None:
        list_songs(loader, args.list if args.list else None)
        return

    if not args.query:
        parser.print_help()
        print("\nExamples:")
        print('  python3 play_song.py "smoke on the water"')
        print('  python3 play_song.py "hotel california" --bars 2')
        print('  python3 play_song.py --list')
        print('  python3 play_song.py --list rock')
        return

    # Search for the song
    all_songs = loader.get_song_list()
    if not all_songs:
        print("No songs found. Make sure the songs/ directory exists with manifest.json.")
        return

    matches = fuzzy_search(args.query, all_songs)

    if not matches:
        print(f"No songs matching '{args.query}' found.")
        print("Use --list to see all available songs.")
        return

    # Auto-select if top match is clearly best (exact substring in title)
    if len(matches) == 1 or args.query.lower() in matches[0]['title'].lower():
        selected = matches[0]
    else:
        print(f"\n  Found {len(matches)} matches for '{args.query}':\n")
        for i, song in enumerate(matches[:10], 1):
            print(f"    {i}. {song['title']} - {song.get('composer', '?')}")

        print()
        try:
            choice = input("  Select a song (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(matches[:10]):
                selected = matches[idx]
            else:
                print("Invalid selection.")
                return
        except (ValueError, EOFError):
            print("Invalid input.")
            return

    # Load full song data
    song_data = loader.load_song(selected['song_id'])
    if not song_data:
        print(f"Could not load song data for '{selected['title']}'.")
        return

    play_song(song_data, loader, bars_per_chord=args.bars, repeat=args.repeat)


if __name__ == '__main__':
    main()
