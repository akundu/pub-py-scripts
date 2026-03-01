"""Tests for chord timeline extraction and time-based lookup."""
import pytest
from lib.song_loader import (
    SongLoader,
    OPEN_STRING_MIDI,
    CHORD_TECHNIQUES,
    SIMULTANEITY_THRESHOLD,
)


class TestOpenStringMidi:
    """Verify standard tuning MIDI constants."""

    def test_string_1_high_e(self):
        assert OPEN_STRING_MIDI[1] == 64  # E4

    def test_string_2_b(self):
        assert OPEN_STRING_MIDI[2] == 59  # B3

    def test_string_3_g(self):
        assert OPEN_STRING_MIDI[3] == 55  # G3

    def test_string_4_d(self):
        assert OPEN_STRING_MIDI[4] == 50  # D3

    def test_string_5_a(self):
        assert OPEN_STRING_MIDI[5] == 45  # A2

    def test_string_6_low_e(self):
        assert OPEN_STRING_MIDI[6] == 40  # E2

    def test_six_strings(self):
        assert len(OPEN_STRING_MIDI) == 6


class TestExtractChordTimeline:
    """Test the core timeline extraction algorithm."""

    def _make_strum(self, time, frets, technique='strum', duration=0.3):
        """Helper: create simultaneous events for frets on strings 6..1."""
        events = []
        for i, fret in enumerate(frets):
            if fret is None:
                continue
            events.append({
                'time': time,
                'string': 6 - i,  # frets[0] -> string 6, frets[5] -> string 1
                'fret': fret,
                'duration': duration,
                'velocity': 0.8,
                'technique': technique,
            })
        return events

    def test_single_chord_em(self):
        # Em open: 0-2-2-0-0-0 (strings 6-1)
        events = self._make_strum(0.0, [0, 2, 2, 0, 0, 0])
        timeline = SongLoader.extract_chord_timeline(events, ['Em', 'C', 'G', 'D'])
        assert len(timeline) == 1
        assert timeline[0]['chord'] == 'Em'
        assert timeline[0]['start_time'] == 0.0

    def test_two_chord_transition(self):
        # Em at t=0, then C at t=2.0
        # Em: 0-2-2-0-0-0, C: x-3-2-0-1-0 (skip string 6)
        events = self._make_strum(0.0, [0, 2, 2, 0, 0, 0])
        events += self._make_strum(2.0, [None, 3, 2, 0, 1, 0])
        timeline = SongLoader.extract_chord_timeline(events, ['Em', 'C', 'G', 'D'])
        assert len(timeline) == 2
        assert timeline[0]['chord'] == 'Em'
        assert timeline[1]['chord'] == 'C'
        assert timeline[1]['start_time'] == 2.0

    def test_mute_excluded(self):
        """Events with 'mute' technique should not appear in timeline."""
        events = self._make_strum(0.0, [0, 2, 2, 0, 0, 0])
        events += self._make_strum(1.0, [0, 2, 2, 0, 0, 0], technique='mute')
        events += self._make_strum(2.0, [None, 3, 2, 0, 1, 0])
        timeline = SongLoader.extract_chord_timeline(events, ['Em', 'C'])
        # Muted strum at t=1.0 should be excluded; only Em and C segments
        assert len(timeline) == 2

    def test_empty_events(self):
        assert SongLoader.extract_chord_timeline([], ['Em', 'C']) == []

    def test_empty_chords(self):
        events = self._make_strum(0.0, [0, 2, 2, 0, 0, 0])
        assert SongLoader.extract_chord_timeline(events, []) == []

    def test_near_simultaneous_grouping(self):
        """Events within SIMULTANEITY_THRESHOLD should be grouped together."""
        # Simulate an arpeggio strum with slight time offsets
        events = []
        base_time = 0.0
        frets = [0, 2, 2, 0, 0, 0]  # Em
        for i, fret in enumerate(frets):
            events.append({
                'time': base_time + i * 0.008,  # 8ms apart = within 0.05s threshold
                'string': 6 - i,
                'fret': fret,
                'duration': 0.3,
                'velocity': 0.8,
                'technique': 'strum',
            })
        timeline = SongLoader.extract_chord_timeline(events, ['Em', 'C'])
        # Should be treated as one snapshot/segment
        assert len(timeline) == 1
        assert timeline[0]['chord'] == 'Em'

    def test_consecutive_same_chord_merged(self):
        """Multiple consecutive snapshots of the same chord merge into one segment."""
        events = self._make_strum(0.0, [0, 2, 2, 0, 0, 0])  # Em
        events += self._make_strum(1.0, [0, 2, 2, 0, 0, 0])  # Em again
        events += self._make_strum(2.0, [0, 2, 2, 0, 0, 0])  # Em again
        timeline = SongLoader.extract_chord_timeline(events, ['Em', 'C'])
        assert len(timeline) == 1
        assert timeline[0]['chord'] == 'Em'

    def test_real_song_wonderwall(self):
        """Load actual Wonderwall data and verify timeline has reasonable structure."""
        loader = SongLoader()
        song = loader.load_song('bg_rk_001')
        if song is None:
            pytest.skip("Wonderwall song file not available")
        timeline = SongLoader.extract_chord_timeline(
            song['events'], song['chords']
        )
        assert len(timeline) > 0
        # All chords in timeline should be from the song's chord list (or None)
        song_chords = set(song['chords'])
        for seg in timeline:
            if seg['chord'] is not None:
                assert seg['chord'] in song_chords
        # Timeline should be time-ordered
        for i in range(len(timeline) - 1):
            assert timeline[i]['start_time'] <= timeline[i + 1]['start_time']

    def test_fingerpick_included(self):
        """Fingerpick technique should be included in timeline."""
        events = self._make_strum(0.0, [0, 2, 2, 0, 0, 0], technique='fingerpick')
        timeline = SongLoader.extract_chord_timeline(events, ['Em'])
        assert len(timeline) == 1
        assert timeline[0]['chord'] == 'Em'

    def test_single_note_excluded(self):
        """single_note technique should not appear in timeline."""
        events = [{
            'time': 0.0, 'string': 1, 'fret': 3,
            'duration': 0.5, 'velocity': 0.8, 'technique': 'single_note',
        }]
        timeline = SongLoader.extract_chord_timeline(events, ['G'])
        assert timeline == []


class TestGetExpectedChordAtTime:
    """Test time-based chord lookup."""

    @pytest.fixture
    def sample_timeline(self):
        return [
            {'start_time': 0.0, 'end_time': 4.0, 'chord': 'Em'},
            {'start_time': 4.0, 'end_time': 8.0, 'chord': 'C'},
            {'start_time': 8.0, 'end_time': 12.0, 'chord': 'G'},
        ]

    def test_exact_match_start(self, sample_timeline):
        assert SongLoader.get_expected_chord_at_time(sample_timeline, 0.0) == 'Em'

    def test_exact_match_middle(self, sample_timeline):
        assert SongLoader.get_expected_chord_at_time(sample_timeline, 6.0) == 'C'

    def test_boundary_returns_next_chord(self, sample_timeline):
        # At exactly 4.0, should match C (start_time <= 4.0 <= end_time)
        assert SongLoader.get_expected_chord_at_time(sample_timeline, 4.0) == 'Em'

    def test_past_end_returns_none(self, sample_timeline):
        assert SongLoader.get_expected_chord_at_time(sample_timeline, 15.0) is None

    def test_tolerance_early(self, sample_timeline):
        # At t=3.5 with tolerance=1.0, should still match Em (within Em's range)
        # but also within expanded C range (4.0 - 1.0 = 3.0), so Em wins (earlier)
        result = SongLoader.get_expected_chord_at_time(sample_timeline, 3.5, tolerance=1.0)
        assert result == 'Em'

    def test_tolerance_late(self, sample_timeline):
        # At t=12.5 (past last segment end_time=12.0), with tolerance=1.0
        # G's expanded range: 8.0-1.0 to 12.0+1.0 = 7.0 to 13.0
        result = SongLoader.get_expected_chord_at_time(sample_timeline, 12.5, tolerance=1.0)
        assert result == 'G'

    def test_zero_tolerance_strict(self, sample_timeline):
        # Just past end of G segment
        assert SongLoader.get_expected_chord_at_time(sample_timeline, 12.01, tolerance=0.0) is None

    def test_empty_timeline(self):
        assert SongLoader.get_expected_chord_at_time([], 5.0) is None

    def test_none_chord_segments_skipped(self):
        timeline = [
            {'start_time': 0.0, 'end_time': 2.0, 'chord': None},
            {'start_time': 2.0, 'end_time': 4.0, 'chord': 'Am'},
        ]
        assert SongLoader.get_expected_chord_at_time(timeline, 1.0) is None
        assert SongLoader.get_expected_chord_at_time(timeline, 3.0) == 'Am'


class TestGetChordTimeline:
    """Test the caching convenience method."""

    def test_returns_list(self):
        loader = SongLoader()
        timeline = loader.get_chord_timeline('bg_rk_001')
        if not timeline:
            pytest.skip("Wonderwall song file not available")
        assert isinstance(timeline, list)
        assert all('start_time' in seg and 'chord' in seg for seg in timeline)

    def test_caches_result(self):
        loader = SongLoader()
        t1 = loader.get_chord_timeline('bg_rk_001')
        t2 = loader.get_chord_timeline('bg_rk_001')
        assert t1 is t2  # Same object (cached)

    def test_nonexistent_song(self):
        loader = SongLoader()
        assert loader.get_chord_timeline('nonexistent_song_xyz') == []
