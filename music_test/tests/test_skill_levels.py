"""Tests for the skill level system."""
import pytest
from lib.skill_levels import (
    SKILL_LEVELS,
    DIFFICULTY_TO_SKILL,
    ALL_SKILL_NAMES,
    get_skill_preset,
    resolve_skill_level,
    suggest_skill_for_song,
)


class TestSkillPresets:
    """Test preset definitions and retrieval."""

    def test_three_levels_exist(self):
        assert set(SKILL_LEVELS.keys()) == {'beginner', 'intermediate', 'advanced'}

    def test_all_presets_have_same_keys(self):
        keys = set(SKILL_LEVELS['intermediate'].keys())
        for level in SKILL_LEVELS:
            assert set(SKILL_LEVELS[level].keys()) == keys, f"{level} has different keys"

    def test_get_skill_preset_returns_copy(self):
        preset = get_skill_preset('beginner')
        preset['confidence_threshold'] = 999
        assert SKILL_LEVELS['beginner']['confidence_threshold'] != 999

    def test_get_skill_preset_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown skill level"):
            get_skill_preset('master')

    def test_intermediate_matches_original_defaults(self):
        """Intermediate must match the original hardcoded defaults for backward compatibility."""
        preset = get_skill_preset('intermediate')
        assert preset['confidence_threshold'] == 0.45
        assert preset['chord_window'] == 0.3
        assert preset['chord_window_confidence'] == 0.45
        assert preset['song_influence'] == 0.70
        assert preset['decay_rate'] == 2.3
        assert preset['hysteresis_bonus'] == 0.15

    def test_beginner_more_forgiving_than_intermediate(self):
        b = get_skill_preset('beginner')
        i = get_skill_preset('intermediate')
        assert b['confidence_threshold'] < i['confidence_threshold']
        assert b['chord_window'] > i['chord_window']
        assert b['decay_rate'] < i['decay_rate']
        assert b['hysteresis_bonus'] > i['hysteresis_bonus']

    def test_advanced_stricter_than_intermediate(self):
        a = get_skill_preset('advanced')
        i = get_skill_preset('intermediate')
        assert a['confidence_threshold'] > i['confidence_threshold']
        assert a['chord_window'] < i['chord_window']
        assert a['decay_rate'] > i['decay_rate']
        assert a['hysteresis_bonus'] < i['hysteresis_bonus']

    def test_get_skill_preset_accepts_aliases(self):
        """All 7 difficulty names should be accepted by get_skill_preset."""
        for name in ALL_SKILL_NAMES:
            preset = get_skill_preset(name)
            assert 'confidence_threshold' in preset

    def test_alias_returns_same_preset_as_canonical(self):
        assert get_skill_preset('elementary') == get_skill_preset('beginner')
        assert get_skill_preset('novice') == get_skill_preset('beginner')
        assert get_skill_preset('proficient') == get_skill_preset('intermediate')
        assert get_skill_preset('expert') == get_skill_preset('advanced')


class TestResolveSkillLevel:
    """Test resolve_skill_level mapping."""

    def test_canonical_levels_resolve_to_themselves(self):
        assert resolve_skill_level('beginner') == 'beginner'
        assert resolve_skill_level('intermediate') == 'intermediate'
        assert resolve_skill_level('advanced') == 'advanced'

    def test_aliases_resolve_correctly(self):
        assert resolve_skill_level('elementary') == 'beginner'
        assert resolve_skill_level('novice') == 'beginner'
        assert resolve_skill_level('proficient') == 'intermediate'
        assert resolve_skill_level('expert') == 'advanced'

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Unknown skill level"):
            resolve_skill_level('master')

    def test_all_skill_names_list(self):
        assert len(ALL_SKILL_NAMES) == 7
        for name in ALL_SKILL_NAMES:
            resolve_skill_level(name)  # should not raise


class TestDifficultyMapping:
    """Test song difficulty -> skill level mapping."""

    def test_all_seven_difficulties_mapped(self):
        expected = ['beginner', 'elementary', 'novice', 'intermediate',
                    'proficient', 'advanced', 'expert']
        for diff in expected:
            assert diff in DIFFICULTY_TO_SKILL

    def test_beginner_group(self):
        for diff in ('beginner', 'elementary', 'novice'):
            assert suggest_skill_for_song(diff) == 'beginner'

    def test_intermediate_group(self):
        for diff in ('intermediate', 'proficient'):
            assert suggest_skill_for_song(diff) == 'intermediate'

    def test_advanced_group(self):
        for diff in ('advanced', 'expert'):
            assert suggest_skill_for_song(diff) == 'advanced'

    def test_unknown_difficulty_defaults_to_intermediate(self):
        assert suggest_skill_for_song('virtuoso') == 'intermediate'
        assert suggest_skill_for_song('') == 'intermediate'


class TestConfigApplication:
    """Test that skill presets integrate with config correctly."""

    def test_state_uses_config_decay_rate(self):
        """AudioProcessingState.get_best_chord() should use config's decay_rate."""
        from lib.state import AudioProcessingState
        import time

        config = {
            'instrument': 'guitar',
            'decay_rate': 1.0,
            'hysteresis_bonus': 0.10,
        }
        state = AudioProcessingState(config)
        # Add a detection so get_best_chord has data
        state.accumulate_chord('Am', 0.8)
        result = state.get_best_chord()
        assert result is not None
        assert result['chord'] == 'Am'

    def test_state_default_decay_rate_without_config(self):
        """When config doesn't specify decay_rate, default 2.3 is used."""
        from lib.state import AudioProcessingState

        config = {'instrument': 'guitar'}
        state = AudioProcessingState(config)
        state.accumulate_chord('C', 0.9)
        result = state.get_best_chord()
        assert result is not None
        assert result['chord'] == 'C'

    def test_cli_skill_level_overrides_defaults(self):
        """Verify that skill preset values differ from parser defaults."""
        preset = get_skill_preset('beginner')
        # Beginner confidence_threshold should be 0.30, not the CLI default 0.45
        assert preset['confidence_threshold'] == 0.30
        assert preset['chord_window'] == 0.6
