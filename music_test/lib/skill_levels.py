"""
Skill level presets for chord detection.

Three levels adjust detection parameters to match player ability:
- Beginner: forgiving detection for learners (longer windows, lower thresholds)
- Intermediate: balanced defaults (matches original hardcoded values)
- Advanced: precise detection for experienced players (short windows, strict thresholds)
"""

SKILL_LEVELS = {
    'beginner': {
        'confidence_threshold': 0.30,
        'chord_window': 0.6,
        'chord_window_confidence': 0.30,
        'song_influence': 0.85,
        'decay_rate': 1.2,
        'hysteresis_bonus': 0.30,
    },
    'intermediate': {
        'confidence_threshold': 0.45,
        'chord_window': 0.3,
        'chord_window_confidence': 0.45,
        'song_influence': 0.70,
        'decay_rate': 2.3,
        'hysteresis_bonus': 0.15,
    },
    'advanced': {
        'confidence_threshold': 0.55,
        'chord_window': 0.15,
        'chord_window_confidence': 0.50,
        'song_influence': 0.50,
        'decay_rate': 3.5,
        'hysteresis_bonus': 0.05,
    },
}

# Map 7 song difficulty levels to 3 skill levels
DIFFICULTY_TO_SKILL = {
    'beginner': 'beginner',
    'elementary': 'beginner',
    'novice': 'beginner',
    'intermediate': 'intermediate',
    'proficient': 'intermediate',
    'advanced': 'advanced',
    'expert': 'advanced',
}


ALL_SKILL_NAMES = list(DIFFICULTY_TO_SKILL.keys())


def resolve_skill_level(level):
    """
    Resolve any accepted skill name to one of the 3 canonical levels.

    Accepts all 7 difficulty names (beginner, elementary, novice,
    intermediate, proficient, advanced, expert) plus the 3 canonical
    level names.

    Args:
        level: any accepted skill/difficulty name

    Returns:
        canonical skill level ('beginner', 'intermediate', or 'advanced')

    Raises:
        ValueError: if level is not recognized
    """
    if level in SKILL_LEVELS:
        return level
    if level in DIFFICULTY_TO_SKILL:
        return DIFFICULTY_TO_SKILL[level]
    raise ValueError(
        f"Unknown skill level '{level}'. Choose from: {', '.join(ALL_SKILL_NAMES)}"
    )


def get_skill_preset(level):
    """
    Return the parameter dict for a skill level.

    Accepts all 7 difficulty names (beginner, elementary, novice,
    intermediate, proficient, advanced, expert) plus the 3 canonical
    level names. Aliases are resolved to the canonical level first.

    Args:
        level: any accepted skill/difficulty name

    Returns:
        dict of parameter names to values

    Raises:
        ValueError: if level is not recognized
    """
    canonical = resolve_skill_level(level)
    return dict(SKILL_LEVELS[canonical])


def suggest_skill_for_song(difficulty):
    """
    Suggest a skill level based on a song's difficulty field.

    Args:
        difficulty: song difficulty string (e.g. 'beginner', 'intermediate', 'expert')

    Returns:
        suggested skill level string, or 'intermediate' if difficulty is unknown
    """
    return DIFFICULTY_TO_SKILL.get(difficulty, 'intermediate')
