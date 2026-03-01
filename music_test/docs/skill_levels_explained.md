# Skill Level System -- How Parameters Affect Detection

## Overview

Three skill levels (beginner, intermediate, advanced) adjust 6 parameters to create a cohesive experience from forgiving to precise. Seven input names map to these 3 levels:

- **beginner, elementary, novice** → Beginner preset
- **intermediate, proficient** → Intermediate preset
- **advanced, expert** → Advanced preset

## Parameter Reference

| Parameter | Beginner | Intermediate | Advanced |
|-----------|----------|--------------|----------|
| `confidence_threshold` | 0.30 | 0.45 | 0.55 |
| `chord_window` | 0.6s | 0.3s | 0.15s |
| `chord_window_confidence` | 0.30 | 0.45 | 0.50 |
| `song_influence` | 0.85 | 0.70 | 0.50 |
| `decay_rate` | 1.2 | 2.3 | 3.5 |
| `hysteresis_bonus` | 0.30 | 0.15 | 0.05 |

## What Each Parameter Does

### 1. `confidence_threshold` (0.30 vs 0.55)

The minimum score a chord detection needs before it's shown at all. Beginner accepts weaker, muddier signals -- a sloppy G chord that only scores 0.35 still shows up. Advanced rejects anything below 0.55, so only clean voicings register. This means beginners see feedback even when their fretting is imprecise, while advanced players get silence instead of noise.

### 2. `chord_window` (0.6s vs 0.15s)

How long the system collects detections before picking a winner. Beginner accumulates over 0.6 seconds -- if you fumble the first 0.3s of a chord change but land it for the last 0.3s, the good detections outvote the bad ones. Advanced uses just 0.15s, giving near-instant response so fast chord changes (strumming patterns, quick transitions) show up immediately without lag.

### 3. `chord_window_confidence` (0.30 vs 0.50)

The minimum average confidence the winning chord needs *within* the window. This is a second gate after `confidence_threshold`. Beginner is lenient (0.30) -- even if most frames in the window were weak, it still shows something. Advanced demands 0.50, filtering out windows where the detection was shaky overall.

### 4. `song_influence` (0.85 vs 0.50)

When a song is loaded, how much the known song chords pull ambiguous detections. At 0.85, if the detector sees "Am7" and the song has "Am", it strongly maps to Am -- the system assumes you're *trying* to play the song chord. At 0.50, it's an even split between raw detection and song context, so if an advanced player intentionally plays a substitution (e.g., Am7 instead of Am, or a passing chord), the system respects that rather than correcting it.

### 5. `decay_rate` (1.2 vs 3.5)

Controls how fast older detections lose weight within the chord window. It's an exponential decay where the half-life = ln(2) / rate:

- **Beginner (1.2)**: half-life ~0.58s. A detection from half a second ago still counts at 50% weight. This means if you briefly mute the strings during a transition (common for beginners), the previous chord's detections don't vanish instantly -- the system "remembers" what you were playing through brief gaps.
- **Advanced (3.5)**: half-life ~0.20s. Old detections fade fast. If you stop playing a chord, the system reflects that almost immediately. This gives responsive, real-time feedback for fast playing.

### 6. `hysteresis_bonus` (0.30 vs 0.05)

A score boost given to whatever chord is currently displayed, making it "sticky." A new chord has to win by this margin to take over the display.

- **Beginner (30%)**: If Am is showing with a weighted score of 1.0, a competing chord needs to score above 1.30 to replace it. This prevents flutter -- where the display rapidly alternates between Am and Am7 (or Am and E) because tiny noise fluctuations swap the winner each frame. Beginners produce inconsistent signals, so this keeps the display stable.
- **Advanced (5%)**: Almost no stickiness. If you switch from Am to C, the display follows with minimal resistance. Advanced players produce clean, decisive chord changes so flutter isn't a problem, and the low hysteresis lets rapid chord changes register immediately.

## How They Work Together

For a **beginner** playing a song: you strum a rough G, the long window smooths out the initial fumble, the low confidence thresholds let the weak signal through, the slow decay keeps it alive through brief mutes as you move to the next chord, high song influence nudges ambiguous detections toward the expected chord, and high hysteresis keeps the display from flickering. The result feels like the system is patient and encouraging.

For an **advanced** player: short window and fast decay give instant response, high confidence thresholds filter noise, low song influence respects intentional deviations, and minimal hysteresis lets rapid changes through. The result feels tight and precise.

## Usage

**CLI:**
```bash
python chord_detector.py --skill-level beginner --song bg_rk_001
python chord_detector.py --skill-level expert
```

**URL parameter:**
```
?skill_level=beginner
?skill_level=novice
?skill_level=expert
```

**Web UI:** Click the Beginner / Intermediate / Advanced buttons near the song controls.

Individual parameters can always override the preset:
```
?skill_level=beginner&confidence_threshold=0.40
```
