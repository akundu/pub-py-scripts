#!/usr/bin/env python3
"""
Play the corrected Smoke on the Water riff.
Based on the actual Deep Purple tab: 0-3-5, 0-3-6-5, 0-3-5, 3-0
"""
import json
import sys
import play_song as ps

# Load corrected file
with open('smoke_on_water_corrected.json') as f:
    song = json.load(f)

# Play with optional repeat flag
repeat = '--repeat' in sys.argv or '-r' in sys.argv

ps.play_song(song, None, bars_per_chord=1, repeat=repeat)
