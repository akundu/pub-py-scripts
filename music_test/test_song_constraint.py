#!/usr/bin/env python3
"""
Test script for song-constrained chord detection.
"""
from lib.song_loader import get_song_loader, SongLoader
from lib.music_understanding import constrain_chord_to_song

def test_song_loader():
    """Test basic song loading functionality."""
    print("=" * 60)
    print("Testing Song Loader")
    print("=" * 60)
    
    loader = get_song_loader()
    
    # Test getting song list
    songs = loader.get_song_list()
    print(f"\n✓ Loaded {len(songs)} songs from manifest")
    
    # Test loading a specific song
    song_id = "bg_rk_001"  # Wonderwall
    song_data = loader.load_song(song_id)
    if song_data:
        print(f"✓ Loaded song: {song_data['title']} by {song_data['composer']}")
        print(f"  Chords: {', '.join(song_data['chords'])}")
    else:
        print(f"✗ Failed to load song: {song_id}")
        return False
    
    # Test getting song chords
    chords = loader.get_song_chords(song_id)
    print(f"✓ Retrieved {len(chords)} chords: {chords}")
    
    return True

def test_pitch_classes():
    """Test pitch class calculation."""
    print("\n" + "=" * 60)
    print("Testing Pitch Class Calculation")
    print("=" * 60)
    
    loader = SongLoader()
    
    test_cases = [
        ("C", {0, 4, 7}),      # C major: C, E, G
        ("Em", {4, 7, 11}),    # E minor: E, G, B
        ("G7", {7, 11, 2, 5}), # G7: G, B, D, F
    ]
    
    for chord, expected_pcs in test_cases:
        pcs = loader.get_chord_pitch_classes(chord)
        status = "✓" if pcs == expected_pcs else "✗"
        print(f"{status} {chord}: {sorted(pcs)} (expected {sorted(expected_pcs)})")
    
    return True

def test_chord_similarity():
    """Test chord similarity calculation."""
    print("\n" + "=" * 60)
    print("Testing Chord Similarity")
    print("=" * 60)
    
    loader = SongLoader()
    
    # Em and G share 2 notes (G, B) out of 3
    em_pcs = loader.get_chord_pitch_classes("Em")
    g_pcs = loader.get_chord_pitch_classes("G")
    similarity = loader.calculate_chord_similarity(em_pcs, g_pcs)
    print(f"✓ Em vs G similarity: {similarity:.2f} (should be ~0.5-0.6)")
    
    # Em and Em7 are very similar
    em7_pcs = loader.get_chord_pitch_classes("Em7")
    similarity = loader.calculate_chord_similarity(em_pcs, em7_pcs)
    print(f"✓ Em vs Em7 similarity: {similarity:.2f} (should be ~0.75)")
    
    # C and F# are very different
    c_pcs = loader.get_chord_pitch_classes("C")
    fs_pcs = loader.get_chord_pitch_classes("F#")
    similarity = loader.calculate_chord_similarity(c_pcs, fs_pcs)
    print(f"✓ C vs F# similarity: {similarity:.2f} (should be ~0)")
    
    return True

def test_best_match():
    """Test finding best matching chord in a song."""
    print("\n" + "=" * 60)
    print("Testing Best Match Finding")
    print("=" * 60)
    
    loader = SongLoader()
    
    # Wonderwall chords: Em, C, D, G
    song_chords = ["Em", "C", "D", "G"]
    
    # Test exact match
    match = loader.find_best_match_in_song("Em", song_chords)
    print(f"✓ Detected Em → {match['chord']} (similarity: {match['similarity']:.2f}, type: {match['match_type']})")
    
    # Test close match (Em7 should match Em)
    match = loader.find_best_match_in_song("Em7", song_chords)
    print(f"✓ Detected Em7 → {match['chord']} (similarity: {match['similarity']:.2f}, type: {match['match_type']})")
    
    # Test distant match (F#m should probably match Em or G)
    match = loader.find_best_match_in_song("F#m", song_chords)
    print(f"✓ Detected F#m → {match['chord']} (similarity: {match['similarity']:.2f}, type: {match['match_type']})")
    
    return True

def test_constraint_function():
    """Test the chord constraint function."""
    print("\n" + "=" * 60)
    print("Testing Chord Constraint Function")
    print("=" * 60)
    
    song_chords = ["Em", "C", "D", "G"]
    
    # Test 1: Chord in song (Em)
    raw_result = {
        'primary_chords': ['Em'],
        'chord_confidence': 0.75
    }
    constrained = constrain_chord_to_song(raw_result, song_chords, verbose=True)
    print(f"\nTest 1: Em detected (in song)")
    print(f"  Raw: {constrained['raw_chord']} ({constrained['raw_confidence']:.2f})")
    print(f"  Final: {constrained['final_chord']} ({constrained['final_confidence']:.2f})")
    print(f"  Match: {constrained['song_match']} ({constrained['match_type']})")
    
    # Test 2: Chord not in song (F#m)
    raw_result = {
        'primary_chords': ['F#m'],
        'chord_confidence': 0.65
    }
    constrained = constrain_chord_to_song(raw_result, song_chords, verbose=True)
    print(f"\nTest 2: F#m detected (not in song)")
    print(f"  Raw: {constrained['raw_chord']} ({constrained['raw_confidence']:.2f})")
    print(f"  Final: {constrained['final_chord']} ({constrained['final_confidence']:.2f})")
    print(f"  Match: {constrained['song_match']} ({constrained['match_type']})")
    if constrained.get('suggested_chord'):
        print(f"  Suggested: {constrained['suggested_chord']}")
    
    # Test 3: Related chord (Em7)
    raw_result = {
        'primary_chords': ['Em7'],
        'chord_confidence': 0.80
    }
    constrained = constrain_chord_to_song(raw_result, song_chords, verbose=True)
    print(f"\nTest 3: Em7 detected (related to Em)")
    print(f"  Raw: {constrained['raw_chord']} ({constrained['raw_confidence']:.2f})")
    print(f"  Final: {constrained['final_chord']} ({constrained['final_confidence']:.2f})")
    print(f"  Match: {constrained['song_match']} ({constrained['match_type']})")
    
    return True

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SONG-CONSTRAINED CHORD DETECTION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Song Loader", test_song_loader),
        ("Pitch Classes", test_pitch_classes),
        ("Chord Similarity", test_chord_similarity),
        ("Best Match Finding", test_best_match),
        ("Constraint Function", test_constraint_function),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return all_passed

if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
