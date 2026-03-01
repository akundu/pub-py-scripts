// Main application logic
let ws = null;
let audioContext = null;
let mediaStream = null;
let isRecording = false;
let config = {};
let currentSongId = null;
let currentSkillLevel = 'intermediate';

// Map all 7 difficulty/skill names to the 3 canonical levels
const SKILL_NAME_MAP = {
  beginner: 'beginner',
  elementary: 'beginner',
  novice: 'beginner',
  intermediate: 'intermediate',
  proficient: 'intermediate',
  advanced: 'advanced',
  expert: 'advanced',
};

function resolveSkillLevel(name) {
  return SKILL_NAME_MAP[name] || 'intermediate';
}

// Skill level presets (mirrors server-side lib/skill_levels.py)
const SKILL_PRESETS = {
  beginner: {
    confidence_threshold: 0.30,
    chord_window: 0.6,
    chord_window_confidence: 0.30,
    song_influence: 0.85,
    decay_rate: 1.2,
    hysteresis_bonus: 0.30,
    timing_tolerance: 1.5,
  },
  intermediate: {
    confidence_threshold: 0.45,
    chord_window: 0.3,
    chord_window_confidence: 0.45,
    song_influence: 0.70,
    decay_rate: 2.3,
    hysteresis_bonus: 0.15,
    timing_tolerance: 0.75,
  },
  advanced: {
    confidence_threshold: 0.55,
    chord_window: 0.15,
    chord_window_confidence: 0.50,
    song_influence: 0.50,
    decay_rate: 3.5,
    hysteresis_bonus: 0.05,
    timing_tolerance: 0.25,
  },
};

// Helper function to get timestamp for logging
function getTimestamp() {
  const now = new Date();
  const year = now.getFullYear();
  const month = String(now.getMonth() + 1).padStart(2, '0');
  const day = String(now.getDate()).padStart(2, '0');
  const hours = String(now.getHours()).padStart(2, '0');
  const minutes = String(now.getMinutes()).padStart(2, '0');
  const seconds = String(now.getSeconds()).padStart(2, '0');
  return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

// Helper function for timestamped console.log
function logWithTimestamp(...args) {
  console.log(`[${getTimestamp()}]`, ...args);
}

// Helper function for timestamped console.error
function errorWithTimestamp(...args) {
  console.error(`[${getTimestamp()}]`, ...args);
}

// Helper function for debug-only logging (only logs when config.debug is true)
function debugLog(...args) {
  if (config && config.debug) {
    console.log(`[${getTimestamp()}] [DEBUG]`, ...args);
  }
}

// Log all active settings to the console (source: url or default)
function logAllSettings() {
  if (!config || typeof config !== 'object') return;
  logWithTimestamp('📋 All settings in use:');
  const keys = Object.keys(config).sort();
  for (const key of keys) {
    const source = (configSource && configSource[key]) || 'default';
    const val = config[key];
    const display = typeof val === 'boolean' ? (val ? 'true' : 'false') : String(val);
    logWithTimestamp(`  ${key}: ${display} (${source})`);
  }
  logWithTimestamp('📋 Full config object:', config);
}

// Track which parameters came from URL
let urlParams = {};
let configSource = {};

// Get config from URL parameters or use defaults
function getConfigFromURL() {
  const params = new URLSearchParams(window.location.search);
  urlParams = Object.fromEntries(params);
  
  const defaultConfig = {
    instrument: 'guitar',
    silence_threshold: 0.005,
    confidence_threshold: 0.45,
    chord_window: 0.3,  // chord smoothing window in seconds
    chord_window_confidence: 0.45,  // minimum confidence for chord-window results
    overlap: 0.75,
    show_frequencies: false,
    show_chroma: false,
    frequencies_only: false,
    notes_only: false,
    debug: false,
    log: false,
    log_interval: 0.5,
    song_influence: 0.7,  // 0=raw detection wins, 1=song chords dominate (only when song selected)
    map_similar_variants: true,  // map Em/Em7/Emin to one form for stable display
    // Frequencies/notes-only mode params
    sensitivity: 1.0,
    multi_pitch: true,
    single_pitch: false,
    show_fft: false,
    raw_frequencies: false,
    decay_rate: 2.3,
    hysteresis_bonus: 0.15,
    timing_tolerance: 0.75,
    timeline_sync: 'auto',  // auto, manual, off
  };

  const config = { ...defaultConfig };

  // Apply skill_level preset first (URL params override afterward)
  const skillParam = params.get('skill_level');
  if (skillParam && SKILL_NAME_MAP[skillParam]) {
    const resolved = resolveSkillLevel(skillParam);
    Object.assign(config, SKILL_PRESETS[resolved]);
    currentSkillLevel = resolved;
  }

  // Override with URL parameters and track source
  for (const [key, value] of params.entries()) {
    if (key === 'skill_level') continue; // already handled above
    if (key in defaultConfig) {
      configSource[key] = 'url';
      if (key === 'map_similar_variants' || (typeof defaultConfig[key] === 'boolean')) {
        config[key] = value.toLowerCase() === 'true';
      } else if (typeof defaultConfig[key] === 'number') {
        config[key] = parseFloat(value);
      } else {
        config[key] = value;
      }
    }
  }

  // Explicitly apply song_influence from URL so it's never missed (e.g. caching or param order)
  const urlSongInfluence = params.get('song_influence');
  if (urlSongInfluence !== null && urlSongInfluence !== '') {
    const num = parseFloat(urlSongInfluence);
    if (!Number.isNaN(num)) {
      config.song_influence = Math.max(0, Math.min(1, num));
      configSource.song_influence = 'url';
    }
  }

  // Mark defaults
  for (const key in defaultConfig) {
    if (!configSource[key]) {
      configSource[key] = 'default';
    }
  }

  // amplitude_threshold is obsolete - silence_threshold is used by the processing pipeline

  return config;
}

// Get song ID from URL parameters
function getSongFromURL() {
  const params = new URLSearchParams(window.location.search);
  return params.get('song') || params.get('song_id');
}

// Initialize settings from URL or defaults
function initializeSettings() {
  config = getConfigFromURL();

  // Update UI with config values
  document.getElementById('instrument').value = config.instrument || 'guitar';
  document.getElementById('sensitivity').value = config.sensitivity || 0.8;
  document.getElementById('sensitivityValue').textContent = config.sensitivity || 0.8;
  document.getElementById('confidenceThreshold').value = config.confidence_threshold || 0.2;
  document.getElementById('confidenceValue').textContent = config.confidence_threshold || 0.2;
  document.getElementById('silenceThreshold').value = config.silence_threshold || 0.005;
  document.getElementById('silenceValue').textContent = config.silence_threshold || 0.005;
  document.getElementById('overlap').value = config.overlap || 0.8;
  document.getElementById('overlapValue').textContent = config.overlap || 0.8;
  document.getElementById('progression').checked = config.progression !== false;
  document.getElementById('multiPitch').checked = config.multi_pitch !== false;
  document.getElementById('showFrequencies').checked = config.show_frequencies === true;
  document.getElementById('showChroma').checked = config.show_chroma === true;
  document.getElementById('debug').checked = config.debug === true;
  document.getElementById('log').checked = config.log === true;

  // Chord window (smoothing) setting
  const chordWindowEl = document.getElementById('chordWindow');
  const chordWindowValueEl = document.getElementById('chordWindowValue');
  if (chordWindowEl && chordWindowValueEl) {
    chordWindowEl.value = config.chord_window || 0.3;
    chordWindowValueEl.textContent = config.chord_window || 0.3;
  }
  
  // Chord window confidence setting
  const chordWindowConfidenceEl = document.getElementById('chordWindowConfidence');
  const chordWindowConfidenceValueEl = document.getElementById('chordWindowConfidenceValue');
  if (chordWindowConfidenceEl && chordWindowConfidenceValueEl) {
    chordWindowConfidenceEl.value = config.chord_window_confidence || 0.45;
    chordWindowConfidenceValueEl.textContent = config.chord_window_confidence || 0.45;
  }

  // Song influence (when a song is selected) - Settings modal
  const songInfluenceEl = document.getElementById('songInfluence');
  const songInfluenceValueEl = document.getElementById('songInfluenceValue');
  if (songInfluenceEl && songInfluenceValueEl) {
    const v = parseFloat(config.song_influence);
    const val = (v >= 0 && v <= 1) ? v : 0.7;
    songInfluenceEl.value = val;
    songInfluenceValueEl.textContent = val.toFixed(2);
  }
  // Main-page Song Influence slider (live control)
  const songInfluenceLiveEl = document.getElementById('songInfluenceLive');
  const songInfluenceLiveValueEl = document.getElementById('songInfluenceLiveValue');
  if (songInfluenceLiveEl && songInfluenceLiveValueEl) {
    const v = parseFloat(config.song_influence);
    const val = (v >= 0 && v <= 1) ? v : 0.7;
    songInfluenceLiveEl.value = val;
    songInfluenceLiveValueEl.textContent = val.toFixed(2);
  }
  // Map similar variants (main page + Settings)
  const mapVariantsLiveEl = document.getElementById('mapSimilarVariantsLive');
  const mapVariantsModalEl = document.getElementById('mapSimilarVariants');
  if (mapVariantsLiveEl) mapVariantsLiveEl.checked = config.map_similar_variants !== false;
  if (mapVariantsModalEl) mapVariantsModalEl.checked = config.map_similar_variants !== false;

  // Sync skill level buttons
  document.querySelectorAll('.btn-skill').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.skill === currentSkillLevel);
  });

  logAllSettings();
}

// Update status display
function updateStatus(connected, text) {
  const indicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('statusText');

  indicator.className = 'status-indicator ' + (connected ? 'connected' : 'disconnected');
  statusText.textContent = text;
}

// Store songs globally for search
let allSongs = [];

// Fuzzy search function
function fuzzySearch(query, text) {
  query = query.toLowerCase();
  text = text.toLowerCase();
  
  // Exact match gets highest score
  if (text.includes(query)) {
    return 100 - text.indexOf(query);
  }
  
  // Character-by-character fuzzy match
  let queryIndex = 0;
  let matchScore = 0;
  let consecutiveMatches = 0;
  
  for (let i = 0; i < text.length && queryIndex < query.length; i++) {
    if (text[i] === query[queryIndex]) {
      queryIndex++;
      consecutiveMatches++;
      matchScore += consecutiveMatches * 2; // Bonus for consecutive matches
    } else {
      consecutiveMatches = 0;
    }
  }
  
  // Return score only if all characters matched
  return queryIndex === query.length ? matchScore : 0;
}

// Search songs
function searchSongs(query) {
  if (!query || query.trim() === '') {
    return allSongs.slice(0, 50); // Show first 50 if no query
  }
  
  const results = allSongs.map(song => {
    // Search in multiple fields
    const idScore = fuzzySearch(query, song.song_id) * 2; // ID matches weighted more
    const titleScore = fuzzySearch(query, song.title);
    const composerScore = fuzzySearch(query, song.composer) * 0.8;
    const genreScore = fuzzySearch(query, song.genre) * 0.6;
    const difficultyScore = fuzzySearch(query, song.difficulty) * 0.5;
    
    const totalScore = idScore + titleScore + composerScore + genreScore + difficultyScore;
    
    return { song, score: totalScore };
  })
  .filter(item => item.score > 0)
  .sort((a, b) => b.score - a.score)
  .slice(0, 20) // Top 20 results
  .map(item => item.song);
  
  return results;
}

// Load songs list
async function loadSongs() {
  try {
    const basePath = getBasePath();
    const response = await fetch(`${basePath}/api/songs`);
    const data = await response.json();
    
    if (data.songs && data.songs.length > 0) {
      allSongs = data.songs;
      logWithTimestamp(`📚 Loaded ${data.songs.length} songs`);
      
      // Initialize search
      setupSongSearch();
    }
  } catch (error) {
    console.error('Error loading songs:', error);
  }
}

// Setup song search autocomplete
function setupSongSearch() {
  const searchInput = document.getElementById('songSearch');
  const dropdown = document.getElementById('songDropdown');
  const clearBtn = document.getElementById('clearSongBtn');
  const selectedSongIdInput = document.getElementById('selectedSongId');
  let currentSelection = -1;
  
  // Show dropdown on focus
  searchInput.addEventListener('focus', () => {
    if (!selectedSongIdInput.value) {
      updateDropdown('');
    }
  });
  
  // Search as user types
  searchInput.addEventListener('input', (e) => {
    selectedSongIdInput.value = ''; // Clear selection when typing
    searchInput.classList.remove('has-selection');
    clearBtn.classList.add('hidden');
    currentSelection = -1;
    updateDropdown(e.target.value);
  });
  
  // Keyboard navigation
  searchInput.addEventListener('keydown', (e) => {
    const items = dropdown.querySelectorAll('.song-dropdown-item[data-song-id]');
    
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      currentSelection = Math.min(currentSelection + 1, items.length - 1);
      updateSelection(items);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      currentSelection = Math.max(currentSelection - 1, -1);
      updateSelection(items);
    } else if (e.key === 'Enter' && currentSelection >= 0 && items[currentSelection]) {
      e.preventDefault();
      items[currentSelection].click();
    } else if (e.key === 'Escape') {
      dropdown.classList.add('hidden');
      currentSelection = -1;
    }
  });
  
  // Hide dropdown when clicking outside
  document.addEventListener('click', (e) => {
    if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
      dropdown.classList.add('hidden');
      currentSelection = -1;
    }
  });
  
  // Clear button
  clearBtn.addEventListener('click', () => {
    searchInput.value = '';
    searchInput.classList.remove('has-selection');
    selectedSongIdInput.value = '';
    clearBtn.classList.add('hidden');
    currentSelection = -1;
    updateURLWithSong(''); // Clear URL parameter
    handleSongSelect(''); // Clear song selection
  });
  
  function updateSelection(items) {
    items.forEach((item, index) => {
      if (index === currentSelection) {
        item.classList.add('selected');
        item.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      } else {
        item.classList.remove('selected');
      }
    });
  }
  
  function updateDropdown(query) {
    const results = searchSongs(query);
    
    if (results.length === 0) {
      dropdown.innerHTML = '<div class="song-dropdown-item no-results">No songs found</div>';
      dropdown.classList.remove('hidden');
      return;
    }
    
    dropdown.innerHTML = results.map(song => `
      <div class="song-dropdown-item" data-song-id="${song.song_id}">
        <div class="song-item-title">${song.title}</div>
        <div class="song-item-meta">
          ${song.composer} • ${song.genre} • ${song.difficulty}
          <span class="song-item-id">${song.song_id}</span>
        </div>
      </div>
    `).join('');
    
    // Add click handlers
    dropdown.querySelectorAll('.song-dropdown-item[data-song-id]').forEach(item => {
      item.addEventListener('click', () => {
        const songId = item.getAttribute('data-song-id');
        const song = allSongs.find(s => s.song_id === songId);
        if (song) {
          selectSong(song);
        }
      });
    });
    
    dropdown.classList.remove('hidden');
  }
  
  function selectSong(song) {
    searchInput.value = `${song.title} - ${song.composer}`;
    searchInput.classList.add('has-selection');
    selectedSongIdInput.value = song.song_id;
    dropdown.classList.add('hidden');
    clearBtn.classList.remove('hidden');
    currentSelection = -1;
    updateURLWithSong(song.song_id);
    handleSongSelect(song.song_id);
  }
  
  // Auto-select song from URL if present
  const urlSongId = getSongFromURL();
  if (urlSongId) {
    const song = allSongs.find(s => s.song_id === urlSongId);
    if (song) {
      // Delay to ensure WebSocket is connected
      setTimeout(() => {
        selectSong(song);
        logWithTimestamp(`🎵 Auto-loaded song from URL: ${song.title}`);
      }, 1000);
    } else {
      console.warn(`Song ID from URL not found: ${urlSongId}`);
    }
  }
}

// Update URL with song parameter (without reloading page)
function updateURLWithSong(songId) {
  const url = new URL(window.location);
  if (songId) {
    url.searchParams.set('song', songId);
  } else {
    url.searchParams.delete('song');
    url.searchParams.delete('song_id');
  }
  window.history.pushState({}, '', url);
}

// Handle song selection
function handleSongSelect(songId) {
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    console.error('Cannot set song: WebSocket not connected');
    return;
  }
  
  currentSongId = songId;
  
  const message = {
    type: 'set_song',
    song_id: songId || null
  };
  
  ws.send(JSON.stringify(message));
  
  // Update display mode
  const constrainedDisplay = document.getElementById('constrainedDisplay');
  const chordDisplay = document.getElementById('chordDisplay');
  
  if (songId) {
    constrainedDisplay.style.display = 'flex';
    chordDisplay.style.display = 'none';
  } else {
    constrainedDisplay.style.display = 'none';
    chordDisplay.style.display = 'block';
  }
}

// Get base path for WebSocket (handles Envoy prefix)
function getBasePath() {
  const pathname = window.location.pathname;
  // If we're under /chord_detector/ or exactly /chord_detector, use that as base
  if (pathname.startsWith('/chord_detector/') || pathname === '/chord_detector') {
    return '/chord_detector';
  }
  // If we're under /c_d/ or exactly /c_d, use that as base (alias for chord_detector)
  if (pathname.startsWith('/c_d/') || pathname === '/c_d') {
    return '/c_d';
  }
  // Otherwise use root
  return '';
}

// Track if we've received the server's "connected" message (module-level so handleWebSocketMessage can access it)
let serverConnected = false;
let connectionTimeout = null;

// Connect WebSocket
async function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const basePath = getBasePath();
  const wsUrl = `${protocol}//${window.location.host}${basePath}/ws`;

  if (config.debug) {
    logWithTimestamp(`🔌 Attempting WebSocket connection to: ${wsUrl}`);
    logWithTimestamp(`   Protocol: ${protocol}, Base path: ${basePath}, Host: ${window.location.host}`);
  }

  // Reset connection tracking
  serverConnected = false;
  if (connectionTimeout) {
    clearTimeout(connectionTimeout);
    connectionTimeout = null;
  }

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    if (config.debug) {
      logWithTimestamp('🔌 WebSocket connection opened, sending configuration...');
    }
    updateStatus(true, 'Connecting...');

    // Send initial configuration
    try {
      ws.send(JSON.stringify(config));
      
      // Set up a timeout to detect if server doesn't respond
      connectionTimeout = setTimeout(() => {
        if (!serverConnected) {
          errorWithTimestamp('Server did not respond. Check server logs.');
          updateStatus(false, 'Connection Timeout');
        }
      }, 5000);
    } catch (error) {
      console.error('Error sending configuration:', error);
      updateStatus(false, 'Send Error');
    }
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handleWebSocketMessage(data);
    } catch (error) {
      console.error('Error parsing WebSocket message:', error);
    }
  };

  ws.onerror = (error) => {
    console.error('WebSocket error:', error);
    updateStatus(false, 'Connection Error');
  };

  ws.onclose = (event) => {
    if (config.debug) {
      logWithTimestamp(`WebSocket closed. Code: ${event.code}, Clean: ${event.wasClean}`);
    }
    updateStatus(false, 'Disconnected');
    if (isRecording) {
      stopRecording();
    }
  };
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
  if (data.type === 'connected') {
    // Mark that server has confirmed connection
    serverConnected = true;
    if (connectionTimeout) {
      clearTimeout(connectionTimeout);
      connectionTimeout = null;
    }
    
    // Update status to Connected and enable start button
    updateStatus(true, 'Connected');
    document.getElementById('startBtn').disabled = false;
    
    // Log connection info
    logWithTimestamp('✅ Connected:', data.message);
    
    // Log configuration parameters if debug is enabled OR server is in DEBUG mode
    const serverDebug = data.server_log_level === 'DEBUG';
    if (config.debug || serverDebug) {
      logWithTimestamp('📊 Configuration:');
      logWithTimestamp('  Instrument:', config.instrument);
      logWithTimestamp('  Silence Threshold:', config.silence_threshold);
      logWithTimestamp('  Confidence Threshold:', config.confidence_threshold);
      logWithTimestamp('  Chord Window:', config.chord_window, config.chord_window > 0 ? '(smoothing enabled)' : '(instant)');
      if (config.chord_window > 0) {
        logWithTimestamp('  Chord Window Confidence:', config.chord_window_confidence || 0.45);
      }
      logWithTimestamp('  Overlap:', config.overlap);
      if (config.frequencies_only || config.notes_only) {
        logWithTimestamp('  Mode:', config.frequencies_only ? 'frequencies_only' : 'notes_only');
        logWithTimestamp('  Sensitivity:', config.sensitivity);
        logWithTimestamp('  Multi-pitch:', config.multi_pitch);
      }
      if (serverDebug) {
        logWithTimestamp('  Server Log Level: DEBUG (server-side debug enabled)');
      }
      logWithTimestamp('📊 Full Config Object:', JSON.stringify(config, null, 2));
    }
    return;
  }

  if (data.type === 'error') {
    errorWithTimestamp('Server error:', data.message);
    return;
  }

  if (data.type === 'song_loaded') {
    logWithTimestamp(`🎵 Song loaded: ${data.song_info.title} by ${data.song_info.composer}`);
    logWithTimestamp(`   Chords: ${data.chords.join(', ')}`);
    // Show skill level suggestion based on song difficulty
    const suggestionEl = document.getElementById('skillSuggestion');
    if (data.suggested_skill && suggestionEl) {
      const label = data.suggested_skill.charAt(0).toUpperCase() + data.suggested_skill.slice(1);
      suggestionEl.textContent = `Suggested: ${label}`;
      suggestionEl.classList.remove('hidden');
      logWithTimestamp(`   Suggested skill: ${data.suggested_skill}`);
    }
    return;
  }

  if (data.type === 'song_cleared') {
    logWithTimestamp('🎵 Song cleared - using raw detection only');
    const suggestionEl = document.getElementById('skillSuggestion');
    if (suggestionEl) suggestionEl.classList.add('hidden');
    return;
  }

  // Log detection results only when log mode or debug mode is enabled
  if (data.type === 'chord' || data.type === 'notes' || data.type === 'frequencies') {
    if (config.log || config.debug) {
      if (data.type === 'chord' && data.chord) {
        const chromaStr = data.chroma ? `Chroma: [${data.chroma.map(v => v.toFixed(3)).join(', ')}]` : '';
        const notesStr = data.notes ? `Notes: ${data.notes.map(n => Array.isArray(n) ? `${n[0]}(${n[1].toFixed(0)}Hz)` : n).join(', ')}` : '';
        const parts = [chromaStr, notesStr].filter(s => s);
        const output = parts.length > 0 ? `${parts.join(' | ')} -> ${data.chord}` : data.chord;
        const stability = data.stability >= 2 ? '' : ' [unstable]';
        logWithTimestamp(output + stability);
      } else if (data.type === 'notes' && data.notes && data.notes.length > 0) {
        const notesStr = data.notes.map(n => Array.isArray(n) ? `${n[0]}(${n[1].toFixed(0)}Hz)` : n).join(', ');
        logWithTimestamp('Notes:', notesStr);
      } else if (data.type === 'frequencies' && data.frequencies && data.frequencies.length > 0) {
        const freqStr = data.frequencies.map(f => Array.isArray(f) ? `${f[0]}(${f[1].toFixed(0)}Hz)` : f).join(', ');
        logWithTimestamp('Frequencies:', freqStr);
      }
    }
  }

  // Update UI based on detection type
  if (data.type === 'chord') {
    updateChordDisplay(data);
    // Also update notes if provided
    if (data.notes) {
      updateNotesDisplay(data.notes);
    }
  } else if (data.type === 'notes') {
    updateNotesDisplay(data.notes || []);
    // If there's a chord candidate with low confidence, show it
    if (data.chord_candidate) {
      updateChordDisplay({
        chord: data.chord_candidate,
        confidence: data.confidence || 0,
        stability: 0
      });
    } else {
      updateChordDisplay(null);
    }
  } else if (data.type === 'frequencies') {
    updateFrequenciesDisplay(data.frequencies || []);
  } else if (data.type === 'listening') {
    // Don't clear everything on listening - keep last state
    // updateChordDisplay(null);
  }

  // Update visualizations
  if (data.chroma) {
    updateChromaVisualization(data.chroma);
  }
  if (data.notes) {
    updateFrequencyVisualization(data.notes);
  }
}

// Update chord display
function updateChordDisplay(data) {
  if (data && data.song_constrained && currentSongId) {
    // Show constrained display
    updateConstrainedDisplay(data);
  } else {
    // Show regular display
    const chordNameEl = document.getElementById('chordName');
    const chordConfidenceEl = document.getElementById('chordConfidence');
    const chordStabilityEl = document.getElementById('chordStability');

    if (data && data.chord) {
      chordNameEl.textContent = data.chord;
      chordNameEl.className = 'chord-name detected';

      if (data.confidence !== undefined) {
        // Show votes info if available (when using chord_window smoothing)
        let confidenceText = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        if (data.votes !== undefined && data.total_votes !== undefined) {
          confidenceText += ` (${data.votes}/${data.total_votes} votes)`;
        }
        chordConfidenceEl.textContent = confidenceText;
        chordConfidenceEl.style.display = 'block';
      } else {
        chordConfidenceEl.style.display = 'none';
      }

      if (data.stability !== undefined) {
        const stabilityText = data.stability >= 2 ? 'Stable' : 'Unstable';
        chordStabilityEl.textContent = stabilityText;
        chordStabilityEl.className = data.stability >= 2 ? 'chord-stability stable' : 'chord-stability unstable';
      }
    } else {
      chordNameEl.textContent = '--';
      chordNameEl.className = 'chord-name';
      chordConfidenceEl.style.display = 'none';
      chordStabilityEl.textContent = '';
    }
  }
}

// Update constrained display (both raw and song-constrained results)
function updateConstrainedDisplay(data) {
  const rawChordEl = document.getElementById('rawChord');
  const rawConfidenceEl = document.getElementById('rawConfidence');
  const finalChordEl = document.getElementById('finalChord');
  const finalConfidenceEl = document.getElementById('finalConfidence');
  const matchIndicatorEl = document.getElementById('matchIndicator');
  const matchTypeEl = document.getElementById('matchType');
  
  if (data.raw_chord) {
    rawChordEl.textContent = data.raw_chord;
    rawConfidenceEl.textContent = `Confidence: ${(data.raw_confidence * 100).toFixed(1)}%`;
  } else {
    rawChordEl.textContent = '--';
    rawConfidenceEl.textContent = '';
  }
  
  if (data.final_chord) {
    finalChordEl.textContent = data.final_chord;
    finalConfidenceEl.textContent = `Confidence: ${(data.final_confidence * 100).toFixed(1)}%`;
    
    // Update match indicator
    if (data.song_match) {
      matchIndicatorEl.textContent = '✓';
      matchIndicatorEl.className = 'match-indicator match';
      matchIndicatorEl.title = 'Chord is in song';
    } else {
      matchIndicatorEl.textContent = '⚠';
      matchIndicatorEl.className = 'match-indicator no-match';
      matchIndicatorEl.title = 'Chord not in song';
    }
    
    // Update match type
    const matchTypeText = {
      'exact': 'Exact match',
      'related': 'Related chord',
      'partial': 'Partial match',
      'none': 'No match'
    };
    matchTypeEl.textContent = matchTypeText[data.match_type] || '';
    matchTypeEl.className = `match-type match-${data.match_type}`;
  } else {
    finalChordEl.textContent = '--';
    finalConfidenceEl.textContent = '';
    matchIndicatorEl.textContent = '';
    matchTypeEl.textContent = '';
  }
}

// Update notes display
function updateNotesDisplay(notes) {
  const notesListEl = document.getElementById('notesList');

  if (notes.length > 0) {
    notesListEl.innerHTML = notes.map(note => {
      const [noteName, freq] = Array.isArray(note) ? note : [note, null];
      return `<span class="note-badge">${noteName}${freq ? ` (${freq.toFixed(0)}Hz)` : ''}</span>`;
    }).join('');
  } else {
    notesListEl.textContent = 'No notes detected';
  }
}

// Update frequencies display
function updateFrequenciesDisplay(frequencies) {
  updateNotesDisplay(frequencies);
}

// Start recording
async function startRecording() {
  try {
    // Check if getUserMedia is available
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('getUserMedia is not supported in this browser. Please use a modern browser like Chrome, Firefox, or Edge.');
    }

    // Check if WebSocket is connected
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      throw new Error('WebSocket not connected. Please wait for connection or refresh the page.');
    }

    // Request microphone access with RAW audio settings
    // CRITICAL: Disable all audio processing for accurate chord detection
    // These features are designed for voice calls and destroy harmonic content
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: false,  // Don't remove echoes - they contain harmonic info
        noiseSuppression: false,  // Don't suppress "noise" - it includes harmonics!
        autoGainControl: false,   // Don't adjust gain - we need consistent levels
        sampleRate: 44100         // Match CLI sample rate
      }
    });

    // Use 44100 Hz sample rate to match CLI
    audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });

    // Log the actual sample rate being used (debug only, except for warnings)
    debugLog(`🎵 Audio Context initialized: sample rate = ${audioContext.sampleRate} Hz`);
    if (audioContext.sampleRate !== 44100) {
      logWithTimestamp(`⚠️ Warning: Sample rate is ${audioContext.sampleRate} Hz, expected 44100 Hz. Detection may be affected.`);
    }

    const source = audioContext.createMediaStreamSource(mediaStream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    let audioChunkCount = 0;
    processor.onaudioprocess = (e) => {
      if (ws && ws.readyState === WebSocket.OPEN && isRecording) {
        const inputData = e.inputBuffer.getChannelData(0);

        // Log first few chunks for debugging (debug mode only)
        audioChunkCount++;
        if (config.debug && audioChunkCount <= 3) {
          const rms = Math.sqrt(inputData.reduce((sum, val) => sum + val * val, 0) / inputData.length);
          debugLog(`Audio chunk #${audioChunkCount}: ${inputData.length} samples, RMS: ${rms.toFixed(4)}`);
        }

        // Convert Float32Array to binary
        // Note: inputData.buffer is the underlying ArrayBuffer
        // We need to send it as binary
        try {
          ws.send(inputData.buffer);
        } catch (error) {
          errorWithTimestamp('Error sending audio data:', error);
          stopRecording();
        }
        // Update waveform on next frame (don't block audio callback)
        const drawWaveform = window.updateWaveformVisualization;
        if (typeof drawWaveform === 'function') {
          const step = Math.max(1, Math.floor(inputData.length / 800));
          const sampled = [];
          for (let i = 0; i < inputData.length; i += step) sampled.push(inputData[i]);
          requestAnimationFrame(() => { drawWaveform(sampled); });
        }
      }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);

    isRecording = true;
    document.getElementById('startBtn').disabled = true;
    document.getElementById('stopBtn').disabled = false;
    updateStatus(true, 'Recording');

  } catch (error) {
    errorWithTimestamp('Error starting recording:', error);

    let errorMessage = 'Failed to access microphone. ';

    // Provide specific error messages
    if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
      errorMessage += 'Microphone permission was denied. Please:\n' +
        '1. Click the lock icon in your browser\'s address bar\n' +
        '2. Allow microphone access\n' +
        '3. Refresh the page and try again';
    } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
      errorMessage += 'No microphone found. Please connect a microphone and try again.';
    } else if (error.name === 'NotReadableError' || error.name === 'TrackStartError') {
      errorMessage += 'Microphone is already in use by another application. Please close other applications using the microphone.';
    } else if (error.name === 'OverconstrainedError' || error.name === 'ConstraintNotSatisfiedError') {
      errorMessage += 'Microphone constraints could not be satisfied. Please try a different microphone.';
    } else if (error.message && error.message.includes('getUserMedia')) {
      errorMessage += 'getUserMedia is not supported. If you\'re not on localhost, you may need HTTPS.';
    } else if (error.message) {
      errorMessage += error.message;
    } else {
      errorMessage += 'Please check your browser settings and microphone permissions.';
    }

    // Check if HTTPS is required (for non-localhost)
    const isLocalhost = window.location.hostname === 'localhost' ||
      window.location.hostname === '127.0.0.1' ||
      window.location.hostname.startsWith('192.168.');

    if (!isLocalhost && window.location.protocol !== 'https:') {
      errorMessage += '\n\nNote: Some browsers require HTTPS for microphone access when not on localhost.';
    }

    alert(errorMessage);
    updateStatus(false, 'Microphone Error');
  }
}

// Stop recording
function stopRecording() {
  isRecording = false;

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
    mediaStream = null;
  }

  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }

  document.getElementById('startBtn').disabled = false;
  document.getElementById('stopBtn').disabled = true;
  updateStatus(ws && ws.readyState === WebSocket.OPEN, 'Connected');
}

// Settings modal
const modal = document.getElementById('settingsModal');
const settingsBtn = document.getElementById('settingsBtn');
const closeBtn = document.querySelector('.close');
const settingsForm = document.getElementById('settingsForm');
const resetSettingsBtn = document.getElementById('resetSettings');

settingsBtn.onclick = () => {
  modal.style.display = 'block';
  // Sync current config into form so modal reflects live slider / URL
  const songInfluenceFormEl = document.getElementById('songInfluence');
  const songInfluenceFormValueEl = document.getElementById('songInfluenceValue');
  if (songInfluenceFormEl && songInfluenceFormValueEl && config.song_influence != null) {
    const v = parseFloat(config.song_influence);
    const val = (v >= 0 && v <= 1) ? v : 0.7;
    songInfluenceFormEl.value = val;
    songInfluenceFormValueEl.textContent = val.toFixed(2);
  }
  const mapVariantsModal = document.getElementById('mapSimilarVariants');
  if (mapVariantsModal) mapVariantsModal.checked = config.map_similar_variants !== false;
};

closeBtn.onclick = () => {
  modal.style.display = 'none';
};

window.onclick = (event) => {
  if (event.target === modal) {
    modal.style.display = 'none';
  }
};

// Update range value displays
document.getElementById('sensitivity').addEventListener('input', (e) => {
  document.getElementById('sensitivityValue').textContent = e.target.value;
});

document.getElementById('confidenceThreshold').addEventListener('input', (e) => {
  document.getElementById('confidenceValue').textContent = e.target.value;
});

document.getElementById('silenceThreshold').addEventListener('input', (e) => {
  document.getElementById('silenceValue').textContent = e.target.value;
});

document.getElementById('overlap').addEventListener('input', (e) => {
  document.getElementById('overlapValue').textContent = e.target.value;
});

// Chord window slider (if it exists)
const chordWindowSlider = document.getElementById('chordWindow');
if (chordWindowSlider) {
  chordWindowSlider.addEventListener('input', (e) => {
    document.getElementById('chordWindowValue').textContent = e.target.value;
  });
}

// Chord window confidence slider (if it exists)
const chordWindowConfidenceSlider = document.getElementById('chordWindowConfidence');
if (chordWindowConfidenceSlider) {
  chordWindowConfidenceSlider.addEventListener('input', (e) => {
    document.getElementById('chordWindowConfidenceValue').textContent = e.target.value;
  });
}

const songInfluenceSlider = document.getElementById('songInfluence');
if (songInfluenceSlider) {
  ['input', 'change'].forEach(ev => {
    songInfluenceSlider.addEventListener(ev, (e) => {
      const el = e.target;
      const val = Math.max(0, Math.min(1, parseFloat(el.value) || 0.5));
      el.value = val;
      document.getElementById('songInfluenceValue').textContent = val.toFixed(2);
      config.song_influence = val;
      const liveSlider = document.getElementById('songInfluenceLive');
      const liveSpan = document.getElementById('songInfluenceLiveValue');
      if (liveSlider) liveSlider.value = val;
      if (liveSpan) liveSpan.textContent = val.toFixed(2);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'config_update', config: { song_influence: val } }));
      }
    });
  });
}

// Main-page Song Influence slider: live update (input + change so value always commits)
const songInfluenceLiveSlider = document.getElementById('songInfluenceLive');
if (songInfluenceLiveSlider) {
  ['input', 'change'].forEach(ev => {
    songInfluenceLiveSlider.addEventListener(ev, (e) => {
      const el = e.target;
      const val = Math.max(0, Math.min(1, parseFloat(el.value) || 0.5));
      el.value = val; // ensure DOM value is exact
      config.song_influence = val;
      const liveValueEl = document.getElementById('songInfluenceLiveValue');
      if (liveValueEl) liveValueEl.textContent = val.toFixed(2);
      const modalSlider = document.getElementById('songInfluence');
      const modalSpan = document.getElementById('songInfluenceValue');
      if (modalSlider) modalSlider.value = val;
      if (modalSpan) modalSpan.textContent = val.toFixed(2);
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'config_update', config: { song_influence: val } }));
      }
    });
  });
}

// Map similar variants checkbox (main page): live config update
const mapSimilarVariantsLiveEl = document.getElementById('mapSimilarVariantsLive');
if (mapSimilarVariantsLiveEl) {
  mapSimilarVariantsLiveEl.addEventListener('change', () => {
    config.map_similar_variants = mapSimilarVariantsLiveEl.checked;
    const modalCheck = document.getElementById('mapSimilarVariants');
    if (modalCheck) modalCheck.checked = config.map_similar_variants;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: 'config_update', config: { map_similar_variants: config.map_similar_variants } }));
    }
  });
}

// Handle settings form submission
settingsForm.addEventListener('submit', (e) => {
  e.preventDefault();

  const formData = new FormData(settingsForm);
  config = {};

  for (const [key, value] of formData.entries()) {
    if (key === 'progression' || key === 'multi_pitch' || key === 'show_frequencies' ||
      key === 'show_chroma' || key === 'debug' || key === 'log' || key === 'map_similar_variants') {
      const id = key === 'multi_pitch' ? 'multiPitch' : key === 'show_frequencies' ? 'showFrequencies' :
        key === 'show_chroma' ? 'showChroma' : key === 'map_similar_variants' ? 'mapSimilarVariants' : key;
      config[key] = document.getElementById(id) ? document.getElementById(id).checked : (value === 'true' || value === true);
    } else {
      config[key] = value;
    }
  }

  // Convert string numbers to actual numbers
  config.sensitivity = parseFloat(config.sensitivity);
  config.confidence_threshold = parseFloat(config.confidence_threshold);
  config.silence_threshold = parseFloat(config.silence_threshold);
  config.overlap = parseFloat(config.overlap);

  // Get chord_window value from the form
  const chordWindowEl = document.getElementById('chordWindow');
  if (chordWindowEl) {
    config.chord_window = parseFloat(chordWindowEl.value) || 0.3;
  }
  
  // Get chord_window_confidence value from the form
  const chordWindowConfidenceEl = document.getElementById('chordWindowConfidence');
  if (chordWindowConfidenceEl) {
    config.chord_window_confidence = parseFloat(chordWindowConfidenceEl.value) || 0.45;
  } else {
    config.chord_window_confidence = parseFloat(config.chord_window_confidence) || 0.45;
  }

  // Song influence (when a song is selected)
  const songInfluenceFormEl = document.getElementById('songInfluence');
  if (songInfluenceFormEl) {
    const v = parseFloat(songInfluenceFormEl.value);
    config.song_influence = (v >= 0 && v <= 1) ? v : 0.7;
  } else {
    config.song_influence = parseFloat(config.song_influence) || 0.7;
  }

  // Ensure all other defaults are set if missing
  const defaultConfig = {
    instrument: 'guitar',
    silence_threshold: 0.005,
    confidence_threshold: 0.45,
    chord_window: 0.3,
    chord_window_confidence: 0.45,
    overlap: 0.75,
    show_frequencies: false,
    show_chroma: false,
    frequencies_only: false,
    notes_only: false,
    debug: false,
    log: false,
    log_interval: 0.5,
    song_influence: 0.7,
    map_similar_variants: true,
    sensitivity: 1.0,
    multi_pitch: true,
    single_pitch: false,
    show_fft: false,
    raw_frequencies: false,
    decay_rate: 2.3,
    hysteresis_bonus: 0.15,
  };

  // Merge with defaults to ensure all values are set
  config = { ...defaultConfig, ...config };

  // Keep main-page Song Influence slider and Map similar variants in sync
  const liveSlider = document.getElementById('songInfluenceLive');
  const liveValueSpan = document.getElementById('songInfluenceLiveValue');
  if (liveSlider && liveValueSpan) {
    const v = parseFloat(config.song_influence);
    const val = (v >= 0 && v <= 1) ? v : 0.7;
    liveSlider.value = val;
    liveValueSpan.textContent = val.toFixed(2);
  }
  const mapLiveCheck = document.getElementById('mapSimilarVariantsLive');
  if (mapLiveCheck) mapLiveCheck.checked = config.map_similar_variants !== false;

  // Reconnect with new config
  if (ws) {
    ws.close();
  }
  connectWebSocket();

  logWithTimestamp('📋 Settings applied from form:');
  logAllSettings();

  modal.style.display = 'none';
});

// Reset settings
resetSettingsBtn.addEventListener('click', () => {
  initializeSettings();
});

// Display configuration panel
function displayConfigPanel() {
  const panel = document.getElementById('configPanel');
  
  // Song Info
  const songInfoEl = document.getElementById('configSongInfo');
  if (currentSongId) {
    const song = allSongs.find(s => s.song_id === currentSongId);
    if (song) {
      songInfoEl.innerHTML = `
        <div class="config-item">
          <span class="config-item-label">Song ID:</span>
          <span class="config-item-value from-url">${song.song_id}</span>
        </div>
        <div class="config-item">
          <span class="config-item-label">Title:</span>
          <span class="config-item-value">${song.title}</span>
        </div>
        <div class="config-item">
          <span class="config-item-label">Composer:</span>
          <span class="config-item-value">${song.composer}</span>
        </div>
        <div class="config-item">
          <span class="config-item-label">Genre:</span>
          <span class="config-item-value">${song.genre}</span>
        </div>
        <div class="config-item">
          <span class="config-item-label">Difficulty:</span>
          <span class="config-item-value">${song.difficulty}</span>
        </div>
      `;
    }
  } else {
    songInfoEl.innerHTML = '<div class="config-empty">No song selected (raw detection only)</div>';
  }
  songInfoEl.innerHTML += configItem('Song Influence', (config.song_influence != null ? parseFloat(config.song_influence).toFixed(2) : '0.70'), configSource.song_influence || 'default');
  songInfoEl.innerHTML += configItem('Map similar variants', config.map_similar_variants !== false, configSource.map_similar_variants || 'default', true);
  
  // Detection Parameters
  const detectionEl = document.getElementById('configDetectionParams');
  detectionEl.innerHTML = `
    ${configItem('Instrument', config.instrument, configSource.instrument)}
    ${configItem('Overlap Ratio', config.overlap, configSource.overlap)}
  `;

  // Thresholds
  const thresholdsEl = document.getElementById('configThresholds');
  thresholdsEl.innerHTML = `
    ${configItem('Silence Threshold', config.silence_threshold, configSource.silence_threshold)}
    ${configItem('Confidence Threshold', config.confidence_threshold, configSource.confidence_threshold)}
    ${configItem('Chord Window', config.chord_window + 's', configSource.chord_window)}
    ${configItem('Window Confidence', config.chord_window_confidence, configSource.chord_window_confidence)}
  `;
  
  // Advanced
  const advancedEl = document.getElementById('configAdvanced');
  advancedEl.innerHTML = `
    ${configItem('Show Frequencies', config.show_frequencies, configSource.show_frequencies, true)}
    ${configItem('Show Chroma', config.show_chroma, configSource.show_chroma, true)}
    ${configItem('Show FFT', config.show_fft, configSource.show_fft, true)}
    ${configItem('Raw Frequencies', config.raw_frequencies, configSource.raw_frequencies, true)}
    ${configItem('Frequencies Only', config.frequencies_only, configSource.frequencies_only, true)}
    ${configItem('Notes Only', config.notes_only, configSource.notes_only, true)}
    ${configItem('Debug Mode', config.debug, configSource.debug, true)}
    ${configItem('Log Mode', config.log, configSource.log, true)}
    ${configItem('Log Interval', config.log_interval + 's', configSource.log_interval)}
  `;
  
  panel.classList.remove('hidden');
}

function configItem(label, value, source, isBoolean = false) {
  let valueClass = 'config-item-value';
  let displayValue = value;
  
  if (isBoolean) {
    valueClass += value ? ' boolean-true' : ' boolean-false';
    displayValue = value ? '✓ Enabled' : '✗ Disabled';
  } else if (source === 'url') {
    valueClass += ' from-url';
  }
  
  const sourceLabel = source === 'url' ? '<span class="config-source">(from URL)</span>' : '';
  
  return `
    <div class="config-item">
      <span class="config-item-label">${label}:</span>
      <span class="${valueClass}">${displayValue}${sourceLabel}</span>
    </div>
  `;
}

// --- Skill Level ---
function applySkillLevel(level) {
  const preset = SKILL_PRESETS[level];
  if (!preset) return;

  currentSkillLevel = level;

  // Apply preset values to config
  Object.assign(config, preset);

  // Update UI sliders to match preset
  const sliderMap = {
    confidence_threshold: { slider: 'confidenceThreshold', display: 'confidenceValue' },
    chord_window: { slider: 'chordWindow', display: 'chordWindowValue' },
    chord_window_confidence: { slider: 'chordWindowConfidence', display: 'chordWindowConfidenceValue' },
    song_influence: { slider: 'songInfluence', display: 'songInfluenceValue' },
  };

  for (const [key, ids] of Object.entries(sliderMap)) {
    const el = document.getElementById(ids.slider);
    const valEl = document.getElementById(ids.display);
    if (el) el.value = preset[key];
    if (valEl) valEl.textContent = typeof preset[key] === 'number' && key !== 'chord_window' ? preset[key].toFixed(2) : preset[key];
  }

  // Sync live song influence slider
  const liveSlider = document.getElementById('songInfluenceLive');
  const liveVal = document.getElementById('songInfluenceLiveValue');
  if (liveSlider) liveSlider.value = preset.song_influence;
  if (liveVal) liveVal.textContent = preset.song_influence.toFixed(2);

  // Update active button
  document.querySelectorAll('.btn-skill').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.skill === level);
  });

  // Send config update via WebSocket
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'config_update', config: preset }));
  }

  logWithTimestamp(`🎯 Skill level set to: ${level}`);
}

// Skill level button handlers
document.querySelectorAll('.btn-skill').forEach(btn => {
  btn.addEventListener('click', () => {
    applySkillLevel(btn.dataset.skill);
  });
});

// Deselect skill buttons when user manually adjusts a skill-related slider
['confidenceThreshold', 'chordWindow', 'chordWindowConfidence'].forEach(id => {
  const el = document.getElementById(id);
  if (el) {
    el.addEventListener('input', () => {
      currentSkillLevel = null;
      document.querySelectorAll('.btn-skill').forEach(btn => btn.classList.remove('active'));
    });
  }
});

// Event listeners
document.getElementById('startBtn').addEventListener('click', startRecording);
document.getElementById('stopBtn').addEventListener('click', stopRecording);

// Config panel toggle
document.getElementById('configBtn').addEventListener('click', displayConfigPanel);
document.getElementById('closeConfigBtn').addEventListener('click', () => {
  document.getElementById('configPanel').classList.add('hidden');
});

// Initialize on page load
window.addEventListener('load', () => {
  initializeSettings();
  loadSongs();
  // Disable start button until WebSocket connects
  document.getElementById('startBtn').disabled = true;
  connectWebSocket();
});

