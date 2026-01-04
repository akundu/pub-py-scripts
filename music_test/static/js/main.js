// Main application logic
let ws = null;
let audioContext = null;
let mediaStream = null;
let isRecording = false;
let config = {};

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

// Get config from URL parameters or use defaults
function getConfigFromURL() {
  const params = new URLSearchParams(window.location.search);
  const defaultConfig = {
    instrument: 'guitar',
    sensitivity: 1.0,
    confidence_threshold: 0.6,
    silence_threshold: 0.005,  // Match CLI default
    amplitude_threshold: 0.005,  // Match CLI default
    overlap: 0.75,
    progression: true,
    multi_pitch: true,
    show_frequencies: false,
    show_chroma: false,
    show_fft: false,
    raw_frequencies: false,
    frequencies_only: false,
    notes_only: false,
    single_pitch: false,
    debug: false,
    log: false,
    log_interval: 0.5
  };

  const config = { ...defaultConfig };

  // Override with URL parameters
  for (const [key, value] of params.entries()) {
    if (key in defaultConfig) {
      if (typeof defaultConfig[key] === 'boolean') {
        config[key] = value.toLowerCase() === 'true';
      } else if (typeof defaultConfig[key] === 'number') {
        config[key] = parseFloat(value);
      } else {
        config[key] = value;
      }
    }
  }

  // Ensure amplitude_threshold matches silence_threshold if not explicitly set
  // This matches CLI behavior where both default to the same value (0.005)
  if (!params.has('amplitude_threshold')) {
    config.amplitude_threshold = config.silence_threshold;
  }

  return config;
}

// Initialize settings from URL or defaults
function initializeSettings() {
  config = getConfigFromURL();

  // Update UI with config values
  document.getElementById('instrument').value = config.instrument || 'guitar';
  document.getElementById('sensitivity').value = config.sensitivity || 1.0;
  document.getElementById('sensitivityValue').textContent = config.sensitivity || 1.0;
  document.getElementById('confidenceThreshold').value = config.confidence_threshold || 0.6;
  document.getElementById('confidenceValue').textContent = config.confidence_threshold || 0.6;
  document.getElementById('silenceThreshold').value = config.silence_threshold || 0.005;
  document.getElementById('silenceValue').textContent = config.silence_threshold || 0.005;
  document.getElementById('overlap').value = config.overlap || 0.75;
  document.getElementById('overlapValue').textContent = config.overlap || 0.75;
  document.getElementById('progression').checked = config.progression !== false;
  document.getElementById('multiPitch').checked = config.multi_pitch !== false;
  document.getElementById('showFrequencies').checked = config.show_frequencies === true;
  document.getElementById('showChroma').checked = config.show_chroma === true;
  document.getElementById('debug').checked = config.debug === true;
  document.getElementById('log').checked = config.log === true;
}

// Update status display
function updateStatus(connected, text) {
  const indicator = document.getElementById('statusIndicator');
  const statusText = document.getElementById('statusText');

  indicator.className = 'status-indicator ' + (connected ? 'connected' : 'disconnected');
  statusText.textContent = text;
}

// Get base path for WebSocket (handles Envoy prefix)
function getBasePath() {
  const pathname = window.location.pathname;
  // If we're under /chord_detector/ or exactly /chord_detector, use that as base
  if (pathname.startsWith('/chord_detector/') || pathname === '/chord_detector') {
    return '/chord_detector';
  }
  // Otherwise use root
  return '';
}

// Connect WebSocket
async function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const basePath = getBasePath();
  const wsUrl = `${protocol}//${window.location.host}${basePath}/ws`;

  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    updateStatus(true, 'Connected');

    // Log configuration parameters only in debug mode
    if (config.debug) {
      logWithTimestamp('📊 Configuration Parameters:');
      logWithTimestamp('  Instrument:', config.instrument);
      logWithTimestamp('  Sensitivity:', config.sensitivity);
      logWithTimestamp('  Confidence Threshold:', config.confidence_threshold);
      logWithTimestamp('  Silence Threshold:', config.silence_threshold);
      logWithTimestamp('  Amplitude Threshold:', config.amplitude_threshold);
      logWithTimestamp('  Overlap:', config.overlap);
      logWithTimestamp('  Progression:', config.progression);
      logWithTimestamp('  Multi-pitch:', config.multi_pitch);
      logWithTimestamp('  Single-pitch:', config.single_pitch);
      logWithTimestamp('  Show Frequencies:', config.show_frequencies);
      logWithTimestamp('  Show Chroma:', config.show_chroma);
      logWithTimestamp('  Show FFT:', config.show_fft);
      logWithTimestamp('  Raw Frequencies:', config.raw_frequencies);
      logWithTimestamp('  Frequencies Only:', config.frequencies_only);
      logWithTimestamp('  Notes Only:', config.notes_only);
      logWithTimestamp('  Debug:', config.debug);
      logWithTimestamp('  Log:', config.log);
      logWithTimestamp('  Log Interval:', config.log_interval);
      logWithTimestamp('📊 Full Config Object:', JSON.stringify(config, null, 2));
    }

    // Send initial configuration
    ws.send(JSON.stringify(config));
    // Enable start button when connected
    document.getElementById('startBtn').disabled = false;
  };

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    handleWebSocketMessage(data);
  };

  ws.onerror = (error) => {
    errorWithTimestamp('WebSocket error:', error);
    updateStatus(false, 'Connection Error');
  };

  ws.onclose = () => {
    updateStatus(false, 'Disconnected');
    if (isRecording) {
      stopRecording();
    }
  };
}

// Handle WebSocket messages
function handleWebSocketMessage(data) {
  if (data.type === 'connected') {
    debugLog('Connected:', data.message);
    return;
  }

  if (data.type === 'error') {
    errorWithTimestamp('Server error:', data.message);
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
  const chordNameEl = document.getElementById('chordName');
  const chordConfidenceEl = document.getElementById('chordConfidence');
  const chordStabilityEl = document.getElementById('chordStability');

  if (data && data.chord) {
    chordNameEl.textContent = data.chord;
    chordNameEl.className = 'chord-name detected';

    if (data.confidence !== undefined) {
      chordConfidenceEl.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
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
          // If sending fails, stop recording
          stopRecording();
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

// Handle settings form submission
settingsForm.addEventListener('submit', (e) => {
  e.preventDefault();

  const formData = new FormData(settingsForm);
  config = {};

  for (const [key, value] of formData.entries()) {
    if (key === 'progression' || key === 'multi_pitch' || key === 'show_frequencies' ||
      key === 'show_chroma' || key === 'debug' || key === 'log') {
      config[key] = document.getElementById(key === 'multi_pitch' ? 'multiPitch' :
        key === 'show_frequencies' ? 'showFrequencies' :
          key === 'show_chroma' ? 'showChroma' : key).checked;
    } else {
      config[key] = value;
    }
  }

  // Convert string numbers to actual numbers
  config.sensitivity = parseFloat(config.sensitivity);
  config.confidence_threshold = parseFloat(config.confidence_threshold);
  config.silence_threshold = parseFloat(config.silence_threshold);
  config.overlap = parseFloat(config.overlap);

  // Set amplitude_threshold to match silence_threshold (CLI behavior)
  // If not explicitly set, use the same value as silence_threshold
  if (!config.amplitude_threshold) {
    config.amplitude_threshold = config.silence_threshold;
  } else {
    config.amplitude_threshold = parseFloat(config.amplitude_threshold);
  }

  // Ensure all other defaults are set if missing
  const defaultConfig = {
    instrument: 'guitar',
    sensitivity: 1.0,
    confidence_threshold: 0.6,
    silence_threshold: 0.005,
    amplitude_threshold: 0.005,
    overlap: 0.75,
    progression: true,
    multi_pitch: true,
    show_frequencies: false,
    show_chroma: false,
    show_fft: false,
    raw_frequencies: false,
    frequencies_only: false,
    notes_only: false,
    single_pitch: false,
    debug: false,
    log: false,
    log_interval: 0.5
  };

  // Merge with defaults to ensure all values are set
  config = { ...defaultConfig, ...config };

  // Reconnect with new config
  if (ws) {
    ws.close();
  }
  connectWebSocket();

  modal.style.display = 'none';
});

// Reset settings
resetSettingsBtn.addEventListener('click', () => {
  initializeSettings();
});

// Event listeners
document.getElementById('startBtn').addEventListener('click', startRecording);
document.getElementById('stopBtn').addEventListener('click', stopRecording);

// Initialize on page load
window.addEventListener('load', () => {
  initializeSettings();
  // Disable start button until WebSocket connects
  document.getElementById('startBtn').disabled = true;
  connectWebSocket();
});

