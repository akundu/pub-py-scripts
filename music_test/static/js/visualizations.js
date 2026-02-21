// Visualization functions

// Frequency spectrum visualization
const frequencyCanvas = document.getElementById('frequencyCanvas');
const frequencyCtx = frequencyCanvas.getContext('2d');

function updateFrequencyVisualization(notes) {
    if (!notes || notes.length === 0) {
        clearCanvas(frequencyCtx, frequencyCanvas);
        return;
    }
    
    const width = frequencyCanvas.width;
    const height = frequencyCanvas.height;
    
    // Clear canvas
    frequencyCtx.fillStyle = '#1a1a1a';
    frequencyCtx.fillRect(0, 0, width, height);
    
    // Draw grid
    frequencyCtx.strokeStyle = '#333';
    frequencyCtx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
        const y = (height / 10) * i;
        frequencyCtx.beginPath();
        frequencyCtx.moveTo(0, y);
        frequencyCtx.lineTo(width, y);
        frequencyCtx.stroke();
    }
    
    // Draw frequency bars
    const barWidth = width / notes.length;
    notes.forEach((note, index) => {
        const [noteName, freq] = Array.isArray(note) ? note : [note, null];
        if (!freq) return;
        
        // Normalize frequency (assuming range 80-2000 Hz)
        const normalizedFreq = (freq - 80) / (2000 - 80);
        const barHeight = normalizedFreq * height * 0.8;
        
        // Color based on note
        const hue = (index * 360 / notes.length) % 360;
        frequencyCtx.fillStyle = `hsl(${hue}, 70%, 60%)`;
        
        frequencyCtx.fillRect(index * barWidth, height - barHeight, barWidth - 2, barHeight);
        
        // Label
        frequencyCtx.fillStyle = '#fff';
        frequencyCtx.font = '12px Arial';
        frequencyCtx.textAlign = 'center';
        frequencyCtx.fillText(noteName, index * barWidth + barWidth / 2, height - 5);
    });
}

// Chroma vector visualization
const chromaCanvas = document.getElementById('chromaCanvas');
const chromaCtx = chromaCanvas.getContext('2d');

const CHROMA_NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function updateChromaVisualization(chromaVector) {
    if (!chromaVector || chromaVector.length !== 12) {
        clearCanvas(chromaCtx, chromaCanvas);
        return;
    }
    
    const width = chromaCanvas.width;
    const height = chromaCanvas.height;
    const barWidth = width / 12;
    
    // Clear canvas
    chromaCtx.fillStyle = '#1a1a1a';
    chromaCtx.fillRect(0, 0, width, height);
    
    // Find max value for normalization
    const maxValue = Math.max(...chromaVector);
    
    // Draw chroma bars
    chromaVector.forEach((value, index) => {
        const normalizedValue = maxValue > 0 ? value / maxValue : 0;
        const barHeight = normalizedValue * height * 0.8;
        
        // Color intensity based on value
        const intensity = Math.floor(normalizedValue * 255);
        chromaCtx.fillStyle = `rgb(${intensity}, ${intensity}, ${200})`;
        
        chromaCtx.fillRect(index * barWidth, height - barHeight, barWidth - 2, barHeight);
        
        // Label
        chromaCtx.fillStyle = '#fff';
        chromaCtx.font = '12px Arial';
        chromaCtx.textAlign = 'center';
        chromaCtx.fillText(CHROMA_NOTES[index], index * barWidth + barWidth / 2, height - 5);
        
        // Value label
        chromaCtx.fillStyle = '#888';
        chromaCtx.font = '10px Arial';
        chromaCtx.fillText(value.toFixed(2), index * barWidth + barWidth / 2, height - barHeight - 5);
    });
}

// Waveform visualization
const waveformCanvas = document.getElementById('waveformCanvas');
const waveformCtx = waveformCanvas ? waveformCanvas.getContext('2d') : null;

function updateWaveformVisualization(audioData) {
    if (!waveformCtx || !waveformCanvas) return;
    if (!audioData || audioData.length === 0) {
        clearCanvas(waveformCtx, waveformCanvas);
        return;
    }
    
    const width = waveformCanvas.width;
    const height = waveformCanvas.height;
    const centerY = height / 2;
    // Amplify so quiet audio is visible (raw mic is often -0.01 to 0.01)
    const gain = 3;
    const maxAmplitude = height / 2 - 4;
    
    // Clear canvas
    waveformCtx.fillStyle = '#1a1a1a';
    waveformCtx.fillRect(0, 0, width, height);
    
    // Draw center line (optional, helps see silence)
    waveformCtx.strokeStyle = '#333';
    waveformCtx.lineWidth = 1;
    waveformCtx.beginPath();
    waveformCtx.moveTo(0, centerY);
    waveformCtx.lineTo(width, centerY);
    waveformCtx.stroke();
    
    // Draw waveform centered: y = centerY + sample * gain * maxAmplitude, clamped
    waveformCtx.strokeStyle = '#4CAF50';
    waveformCtx.lineWidth = 2;
    waveformCtx.beginPath();
    
    const sliceWidth = width / Math.max(1, audioData.length - 1);
    
    for (let i = 0; i < audioData.length; i++) {
        const sample = Math.max(-1, Math.min(1, audioData[i] * gain));
        const y = centerY + sample * maxAmplitude;
        const x = i * sliceWidth;
        
        if (i === 0) {
            waveformCtx.moveTo(x, y);
        } else {
            waveformCtx.lineTo(x, y);
        }
    }
    
    waveformCtx.stroke();
}

// Expose for main.js (in case of script load order / scope)
if (typeof window !== 'undefined') {
    window.updateWaveformVisualization = updateWaveformVisualization;
}

function clearCanvas(ctx, canvas) {
    ctx.fillStyle = '#1a1a1a';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw placeholder text
    ctx.fillStyle = '#666';
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('No data', canvas.width / 2, canvas.height / 2);
}

// Initialize canvases (guard against missing elements)
[frequencyCanvas, chromaCanvas, waveformCanvas].filter(Boolean).forEach(canvas => {
    const ctx = canvas.getContext('2d');
    if (ctx) clearCanvas(ctx, canvas);
});

