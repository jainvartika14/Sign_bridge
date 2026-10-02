/**
 * SignBridge Frontend Application v3.0
 *
 * WEBCAM PIPELINE:
 *  Browser webcam → getUserMedia() → MediaPipe Hands → 42 landmarks/frame
 *  → 2.0-second gesture capture & rolling history buffer (~60 frames)
 *  → Uniform 32-frame downsampling: np.linspace(0, total - 1, 32)
 *  → Preserves natural lead-in and lead-out resting zeros (matches training data)
 *  → POST /api/predict/sequence [32, 42, 3]
 *  → FastAPI → normalize_sequence() → GRU → 2,731-class prediction → UI
 *
 * MODES:
 *  1. Interactive "Sign Now" (Spacebar / Click): 2.0-second timed gesture capture with progress bar
 *  2. Continuous Live Stream: Resamples last 2.0s rolling history every 600ms with temporal debouncing
 *  3. Diagnostic HUD: Toggle with 'D' key or '🛠️ Debug' button
 */

document.addEventListener('DOMContentLoaded', () => {

  // --------------------------------------------------------------------------
  // DOM Elements
  // --------------------------------------------------------------------------
  const navConversation        = document.getElementById('nav-conversation');
  const navRecognize           = document.getElementById('nav-recognize');
  const navLearn               = document.getElementById('nav-learn');
  const btnTeachMeTop          = document.getElementById('btn-teach-me-top');
  const btnTeachMeChip         = document.getElementById('btn-teach-me-chip');
  const viewConversation       = document.getElementById('view-conversation');
  const viewRecognize          = document.getElementById('view-recognize');
  const viewLearn              = document.getElementById('view-learn');

  const tabCamera               = document.getElementById('tab-camera');
  const tabUpload               = document.getElementById('tab-upload');
  const panelCamera             = document.getElementById('panel-camera');
  const panelUpload             = document.getElementById('panel-upload');

  const webcamFeed              = document.getElementById('webcam-feed');
  const landmarkCanvas          = document.getElementById('landmark-canvas');
  const canvasCtx               = landmarkCanvas ? landmarkCanvas.getContext('2d') : null;

  const convWebcamFeed          = document.getElementById('conv-webcam-feed');
  const convLandmarkCanvas      = document.getElementById('conv-landmark-canvas');
  const convCanvasCtx           = convLandmarkCanvas ? convLandmarkCanvas.getContext('2d') : null;
  const convCameraOverlay       = document.getElementById('conv-camera-overlay');

  const btnStartCamera          = document.getElementById('btn-start-camera');
  const btnToggleCamera         = document.getElementById('btn-toggle-camera');
  const btnCaptureSign          = document.getElementById('btn-capture-sign');
  const btnToggleDebug          = document.getElementById('btn-toggle-debug');
  const cameraPromptOverlay     = document.getElementById('camera-prompt-overlay');
  const cameraStatusBadge       = document.getElementById('camera-status-badge');
  const cameraStatusText        = document.getElementById('camera-status-text');

  const captureProgressContainer= document.getElementById('capture-progress-container');
  const captureProgressFill     = document.getElementById('capture-progress-fill');
  const captureStatusText       = document.getElementById('capture-status-text');
  const captureTimeText         = document.getElementById('capture-time-text');

  const dropzone                = document.getElementById('dropzone');
  const videoFileInput          = document.getElementById('video-file-input');
  const uploadPreviewContainer  = document.getElementById('upload-preview-container');
  const uploadVideoPlayer       = document.getElementById('upload-video-player');
  const uploadFileName          = document.getElementById('upload-file-name');
  const btnReupload             = document.getElementById('btn-reupload');

  const primaryGloss            = document.getElementById('primary-gloss');
  const primaryConfidence       = document.getElementById('primary-confidence');
  const confidenceAssessment    = document.getElementById('confidence-assessment');
  const confidenceBadge         = document.getElementById('confidence-badge');
  const top5List                = document.getElementById('top5-list');

  const learnSearchInput        = document.getElementById('learn-search-input');
  const vocabList               = document.getElementById('vocab-list');
  const vocabCountBadge         = document.getElementById('vocab-count-badge');
  const demoGlossTitle          = document.getElementById('demo-gloss-title');
  const demoTierBadge           = document.getElementById('demo-tier-badge');
  const demoVideoPlayer         = document.getElementById('demo-video-player');
  const demoVideoPlaceholder    = document.getElementById('demo-video-placeholder');
  const demoPlaceholderText     = document.getElementById('demo-placeholder-text');
  const btnTrySign              = document.getElementById('btn-try-sign');

  const practiceTargetBanner    = document.getElementById('practice-target-banner');
  const practiceTargetGloss     = document.getElementById('practice-target-gloss');
  const btnCancelPractice       = document.getElementById('btn-cancel-practice');

  // Sentence Builder DOM Elements
  const sbWordCount             = document.getElementById('sb-word-count');
  const sbAutoAddToggle         = document.getElementById('sb-auto-add-toggle');
  const sbEmptyState            = document.getElementById('sb-empty-state');
  const sbWordsContainer        = document.getElementById('sb-words-container');
  const sbNlpBox                = document.getElementById('sb-nlp-box');
  const sbNlpText               = document.getElementById('sb-nlp-text');
  const sbNlpStatus             = document.getElementById('sb-nlp-status');
  const sbBtnSpeak              = document.getElementById('sb-btn-speak');
  const sbBtnCopy               = document.getElementById('sb-btn-copy');
  const sbCopyText              = document.getElementById('sb-copy-text');
  const sbBtnAddCurrent         = document.getElementById('sb-btn-add-current');
  const sbBtnBackspace          = document.getElementById('sb-btn-backspace');
  const sbBtnClear              = document.getElementById('sb-btn-clear');
  const sbTtsVoiceSelect        = document.getElementById('sb-tts-voice-select');
  const sbTtsRate               = document.getElementById('sb-tts-rate');
  const sbTtsRateVal            = document.getElementById('sb-tts-rate-val');
  const sbQuickChips            = document.querySelectorAll('.sb-quick-chip');

  // Hearing Partner Speech-to-Text (STT) Elements
  const hpCard                  = document.getElementById('hearing-partner-card');
  const hpBtnMic                = document.getElementById('hp-btn-mic');
  const hpMicBadge              = document.getElementById('hp-mic-badge');
  const hpMicLabel              = document.getElementById('hp-mic-label');
  const hpSpeechBox             = document.getElementById('hp-speech-box');
  const hpEmptyState            = document.getElementById('hp-empty-state');
  const hpTranscriptionText     = document.getElementById('hp-transcription-text');
  const hpMatchedSignsRow       = document.getElementById('hp-matched-signs-row');
  const hpMatchedChips          = document.getElementById('hp-matched-chips');

  // --------------------------------------------------------------------------
  // Pipeline Parameters & State
  // --------------------------------------------------------------------------
  const SEQUENCE_LENGTH         = 32;       // Target temporal length for GRU model
  const HISTORY_WINDOW_FRAMES   = 90;       // ~3.0 seconds at 30 FPS
  const GESTURE_CAPTURE_MS      = 2000;     // 2.0s capture window for "Sign Now"
  const CONTINUOUS_INTERVAL_MS  = 180;      // Continuous streaming inference check (~5 Hz)
  const SLIDING_WINDOW_FRAMES   = 48;       // Overlapping sliding window size (~1.6s)
  const STRIDE_FRAMES           = 6;        // Sliding window stride (~200ms / 6 frames)
  const MIN_ACTIVE_FRAMES       = 8;        // Min hand-active frames in window to trigger inference
  const IDLE_GATING_FRAMES      = 12;       // Frames without hands before marking state as IDLE

  let isCameraRunning           = false;
  let mediaStream               = null;
  let handsDetector             = null;
  let animationFrameId          = null;

  // Buffering
  let rollingHistory            = [];       // Rolling buffer of last ~90 raw frames (including zeros)
  let isRecordingGesture        = false;    // True during 2-second "Sign Now" capture
  let gestureRecordedFrames     = [];       // Accumulated frames during interactive capture
  let gestureStartTime          = 0;
  let gestureProgressInterval   = null;

  let lastContinuousInference   = 0;
  let framesSinceLastInference  = 0;
  let idleNoHandsCount          = 0;
  let isInferencing             = false;

  // Smoothing & Stability — 4-Stage Prediction Stabilizer
  const stabilizer              = new window.SignBridgeStabilizer();
  window.SignBridgeStabilizerInstance = stabilizer;
  let voteBuffer                = [];       // Array of recent predictions
  const VOTE_BUFFER_SIZE        = 6;        // Number of recent sliding-window predictions to consider
  let lastStableGloss           = '—';
  let consecutiveMatches        = 0;
  const CONSECUTIVE_REQUIRED    = 2;

  // Target sign for practice mode
  let targetPracticeGloss       = null;
  let currentLearnGloss         = 'APPLE';
  let searchDebounceTimeout     = null;

  // Diagnostic State
  let debugVisible              = false;
  const DS = {
    cameraReady:        false,
    mediaPipeReady:     false,
    mediaPipeError:     '',
    handsDetected:      0,
    historyLength:      0,
    activeFramesInHist: 0,
    apiStatus:          'IDLE',
    lastPrediction:     '—',
    apiLatencyMs:       0,
    totalFrames:        0,
    samplingMode:       'IDLE',
  };

  // --------------------------------------------------------------------------
  // Diagnostic HUD Overlay
  // --------------------------------------------------------------------------
  (function initDebugHUD() {
    const panel = document.getElementById('panel-camera');
    if (!panel) return;

    const wrap = document.createElement('div');
    wrap.id = 'sb-debug-hud';
    wrap.style.cssText = `
      display: none;
      margin-top: 10px;
      background: #0f1117;
      border: 1px solid #2d3748;
      border-radius: 8px;
      padding: 12px 16px;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 11px;
      color: #cbd5e1;
      line-height: 1.7;
    `;

    wrap.innerHTML = `
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; border-bottom:1px solid #1e293b; padding-bottom:6px;">
        <span style="font-weight:700; color:#818cf8; font-size:12px;">🛠️ PIPELINE TELEMETRY</span>
        <span style="color:#64748b; font-size:10px;">Toggle with 'D' key</span>
      </div>
      <div id="sb-debug-metrics" style="display:grid; grid-template-columns: 1fr 1fr; gap:6px 16px;"></div>
    `;

    panel.appendChild(wrap);
  })();

  function toggleDebugHUD() {
    debugVisible = !debugVisible;
    const hud = document.getElementById('sb-debug-hud');
    if (hud) hud.style.display = debugVisible ? 'block' : 'none';
  }

  function updateDebugHUD() {
    if (!debugVisible) return;
    const metrics = document.getElementById('sb-debug-metrics');
    if (!metrics) return;

    const row = (label, val, col = '#e2e8f0') =>
      `<div style="display:flex; justify-content:space-between;">
        <span style="color:#64748b;">${label}:</span>
        <span style="color:${col}; font-weight:600;">${val}</span>
      </div>`;

    metrics.innerHTML = [
      row('Camera', DS.cameraReady ? 'READY' : 'OFF', DS.cameraReady ? '#4ade80' : '#f87171'),
      row('MediaPipe', DS.mediaPipeReady ? 'READY' : (DS.mediaPipeError || 'INIT'), DS.mediaPipeReady ? '#4ade80' : '#f87171'),
      row('Hands Detected', `${DS.handsDetected}`, DS.handsDetected > 0 ? '#38bdf8' : '#64748b'),
      row('History Window', `${DS.historyLength}/${HISTORY_WINDOW_FRAMES}`, '#fbbf24'),
      row('Active Hand Frames', `${DS.activeFramesInHist}`, '#a78bfa'),
      row('Capture Mode', DS.samplingMode, '#c084fc'),
      row('API Status', DS.apiStatus, DS.apiStatus === 'SUCCESS' ? '#4ade80' : '#f87171'),
      row('Latency', DS.apiLatencyMs ? `${DS.apiLatencyMs}ms` : '—', '#94a3b8'),
      row('Last Prediction', DS.lastPrediction, '#f1f5f9'),
      row('Frames Processed', `${DS.totalFrames}`, '#64748b'),
    ].join('');
  }

  // Keyboard shortcut 'D' to toggle debug panel
  window.addEventListener('keydown', (e) => {
    if (e.key === 'd' || e.key === 'D') {
      if (document.activeElement?.tagName !== 'INPUT') {
        toggleDebugHUD();
      }
    }
    // Spacebar to trigger "Sign Now" if camera is running
    if (e.code === 'Space' && isCameraRunning && !isRecordingGesture) {
      if (document.activeElement?.tagName !== 'INPUT') {
        e.preventDefault();
        startGestureCapture();
      }
    }
  });

  if (btnToggleDebug) {
    btnToggleDebug.addEventListener('click', toggleDebugHUD);
  }

  // --------------------------------------------------------------------------
  // Navigation
  // --------------------------------------------------------------------------
  if (navConversation) navConversation.addEventListener('click', () => switchMainView('conversation'));
  if (navRecognize)    navRecognize.addEventListener('click',    () => switchMainView('recognize'));
  if (navLearn)        navLearn.addEventListener('click',        () => switchMainView('learn'));
  if (btnTeachMeTop)   btnTeachMeTop.addEventListener('click',   triggerTeachMeSomething);
  if (btnTeachMeChip)  btnTeachMeChip.addEventListener('click',  triggerTeachMeSomething);

  function switchMainView(viewName) {
    const allNavs = [navConversation, navRecognize, navLearn];
    const allViews = [viewConversation, viewRecognize, viewLearn];

    allNavs.forEach(btn => {
      if (btn) {
        btn.classList.remove('active');
        btn.removeAttribute('aria-current');
      }
    });
    allViews.forEach(v => {
      if (v) v.classList.remove('active');
    });

    if (viewName === 'conversation') {
      if (navConversation) {
        navConversation.classList.add('active');
        navConversation.setAttribute('aria-current', 'page');
      }
      if (viewConversation) viewConversation.classList.add('active');
      syncConversationCameraViewport();
    } else if (viewName === 'recognize') {
      if (navRecognize) {
        navRecognize.classList.add('active');
        navRecognize.setAttribute('aria-current', 'page');
      }
      if (viewRecognize) viewRecognize.classList.add('active');
    } else {
      if (navLearn) {
        navLearn.classList.add('active');
        navLearn.setAttribute('aria-current', 'page');
      }
      if (viewLearn) viewLearn.classList.add('active');
      if (isCameraRunning && !targetPracticeGloss) stopCamera();
      if (vocabList && vocabList.children.length === 0) {
        fetchVocabulary();
        loadDemonstration('APPLE');
      }
    }
  }

  if (tabCamera) tabCamera.addEventListener('click', () => switchRecognizeTab('camera'));
  if (tabUpload) tabUpload.addEventListener('click', () => switchRecognizeTab('upload'));

  function switchRecognizeTab(mode) {
    if (mode === 'camera') {
      if (tabCamera)  { tabCamera.classList.add('active');    tabCamera.setAttribute('aria-selected', 'true'); }
      if (tabUpload)  { tabUpload.classList.remove('active'); tabUpload.setAttribute('aria-selected', 'false'); }
      if (panelCamera) panelCamera.classList.add('active');
      if (panelUpload) panelUpload.classList.remove('active');
    } else {
      if (tabUpload)  { tabUpload.classList.add('active');    tabUpload.setAttribute('aria-selected', 'true'); }
      if (tabCamera)  { tabCamera.classList.remove('active'); tabCamera.setAttribute('aria-selected', 'false'); }
      if (panelUpload) panelUpload.classList.add('active');
      if (panelCamera) panelCamera.classList.remove('active');
      if (isCameraRunning) stopCamera();
    }
  }

  // --------------------------------------------------------------------------
  // Camera Management
  // --------------------------------------------------------------------------
  if (btnStartCamera)  btnStartCamera.addEventListener('click', startCamera);
  if (btnToggleCamera) btnToggleCamera.addEventListener('click', () => isCameraRunning ? stopCamera() : startCamera());
  if (btnCaptureSign)  btnCaptureSign.addEventListener('click', () => {
    if (isCameraRunning && !isRecordingGesture) startGestureCapture();
  });

  function syncConversationCameraViewport() {
    if (!mediaStream || !isCameraRunning) return;
    if (convWebcamFeed && convWebcamFeed.srcObject !== mediaStream) {
      convWebcamFeed.srcObject = mediaStream;
      convWebcamFeed.play().catch(() => {});
    }
    if (convLandmarkCanvas && landmarkCanvas) {
      convLandmarkCanvas.width  = landmarkCanvas.width  || 640;
      convLandmarkCanvas.height = landmarkCanvas.height || 480;
    }
    if (convCameraOverlay) {
      convCameraOverlay.classList.add('hidden');
    }
    const chipCam = document.getElementById('chip-camera-live');
    if (chipCam) {
      chipCam.textContent = 'Camera Live';
      chipCam.classList.add('active');
    }
  }

  async function startCamera() {
    try {
      updateCameraStatus('Connecting camera…', 'neutral');
      DS.cameraReady = false;

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("Camera API not available. Please access the site via http://localhost:8000 or HTTPS.");
      }

      mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
        audio: false,
      });

      webcamFeed.srcObject = mediaStream;
      await webcamFeed.play();

      landmarkCanvas.width  = webcamFeed.videoWidth  || 640;
      landmarkCanvas.height = webcamFeed.videoHeight || 480;

      DS.cameraReady = true;
      console.log('[SB] Camera connected:', landmarkCanvas.width, 'x', landmarkCanvas.height);

      if (!handsDetector) initMediaPipeHands();

      isCameraRunning        = true;
      rollingHistory         = [];
      gestureRecordedFrames  = [];
      isRecordingGesture     = false;
      DS.totalFrames         = 0;

      cameraPromptOverlay.classList.add('hidden');
      btnToggleCamera.disabled = false;
      btnCaptureSign.disabled  = false;
      btnToggleCamera.querySelector('span').textContent = 'Stop Camera';
      updateCameraStatus('Ready to sign — click Sign Now or hold Space', 'active');

      syncConversationCameraViewport();

      processCameraLoop();

    } catch (err) {
      DS.cameraReady = false;
      console.error('[SB] Camera error:', err);
      updateCameraStatus('Camera error: ' + err.message, 'error');
      alert('Camera access failed: ' + err.message);
    }
  }

  function stopCamera() {
    isCameraRunning    = false;
    isRecordingGesture = false;
    if (gestureProgressInterval) { clearInterval(gestureProgressInterval); gestureProgressInterval = null; }
    if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
    if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
    webcamFeed.srcObject = null;
    if (canvasCtx) canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);

    if (convWebcamFeed) convWebcamFeed.srcObject = null;
    if (convCanvasCtx && convLandmarkCanvas) convCanvasCtx.clearRect(0, 0, convLandmarkCanvas.width, convLandmarkCanvas.height);
    if (convCameraOverlay) convCameraOverlay.classList.remove('hidden');
    const chipCam = document.getElementById('chip-camera-live');
    if (chipCam) { chipCam.textContent = 'Camera Off'; chipCam.classList.remove('active'); }

    rollingHistory        = [];
    gestureRecordedFrames = [];
    voteBuffer            = [];
    consecutiveMatches    = 0;
    lastStableGloss       = '—';
    Object.assign(DS, {
      cameraReady: false, handsDetected: 0, historyLength: 0,
      activeFramesInHist: 0, apiStatus: 'IDLE', samplingMode: 'IDLE'
    });
    updateDebugHUD();

    captureProgressContainer.classList.add('hidden');
    btnCaptureSign.classList.remove('recording');
    btnCaptureSign.disabled  = true;
    cameraPromptOverlay.classList.remove('hidden');
    btnToggleCamera.disabled = true;
    btnToggleCamera.querySelector('span').textContent = 'Start Camera';
    updateCameraStatus('Camera off', 'neutral');
    resetPredictionUI();
  }

  function updateCameraStatus(text, state = 'neutral') {
    cameraStatusText.textContent = text;
    state === 'active'
      ? cameraStatusBadge.classList.add('active')
      : cameraStatusBadge.classList.remove('active');
  }

  // --------------------------------------------------------------------------
  // MediaPipe Hands Initialization
  // --------------------------------------------------------------------------
  function initMediaPipeHands() {
    if (typeof Hands === 'undefined') {
      DS.mediaPipeError = 'CDN not loaded';
      DS.mediaPipeReady = false;
      updateDebugHUD();
      console.error('[SB] MediaPipe Hands script not loaded from CDN.');
      return;
    }

    try {
      handsDetector = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
      });
      handsDetector.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
      handsDetector.onResults(onHandResults);
      DS.mediaPipeReady = true;
      DS.mediaPipeError = '';
      updateDebugHUD();
      console.log('[SB] MediaPipe Hands initialized successfully.');
    } catch (e) {
      DS.mediaPipeError = e.message;
      DS.mediaPipeReady = false;
      updateDebugHUD();
      console.error('[SB] MediaPipe init failed:', e);
    }
  }

  async function processCameraLoop() {
    if (!isCameraRunning) return;
    if (webcamFeed.readyState >= 2 && handsDetector) {
      try {
        await handsDetector.send({ image: webcamFeed });
      } catch (e) {
        // Suppress frame drop errors during resize/teardown
      }
    }
    animationFrameId = requestAnimationFrame(processCameraLoop);
  }

  // --------------------------------------------------------------------------
  // MediaPipe Frame Callback
  // --------------------------------------------------------------------------
  /**
   * Processes each camera frame:
   * 1. Extracts landmarks [42, 3] (21 landmarks x 2 hands, sorted left-to-right by wrist X)
   * 2. Preserves zeros when hands are not present (vital for GRU recurrent state)
   * 3. Buffers into gestureRecordedFrames (if interactive capture active)
   * 4. Buffers into rollingHistory (continuous sliding window of ~60 frames)
   * 5. Triggers continuous streaming inference every 650ms if active motion detected
   */
  function onHandResults(results) {
    if (!canvasCtx) return;
    canvasCtx.clearRect(0, 0, landmarkCanvas.width, landmarkCanvas.height);
    if (convCanvasCtx && convLandmarkCanvas) {
      convCanvasCtx.clearRect(0, 0, convLandmarkCanvas.width, convLandmarkCanvas.height);
    }

    DS.totalFrames++;

    // Default zero-initialized frame [42, 3]
    const frameLandmarks = Array.from({ length: 42 }, () => [0.0, 0.0, 0.0]);
    let hasHands = false;

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      hasHands = true;

      // Extract and sort detected hands by wrist X (ascending: leftmost hand first)
      const detectedHands = results.multiHandLandmarks.map((handLms) => ({
        wristX: handLms[0].x,
        coords: handLms.map(lm => [lm.x, lm.y, lm.z]),
        rawHand: handLms,
      }));

      detectedHands.sort((a, b) => a.wristX - b.wristX);

      detectedHands.slice(0, 2).forEach((hand, idx) => {
        const offset = idx * 21;
        hand.coords.forEach((coord, lmIdx) => {
          frameLandmarks[offset + lmIdx] = coord;
        });
        drawHandOverlay(hand.rawHand);
      });

      DS.handsDetected = Math.min(detectedHands.length, 2);
    } else {
      DS.handsDetected = 0;
    }

    // Record hand presence to stabilizer for natural sign boundary detection
    if (stabilizer) {
      stabilizer.recordHandPresence(DS.handsDetected);
    }

    // ── Explicit Idle / No-Hands Gating ──────────────────────────────────────
    if (DS.handsDetected === 0) {
      idleNoHandsCount++;
      if (idleNoHandsCount >= IDLE_GATING_FRAMES && !isRecordingGesture) {
        if (voteBuffer.length > 0) {
          // Gracefully decay vote buffer during rest periods to prevent stale triggers
          voteBuffer.shift();
          consecutiveMatches = 0;
        }
        DS.samplingMode = 'IDLE (No Hands)';
      }
    } else {
      idleNoHandsCount = 0;
    }

    // ── Mode 1: Interactive Timed Gesture Capture ────────────────────────────
    if (isRecordingGesture) {
      gestureRecordedFrames.push(frameLandmarks);
    }

    // ── Mode 2: Continuous Rolling History Buffer (~90 frames / 3.0s) ────────
    // We intentionally push ALL frames (including zeros) to capture natural rest/sign/rest motion
    rollingHistory.push(frameLandmarks);
    if (rollingHistory.length > HISTORY_WINDOW_FRAMES) {
      rollingHistory.shift();
    }

    // Update telemetry counters
    DS.historyLength = rollingHistory.length;
    DS.activeFramesInHist = rollingHistory.filter(f => !isZeroFrame(f)).length;
    updateDebugHUD();

    // ── Autonomous Overlapping Sliding Window Inference ──────────────────────
    framesSinceLastInference++;
    const now = Date.now();
    if (
      !isRecordingGesture &&
      !isInferencing &&
      rollingHistory.length >= 32 &&
      framesSinceLastInference >= STRIDE_FRAMES &&
      now - lastContinuousInference >= CONTINUOUS_INTERVAL_MS
    ) {
      // Extract the sliding window of recent frames (overlapping 48 frames)
      const slidingWindow = rollingHistory.slice(-SLIDING_WINDOW_FRAMES);
      const activeCount = slidingWindow.filter(f => !isZeroFrame(f)).length;

      // Gate inference on hand presence and activity (prevent inferencing on empty air)
      if (activeCount >= MIN_ACTIVE_FRAMES && idleNoHandsCount < IDLE_GATING_FRAMES) {
        lastContinuousInference = now;
        framesSinceLastInference = 0;
        DS.samplingMode = `SLIDING (${slidingWindow.length}f [s=${STRIDE_FRAMES}]→32f)`;
        const sampledSequence = resampleTo32Frames(slidingWindow);
        executeSequenceInference(sampledSequence, false);
      }
    }
  }

  // --------------------------------------------------------------------------
  // Temporal Resampling with Linear Interpolation (Preserves Continuity)
  // --------------------------------------------------------------------------
  /**
   * Resamples an arbitrary sequence of frames (e.g. 40-70 frames from a capture or sliding window)
   * into exactly 32 frames using smooth linear interpolation instead of nearest-neighbor snapping.
   * Preserves hand coordinate continuity and avoids artificial velocity jumps.
   */
  function resampleTo32Frames(frames) {
    if (!frames || frames.length === 0) {
      return Array.from({ length: SEQUENCE_LENGTH }, () => Array.from({ length: 42 }, () => [0, 0, 0]));
    }
    if (frames.length === 1) {
      return Array.from({ length: SEQUENCE_LENGTH }, () => frames[0]);
    }

    const result = [];
    const total = frames.length;
    for (let i = 0; i < SEQUENCE_LENGTH; i++) {
      const t = (i * (total - 1)) / (SEQUENCE_LENGTH - 1);
      const i0 = Math.floor(t);
      const i1 = Math.min(i0 + 1, total - 1);
      const alpha = t - i0;

      if (alpha === 0 || i0 === i1) {
        result.push(frames[i0]);
        continue;
      }

      const f0 = frames[i0];
      const f1 = frames[i1];
      const interpolated = [];

      // Interpolate per-hand to preserve coordinate continuity & handle missing hand transitions
      for (let h = 0; h < 2; h++) {
        const offset = h * 21;
        const h0Active = f0[offset][0] !== 0 || f0[offset][1] !== 0;
        const h1Active = f1[offset][0] !== 0 || f1[offset][1] !== 0;

        if (h0Active && h1Active) {
          // Both frames have hand detected -> smooth linear interpolation
          for (let lm = 0; lm < 21; lm++) {
            const p0 = f0[offset + lm];
            const p1 = f1[offset + lm];
            interpolated.push([
              (1 - alpha) * p0[0] + alpha * p1[0],
              (1 - alpha) * p0[1] + alpha * p1[1],
              (1 - alpha) * p0[2] + alpha * p1[2],
            ]);
          }
        } else if (h0Active && !h1Active) {
          const useH0 = alpha < 0.5;
          for (let lm = 0; lm < 21; lm++) {
            interpolated.push(useH0 ? f0[offset + lm] : [0, 0, 0]);
          }
        } else if (!h0Active && h1Active) {
          const useH1 = alpha >= 0.5;
          for (let lm = 0; lm < 21; lm++) {
            interpolated.push(useH1 ? f1[offset + lm] : [0, 0, 0]);
          }
        } else {
          for (let lm = 0; lm < 21; lm++) {
            interpolated.push([0, 0, 0]);
          }
        }
      }
      result.push(interpolated);
    }
    return result;
  }

  /**
   * Extract the peak signing window from rolling history.
   * Finds the contiguous window of `windowSize` frames with the most hand-active frames.
   * This focuses inference on the most informative segment rather than random sampling.
   */
  function extractPeakWindow(history, windowSize) {
    if (history.length <= windowSize) return history;

    // Count active frames in each possible window position
    let bestStart = 0;
    let bestCount = 0;

    // Pre-compute active flags
    const active = history.map(f => !isZeroFrame(f));

    // Sliding window sum
    let count = 0;
    for (let i = 0; i < windowSize && i < active.length; i++) {
      if (active[i]) count++;
    }
    bestCount = count;

    for (let start = 1; start + windowSize <= history.length; start++) {
      if (active[start - 1]) count--;
      if (active[start + windowSize - 1]) count++;
      if (count > bestCount) {
        bestCount = count;
        bestStart = start;
      }
    }

    return history.slice(bestStart, bestStart + windowSize);
  }

  function isZeroFrame(frame) {
    // Check if both hand slots are all zeros
    return frame[0][0] === 0 && frame[0][1] === 0 && frame[21][0] === 0 && frame[21][1] === 0;
  }

  // --------------------------------------------------------------------------
  // Interactive Sign Gesture Capture (2.0-Second Timed Capture)
  // --------------------------------------------------------------------------
  function startGestureCapture() {
    if (!isCameraRunning || isRecordingGesture) return;

    isRecordingGesture    = true;
    gestureRecordedFrames = [];
    gestureStartTime      = Date.now();
    DS.samplingMode       = 'RECORDING (2.0s)';

    // Update UI elements
    btnCaptureSign.classList.add('recording');
    btnCaptureSign.querySelector('span').textContent = 'Recording…';
    captureProgressContainer.classList.remove('hidden');
    captureProgressFill.style.width = '0%';
    captureStatusText.textContent   = 'Perform sign clearly…';
    updateCameraStatus('Recording your sign gesture (2.0s)…', 'active');
    setProcessingUI('Capturing sign gesture…');

    // Smooth visual progress bar animation
    const intervalMs = 20;
    gestureProgressInterval = setInterval(() => {
      const elapsed = Date.now() - gestureStartTime;
      const progress = Math.min(100, (elapsed / GESTURE_CAPTURE_MS) * 100);
      captureProgressFill.style.width = `${progress}%`;
      const remainingSec = Math.max(0, ((GESTURE_CAPTURE_MS - elapsed) / 1000)).toFixed(1);
      captureTimeText.textContent = `${remainingSec}s`;

      if (elapsed >= GESTURE_CAPTURE_MS) {
        clearInterval(gestureProgressInterval);
        gestureProgressInterval = null;
        finishGestureCapture();
      }
    }, intervalMs);
  }

  function finishGestureCapture() {
    isRecordingGesture = false;
    btnCaptureSign.classList.remove('recording');
    btnCaptureSign.querySelector('span').textContent = '⚡ Sign Now (Space)';
    captureProgressContainer.classList.add('hidden');
    updateCameraStatus('Analyzing sign gesture…', 'active');
    setProcessingUI('Analyzing captured gesture…');

    const totalCaptured = gestureRecordedFrames.length;
    console.log(`[SB] Gesture capture completed: ${totalCaptured} frames recorded.`);

    // Resample the full 2.0-second gesture into exactly 32 frames
    const sampledSequence = resampleTo32Frames(gestureRecordedFrames);
    DS.samplingMode = `GESTURE (${totalCaptured}f→32f)`;
    executeSequenceInference(sampledSequence, true);
  }

  // --------------------------------------------------------------------------
  // Inference API Execution
  // --------------------------------------------------------------------------
  async function executeSequenceInference(sequence32, isExplicitCapture = false) {
    if (sequence32.length !== 32 || sequence32[0].length !== 42) {
      console.error('[SB] Invalid sequence dimensions before API dispatch:', sequence32.length, sequence32[0]?.length);
      return;
    }

    isInferencing = true;
    DS.apiStatus = 'SENDING';
    updateDebugHUD();

    const t0 = Date.now();
    try {
      const res = await fetch('/api/predict/sequence', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sequence: sequence32, top_k: 5 }),
      });

      DS.apiLatencyMs = Date.now() - t0;

      if (!res.ok) {
        DS.apiStatus = `ERR_${res.status}`;
        updateDebugHUD();
        console.error('[SB] Inference error:', res.status);
        return;
      }

      const data = await res.json();
      DS.apiStatus      = 'SUCCESS';
      DS.lastPrediction = `${data.predicted_gloss} (${data.confidence.toFixed(1)}%)`;
      updateDebugHUD();

      if (isExplicitCapture) {
        // Direct capture: update immediately with full confidence presentation
        updatePredictionUI(data);
        updateCameraStatus(`Identified: ${data.predicted_gloss} (${data.confidence.toFixed(1)}%)`, 'active');
      } else {
        // Continuous stream: apply temporal smoothing to prevent UI flicker
        applyTemporalSmoothing(data);
      }

    } catch (err) {
      DS.apiLatencyMs = Date.now() - t0;
      DS.apiStatus = 'FETCH_ERR';
      updateDebugHUD();
      console.error('[SB] Network inference error:', err);
    } finally {
      isInferencing = false;
    }
  }

  // --------------------------------------------------------------------------
  // Temporal Smoothing & Margin-Aware Confidence-Weighted Voting
  // --------------------------------------------------------------------------
  function computeTop1Top2Margin(topK) {
    if (!topK || topK.length < 2) return topK?.[0]?.confidence || 0.0;
    return Math.max(0.0, (topK[0]?.confidence || 0.0) - (topK[1]?.confidence || 0.0));
  }

  function applyTemporalSmoothing(result) {
    if (!result || !result.predicted_gloss) return;

    // 1. Immediately render Top-5 list for raw model responsiveness (Stage 1: Raw Prediction)
    renderTop5List(result.top_k);

    // 2. Process through 4-stage Stabilization Engine
    const status = stabilizer.processPrediction(result);

    // Telemetry: record margin in debug HUD
    const margin = stabilizer.computeMargin(result.top_k);
    DS.lastPrediction = `${result.predicted_gloss} (${result.confidence.toFixed(1)}%, Δ${margin.toFixed(1)}%)`;
    updateDebugHUD();

    // 3. Stage 3: If candidate is stabilized, update UI with stable candidate
    if (status.stableCandidate) {
      updatePredictionUI({
        predicted_gloss: status.stableCandidate.gloss,
        confidence: status.stableCandidate.confidence,
        top_k: result.top_k,
      });
    }

    // 4. Stage 4: If sign was just committed (boundary/transition satisfied, duplicate suppressed)
    if (status.justCommitted && status.committedGloss) {
      console.log(`[Stabilizer] Sign Committed: ${status.committedGloss}`);

      // Auto-add to Sentence Builder in Recognize view (if toggle enabled)
      if (sbAutoAddToggle && sbAutoAddToggle.checked) {
        addWordToSentence(status.committedGloss, true);
      }

      // Add to Two-Way Conversation buffer
      addCommittedGlossToConversation(status.committedGloss);
    }
  }

  // --------------------------------------------------------------------------
  // Canvas Hand Skeleton Overlay
  // --------------------------------------------------------------------------
  function drawHandOverlay(landmarks) {
    const targets = [
      { ctx: canvasCtx, W: landmarkCanvas.width, H: landmarkCanvas.height }
    ];
    if (convCanvasCtx && convLandmarkCanvas && viewConversation && viewConversation.classList.contains('active')) {
      targets.push({ ctx: convCanvasCtx, W: convLandmarkCanvas.width, H: convLandmarkCanvas.height });
    }

    const CONNECTIONS = [
      [0,1],[1,2],[2,3],[3,4],
      [0,5],[5,6],[6,7],[7,8],
      [5,9],[9,10],[10,11],[11,12],
      [9,13],[13,14],[14,15],[15,16],
      [13,17],[17,18],[18,19],[19,20],[0,17],
    ];

    targets.forEach(({ ctx, W, H }) => {
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.65)';
      ctx.lineWidth = 2.5;

      CONNECTIONS.forEach(([i, j]) => {
        ctx.beginPath();
        ctx.moveTo(landmarks[i].x * W, landmarks[i].y * H);
        ctx.lineTo(landmarks[j].x * W, landmarks[j].y * H);
        ctx.stroke();
      });

      landmarks.forEach((lm, idx) => {
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, idx === 0 ? 4.5 : 3.0, 0, 2 * Math.PI);
        ctx.fillStyle = idx === 0 ? '#ec4899' : '#6366f1';
        ctx.fill();
      });
    });
  }

  // --------------------------------------------------------------------------
  // Video Upload Handlers
  // --------------------------------------------------------------------------
  dropzone.addEventListener('click',  () => videoFileInput.click());
  dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', e => {
    e.preventDefault(); dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files?.[0]) handleVideoFile(e.dataTransfer.files[0]);
  });
  videoFileInput.addEventListener('change', e => { if (e.target.files?.[0]) handleVideoFile(e.target.files[0]); });
  btnReupload.addEventListener('click', () => {
    uploadPreviewContainer.classList.add('hidden');
    dropzone.classList.remove('hidden');
    videoFileInput.value = '';
    resetPredictionUI();
  });

  async function handleVideoFile(file) {
    uploadVideoPlayer.src = URL.createObjectURL(file);
    uploadFileName.textContent = file.name;
    dropzone.classList.add('hidden');
    uploadPreviewContainer.classList.remove('hidden');
    setProcessingUI('Extracting landmarks & analyzing sign…');

    const fd = new FormData();
    fd.append('file', file);
    fd.append('top_k', 5);

    try {
      const res = await fetch('/api/predict/video', { method: 'POST', body: fd });
      if (!res.ok) { const e = await res.json(); throw new Error(e.detail || 'Inference failed'); }
      updatePredictionUI(await res.json());
    } catch (err) {
      console.error('[SB] Video processing error:', err);
      alert('Video processing failed: ' + err.message);
      resetPredictionUI();
    }
  }

  // --------------------------------------------------------------------------
  // Learn & Practice Module
  // --------------------------------------------------------------------------
  learnSearchInput.addEventListener('input', e => {
    clearTimeout(searchDebounceTimeout);
    searchDebounceTimeout = setTimeout(() => fetchVocabulary(e.target.value), 200);
  });

  document.querySelectorAll('.chip-btn[data-gloss]').forEach(btn => {
    btn.addEventListener('click', () => loadDemonstration(btn.getAttribute('data-gloss')));
  });

  async function fetchVocabulary(query = '') {
    try {
      const url = query ? `/api/vocabulary?q=${encodeURIComponent(query)}&limit=60` : `/api/vocabulary?limit=60`;
      const res = await fetch(url);
      if (!res.ok) return;
      const data = await res.json();
      renderVocabList(data.items);
      vocabCountBadge.textContent = `${data.total_matches} Signs`;
    } catch (err) { console.warn('[SB] Vocab fetch error:', err); }
  }

  function renderVocabList(items) {
    if (!items?.length) {
      vocabList.innerHTML = `<p style="padding:16px; color:var(--text-muted); text-align:center;">No signs match your search.</p>`;
      return;
    }
    vocabList.innerHTML = items.map(item => `
      <div class="vocab-item ${item.gloss === currentLearnGloss ? 'selected' : ''}"
           data-gloss="${item.gloss}" role="button" tabindex="0">
        <span class="vocab-name">${item.gloss}</span>
        <span class="vocab-tier-tag">${item.tier}</span>
      </div>`).join('');
    vocabList.querySelectorAll('.vocab-item').forEach(el =>
      el.addEventListener('click', () => loadDemonstration(el.getAttribute('data-gloss')))
    );
  }

  async function loadDemonstration(gloss) {
    currentLearnGloss = gloss;
    vocabList.querySelectorAll('.vocab-item').forEach(el =>
      el.classList.toggle('selected', el.getAttribute('data-gloss') === gloss)
    );
    demoGlossTitle.textContent = gloss;
    btnTrySign.querySelector('span').textContent = `Try Signing ${gloss}`;
    try {
      const res = await fetch(`/api/demonstration/${encodeURIComponent(gloss)}`);
      if (!res.ok) return;
      const data = await res.json();
      demoTierBadge.textContent = data.tier || 'Standard';
      if (data.has_demonstration && data.video_url) {
        demoVideoPlayer.src = data.video_url;
        demoVideoPlayer.classList.remove('hidden');
        demoVideoPlaceholder.classList.add('hidden');
        demoVideoPlayer.play().catch(() => {});
      } else {
        demoVideoPlayer.pause();
        demoVideoPlayer.classList.add('hidden');
        demoVideoPlaceholder.classList.remove('hidden');
        demoPlaceholderText.textContent = data.message || 'No demonstration video available.';
      }
    } catch (err) { console.warn('[SB] Demo load error:', err); }
  }

  async function triggerTeachMeSomething() {
    try {
      const res = await fetch('/api/teach-me');
      if (!res.ok) return;
      const data = await res.json();
      switchMainView('learn');
      loadDemonstration(data.gloss);
    } catch (err) { console.warn('[SB] Teach-me error:', err); }
  }

  btnTrySign.addEventListener('click', () => {
    targetPracticeGloss = currentLearnGloss;
    practiceTargetGloss.textContent = targetPracticeGloss;
    practiceTargetBanner.classList.remove('hidden');
    switchMainView('recognize');
    switchRecognizeTab('camera');
    if (!isCameraRunning) startCamera();
  });

  if (btnCancelPractice) {
    btnCancelPractice.addEventListener('click', () => {
      targetPracticeGloss = null;
      practiceTargetBanner.classList.add('hidden');
      resetPredictionUI();
    });
  }

  // --------------------------------------------------------------------------
  // Prediction Presentation UI
  // --------------------------------------------------------------------------
  function updatePredictionUI(result) {
    const gloss = result.predicted_gloss;
    const conf  = result.confidence;

    if (primaryGloss.textContent !== gloss) {
      primaryGloss.classList.remove('update-pulse');
      void primaryGloss.offsetWidth;
      primaryGloss.classList.add('update-pulse');
    }
    primaryGloss.textContent      = gloss;
    primaryConfidence.textContent = `${conf.toFixed(1)}%`;

    // Update Conversation View UI
    const convPrimaryGloss = document.getElementById('conv-primary-gloss');
    const convPrimaryConf  = document.getElementById('conv-primary-conf');
    if (convPrimaryGloss) convPrimaryGloss.textContent = gloss;
    if (convPrimaryConf)  convPrimaryConf.textContent = `${conf.toFixed(1)}%`;

    // Notify Conversation Controller
    if (window.SignBridgeConversation) {
      window.SignBridgeConversation.receiveSignPrediction(gloss, conf);
    }

    if (targetPracticeGloss) {
      const isTop1 = gloss.toUpperCase() === targetPracticeGloss.toUpperCase();
      const isTop5 = result.top_k.some(i => i.gloss.toUpperCase() === targetPracticeGloss.toUpperCase());
      if (isTop1 && conf >= 70) {
        confidenceBadge.className = 'confidence-badge high';
        confidenceBadge.textContent = 'Great Job!';
        confidenceAssessment.textContent = `🎉 Excellent! That looks like ${targetPracticeGloss}.`;
      } else if (isTop1 || isTop5) {
        confidenceBadge.className = 'confidence-badge medium';
        confidenceBadge.textContent = 'Getting Close!';
        confidenceAssessment.textContent = `💡 Good effort! ${targetPracticeGloss} detected in top candidates.`;
      } else {
        confidenceBadge.className = 'confidence-badge neutral';
        confidenceBadge.textContent = 'Try Again';
        confidenceAssessment.textContent = `🔁 Not recognized as ${targetPracticeGloss}. Try watching the demo again!`;
      }
    } else {
      if (conf >= 60) {
        confidenceBadge.className = 'confidence-badge high';
        confidenceBadge.textContent = 'High confidence';
        confidenceAssessment.textContent = 'Very strong recognition match';
      } else if (conf >= 25) {
        confidenceBadge.className = 'confidence-badge medium';
        confidenceBadge.textContent = 'Moderate confidence';
        confidenceAssessment.textContent = 'Good match — check alternatives';
      } else {
        confidenceBadge.className = 'confidence-badge neutral';
        confidenceBadge.textContent = 'Low confidence';
        confidenceAssessment.textContent = 'Sign clearly and complete the full motion';
      }
    }

    // Auto-add high-confidence signs to Sentence Builder
    if (sbAutoAddToggle && sbAutoAddToggle.checked && conf >= 45.0) {
      addWordToSentence(gloss, true);
    }

    renderTop5List(result.top_k);
  }

  function renderTop5List(items) {
    if (!items?.length) return;
    top5List.innerHTML = items.map((item, i) => {
      const prob   = item.confidence.toFixed(1);
      const rank   = String(i + 1).padStart(2, '0');
      const target = targetPracticeGloss && item.gloss.toUpperCase() === targetPracticeGloss.toUpperCase();
      return `
        <div class="ranking-item ${target ? 'target-match' : ''}" role="listitem" data-gloss="${item.gloss}" title="Click to add '${item.gloss}' to sentence">
          <span class="rank-num">${rank}</span>
          <span class="rank-gloss" title="${item.gloss}">${item.gloss}</span>
          <div class="rank-bar-bg" aria-hidden="true">
            <div class="rank-bar-fill" style="width:${Math.max(prob, 2)}%"></div>
          </div>
          <span class="rank-prob">${prob}%</span>
        </div>`;
    }).join('');

    // Clicking any candidate in Top-5 list immediately selects it and adds to sentence
    top5List.querySelectorAll('.ranking-item:not(.empty)').forEach(el => {
      el.addEventListener('click', () => {
        const selectedGloss = el.getAttribute('data-gloss');
        if (selectedGloss) {
          primaryGloss.textContent = selectedGloss;
          primaryGloss.classList.remove('update-pulse');
          void primaryGloss.offsetWidth;
          primaryGloss.classList.add('update-pulse');
          addWordToSentence(selectedGloss, false);
          updateCameraStatus(`Selected '${selectedGloss}' → Added to sentence`, 'active');
        }
      });
    });
  }

  function setProcessingUI(msg) {
    primaryGloss.textContent       = 'Analyzing…';
    primaryConfidence.textContent  = '—';
    confidenceBadge.className      = 'confidence-badge medium';
    confidenceBadge.textContent    = 'Processing';
    confidenceAssessment.textContent = msg;
  }

  function resetPredictionUI() {
    primaryGloss.textContent       = '—';
    primaryConfidence.textContent  = '0.0%';
    confidenceBadge.className      = 'confidence-badge neutral';
    confidenceBadge.textContent    = 'Awaiting Input';
    confidenceAssessment.textContent = targetPracticeGloss
      ? `Perform ${targetPracticeGloss} for the camera`
      : 'Show a sign or click Sign Now';
    top5List.innerHTML = [1,2,3,4,5].map((n, i) => `
      <div class="ranking-item empty"><span class="rank-num">0${n}</span>
        <span class="rank-gloss">${i === 0 ? 'Waiting for sign input…' : '—'}</span>
        <div class="rank-bar-bg"><div class="rank-bar-fill" style="width:0%"></div></div>
        <span class="rank-prob">0.0%</span></div>`).join('');
    lastStableGloss    = '—';
    consecutiveMatches = 0;
  }

  // --------------------------------------------------------------------------
  // --------------------------------------------------------------------------
  // ASL Vocabulary Set for NLP & Speech-to-Text
  // --------------------------------------------------------------------------
  let allASLGlossesSet = new Set(['APPLE', 'BLUE', 'LABEL', 'FAMILY', 'BREAK', 'DEPARTMENT', 'INDEPENDENT', 'BADGE', 'HUNDRED', 'EIGHT', 'WANT', 'EAT', 'GO', 'HELP', 'LIKE', 'NEED', 'PLEASE', 'THANKYOU', 'SORRY', 'HELLO', 'YES', 'NO', 'TIME', 'WHAT', 'WHERE', 'WHO', 'WHY', 'HOW']);

  fetch('/api/classes')
    .then(res => res.json())
    .then(data => {
      if (data && data.classes && Array.isArray(data.classes)) {
        allASLGlossesSet = new Set(data.classes.map(c => c.toUpperCase()));
      }
    })
    .catch(() => {});

  // --------------------------------------------------------------------------
  // Sentence Builder & NLP Translation Logic
  // --------------------------------------------------------------------------
  let sentenceWords      = [];
  let lastAutoAddedWord  = '';
  let lastAutoAddedTime  = 0;
  let currentNLPEnglish  = '';

  function addWordToSentence(word, isAuto = false) {
    if (!word || word === '—' || word === 'Analyzing…' || word.startsWith('Waiting')) return;
    const cleanWord = word.trim().toUpperCase();
    const now = Date.now();

    // Prevent adding same sign continuously in live streaming within 2.5 seconds
    if (isAuto) {
      if (cleanWord === lastAutoAddedWord && (now - lastAutoAddedTime < 2500)) {
        return;
      }
    }

    sentenceWords.push(cleanWord);
    lastAutoAddedWord = cleanWord;
    lastAutoAddedTime = now;
    renderSentenceUI();
    updateNLPTranslation();
  }

  function removeWordFromSentence(index) {
    if (index >= 0 && index < sentenceWords.length) {
      sentenceWords.splice(index, 1);
      renderSentenceUI();
      updateNLPTranslation();
    }
  }

  function backspaceSentence() {
    if (sentenceWords.length > 0) {
      sentenceWords.pop();
      renderSentenceUI();
      updateNLPTranslation();
    }
  }

  function clearSentence() {
    sentenceWords = [];
    lastAutoAddedWord = '';
    renderSentenceUI();
    updateNLPTranslation();
  }

  function updateNLPTranslation() {
    if (!sbNlpText) return;

    if (sentenceWords.length === 0) {
      sbNlpText.innerHTML = '<em>Assemble signs to generate natural English grammar...</em>';
      if (sbNlpStatus) sbNlpStatus.textContent = 'Ready';
      currentNLPEnglish = '';
      return;
    }

    // 1. Instant local NLP translation (0ms latency)
    if (window.SignBridgeNLP) {
      currentNLPEnglish = window.SignBridgeNLP.glossesToEnglish(sentenceWords);
      sbNlpText.textContent = `"${currentNLPEnglish}"`;
      if (sbNlpStatus) sbNlpStatus.textContent = 'Instant NLP';
    } else {
      currentNLPEnglish = sentenceWords.join(' ');
      sbNlpText.textContent = `"${currentNLPEnglish}"`;
    }

    // 2. Asynchronous backend NLP smoothing
    fetch('/api/nlp/smooth', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ glosses: sentenceWords })
    })
    .then(res => res.json())
    .then(data => {
      if (data && data.english) {
        currentNLPEnglish = data.english;
        sbNlpText.textContent = `"${currentNLPEnglish}"`;
        if (sbNlpStatus) sbNlpStatus.textContent = data.idiom ? 'ASL Idiom' : 'NLP Refined';
      }
    })
    .catch(() => {});
  }

  function renderSentenceUI() {
    if (!sbWordsContainer) return;
    const count = sentenceWords.length;
    if (sbWordCount) sbWordCount.textContent = `${count} word${count === 1 ? '' : 's'}`;

    if (count === 0) {
      if (sbEmptyState) sbEmptyState.classList.remove('hidden');
      sbWordsContainer.innerHTML = '';
    } else {
      if (sbEmptyState) sbEmptyState.classList.add('hidden');
      sbWordsContainer.innerHTML = sentenceWords.map((word, idx) => `
        <span class="sb-word-chip" data-idx="${idx}">
          <span>${word}</span>
          <button class="sb-word-remove" title="Remove word" aria-label="Remove ${word}">&times;</button>
        </span>
      `).join('');

      sbWordsContainer.querySelectorAll('.sb-word-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const chip = e.target.closest('.sb-word-chip');
          const idx = parseInt(chip.getAttribute('data-idx'), 10);
          removeWordFromSentence(idx);
        });
      });
    }
  }

  // --------------------------------------------------------------------------
  // Natural Voice Text-to-Speech (TTS) Engine
  // --------------------------------------------------------------------------
  let availableVoices = [];

  function loadNaturalVoices() {
    if (!('speechSynthesis' in window)) return;
    const voices = window.speechSynthesis.getVoices();
    if (!voices || voices.length === 0) return;

    // Filter English voices
    const enVoices = voices.filter(v => v.lang.startsWith('en') || v.lang.startsWith('en-'));
    availableVoices = enVoices.length > 0 ? enVoices : voices;

    if (!sbTtsVoiceSelect) return;
    sbTtsVoiceSelect.innerHTML = '';

    // Sort natural / enhanced voices first
    const preferredKeywords = ['enhanced', 'natural', 'google', 'samantha', 'ava', 'alex', 'daniel', 'serena', 'karen'];
    availableVoices.sort((a, b) => {
      const aName = a.name.toLowerCase();
      const bName = b.name.toLowerCase();
      const aPref = preferredKeywords.some(k => aName.includes(k)) ? 1 : 0;
      const bPref = preferredKeywords.some(k => bName.includes(k)) ? 1 : 0;
      return bPref - aPref;
    });

    availableVoices.forEach((voice, i) => {
      const opt = document.createElement('option');
      opt.value = i;
      opt.textContent = `${voice.name} (${voice.lang})${voice.default ? ' — System Default' : ''}`;
      sbTtsVoiceSelect.appendChild(opt);
    });
  }

  if ('speechSynthesis' in window) {
    loadNaturalVoices();
    window.speechSynthesis.onvoiceschanged = loadNaturalVoices;
  }

  if (sbTtsRate && sbTtsRateVal) {
    sbTtsRate.addEventListener('input', () => {
      sbTtsRateVal.textContent = `${parseFloat(sbTtsRate.value).toFixed(1)}x`;
    });
  }

  function speakSentence() {
    if (sentenceWords.length === 0 && !currentNLPEnglish) {
      updateCameraStatus('Sentence is empty — sign words first!', 'neutral');
      return;
    }

    // Speak the natural NLP English sentence!
    const textToSpeak = currentNLPEnglish || sentenceWords.join(' ');

    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(textToSpeak);

      // Voice selection
      const voiceIdx = sbTtsVoiceSelect ? parseInt(sbTtsVoiceSelect.value, 10) : 0;
      if (availableVoices && availableVoices[voiceIdx]) {
        utterance.voice = availableVoices[voiceIdx];
      }

      // Rate and pitch
      utterance.rate  = parseFloat(sbTtsRate?.value || 1.0);
      utterance.pitch = 1.0;

      if (sbBtnSpeak) sbBtnSpeak.classList.add('speaking');
      utterance.onend = () => { if (sbBtnSpeak) sbBtnSpeak.classList.remove('speaking'); };
      utterance.onerror = () => { if (sbBtnSpeak) sbBtnSpeak.classList.remove('speaking'); };

      window.speechSynthesis.speak(utterance);
      updateCameraStatus(`Speaking: "${textToSpeak}"`, 'active');
    } else {
      alert('Text-to-speech is not supported in this browser.');
    }
  }

  function copySentence() {
    const textToCopy = currentNLPEnglish || sentenceWords.join(' ');
    if (!textToCopy) return;

    if (navigator.clipboard) {
      navigator.clipboard.writeText(textToCopy).then(() => {
        if (sbCopyText) sbCopyText.textContent = 'Copied!';
        setTimeout(() => { if (sbCopyText) sbCopyText.textContent = 'Copy Text'; }, 1800);
      }).catch(() => {});
    }
  }

  // --------------------------------------------------------------------------
  // Speech-to-Text (STT) Hearing Partner Module
  // --------------------------------------------------------------------------
  let speechRecognizer = null;
  let isListeningSTT   = false;

  function initSpeechToText() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      if (hpMicBadge) {
        hpMicBadge.textContent = 'Mic Not Supported';
      }
      if (hpBtnMic) {
        hpBtnMic.disabled = true;
        hpBtnMic.title = 'Speech Recognition is not supported in this browser (Use Chrome or Edge).';
      }
      return;
    }

    speechRecognizer = new SpeechRecognition();
    speechRecognizer.continuous     = true;
    speechRecognizer.interimResults  = true;
    speechRecognizer.lang           = 'en-US';

    speechRecognizer.onstart = () => {
      isListeningSTT = true;
      if (hpMicBadge) {
        hpMicBadge.textContent = 'Listening…';
        hpMicBadge.className = 'hp-badge listening';
      }
      if (hpBtnMic) {
        hpBtnMic.classList.add('active');
        if (hpMicLabel) hpMicLabel.textContent = 'Stop Listening';
      }
      if (hpSpeechBox) hpSpeechBox.classList.add('listening');
      if (hpEmptyState) hpEmptyState.style.display = 'none';
      if (hpTranscriptionText) hpTranscriptionText.style.display = 'block';
    };

    speechRecognizer.onresult = (event) => {
      let finalTranscript = '';
      let interimTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      const fullSpokenText = (finalTranscript || interimTranscript).trim();
      if (!fullSpokenText) return;

      if (hpTranscriptionText) {
        hpTranscriptionText.textContent = fullSpokenText;
      }

      // NLP match spoken English words to ASL signs
      if (window.SignBridgeNLP) {
        const matches = window.SignBridgeNLP.extractASLGlossesFromSpeech(fullSpokenText, allASLGlossesSet);
        renderSTTMatchedSigns(matches);
      }
    };

    speechRecognizer.onerror = (err) => {
      console.warn('Speech recognition error:', err);
      if (err.error === 'not-allowed') {
        updateCameraStatus('Microphone access was denied.', 'error');
      }
    };

    speechRecognizer.onend = () => {
      isListeningSTT = false;
      if (hpMicBadge) {
        hpMicBadge.textContent = 'Microphone Idle';
        hpMicBadge.className = 'hp-badge';
      }
      if (hpBtnMic) {
        hpBtnMic.classList.remove('active');
        if (hpMicLabel) hpMicLabel.textContent = 'Start Listening';
      }
      if (hpSpeechBox) hpSpeechBox.classList.remove('listening');
    };
  }

  function toggleSpeechToText() {
    if (!speechRecognizer) {
      initSpeechToText();
    }
    if (!speechRecognizer) return;

    if (isListeningSTT) {
      speechRecognizer.stop();
    } else {
      try {
        speechRecognizer.start();
      } catch (err) {
        console.warn('Recognition start exception:', err);
      }
    }
  }

  function renderSTTMatchedSigns(matches) {
    if (!hpMatchedSignsRow || !hpMatchedChips) return;

    if (!matches || matches.length === 0) {
      hpMatchedSignsRow.style.display = 'none';
      hpMatchedChips.innerHTML = '';
      return;
    }

    hpMatchedSignsRow.style.display = 'flex';
    hpMatchedChips.innerHTML = matches.map(m => `
      <button class="hp-asl-chip" data-gloss="${m.gloss}" title="Click to add '${m.gloss}' to sentence">
        <span>${m.gloss}</span>
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line></svg>
      </button>
    `).join('');

    hpMatchedChips.querySelectorAll('.hp-asl-chip').forEach(btn => {
      btn.addEventListener('click', () => {
        const gloss = btn.getAttribute('data-gloss');
        if (gloss) {
          addWordToSentence(gloss, false);
          updateCameraStatus(`Added '${gloss}' from partner's speech`, 'active');
        }
      });
    });
  }

  // Initialize Speech-to-Text
  initSpeechToText();
  if (hpBtnMic) {
    hpBtnMic.addEventListener('click', toggleSpeechToText);
  }

  // Sentence Builder event listeners
  if (sbBtnSpeak)      sbBtnSpeak.addEventListener('click', speakSentence);
  if (sbBtnCopy)       sbBtnCopy.addEventListener('click', copySentence);
  if (sbBtnBackspace)  sbBtnBackspace.addEventListener('click', backspaceSentence);
  if (sbBtnClear)      sbBtnClear.addEventListener('click', clearSentence);
  if (sbBtnAddCurrent) {
    sbBtnAddCurrent.addEventListener('click', () => {
      const current = primaryGloss?.textContent;
      if (current && current !== '—' && current !== 'Analyzing…') {
        addWordToSentence(current, false);
      }
    });
  }

  // Quick high-accuracy vocabulary chips
  if (sbQuickChips) {
    sbQuickChips.forEach(chip => {
      chip.addEventListener('click', () => {
        const gloss = chip.getAttribute('data-gloss');
        if (gloss) {
          addWordToSentence(gloss, false);
          updateCameraStatus(`Added '${gloss}' to sentence`, 'active');
        }
      });
    });
  }

  // Initial Sentence Builder render
  renderSentenceUI();
  updateNLPTranslation();

  // ==========================================================================
  // TWO-WAY REAL-TIME CONVERSATION MODE CONTROLLER & UI INTEGRATION
  // ==========================================================================
  const convController = new window.ConversationController();
  window.SignBridgeConversation = convController;

  // DOM Elements for Conversation View
  const btnModeTwoWay        = document.getElementById('btn-mode-twoway');
  const btnModeSignToSpeech  = document.getElementById('btn-mode-signtospeech');
  const btnModeSpeechToText  = document.getElementById('btn-mode-speechtotext');
  const btnConvToggle        = document.getElementById('btn-conv-toggle');
  const btnConvToggleLabel   = document.getElementById('btn-conv-toggle-label');
  const btnConvClear         = document.getElementById('btn-conv-clear');
  const btnConvExport        = document.getElementById('btn-conv-export');
  const btnConvSpeak         = document.getElementById('btn-conv-speak');
  const btnConvAddSign       = document.getElementById('btn-conv-add-sign');
  const btnConvBackspace     = document.getElementById('btn-conv-backspace');
  const btnConvClearSigns    = document.getElementById('btn-conv-clear-signs');
  const btnConvStopAudio     = document.getElementById('btn-conv-stop-audio');
  const convVoiceSelect      = document.getElementById('conv-voice-select');

  const indSign              = document.getElementById('ind-sign');
  const valSign              = document.getElementById('val-sign');
  const indTts               = document.getElementById('ind-tts');
  const valTts               = document.getElementById('val-tts');
  const indStt               = document.getElementById('ind-stt');
  const valStt               = document.getElementById('val-stt');
  const valSpeaker           = document.getElementById('val-speaker');

  const convPrimaryGloss     = document.getElementById('conv-primary-gloss');
  const convPrimaryConf      = document.getElementById('conv-primary-conf');
  const convGlossTokens      = document.getElementById('conv-gloss-tokens');
  const convNlpText          = document.getElementById('conv-nlp-text');
  const convPartialText      = document.getElementById('conv-partial-text');
  const liveTypingTag        = document.getElementById('live-typing-tag');
  const micWaveIndicator     = document.getElementById('mic-wave-indicator');
  const convMessagesList     = document.getElementById('conv-messages-list');
  const convThreadContainer  = document.getElementById('conv-thread-container');
  const convThreadEmpty      = document.getElementById('conv-thread-empty');
  const convMsgCount         = document.getElementById('conv-msg-count');
  const chipMicLive          = document.getElementById('chip-mic-live');

  let convAssembledWords     = [];
  let isConversationRunning  = false;
  let activeSelectedMode     = 'two_way';

  // 1. Populate Voicebox voices in Conversation selector
  if (convVoiceSelect && convController.ttsClient) {
    convController.ttsClient.fetchVoices().then(() => {
      const voices = convController.ttsClient.voices;
      if (voices && voices.length > 0) {
        convVoiceSelect.innerHTML = voices.map(v => `
          <option value="${v.id}">${v.name} (${v.lang})</option>
        `).join('');
      }
    });

    convVoiceSelect.addEventListener('change', (e) => {
      if (convController.ttsClient) {
        convController.ttsClient.defaultVoiceId = e.target.value;
      }
    });
  }

  // 2. Subsystem Telemetry Event Listeners
  convController.on('subsystemChange', (subsystem, state, details) => {
    if (subsystem === 'signRecognition') {
      if (valSign) valSign.textContent = state === 'listening' ? 'Active Tracking' : (state === 'paused' ? 'Paused' : state);
      if (indSign) indSign.className = `telemetry-indicator ${state === 'listening' ? 'active' : 'ready'}`;
    } else if (subsystem === 'tts') {
      if (valTts) valTts.textContent = state === 'generating' ? 'Synthesizing...' : (state === 'playing' ? 'Speaking Aloud' : 'Idle');
      if (indTts) indTts.className = `telemetry-indicator ${state === 'playing' ? 'active' : (state === 'generating' ? 'warn' : 'ready')}`;
      if (btnConvSpeak) {
        if (state === 'generating' || state === 'playing') {
          btnConvSpeak.classList.add('speaking');
          btnConvSpeak.disabled = true;
        } else {
          btnConvSpeak.classList.remove('speaking');
          btnConvSpeak.disabled = false;
        }
      }
    } else if (subsystem === 'stt' || subsystem === 'microphone') {
      const isListening = convController.subsystems.stt === 'listening' || convController.subsystems.stt === 'receiving_partial_transcript';
      const isMuted = convController.subsystems.microphone === 'muted';

      if (valStt) {
        valStt.textContent = isMuted ? 'Echo Suppressed (TTS Playing)' : (isListening ? 'Listening Aloud' : 'Microphone Idle');
      }
      if (indStt) {
        indStt.className = `telemetry-indicator ${isListening ? (isMuted ? 'warn' : 'active') : 'ready'}`;
      }
      if (chipMicLive) {
        chipMicLive.textContent = isMuted ? 'Echo Suppressed' : (isListening ? 'Mic Active' : 'Mic Idle');
        chipMicLive.className = `conv-live-chip ${isListening ? 'active' : ''}`;
      }
      if (micWaveIndicator) {
        micWaveIndicator.className = `mic-wave-indicator ${isListening && !isMuted ? 'active' : ''}`;
      }
    }

    if (valSpeaker) {
      valSpeaker.textContent = convController.activeSpeaker === 'sign_user'
        ? '🤟 Sign Language User'
        : (convController.activeSpeaker === 'hearing_partner' ? '🎙️ Hearing Partner' : '—');
    }
  });

  // 3. Hearing Partner Real-Time Partial (Interim) Speech
  convController.on('partialTranscript', (text) => {
    if (!convPartialText) return;
    if (text) {
      convPartialText.textContent = `"${text}"`;
      if (liveTypingTag) liveTypingTag.style.display = 'inline-block';
    } else {
      convPartialText.textContent = isConversationRunning
        ? 'Listening for hearing partner speech...'
        : 'Click Start Conversation to begin listening.';
      if (liveTypingTag) liveTypingTag.style.display = 'none';
    }
  });

  // 4. Synchronized Conversation Message Thread
  convController.on('messageAdded', (message) => {
    if (!convMessagesList) return;
    if (convThreadEmpty) convThreadEmpty.style.display = 'none';

    const bubble = document.createElement('div');
    bubble.className = `conv-msg-bubble ${message.speaker === 'sign_user' ? 'sign-speaker' : 'hearing-speaker'}`;
    bubble.id = message.id;

    const speakerLabel = message.speaker === 'sign_user' ? '🤟 Deaf / ASL Participant' : '🎙️ Hearing Partner';
    const glossChipsHtml = message.rawGlosses && message.rawGlosses.length > 0
      ? `<div class="msg-bubble-glosses">
          ${message.rawGlosses.map(g => `<span class="msg-gloss-chip">${g}</span>`).join('')}
        </div>`
      : (message.matchedGlosses && message.matchedGlosses.length > 0
          ? `<div class="msg-bubble-glosses">
              <span style="font-size:0.6875rem; color:var(--text-muted); font-weight:600;">Matched ASL:</span>
              ${message.matchedGlosses.map(m => `<span class="msg-gloss-chip" title="Click to add">${m.gloss}</span>`).join('')}
            </div>`
          : '');

    const replayBtnHtml = message.speaker === 'sign_user'
      ? `<button class="msg-replay-btn" data-msg-id="${message.id}" title="Replay spoken audio">
          <span>🔊 Replay</span>
        </button>`
      : '';

    bubble.innerHTML = `
      <div class="msg-bubble-meta">
        <span class="msg-bubble-speaker">${speakerLabel}</span>
        <span class="msg-bubble-time">${message.timestamp}</span>
      </div>
      <div class="msg-bubble-body">"${message.text}"</div>
      ${glossChipsHtml}
      ${replayBtnHtml}
    `;

    // Replay speech button listener
    const replayBtn = bubble.querySelector('.msg-replay-btn');
    if (replayBtn) {
      replayBtn.addEventListener('click', () => {
        if (convController.ttsClient) {
          convController.ttsClient.speak(message.text);
        }
      });
    }

    convMessagesList.appendChild(bubble);

    // Auto-scroll to latest message
    if (convThreadContainer) {
      convThreadContainer.scrollTop = convThreadContainer.scrollHeight;
    }

    if (convMsgCount) {
      const count = convController.messages.length;
      convMsgCount.textContent = `${count} message${count === 1 ? '' : 's'} exchanged`;
    }
  });

  convController.on('messageUpdated', (message) => {
    const el = document.getElementById(message.id);
    if (el) {
      const replayBtn = el.querySelector('.msg-replay-btn');
      if (replayBtn) {
        replayBtn.innerHTML = message.status === 'playing' ? '<span>🔊 Speaking…</span>' : '<span>🔊 Replay</span>';
      }
    }
  });

  convController.on('historyCleared', () => {
    if (convMessagesList) convMessagesList.innerHTML = '';
    if (convThreadEmpty) convThreadEmpty.style.display = 'block';
    if (convMsgCount) convMsgCount.textContent = '0 messages exchanged';
  });

  // 5. Assembling Signs in Conversation View (Connected to Stabilizer)
  function renderConversationSentence() {
    if (!convGlossTokens) return;
    convAssembledWords = stabilizer.getCommittedGlosses();
    if (convAssembledWords.length === 0) {
      convGlossTokens.innerHTML = `<span class="empty-tokens-hint">Signs performed in front of camera assemble here...</span>`;
    } else {
      convGlossTokens.innerHTML = convAssembledWords.map((word, idx) => `
        <span class="sb-word-chip" data-conv-idx="${idx}">
          <span>${word}</span>
          <button class="sb-word-remove" title="Remove" aria-label="Remove ${word}">&times;</button>
        </span>
      `).join('');

      convGlossTokens.querySelectorAll('.sb-word-remove').forEach(btn => {
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          const chip = e.target.closest('.sb-word-chip');
          const idx = parseInt(chip.getAttribute('data-conv-idx'), 10);
          stabilizer.removeCommittedAt(idx);
          convAssembledWords = stabilizer.getCommittedGlosses();
          renderConversationSentence();
          updateConversationNLP();
        });
      });
    }
  }

  function updateConversationNLP() {
    if (!convNlpText) return;
    convAssembledWords = stabilizer.getCommittedGlosses();
    if (convAssembledWords.length === 0) {
      convNlpText.value = '';
      convNlpText.placeholder = 'Natural English translation will generate here (editable for review)...';
      return;
    }

    if (window.SignBridgeNLP && window.SignBridgeNLP.glossesToEnglish) {
      const english = window.SignBridgeNLP.glossesToEnglish(convAssembledWords);
      convNlpText.value = english;
    } else {
      convNlpText.value = convAssembledWords.join(' ');
    }
  }

  function addCommittedGlossToConversation(gloss) {
    if (!gloss || gloss === '—' || gloss === 'Analyzing…') return;
    convController.receiveCommittedGloss(gloss);
    convAssembledWords = stabilizer.getCommittedGlosses();
    renderConversationSentence();
    updateConversationNLP();
  }

  // Buttons in Conversation View
  if (btnConvAddSign) {
    btnConvAddSign.addEventListener('click', () => {
      const current = convPrimaryGloss ? convPrimaryGloss.textContent.trim() : null;
      if (current && current !== '—' && current !== 'Analyzing…') {
        stabilizer.commitManual(current);
        addCommittedGlossToConversation(current);
      }
    });
  }

  if (btnConvBackspace) {
    btnConvBackspace.addEventListener('click', () => {
      stabilizer.undoLastCommitted();
      convAssembledWords = stabilizer.getCommittedGlosses();
      renderConversationSentence();
      updateConversationNLP();
    });
  }

  if (btnConvClearSigns) {
    btnConvClearSigns.addEventListener('click', () => {
      stabilizer.clearCommitted();
      convAssembledWords = [];
      renderConversationSentence();
      updateConversationNLP();
    });
  }

  if (btnConvSpeak) {
    btnConvSpeak.addEventListener('click', async () => {
      if (btnConvSpeak.disabled) return;
      btnConvSpeak.disabled = true;

      try {
        const reviewedText = convNlpText ? convNlpText.value.trim() : '';
        convAssembledWords = stabilizer.getCommittedGlosses();

        if (!reviewedText && convAssembledWords.length === 0) {
          alert('Gloss buffer is empty! Sign words or select quick signs first.');
          return;
        }

        // Execute End-to-End sentence pipeline: Gloss Buffer -> Ollama Qwen3:8B -> Kokoro TTS
        const success = await convController.commitSignUserSentence(convAssembledWords, reviewedText);

        if (success) {
          // Clear buffer ONLY after successful sentence generation & TTS initiation
          stabilizer.clearCommitted();
          convAssembledWords = [];
          renderConversationSentence();
          updateConversationNLP();
        }
      } finally {
        if (convController.conversationState !== 'processing' && convController.conversationState !== 'speaking') {
          btnConvSpeak.disabled = false;
        }
      }
    });
  }

  // Quick signs in Conversation view
  document.querySelectorAll('.conv-phrase-chip').forEach(btn => {
    btn.addEventListener('click', () => {
      const gloss = btn.getAttribute('data-gloss');
      if (gloss) {
        stabilizer.commitManual(gloss);
        addCommittedGlossToConversation(gloss);
      }
    });
  });

  // 6. Mode Buttons
  function setActiveModeButton(mode) {
    activeSelectedMode = mode;
    [btnModeTwoWay, btnModeSignToSpeech, btnModeSpeechToText].forEach(b => {
      if (b) b.classList.remove('active');
    });
    if (mode === 'two_way' && btnModeTwoWay) btnModeTwoWay.classList.add('active');
    if (mode === 'sign_to_speech' && btnModeSignToSpeech) btnModeSignToSpeech.classList.add('active');
    if (mode === 'speech_to_text' && btnModeSpeechToText) btnModeSpeechToText.classList.add('active');

    if (isConversationRunning) {
      convController.setMode(mode);
    }
  }

  if (btnModeTwoWay) btnModeTwoWay.addEventListener('click', () => setActiveModeButton('two_way'));
  if (btnModeSignToSpeech) btnModeSignToSpeech.addEventListener('click', () => setActiveModeButton('sign_to_speech'));
  if (btnModeSpeechToText) btnModeSpeechToText.addEventListener('click', () => setActiveModeButton('speech_to_text'));

  // 7. Toggle Conversation Session
  if (btnConvToggle) {
    btnConvToggle.addEventListener('click', async () => {
      if (!isConversationRunning) {
        // Start Session
        if (!isCameraRunning) {
          await startCamera();
        }
        convController.setMode(activeSelectedMode);
        isConversationRunning = true;
        btnConvToggle.classList.add('recording');
        if (btnConvToggleLabel) btnConvToggleLabel.textContent = 'Pause Conversation';
        if (convPartialText) convPartialText.textContent = 'Listening for hearing partner speech...';
      } else {
        // Pause Session
        convController.setMode('idle');
        isConversationRunning = false;
        btnConvToggle.classList.remove('recording');
        if (btnConvToggleLabel) btnConvToggleLabel.textContent = 'Start Conversation';
        if (convPartialText) convPartialText.textContent = 'Conversation paused. Click Start Conversation to resume.';
      }
    });
  }

  if (btnConvClear) {
    btnConvClear.addEventListener('click', () => {
      convController.clearHistory();
    });
  }

  if (btnConvExport) {
    btnConvExport.addEventListener('click', () => {
      const text = convController.exportTranscriptText();
      const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `SignBridge_Conversation_${Date.now()}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  if (btnConvStopAudio) {
    btnConvStopAudio.addEventListener('click', () => {
      if (convController.ttsClient) convController.ttsClient.stop();
    });
  }

  // --------------------------------------------------------------------------
  // Truthful Model Status & Health Telemetry Check
  // --------------------------------------------------------------------------
  async function checkSystemHealthAndModelStatus() {
    const statusText = document.getElementById('system-status-text');
    const statusDot  = document.getElementById('system-status-dot');

    try {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error(`Health API returned ${res.status}`);
      const data = await res.json();

      if (data.model_loaded) {
        const classCount = data.num_classes ? data.num_classes.toLocaleString() : '2,731';
        if (statusText) statusText.textContent = `Model Ready (${classCount} Glosses)`;
        if (statusDot) statusDot.className = 'status-indicator-dot ready';
        if (valSign) valSign.textContent = `Ready (${classCount} Glosses)`;
        if (indSign) indSign.className = 'telemetry-indicator ready';
      } else {
        if (statusText) statusText.textContent = 'Model Checkpoint Not Loaded (Vocabulary only)';
        if (statusDot) statusDot.className = 'status-indicator-dot warn';
        if (valSign) valSign.textContent = 'Model Checkpoint Not Loaded';
        if (indSign) indSign.className = 'telemetry-indicator warn';
      }
    } catch (e) {
      console.warn('[SB] Health check error:', e);
      if (statusText) statusText.textContent = 'Backend Offline';
      if (statusDot) statusDot.className = 'status-indicator-dot error';
      if (valSign) valSign.textContent = 'Backend Disconnected';
      if (indSign) indSign.className = 'telemetry-indicator error';
    }
  }

  // Initial Conversation Sentence render & Health check
  renderConversationSentence();
  updateConversationNLP();
  checkSystemHealthAndModelStatus();

}); // end DOMContentLoaded

