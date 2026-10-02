/**
 * Real-Time Speech-to-Text (STT) Client & Microphone Controller for SignBridge.
 *
 * Provides:
 * - High-speed, streaming transcription using Web Speech API (primary)
 * - Managed microphone capture lifecycle and device permission states
 * - Separation of interim (partial draft) and final transcription events
 * - Acoustic echo suppression gating (prevents TTS output from being transcribed as hearing input)
 * - Safe reconnect and device disconnection handling
 */

(function(window) {
  'use strict';

  class STTClient {
    constructor(options = {}) {
      this.language = options.language || 'en-US';
      this.sttState = 'idle'; // idle | requesting_permission | connecting | listening | receiving_partial_transcript | finalizing | reconnecting | stopped | error
      this.micState = 'unavailable'; // unavailable | permission_required | permission_granted | active | muted | disconnected

      this.recognizer = null;
      this.mediaStream = null;
      this.isExplicitlyStopped = true;
      this.isEchoSuppressed = false; // When true (e.g. TTS playing), mic audio is ignored
      this.sessionId = null;

      this.reconnectAttempts = 0;
      this.maxReconnectAttempts = 3;
      this.reconnectTimeout = null;

      // Event Listeners
      this.partialListeners = new Set();
      this.finalListeners = new Set();
      this.stateListeners = new Set();
      this.errorListeners = new Set();

      this._checkSupport();
    }

    _checkSupport() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (SpeechRecognition) {
        this.micState = 'permission_required';
      } else {
        this.micState = 'unavailable';
      }
    }

    onPartial(fn) {
      this.partialListeners.add(fn);
      return () => this.partialListeners.delete(fn);
    }

    onFinal(fn) {
      this.finalListeners.add(fn);
      return () => this.finalListeners.delete(fn);
    }

    onStateChange(fn) {
      this.stateListeners.add(fn);
      return () => this.stateListeners.delete(fn);
    }

    onError(fn) {
      this.errorListeners.add(fn);
      return () => this.errorListeners.delete(fn);
    }

    _setStates(sttState, micState = null) {
      if (sttState) this.sttState = sttState;
      if (micState) this.micState = micState;

      for (const listener of this.stateListeners) {
        try {
          listener(this.sttState, this.micState);
        } catch (e) {
          console.error('[STTClient] State listener error:', e);
        }
      }
    }

    _emitError(err, recoverable = true) {
      this._setStates('error');
      for (const listener of this.errorListeners) {
        try {
          listener(err, recoverable);
        } catch (e) {
          console.error('[STTClient] Error listener error:', e);
        }
      }
    }

    /**
     * Set acoustic echo suppression state.
     * When true (TTS output is actively playing aloud), microphone transcription
     * is temporarily ignored to prevent self-looping feedback.
     */
    setEchoSuppression(isSuppressed) {
      this.isEchoSuppressed = !!isSuppressed;
      if (this.isEchoSuppressed && this.sttState === 'listening') {
        this._setStates(null, 'muted');
      } else if (!this.isEchoSuppressed && this.sttState === 'listening') {
        this._setStates(null, 'active');
      }
    }

    /**
     * Start listening to the microphone.
     */
    async start() {
      this.isExplicitlyStopped = false;
      this.sessionId = 'stt_session_' + Date.now();
      const currentSession = this.sessionId;

      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        this._emitError(new Error('Speech recognition not supported in this browser. Please use Chrome or Edge.'), false);
        return;
      }

      this._setStates('requesting_permission', 'permission_required');

      // Request microphone device permission explicitly
      try {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
          this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: true,
              noiseSuppression: true,
              autoGainControl: true,
            }
          });
          this._setStates(null, 'permission_granted');
        }
      } catch (permErr) {
        console.warn('[STTClient] Microphone permission denied:', permErr);
        this._emitError(new Error('Microphone permission was denied. Please allow microphone access to transcribe speech.'), false);
        this._setStates('error', 'unavailable');
        return;
      }

      if (this.isExplicitlyStopped || this.sessionId !== currentSession) return;

      this._setStates('connecting');

      try {
        if (this.recognizer) {
          try { this.recognizer.abort(); } catch (e) {}
        }

        const recognizer = new SpeechRecognition();
        recognizer.continuous = true;
        recognizer.interimResults = true;
        recognizer.lang = this.language;

        recognizer.onstart = () => {
          if (this.sessionId !== currentSession) return;
          this.reconnectAttempts = 0;
          this._setStates('listening', this.isEchoSuppressed ? 'muted' : 'active');
        };

        recognizer.onresult = (event) => {
          if (this.sessionId !== currentSession) return;
          if (this.isEchoSuppressed) {
            // Ignore acoustic feedback during TTS speech playback
            return;
          }

          let interimTranscript = '';
          let finalTranscript = '';

          for (let i = event.resultIndex; i < event.results.length; ++i) {
            const res = event.results[i];
            const text = res[0].transcript;
            if (res.isFinal) {
              finalTranscript += text;
            } else {
              interimTranscript += text;
            }
          }

          if (interimTranscript.trim()) {
            this._setStates('receiving_partial_transcript');
            for (const fn of this.partialListeners) {
              try { fn(interimTranscript.trim()); } catch (e) {}
            }
          }

          if (finalTranscript.trim()) {
            this._setStates('finalizing');
            const conf = event.results[event.results.length - 1][0].confidence || 0.92;
            for (const fn of this.finalListeners) {
              try { fn(finalTranscript.trim(), conf); } catch (e) {}
            }
            // Transition back to listening for next turn
            this._setStates('listening');
          }
        };

        recognizer.onerror = (event) => {
          if (this.sessionId !== currentSession) return;
          console.warn('[STTClient] Recognition error event:', event.error);

          if (event.error === 'not-allowed') {
            this._emitError(new Error('Microphone permission denied.'), false);
            this._setStates('error', 'unavailable');
          } else if (event.error === 'no-speech') {
            // Normal silence timeout in continuous mode; ignore
          } else if (event.error === 'network') {
            this._handleReconnect(currentSession);
          } else {
            this._emitError(new Error(`Speech recognition error: ${event.error}`), true);
          }
        };

        recognizer.onend = () => {
          if (this.sessionId !== currentSession) return;
          if (!this.isExplicitlyStopped) {
            // Browser SpeechRecognition automatically ends after long silence; reconnect smoothly
            this._handleReconnect(currentSession);
          } else {
            this._setStates('stopped', 'permission_granted');
          }
        };

        this.recognizer = recognizer;
        recognizer.start();

      } catch (err) {
        console.error('[STTClient] Start error:', err);
        this._emitError(err, true);
      }
    }

    _handleReconnect(session) {
      if (this.isExplicitlyStopped || this.sessionId !== session) return;

      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        this.reconnectAttempts++;
        const backoffMs = Math.min(1000 * Math.pow(1.5, this.reconnectAttempts), 4000);
        this._setStates('reconnecting');
        console.log(`[STTClient] Reconnecting in ${backoffMs}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        clearTimeout(this.reconnectTimeout);
        this.reconnectTimeout = setTimeout(() => {
          if (!this.isExplicitlyStopped && this.sessionId === session) {
            try {
              if (this.recognizer) this.recognizer.start();
            } catch (e) {
              // If start fails, recreate instance
              this.start();
            }
          }
        }, backoffMs);
      } else {
        this._emitError(new Error('Microphone connection dropped. Click Start Listening to resume.'), true);
        this._setStates('stopped');
      }
    }

    /**
     * Stop listening and release active microphone resources.
     */
    stop() {
      this.isExplicitlyStopped = true;
      this.sessionId = null;
      clearTimeout(this.reconnectTimeout);

      if (this.recognizer) {
        try {
          this.recognizer.stop();
          this.recognizer.abort();
        } catch (e) {}
        this.recognizer = null;
      }

      if (this.mediaStream) {
        try {
          this.mediaStream.getTracks().forEach(track => track.stop());
        } catch (e) {}
        this.mediaStream = null;
      }

      this._setStates('stopped', 'permission_granted');
    }
  }

  window.STTClient = STTClient;

})(window);
