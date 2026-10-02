/**
 * Voicebox TTS Client & Audio Playback Controller for SignBridge.
 *
 * Provides:
 * - Asynchronous synthesis requests to /api/tts/synthesize (Voicebox provider)
 * - Managed audio buffering, queueing, and HTML5 Audio / Web Audio playback
 * - Request cancellation tokens and stale response rejection
 * - Autoplay restriction handling and browser SpeechSynthesis fallback
 * - Explicit playback state synchronization
 */

(function(window) {
  'use strict';

  class VoiceboxTTSClient {
    constructor() {
      this.state = 'idle'; // idle | queued | generating | ready | playing | completed | cancelled | error
      this.currentRequestId = null;
      this.activeAudio = null;
      this.audioQueue = [];
      this.voices = [];
      this.defaultVoiceId = 'natural-en-us-1';
      this.rate = 1.0;
      this.isConfigured = false;
      this.stateListeners = new Set();
      this.errorListeners = new Set();

      this.fetchVoices();
    }

    onStateChange(fn) {
      this.stateListeners.add(fn);
      return () => this.stateListeners.delete(fn);
    }

    onError(fn) {
      this.errorListeners.add(fn);
      return () => this.errorListeners.delete(fn);
    }

    _setState(newState, details = {}) {
      this.state = newState;
      for (const listener of this.stateListeners) {
        try {
          listener(newState, details);
        } catch (e) {
          console.error('[TTSClient] State listener error:', e);
        }
      }
    }

    _emitError(err, recoverable = true) {
      this._setState('error', { error: err, recoverable });
      for (const listener of this.errorListeners) {
        try {
          listener(err, recoverable);
        } catch (e) {
          console.error('[TTSClient] Error listener error:', e);
        }
      }
    }

    async fetchVoices() {
      try {
        const res = await fetch('/api/tts/voices');
        if (res.ok) {
          const data = await res.json();
          this.voices = data.voices || [];
          this.defaultVoiceId = data.default_voice || 'natural-en-us-1';
          this.isConfigured = !!data.is_configured;
        }
      } catch (e) {
        console.warn('[TTSClient] Could not fetch Voicebox voices, using default profile:', e);
      }
    }

    setRate(newRate) {
      this.rate = Math.max(0.5, Math.min(2.0, parseFloat(newRate) || 1.0));
    }

    /**
     * Stop and cancel any currently generating or playing audio.
     */
    stop() {
      this.currentRequestId = null; // Invalidate pending async responses

      if (this.activeAudio) {
        try {
          this.activeAudio.pause();
          this.activeAudio.currentTime = 0;
          this.activeAudio.src = '';
        } catch (e) {}
        this.activeAudio = null;
      }

      // Also cancel browser SpeechSynthesis if active
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }

      this.audioQueue = [];
      this._setState('cancelled');
      this._setState('idle');
    }

    /**
     * Synthesize and play text aloud.
     * Prevents overlapping audio, rejects stale requests, and handles fallback gracefully.
     *
     * @param {string} text - The natural English sentence to speak
     * @param {object} options - Optional { voiceId, rate, onStart, onEnd }
     * @returns {Promise<string>} request_id
     */
    async speak(text, options = {}) {
      if (!text || !text.trim()) return null;

      // Stop previous playback to enforce single-speaker clarity in conversation
      this.stop();

      const requestId = 'tts_' + Date.now() + '_' + Math.random().toString(36).substring(2, 7);
      this.currentRequestId = requestId;
      const voiceId = options.voiceId || this.defaultVoiceId;
      const rate = options.rate || this.rate;

      this._setState('generating', { requestId, text });

      try {
        const response = await fetch('/api/tts/synthesize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: text.trim(),
            voice_id: voiceId,
            rate: rate,
            request_id: requestId,
          }),
        });

        // Check if request was cancelled while waiting for response
        if (this.currentRequestId !== requestId) {
          console.log(`[TTSClient] Request ${requestId} was cancelled while awaiting synthesis.`);
          return requestId;
        }

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const errMsg = errorData.detail?.error || `TTS server error (${response.status})`;
          throw new Error(errMsg);
        }

        const isTTSConfigured = response.headers.get('X-Voicebox-Configured') === 'true' || response.headers.get('X-TTS-Configured') === 'true';
        if (!isTTSConfigured) {
          // Backend TTS engine is unconfigured; use native browser SpeechSynthesis for real spoken audio
          console.log('[TTSClient] Backend TTS engine unconfigured; speaking via browser SpeechSynthesis.');
          this._fallbackBrowserSpeak(text, rate, options, requestId);
          return requestId;
        }

        const audioBlob = await response.blob();

        // Check again for cancellation before playback
        if (this.currentRequestId !== requestId) {
          return requestId;
        }

        this._setState('ready', { requestId, audioBlob });
        await this._playBlob(audioBlob, requestId, text, options);
        return requestId;

      } catch (err) {
        console.warn(`[TTSClient] Backend synthesis error, falling back to Web Speech:`, err);
        // Seamless fallback to browser Web Speech API
        if (this.currentRequestId === requestId) {
          this._fallbackBrowserSpeak(text, rate, options, requestId);
        }
        return requestId;
      }
    }

    _playBlob(blob, requestId, text, options = {}) {
      return new Promise((resolve) => {
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        this.activeAudio = audio;

        this._setState('playing', { requestId, text, audioUrl });
        if (options.onStart) options.onStart();

        const cleanup = () => {
          URL.revokeObjectURL(audioUrl);
          if (this.activeAudio === audio) {
            this.activeAudio = null;
          }
        };

        audio.onended = () => {
          cleanup();
          if (this.currentRequestId === requestId) {
            this._setState('completed', { requestId, text });
            this._setState('idle');
          }
          if (options.onEnd) options.onEnd();
          resolve();
        };

        audio.onerror = (e) => {
          cleanup();
          console.warn('[TTSClient] Audio playback error:', e);
          if (this.currentRequestId === requestId) {
            // Attempt browser fallback on audio tag decode error
            this._fallbackBrowserSpeak(text, options.rate || this.rate, options, requestId);
          }
          resolve();
        };

        audio.play().catch(playErr => {
          cleanup();
          console.warn('[TTSClient] Autoplay prevented or interrupted:', playErr);
          if (this.currentRequestId === requestId) {
            this._fallbackBrowserSpeak(text, options.rate || this.rate, options, requestId);
          }
          resolve();
        });
      });
    }

    _fallbackBrowserSpeak(text, rate, options, requestId) {
      if (!('speechSynthesis' in window)) {
        this._emitError(new Error('Audio synthesis and Web Speech not supported in this browser.'));
        this._setState('idle');
        return;
      }

      try {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = rate;

        this._setState('playing', { requestId, text, provider: 'browser_speech_fallback' });
        if (options.onStart) options.onStart();

        utterance.onend = () => {
          if (this.currentRequestId === requestId) {
            this._setState('completed', { requestId, text });
            this._setState('idle');
          }
          if (options.onEnd) options.onEnd();
        };

        utterance.onerror = (err) => {
          console.warn('[TTSClient] Fallback speech error:', err);
          this._setState('idle');
          if (options.onEnd) options.onEnd();
        };

        window.speechSynthesis.speak(utterance);
      } catch (e) {
        this._setState('idle');
      }
    }
  }

  window.VoiceboxTTSClient = VoiceboxTTSClient;

})(window);
