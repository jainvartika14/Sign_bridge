/**
 * Two-Way Real-Time Conversation Controller for SignBridge.
 *
 * Coordinates:
 * - Communication modes (idle, sign_to_speech, speech_to_text, two_way, stopping, error)
 * - Independent subsystem state machines (Sign Recognition, TTS, STT, Microphone, Audio Output)
 * - Synchronized conversation history with speaker attribution
 * - Acoustic echo suppression gating (prevents TTS output from self-transcribing)
 * - Request session tokens and stale update rejection
 */

(function(window) {
  'use strict';

  class ConversationController {
    constructor(options = {}) {
      this.mode = 'idle'; // idle | sign_to_speech | speech_to_text | two_way | stopping | error
      this.sessionId = null;
      this.activeSpeaker = null; // null | 'sign_user' | 'hearing_partner'

      this.conversationState = 'idle'; // idle | signing | listening | processing | reviewing | speaking

      // Subsystem States
      this.subsystems = {
        signRecognition: 'ready', // initializing | ready | listening | processing | prediction_available | paused | error
        tts: 'idle',             // idle | queued | generating | ready | playing | completed | cancelled | error
        stt: 'idle',             // idle | requesting_permission | connecting | listening | receiving_partial_transcript | finalizing | reconnecting | stopped | error
        microphone: 'unavailable', // unavailable | permission_required | permission_granted | active | muted | disconnected
        audioPlayback: 'idle',   // idle | playing | paused | error
      };

      // Conversation Data (Session-local in memory; recordings are never stored permanently)
      this.messages = [];
      this.currentPartialTranscript = '';
      this.pendingSignGlosses = [];
      this.lastRecognizedGloss = null;
      this.lastRecognizedTime = 0;

      // Clients
      this.ttsClient = options.ttsClient || (window.VoiceboxTTSClient ? new window.VoiceboxTTSClient() : null);
      this.sttClient = options.sttClient || (window.STTClient ? new window.STTClient() : null);

      // Event Subscriptions
      this.listeners = {
        modeChange: new Set(),
        stateChange: new Set(),
        subsystemChange: new Set(),
        messageAdded: new Set(),
        messageUpdated: new Set(),
        partialTranscript: new Set(),
        historyCleared: new Set(),
        error: new Set(),
      };

      this._initSubsystemHooks();
    }

    _initSubsystemHooks() {
      if (this.ttsClient) {
        this.ttsClient.onStateChange((state, details) => {
          this.subsystems.tts = state;
          this.subsystems.audioPlayback = (state === 'playing') ? 'playing' : 'idle';

          // Acoustic Echo Suppression:
          // When Voicebox TTS begins speaking, mute/suppress STT microphone capture
          // to prevent the speaker audio from being transcribed as a hearing user message.
          if (state === 'playing') {
            if (this.sttClient) this.sttClient.setEchoSuppression(true);
            this.activeSpeaker = 'sign_user';
          } else if (state === 'completed' || state === 'idle' || state === 'cancelled') {
            if (this.sttClient) this.sttClient.setEchoSuppression(false);
            if (this.activeSpeaker === 'sign_user') this.activeSpeaker = null;
          }

          this._notifySubsystemChange('tts', state, details);
        });

        this.ttsClient.onError((err) => {
          this._notifyError('TTS', err);
        });
      }

      if (this.sttClient) {
        this.sttClient.onStateChange((sttState, micState) => {
          this.subsystems.stt = sttState;
          if (micState) this.subsystems.microphone = micState;

          if (sttState === 'receiving_partial_transcript' || sttState === 'listening') {
            if (this.subsystems.audioPlayback !== 'playing') {
              this.activeSpeaker = 'hearing_partner';
            }
          }

          this._notifySubsystemChange('stt', sttState);
          if (micState) this._notifySubsystemChange('microphone', micState);
        });

        this.sttClient.onPartial((interimText) => {
          if (this.subsystems.audioPlayback === 'playing') return; // Gate echo
          this.currentPartialTranscript = interimText;
          for (const fn of this.listeners.partialTranscript) {
            try { fn(interimText); } catch (e) {}
          }
        });

        this.sttClient.onFinal((finalText, confidence) => {
          if (this.subsystems.audioPlayback === 'playing') return; // Gate echo
          this.currentPartialTranscript = '';
          for (const fn of this.listeners.partialTranscript) {
            try { fn(''); } catch (e) {}
          }
          this._handleHearingPartnerFinalMessage(finalText, confidence);
        });

        this.sttClient.onError((err) => {
          this._notifyError('STT', err);
        });
      }
    }

    // ------------------------------------------------------------------------
    // Event Subscription API
    // ------------------------------------------------------------------------
    on(event, callback) {
      if (this.listeners[event]) {
        this.listeners[event].add(callback);
        return () => this.listeners[event].delete(callback);
      }
      return () => {};
    }

    _notifyModeChange(mode) {
      for (const fn of this.listeners.modeChange) {
        try { fn(mode); } catch (e) {}
      }
    }

    _notifySubsystemChange(subsystem, state, details = {}) {
      for (const fn of this.listeners.subsystemChange) {
        try { fn(subsystem, state, details); } catch (e) {}
      }
    }

    _notifyError(source, error) {
      for (const fn of this.listeners.error) {
        try { fn(source, error); } catch (e) {}
      }
    }

    // ------------------------------------------------------------------------
    // Communication Mode Management
    // ------------------------------------------------------------------------
    setMode(newMode) {
      if (this.mode === newMode) return;
      const prevMode = this.mode;
      this.mode = newMode;
      console.log(`[ConversationController] Mode transition: ${prevMode} -> ${newMode}`);

      if (newMode === 'two_way') {
        this._activateTwoWayMode();
      } else if (newMode === 'sign_to_speech') {
        this._activateSignToSpeechMode();
      } else if (newMode === 'speech_to_text') {
        this._activateSpeechToTextMode();
      } else if (newMode === 'idle' || newMode === 'stopping') {
        this._deactivateAll();
      }

      this._notifyModeChange(newMode);
    }

    _activateTwoWayMode() {
      this.sessionId = 'conv_' + Date.now();
      // Activate microphone and STT
      if (this.sttClient) {
        this.sttClient.start();
      }
      this.subsystems.signRecognition = 'listening';
      this._notifySubsystemChange('signRecognition', 'listening');
    }

    _activateSignToSpeechMode() {
      // Pause STT microphone
      if (this.sttClient) {
        this.sttClient.stop();
      }
      this.subsystems.signRecognition = 'listening';
      this._notifySubsystemChange('signRecognition', 'listening');
    }

    _activateSpeechToTextMode() {
      // Pause sign recognition activity
      this.subsystems.signRecognition = 'paused';
      this._notifySubsystemChange('signRecognition', 'paused');

      if (this.sttClient) {
        this.sttClient.start();
      }
    }

    _deactivateAll() {
      if (this.ttsClient) this.ttsClient.stop();
      if (this.sttClient) this.sttClient.stop();
      this.subsystems.signRecognition = 'paused';
      this.activeSpeaker = null;
      this.currentPartialTranscript = '';
      this._notifySubsystemChange('signRecognition', 'paused');
    }

    setConversationState(newState) {
      if (this.conversationState === newState) return;
      const oldState = this.conversationState;
      this.conversationState = newState;
      console.log(`[ConversationController] State: ${oldState} -> ${newState}`);
      for (const fn of this.listeners.stateChange) {
        try { fn(newState, oldState); } catch (e) {}
      }
    }

    // ------------------------------------------------------------------------
    // Sign-to-Speech Flow
    // ------------------------------------------------------------------------
    /**
     * Process a candidate or stable sign prediction from the camera recognition pipeline.
     */
    receiveSignPrediction(gloss, confidence) {
      if (this.mode !== 'two_way' && this.mode !== 'sign_to_speech') return;
      if (!gloss || gloss === '—' || gloss === 'Analyzing…') return;

      this.setConversationState('signing');
      this.subsystems.signRecognition = 'prediction_available';
      this._notifySubsystemChange('signRecognition', 'prediction_available', { gloss, confidence });
    }

    /**
     * Process a finalized, stabilized gloss committed by the PredictionStabilizer.
     * Appends to pendingSignGlosses and transitions state to 'reviewing'.
     */
    receiveCommittedGloss(gloss) {
      if (this.mode !== 'two_way' && this.mode !== 'sign_to_speech') return;
      if (!gloss || gloss === '—' || gloss === 'Analyzing…') return;

      const cleanGloss = gloss.trim().toUpperCase();
      this.pendingSignGlosses.push(cleanGloss);
      this.setConversationState('reviewing');
      this._notifySubsystemChange('signRecognition', 'prediction_available', { gloss: cleanGloss });
    }

    /**
     * Finalize and speak the sign user's assembled sentence.
     * Integrates Ollama Qwen3:8B LLM translation and Kokoro TTS audio playback.
     * Gloss buffer is only cleared AFTER successful LLM translation AND successful TTS initiation.
     *
     * @param {Array|null} explicitGlosses - List of glosses to commit
     * @param {string|null} explicitReviewedText - User-reviewed or edited English sentence
     * @returns {Promise<boolean>} Success status
     */
    async commitSignUserSentence(explicitGlosses = null, explicitReviewedText = null) {
      if (this.conversationState === 'processing' || this.conversationState === 'speaking') {
        console.warn('[ConversationController] Cannot speak: Already processing or speaking sentence.');
        return false;
      }

      const glossesToCommit = explicitGlosses || [...this.pendingSignGlosses];
      const hasReviewedText = explicitReviewedText && explicitReviewedText.trim().length > 0;

      if ((!glossesToCommit || glossesToCommit.length === 0) && !hasReviewedText) {
        console.warn('[ConversationController] Cannot speak: Gloss buffer is empty.');
        this._notifyError('Speak', new Error('Gloss buffer is empty. Perform signs or type words first.'));
        return false;
      }

      // Formulate natural English sentence via Ollama Qwen3:8B LLM endpoint (or use explicit reviewed text)
      let naturalEnglish = hasReviewedText ? explicitReviewedText.trim() : '';

      if (!naturalEnglish) {
        try {
          this.setConversationState('processing');
          const res = await fetch('/api/nlp/smooth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ glosses: glossesToCommit, use_llm: true }),
          });

          if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            const errMsg = errData.detail?.error || `Ollama LLM Translation Error (${res.status})`;
            throw new Error(errMsg);
          }

          const data = await res.json();
          naturalEnglish = data.english || '';
        } catch (llmErr) {
          console.error('[ConversationController] Ollama LLM error:', llmErr);
          this._notifyError('OllamaLLM', llmErr);
          this.setConversationState('idle');
          // PRESERVE GLOSS BUFFER ON LLM FAILURE
          return false;
        }
      }

      if (!naturalEnglish) {
        const err = new Error('Generated sentence was empty.');
        this._notifyError('OllamaLLM', err);
        this.setConversationState('idle');
        return false;
      }

      this.setConversationState('speaking');
      const messageId = 'msg_sign_' + Date.now();
      const message = {
        id: messageId,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        speaker: 'sign_user',
        text: naturalEnglish,
        rawGlosses: glossesToCommit,
        status: 'synthesizing', // synthesizing | ready | playing | failed
        audioRequestId: null,
      };

      // Send to Kokoro TTS client
      if (this.ttsClient) {
        try {
          const reqId = await this.ttsClient.speak(naturalEnglish, {
            onStart: () => {
              message.status = 'playing';
              this.setConversationState('speaking');
              this._notifyMessageUpdated(message);
            },
            onEnd: () => {
              message.status = 'ready';
              this.setConversationState('idle');
              this._notifyMessageUpdated(message);
            },
          });
          message.audioRequestId = reqId;
          
          // Successful sentence generation + TTS initiation: commit message and clear buffer
          this.pendingSignGlosses = [];
          this.messages.push(message);
          this._notifyMessageAdded(message);
          return true;

        } catch (ttsErr) {
          console.error('[ConversationController] Kokoro TTS error:', ttsErr);
          message.status = 'failed';
          this.setConversationState('idle');
          this._notifyError('KokoroTTS', ttsErr);
          // PRESERVE GLOSS BUFFER ON TTS FAILURE
          return false;
        }
      } else {
        // Fallback without TTS client
        this.pendingSignGlosses = [];
        this.messages.push(message);
        this._notifyMessageAdded(message);
        this.setConversationState('idle');
        return true;
      }
    }

    // ------------------------------------------------------------------------
    // Speech-to-Text Flow
    // ------------------------------------------------------------------------
    _handleHearingPartnerFinalMessage(text, confidence) {
      if (!text || !text.trim()) return;

      // Extract matched ASL glosses for visual comprehension
      let matchedGlosses = [];
      if (window.SignBridgeNLP && window.SignBridgeNLP.extractASLGlossesFromSpeech) {
        matchedGlosses = window.SignBridgeNLP.extractASLGlossesFromSpeech(text, window.allASLGlossesSet);
      }

      const messageId = 'msg_hearing_' + Date.now();
      const message = {
        id: messageId,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        speaker: 'hearing_partner',
        text: text.trim(),
        matchedGlosses: matchedGlosses,
        confidence: Math.round(confidence * 100),
        status: 'finalized',
      };

      this.messages.push(message);
      this._notifyMessageAdded(message);
    }

    _notifyMessageAdded(message) {
      for (const fn of this.listeners.messageAdded) {
        try { fn(message); } catch (e) {}
      }
    }

    _notifyMessageUpdated(message) {
      for (const fn of this.listeners.messageUpdated) {
        try { fn(message); } catch (e) {}
      }
    }

    clearHistory() {
      this.messages = [];
      this.pendingSignGlosses = [];
      this.currentPartialTranscript = '';
      for (const fn of this.listeners.historyCleared) {
        try { fn(); } catch (e) {}
      }
    }

    exportTranscriptText() {
      if (this.messages.length === 0) return 'No conversation messages recorded.';
      return this.messages.map(m => {
        const speakerLabel = m.speaker === 'sign_user' ? '🤟 Deaf/Signing Participant' : '🎙️ Hearing Partner';
        const glossNote = m.rawGlosses ? ` [ASL Glosses: ${m.rawGlosses.join(', ')}]` : '';
        return `[${m.timestamp}] ${speakerLabel}: "${m.text}"${glossNote}`;
      }).join('\n\n');
    }
  }

  window.ConversationController = ConversationController;

})(window);
