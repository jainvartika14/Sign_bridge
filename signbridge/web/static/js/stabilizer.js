/**
 * SignBridge Prediction Stabilization & Sign-Boundary Engine
 *
 * Implements a 4-stage state progression:
 *   1. raw prediction      — immediate per-frame inference result from model
 *   2. candidate           — leading gloss from the recent sliding vote window
 *   3. stable candidate    — candidate supported by sustained window agreement and margin
 *   4. committed gloss     — finalized sign committed to sentence buffer upon boundary/transition
 *
 * Provides conservative boundary detection, duplicate suppression, and buffer review controls.
 */

(function(window) {
  'use strict';

  // Centralized, tuneable configuration thresholds
  const STABILIZATION_CONFIG = {
    VOTE_BUFFER_SIZE: 6,               // Rolling window of recent sliding-window inferences
    MIN_CONFIDENCE_THRESHOLD: 35.0,    // Minimum confidence % to qualify as a candidate
    STABILITY_VOTE_FRACTION: 0.50,     // Required fraction of weighted votes in buffer (50%)
    MIN_CONSECUTIVE_FRAMES: 2,         // Minimum consecutive occurrences in buffer
    MIN_MARGIN_THRESHOLD: 3.0,         // Minimum Top-1 minus Top-2 confidence margin %
    DECISIVE_CONFIDENCE_THRESHOLD: 65.0,// High confidence threshold for immediate stability
    DECISIVE_MARGIN_THRESHOLD: 12.0,   // Dominant margin threshold for immediate stability
    BOUNDARY_IDLE_FRAMES: 8,           // Consecutive zero/no-hands frames marking a sign boundary
    REPEAT_COOLDOWN_MS: 1800,          // Minimum ms before identical gloss can re-commit without boundary
    LOW_CONFIDENCE_RESET_THRESHOLD: 20.0, // Confidence level below which boundary reset is flagged
  };

  class PredictionStabilizer {
    constructor(config = {}) {
      this.config = Object.assign({}, STABILIZATION_CONFIG, config);

      // Four-stage state separation
      this.rawPrediction = null;      // Stage 1: { gloss, confidence, topK, timestamp }
      this.candidate = null;          // Stage 2: { gloss, score, confidence }
      this.stableCandidate = null;    // Stage 3: { gloss, confidence, sustainedCount }
      this.lastCommittedGloss = null; // Stage 4 tracking: Most recently committed sign
      this.committedGlosses = [];     // Stage 4: Buffer of committed gloss strings

      // Temporal & Boundary State
      this.voteBuffer = [];           // Array of recent { gloss, confidence, margin, weight, timestamp }
      this.lastCommitTime = 0;
      this.idleFrameCount = 0;
      this.boundaryDetected = true;   // Start true so first sign can commit
      this.consecutiveMatches = 0;
      this.lastLeadingGloss = null;

      // Event listeners
      this.listeners = {
        candidateChange: new Set(),
        stableCandidateChange: new Set(),
        glossCommitted: new Set(),
        bufferChanged: new Set(),
        boundaryReset: new Set(),
      };
    }

    on(event, callback) {
      if (this.listeners[event]) {
        this.listeners[event].add(callback);
        return () => this.listeners[event].delete(callback);
      }
      return () => {};
    }

    _notify(event, data) {
      if (this.listeners[event]) {
        for (const fn of this.listeners[event]) {
          try { fn(data); } catch (e) { console.error(`[Stabilizer] Error in ${event} listener:`, e); }
        }
      }
    }

    /**
     * Compute margin between top-1 and top-2 predictions.
     */
    computeMargin(topK) {
      if (!topK || topK.length < 2) return topK?.[0]?.confidence || 0.0;
      return Math.max(0.0, (topK[0]?.confidence || 0.0) - (topK[1]?.confidence || 0.0));
    }

    /**
     * Record frame hand presence to detect natural sign boundaries (hands lowered/at rest).
     */
    recordHandPresence(numHands) {
      if (numHands === 0) {
        this.idleFrameCount++;
        if (this.idleFrameCount >= this.config.BOUNDARY_IDLE_FRAMES) {
          if (!this.boundaryDetected) {
            this.boundaryDetected = true;
            this._notify('boundaryReset', { reason: 'hands_lowered', idleFrames: this.idleFrameCount });
          }
        }
      } else {
        this.idleFrameCount = 0;
      }
    }

    /**
     * Process an incoming raw prediction from the GRU inference endpoint.
     * Evaluates through the 4-stage progression:
     * raw -> candidate -> stable candidate -> committed gloss.
     *
     * @param {object} result - { predicted_gloss, confidence, top_k }
     * @returns {object} Status snapshot of the stabilization pipeline
     */
    processPrediction(result) {
      if (!result || !result.predicted_gloss) {
        return this.getStatus();
      }

      const now = Date.now();
      const rawTop1 = result.predicted_gloss.trim().toUpperCase();
      const rawConf = Number(result.confidence) || 0.0;
      const margin = this.computeMargin(result.top_k);

      // Stage 1: Raw Prediction
      this.rawPrediction = {
        gloss: rawTop1,
        confidence: rawConf,
        margin: margin,
        topK: result.top_k || [],
        timestamp: now,
      };

      // Flag boundary reset if confidence drops significantly (relaxation between signs)
      if (rawConf < this.config.LOW_CONFIDENCE_RESET_THRESHOLD) {
        this.boundaryDetected = true;
      }

      // Check if prediction meets minimum confidence to enter stabilization window
      if (rawConf < this.config.MIN_CONFIDENCE_THRESHOLD) {
        // Decay vote buffer slightly during ambiguous frames
        if (this.voteBuffer.length > 0) {
          this.voteBuffer.shift();
        }
        return this.getStatus();
      }

      // Weight calculation: decisive margins receive up to 2.0x weight amplification
      const marginMultiplier = 1.0 + Math.min(Math.max(margin, 0.0) / 25.0, 1.0);
      const voteWeight = rawConf * marginMultiplier;

      // Add to sliding vote buffer
      this.voteBuffer.push({
        gloss: rawTop1,
        confidence: rawConf,
        margin: margin,
        weight: voteWeight,
        topK: result.top_k,
        timestamp: now,
      });

      if (this.voteBuffer.length > this.config.VOTE_BUFFER_SIZE) {
        this.voteBuffer.shift();
      }

      // Aggregate weighted votes across the sliding buffer
      let totalWeight = 0;
      const votes = {};
      const occurrences = {};

      for (const v of this.voteBuffer) {
        totalWeight += v.weight;
        votes[v.gloss] = (votes[v.gloss] || 0) + v.weight;
        occurrences[v.gloss] = (occurrences[v.gloss] || 0) + 1;
      }

      // Stage 2: Candidate Selection (highest weighted gloss in buffer)
      let leadingGloss = rawTop1;
      let highestScore = 0;
      for (const [gloss, score] of Object.entries(votes)) {
        if (score > highestScore) {
          highestScore = score;
          leadingGloss = gloss;
        }
      }

      const candidateOccurrences = occurrences[leadingGloss] || 1;
      const candidateFraction = totalWeight > 0 ? (highestScore / totalWeight) : 0;

      // Track consecutive occurrences
      if (leadingGloss === this.lastLeadingGloss) {
        this.consecutiveMatches++;
      } else {
        this.consecutiveMatches = 1;
        this.lastLeadingGloss = leadingGloss;
      }

      this.candidate = {
        gloss: leadingGloss,
        score: highestScore,
        confidence: rawConf,
        fraction: candidateFraction,
        consecutive: this.consecutiveMatches,
      };
      this._notify('candidateChange', this.candidate);

      // Stage 3: Stability Evaluation
      // A candidate becomes stable if:
      // 1. Sustained agreement: >= MIN_CONSECUTIVE_FRAMES AND >= STABILITY_VOTE_FRACTION of buffer, OR
      // 2. Decisive prediction: high confidence (>= DECISIVE_CONFIDENCE_THRESHOLD) and strong margin
      const isSustainedStable = (candidateOccurrences >= this.config.MIN_CONSECUTIVE_FRAMES) &&
                                (candidateFraction >= this.config.STABILITY_VOTE_FRACTION) &&
                                (margin >= this.config.MIN_MARGIN_THRESHOLD);

      const isDecisiveStable = (rawConf >= this.config.DECISIVE_CONFIDENCE_THRESHOLD) &&
                               (margin >= this.config.DECISIVE_MARGIN_THRESHOLD);

      let justStabilized = false;
      if (isSustainedStable || isDecisiveStable) {
        const bestEntry = this.voteBuffer
          .filter(v => v.gloss === leadingGloss)
          .reduce((best, v) => (v.confidence > (best?.confidence ?? 0) ? v : best), null);

        this.stableCandidate = {
          gloss: leadingGloss,
          confidence: bestEntry ? bestEntry.confidence : rawConf,
          margin: bestEntry ? bestEntry.margin : margin,
          timestamp: now,
        };
        justStabilized = true;
        this._notify('stableCandidateChange', this.stableCandidate);
      }

      // Stage 4: Sign Commit & Duplicate Suppression
      let justCommitted = false;
      let committedGloss = null;

      if (justStabilized && this.stableCandidate) {
        const stableGloss = this.stableCandidate.gloss;

        // Condition A: Brand new sign (first sign in buffer or transition to different gloss)
        if (this.lastCommittedGloss === null || stableGloss !== this.lastCommittedGloss) {
          committedGloss = this._commitGloss(stableGloss);
          justCommitted = true;
        }
        // Condition B: Same sign repeated — only commit if boundary occurred or cooldown elapsed
        else if (stableGloss === this.lastCommittedGloss) {
          const elapsedMs = now - this.lastCommitTime;
          if (this.boundaryDetected && elapsedMs >= this.config.REPEAT_COOLDOWN_MS) {
            committedGloss = this._commitGloss(stableGloss);
            justCommitted = true;
          } else {
            // SUPPRESS DUPLICATE: User is continuing to hold the sign.
            // Do NOT re-commit while held continuously.
          }
        }
      }

      return {
        rawPrediction: this.rawPrediction,
        candidate: this.candidate,
        stableCandidate: this.stableCandidate,
        lastCommittedGloss: this.lastCommittedGloss,
        committedGlosses: [...this.committedGlosses],
        justStabilized,
        justCommitted,
        committedGloss,
      };
    }

    _commitGloss(gloss) {
      const clean = gloss.trim().toUpperCase();
      this.committedGlosses.push(clean);
      this.lastCommittedGloss = clean;
      this.lastCommitTime = Date.now();
      this.boundaryDetected = false; // Reset boundary flag until next reset event

      this._notify('glossCommitted', { gloss: clean, allGlosses: [...this.committedGlosses] });
      this._notify('bufferChanged', [...this.committedGlosses]);
      return clean;
    }

    /**
     * Manually commit a sign (e.g. from user "+ Add to Sentence" button click).
     */
    commitManual(gloss) {
      if (!gloss || gloss === '—' || gloss === 'Analyzing…') return null;
      return this._commitGloss(gloss);
    }

    /**
     * Remove the most recently committed gloss (Undo / Backspace).
     */
    undoLastCommitted() {
      if (this.committedGlosses.length === 0) return null;
      const removed = this.committedGlosses.pop();
      // Update last committed gloss to previous element
      this.lastCommittedGloss = this.committedGlosses.length > 0
        ? this.committedGlosses[this.committedGlosses.length - 1]
        : null;
      this.boundaryDetected = true; // Allow re-committing after undo
      this._notify('bufferChanged', [...this.committedGlosses]);
      return removed;
    }

    /**
     * Remove a committed gloss at a specific index.
     */
    removeCommittedAt(index) {
      if (index >= 0 && index < this.committedGlosses.length) {
        const removed = this.committedGlosses.splice(index, 1)[0];
        this.lastCommittedGloss = this.committedGlosses.length > 0
          ? this.committedGlosses[this.committedGlosses.length - 1]
          : null;
        this.boundaryDetected = true;
        this._notify('bufferChanged', [...this.committedGlosses]);
        return removed;
      }
      return null;
    }

    /**
     * Clear all committed glosses from the buffer.
     */
    clearCommitted() {
      this.committedGlosses = [];
      this.lastCommittedGloss = null;
      this.boundaryDetected = true;
      this._notify('bufferChanged', []);
    }

    /**
     * Reset vote buffer and temporal tracking state (e.g., when camera stops or restarts).
     */
    resetTracking() {
      this.voteBuffer = [];
      this.rawPrediction = null;
      this.candidate = null;
      this.stableCandidate = null;
      this.consecutiveMatches = 0;
      this.lastLeadingGloss = null;
      this.boundaryDetected = true;
      this.idleFrameCount = 0;
    }

    getStatus() {
      return {
        rawPrediction: this.rawPrediction,
        candidate: this.candidate,
        stableCandidate: this.stableCandidate,
        lastCommittedGloss: this.lastCommittedGloss,
        committedGlosses: [...this.committedGlosses],
        boundaryDetected: this.boundaryDetected,
      };
    }
  }

  window.SignBridgeStabilizer = PredictionStabilizer;

})(typeof window !== 'undefined' ? window : this);
