// Siren and feedback sounds, synthesised with the Web Audio API
// (no audio files, nothing downloaded).

const MAX_GAIN = 0.6; // keeps 100% loud but not ear-damaging

export class SoundEngine {
  constructor() {
    this.ctx = null;
    this.active = null; // { master, nodes, timer }
  }

  // Must be called from a user gesture at least once (browser autoplay rules).
  unlock() {
    if (!this.ctx) this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    if (this.ctx.state === 'suspended') this.ctx.resume();
  }

  get playing() {
    return !!this.active;
  }

  /**
   * @param {object} o
   * @param {'chime'|'alarm'|'siren'|'buzzer'} o.sound
   * @param {number} o.volume 0..100
   * @param {boolean} o.escalate
   * @param {number} o.escalateSec
   */
  start({ sound, volume, escalate, escalateSec }) {
    this.unlock();
    this.stop();
    const ctx = this.ctx;
    const now = ctx.currentTime;
    const target = (volume / 100) * MAX_GAIN;
    const master = ctx.createGain();
    master.gain.setValueAtTime(0, now);
    if (escalate) {
      master.gain.linearRampToValueAtTime(target * 0.25, now + 0.08);
      master.gain.linearRampToValueAtTime(target, now + Math.max(1, escalateSec));
    } else {
      master.gain.linearRampToValueAtTime(target, now + 0.08);
    }
    master.connect(ctx.destination);

    const nodes = [];
    let timer = null;
    const osc = (type, freq) => {
      const o = ctx.createOscillator();
      o.type = type;
      o.frequency.value = freq;
      nodes.push(o);
      return o;
    };
    const gain = (v) => {
      const g = ctx.createGain();
      g.gain.value = v;
      return g;
    };

    if (sound === 'siren') {
      // Classic wail: sawtooth swept 650–1500 Hz by a slow LFO.
      const lp = ctx.createBiquadFilter();
      lp.type = 'lowpass';
      lp.frequency.value = 3200;
      lp.connect(master);
      const lfo = osc('sine', 0.45);
      const lfoAmt = gain(425);
      lfo.connect(lfoAmt);
      [0, 7].forEach((detune) => {
        const o = osc('sawtooth', 1075);
        o.detune.value = detune;
        lfoAmt.connect(o.frequency);
        const g = gain(0.35);
        o.connect(g).connect(lp);
      });
    } else if (sound === 'alarm') {
      // Fast on/off beeping.
      const o = osc('square', 1760);
      const gate = gain(0.5);
      const lfo = osc('square', 4);
      const lfoAmt = gain(0.5);
      lfo.connect(lfoAmt).connect(gate.gain);
      const tone = gain(0.35);
      o.connect(tone).connect(gate).connect(master);
    } else if (sound === 'buzzer') {
      // Low, dissonant, harsh.
      const gate = gain(0.5);
      const lfo = osc('square', 2.2);
      const lfoAmt = gain(0.5);
      lfo.connect(lfoAmt).connect(gate.gain);
      gate.connect(master);
      [130, 138.6, 196].forEach((f) => {
        const o = osc('sawtooth', f);
        o.connect(gain(0.4)).connect(gate);
      });
    } else {
      // Soft repeating chime — the low-stimulation option.
      const ring = () => this._bell(master, [660, 880], 0.7);
      ring();
      timer = setInterval(ring, 1400);
    }

    nodes.forEach((n) => n.start(now));
    this.active = { master, nodes, timer };
  }

  stop() {
    if (!this.active) return;
    const { master, nodes, timer } = this.active;
    this.active = null;
    clearInterval(timer);
    const now = this.ctx.currentTime;
    master.gain.cancelScheduledValues(now);
    master.gain.setValueAtTime(master.gain.value, now);
    master.gain.linearRampToValueAtTime(0, now + 0.12);
    setTimeout(() => {
      nodes.forEach((n) => { try { n.stop(); } catch { /* already stopped */ } });
      master.disconnect();
    }, 200);
  }

  // A short, pleasant two-note chime for warnings, breaks and praise.
  chime(volume = 50, notes = [523.25, 783.99]) {
    this.unlock();
    const g = this.ctx.createGain();
    g.gain.value = Math.max(0.05, (volume / 100) * MAX_GAIN * 0.8);
    g.connect(this.ctx.destination);
    this._bell(g, notes, 1);
    setTimeout(() => g.disconnect(), 2500);
  }

  _bell(dest, freqs, level) {
    const ctx = this.ctx;
    freqs.forEach((f, i) => {
      const t0 = ctx.currentTime + i * 0.18;
      const o = ctx.createOscillator();
      o.type = 'sine';
      o.frequency.value = f;
      const env = ctx.createGain();
      env.gain.setValueAtTime(0.0001, t0);
      env.gain.exponentialRampToValueAtTime(level, t0 + 0.02);
      env.gain.exponentialRampToValueAtTime(0.0001, t0 + 1.1);
      o.connect(env).connect(dest);
      o.start(t0);
      o.stop(t0 + 1.2);
    });
  }
}
