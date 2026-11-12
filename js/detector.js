// On-device hand + phone detection with MediaPipe Tasks Vision.
// The library, its WebAssembly runtime and both models are served from this
// site (vendor/ and models/), so no frame or request ever leaves the device.
//
// How a phone in a hand is found (tuned on real sideways / edge-on holds):
// - Hand landmarks every frame, with low thresholds: a hand gripping a phone
//   hides most of the palm the hand model looks for.
// - The phone detector sees its input at only 320x320 px, so a phone held at
//   desk distance is tiny in the whole frame. It also looks at square crops
//   around each hand (biased toward the fingertips), where the phone is larger.
// - Crops are checked upright, at 90°/270°, and at 45°/135° — landscape and
//   diagonally tilted phones need those angles to match EfficientDet.
// - In a hand crop a "remote" also counts: zoomed in, a held phone is often
//   called one; a mouse almost never is.
// - When the object model still misses (common for edge-on phones), a grip
//   estimator built from hand landmarks synthesises a phone box — that is the
//   "taught" fallback for holds the COCO model was not trained on.
// - Phones are tracked between passes with hysteresis so a box near the
//   threshold does not flicker.

import {
  FilesetResolver,
  HandLandmarker,
  ObjectDetector
} from '../vendor/mediapipe/vision_bundle.mjs';

const WASM_PATH = new URL('../vendor/mediapipe/wasm', import.meta.url).href;
const HAND_MODEL = new URL('../models/hand_landmarker.task', import.meta.url).href;
const OBJECT_MODEL = new URL('../models/efficientdet_lite0.tflite', import.meta.url).href;

const OBJECT_INPUT = 320;   // EfficientDet-Lite0 input size
const HAND_EVERY_MS = 30;   // hand landmarks at most ~33 times a second
const PHONE_EVERY_MS = 90;  // phone pass at most ~11 times a second (slower on slow devices)
const TRACK_TTL_MS = 450;   // forget a phone that has not been seen for this long
const KEEP_RATIO = 0.55;    // a tracked phone stays while its score >= start threshold x this
const CROP_SCALE = 4.2;     // crop side = hand size x this (sideways phones stick out past the palm)
const RIGHT_ANGLES = [90, 270];
const DIAG_ANGLES = [45, 135]; // physical rotate — MediaPipe only accepts 90° steps via API

// Score a new phone needs, per source.
// grip = confidence from the hand-landmark estimator (0..1), not an OD score.
export const SENSITIVITY_THRESHOLD = {
  low: { full: 0.4, crop: 0.28, grip: 0.78 },
  normal: { full: 0.28, crop: 0.16, grip: 0.52 },
  high: { full: 0.18, crop: 0.1, grip: 0.4 }
};

// Fingertips + palm landmarks used to decide whether a hand touches the phone.
const CONTACT_POINTS = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];

// Prefetch model bytes as soon as the module loads so "Start session" can
// often skip the network wait (camera permission prompt overlaps with this).
const modelBuffers = Promise.all([
  fetch(HAND_MODEL).then((r) => { if (!r.ok) throw new Error(r.statusText); return r.arrayBuffer(); }),
  fetch(OBJECT_MODEL).then((r) => { if (!r.ok) throw new Error(r.statusText); return r.arrayBuffer(); })
]).then(([hand, object]) => ({
  hand: new Uint8Array(hand),
  object: new Uint8Array(object)
}));

export class PhoneDetector {
  constructor() {
    this.hands = null;
    this.objects = null;
    this.lastTs = -1;
    this.handAt = -Infinity;
    this.phoneAt = -Infinity;
    this.phoneEvery = PHONE_EVERY_MS;
    this.pass = 0;
    this.lastHands = [];
    this.tracks = [];
    this.failures = 0;
    this.broken = false;
    this.crop = document.createElement('canvas');
    this.crop.width = this.crop.height = OBJECT_INPUT;
    this.cropCtx = this.crop.getContext('2d', { willReadFrequently: false });
    this.rot = document.createElement('canvas');
    this.rot.width = this.rot.height = OBJECT_INPUT;
    this.rotCtx = this.rot.getContext('2d', { willReadFrequently: false });
  }

  async init(threshold, { delegate } = {}) {
    this.threshold = threshold;
    const [, fileset] = await Promise.all([
      modelBuffers,
      FilesetResolver.forVisionTasks(WASM_PATH)
    ]);
    this.fileset = fileset;
    this.buffers = await modelBuffers;
    if (delegate) {
      await this.create(delegate);
    } else {
      try {
        if (!hasWebGL2()) throw new Error('WebGL2 unavailable');
        await this.create('GPU');
      } catch (e) {
        console.warn('GPU delegate unavailable, using CPU', e);
        await this.create('CPU');
      }
    }
  }

  async create(delegate) {
    // Copy buffers: MediaPipe may transfer/detach the underlying ArrayBuffer.
    const handBuf = this.buffers.hand.slice();
    const objectBuf = this.buffers.object.slice();
    const [hands, objects] = await Promise.all([
      HandLandmarker.createFromOptions(this.fileset, {
        baseOptions: { modelAssetBuffer: handBuf, delegate },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.15,
        minHandPresenceConfidence: 0.15,
        minTrackingConfidence: 0.15
      }),
      // IMAGE mode: stateless, so we can run upright + rotated crops each pass.
      // Low floor: real thresholds (and the grip fallback) decide what counts.
      ObjectDetector.createFromOptions(this.fileset, {
        baseOptions: { modelAssetBuffer: objectBuf, delegate },
        runningMode: 'IMAGE',
        scoreThreshold: 0.08,
        maxResults: 8,
        categoryAllowlist: ['cell phone', 'remote']
      })
    ]);
    this.closeTasks();
    this.hands = hands;
    this.objects = objects;
    this.delegate = delegate;
  }

  // GPU init can succeed and still fail on every frame (blocklisted or lost
  // WebGL). Switch to CPU once instead of silently detecting nothing.
  fallBackToCpu(err) {
    if (this.delegate !== 'GPU' || this.switching) return;
    console.warn('GPU detection failed, switching to CPU', err);
    this.switching = true;
    this.create('CPU').catch(console.error).finally(() => { this.switching = false; });
  }

  setThreshold(threshold) {
    this.threshold = threshold;
  }

  /**
   * Runs detection on the current frame of a <video> (or a canvas).
   * Coordinates are normalised to 0..1.
   */
  detect(source, now = performance.now()) {
    const w = source.videoWidth ?? source.width;
    const h = source.videoHeight ?? source.height;
    if (!this.hands || this.switching || this.broken || source.readyState < 2 || !w) return null;
    try {
      const res = this.run(source, w, h, now);
      this.failures = 0;
      return res;
    } catch (e) {
      if (this.delegate === 'GPU') this.fallBackToCpu(e);
      else if (++this.failures >= 20) { this.broken = true; console.error('Detection unavailable', e); }
      return null;
    }
  }

  run(source, w, h, now) {
    if (now - this.handAt >= HAND_EVERY_MS) {
      this.handAt = now;
      const ts = Math.max(now, this.lastTs + 1);
      this.lastTs = ts;
      this.lastHands = this.hands.detectForVideo(source, ts).landmarks || [];
    }
    if (now - this.phoneAt >= this.phoneEvery) {
      this.phoneAt = now;
      const t0 = performance.now();
      this.phonePass(source, w, h, now);
      // On slow devices keep the phone pass to about half of the frame budget.
      this.phoneEvery = Math.min(600, Math.max(PHONE_EVERY_MS, 2 * (performance.now() - t0)));
    }
    // Keep tracks at least across one adaptive phone interval; on slow devices
    // phoneEvery can grow past TRACK_TTL_MS and would otherwise drop a valid hit.
    const ttl = Math.max(TRACK_TTL_MS, this.phoneEvery * 2);
    this.tracks = this.tracks.filter((tr) => now - tr.seen <= ttl);

    const phones = this.tracks.map((tr) => {
      const inHand = tr.src === 'grip' || this.lastHands.some((hand) => touches(hand, tr.b));
      return {
        x: tr.b[0], y: tr.b[1], w: tr.b[2] - tr.b[0], h: tr.b[3] - tr.b[1], score: tr.s,
        inHand, src: tr.src
      };
    });
    return { hands: this.lastHands, phones, phoneVisible: phones.length > 0, inHand: phones.some((p) => p.inHand) };
  }

  phonePass(source, w, h, now) {
    const found = [];
    const rects = cropsFor(this.lastHands, w, h);

    for (let i = 0; i < rects.length; i++) {
      const r = rects[i];
      const hand = this.lastHands[r.handIndex] || this.lastHands[0];
      this.cropCtx.drawImage(source, r.x, r.y, r.s, r.s, 0, 0, OBJECT_INPUT, OBJECT_INPUT);
      const mapCrop = (x, y, bw, bh) => {
        const k = r.s / OBJECT_INPUT;
        return [(r.x + x * k) / w, (r.y + y * k) / h, (r.x + (x + bw) * k) / w, (r.y + (y + bh) * k) / h];
      };
      const before = found.length;
      collect(this.objects.detect(this.crop), found, 'crop', mapCrop);

      if (!hasAccepted(found.slice(before), this.threshold)) {
        for (const deg of RIGHT_ANGLES) {
          collect(this.objects.detect(this.crop, { rotationDegrees: deg }), found, 'crop', mapCrop);
          if (hasAccepted(found.slice(before), this.threshold)) break;
        }
      }

      // Diagonally tilted / edge-on phones need non-90° views. Boxes are mapped
      // back from the physically rotated crop into crop pixel space, then frame.
      if (!hasAccepted(found.slice(before), this.threshold)) {
        for (const deg of DIAG_ANGLES) {
          collect(this.detectCropAngle(deg), found, 'crop', mapCrop);
          if (hasAccepted(found.slice(before), this.threshold)) break;
        }
      }

      // Taught fallback: object model still blind → estimate phone from grip.
      if (!hasAccepted(found.slice(before), this.threshold) && hand) {
        const grip = gripPhoneFromHand(hand);
        if (grip && grip.s >= this.threshold.grip) {
          found.push({ cls: 'cell phone', s: grip.s, src: 'grip', b: grip.b });
        }
      }
    }

    // Whole frame every other pass (every pass when there is no hand): finds
    // phones away from hands, for "any phone" mode and "out of sight".
    if (!rects.length || this.pass++ % 2 === 0) {
      collect(this.objects.detect(source), found, 'full',
        (x, y, bw, bh) => [x / w, y / h, (x + bw) / w, (y + bh) / h]);
      if (!hasAccepted(found, this.threshold)) {
        for (const deg of RIGHT_ANGLES) {
          collect(this.objects.detect(source, { rotationDegrees: deg }), found, 'full',
            (x, y, bw, bh) => [x / w, y / h, (x + bw) / w, (y + bh) / h]);
          if (hasAccepted(found, this.threshold)) break;
        }
      }
    }

    // If there were hands but every crop+OD path failed, still try grip on each.
    if (this.lastHands.length && !hasAccepted(found, this.threshold)) {
      for (const hand of this.lastHands) {
        const grip = gripPhoneFromHand(hand);
        if (grip && grip.s >= this.threshold.grip) {
          found.push({ cls: 'cell phone', s: grip.s, src: 'grip', b: grip.b });
        }
      }
    }

    this.updateTracks(found, now);
  }

  // Run OD on this.crop rotated by `deg` degrees CW; return detections whose
  // bounding boxes are already mapped back into upright crop pixel space.
  detectCropAngle(deg) {
    const S = OBJECT_INPUT;
    const ctx = this.rotCtx;
    ctx.clearRect(0, 0, S, S);
    ctx.save();
    ctx.translate(S / 2, S / 2);
    ctx.rotate((deg * Math.PI) / 180);
    ctx.drawImage(this.crop, -S / 2, -S / 2);
    ctx.restore();
    const result = this.objects.detect(this.rot);
    const rad = (-deg * Math.PI) / 180;
    const cos = Math.cos(rad), sin = Math.sin(rad);
    const mapped = { detections: [] };
    for (const d of result.detections || []) {
      const b = d.boundingBox;
      const corners = [
        [b.originX, b.originY],
        [b.originX + b.width, b.originY],
        [b.originX, b.originY + b.height],
        [b.originX + b.width, b.originY + b.height]
      ].map(([x, y]) => {
        const dx = x - S / 2, dy = y - S / 2;
        return [S / 2 + dx * cos - dy * sin, S / 2 + dx * sin + dy * cos];
      });
      const xs = corners.map((p) => p[0]), ys = corners.map((p) => p[1]);
      const x0 = Math.min(...xs), y0 = Math.min(...ys);
      mapped.detections.push({
        categories: d.categories,
        boundingBox: { originX: x0, originY: y0, width: Math.max(...xs) - x0, height: Math.max(...ys) - y0 }
      });
    }
    return mapped;
  }

  updateTracks(found, now) {
    const phones = found
      .filter((d) => d.cls === 'cell phone' || d.src === 'grip' || (d.cls === 'remote' && d.src === 'crop'))
      .sort((a, b) => b.s - a.s);
    for (const d of phones) {
      const start = d.src === 'grip' ? this.threshold.grip : this.threshold[d.src];
      // When hands are visible, ignore full-frame hits that are nowhere near a
      // hand — rotated frames often invent "phones" from UI chrome / clutter.
      if (d.src === 'full' && this.lastHands.length &&
          !this.lastHands.some((hand) => touches(hand, d.b))) {
        continue;
      }
      const tr = this.tracks.find((t) => iou(t.b, d.b) > 0.25);
      if (tr) {
        if (tr.seen !== now && d.s >= start * KEEP_RATIO) {
          tr.b = tr.b.map((v, i) => v + 0.55 * (d.b[i] - v));
          tr.s = Math.max(tr.s * 0.4, d.s);
          tr.seen = now;
          // Prefer remembering a real OD hit over a grip estimate.
          if (d.src !== 'grip' || tr.src === 'grip') tr.src = d.src;
        }
      } else if (d.s >= start) {
        this.tracks.push({ b: d.b, s: d.s, seen: now, src: d.src });
      }
    }
  }

  closeTasks() {
    this.hands?.close();
    this.objects?.close();
    this.hands = this.objects = null;
  }

  close() {
    this.closeTasks();
  }
}

function hasWebGL2() {
  try {
    return !!document.createElement('canvas').getContext('webgl2');
  } catch {
    return false;
  }
}

// A detection that would be kept by updateTracks at or above the source threshold.
function hasAccepted(found, threshold) {
  return found.some((d) => {
    if (d.src === 'grip') return d.s >= threshold.grip;
    if (d.cls === 'cell phone') return d.s >= threshold[d.src];
    if (d.cls === 'remote' && d.src === 'crop') return d.s >= threshold.crop;
    return false;
  });
}

function collect(result, out, src, map) {
  for (const d of result.detections || []) {
    const c = d.categories[0];
    const b = d.boundingBox;
    out.push({ cls: c.categoryName, s: c.score, src, b: map(b.originX, b.originY, b.width, b.height) });
  }
}

function iou(a, b) {
  const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const i = ix * iy;
  const u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
  return u > 0 ? i / u : 0;
}

function handBox(hand, w, h) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const p of hand) {
    x0 = Math.min(x0, p.x * w); y0 = Math.min(y0, p.y * h);
    x1 = Math.max(x1, p.x * w); y1 = Math.max(y1, p.y * h);
  }
  return { x0, y0, x1, y1 };
}

// Square crops around each hand, shifted toward the fingertips so a sideways
// phone that sticks out past the palm still sits inside the crop.
function cropsFor(hands, w, h) {
  const frame = Math.min(w, h);
  let rects = hands.map((hand, handIndex) => {
    const b = handBox(hand, w, h);
    const s = Math.min(frame, Math.max(0.22 * frame, CROP_SCALE * Math.max(b.x1 - b.x0, b.y1 - b.y0)));
    // Bias toward fingertips (middle MCP ← wrist).
    const dirX = (hand[9].x - hand[0].x) * w;
    const dirY = (hand[9].y - hand[0].y) * h;
    const len = Math.hypot(dirX, dirY) || 1;
    const bias = 0.28 * s;
    const cx = (b.x0 + b.x1) / 2 + (dirX / len) * bias;
    const cy = (b.y0 + b.y1) / 2 + (dirY / len) * bias;
    return { x: cx - s / 2, y: cy - s / 2, s, handIndex };
  });
  if (rects.length === 2) {
    const [a, b] = rects;
    const x0 = Math.min(a.x, b.x), y0 = Math.min(a.y, b.y);
    const s = Math.max(Math.max(a.x + a.s, b.x + b.s) - x0, Math.max(a.y + a.s, b.y + b.s) - y0);
    if (s <= 1.3 * Math.max(a.s, b.s) && s <= frame) {
      rects = [{ x: x0, y: y0, s, handIndex: a.handIndex }];
    }
  }
  return rects
    .map((r) => ({
      x: Math.min(Math.max(0, r.x), w - r.s),
      y: Math.min(Math.max(0, r.y), h - r.s),
      s: r.s,
      handIndex: r.handIndex
    }))
    .filter((r) => r.s < 0.92 * frame);
}

// Estimate a phone box from a gripping hand. Returns null when the pose looks
// like an open palm / fist / point rather than holding a flat device.
function gripPhoneFromHand(hand) {
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const wrist = hand[0];
  const thumbTip = hand[4];
  const indexMcp = hand[5], middleMcp = hand[9], ringMcp = hand[13], pinkyMcp = hand[17];
  const indexTip = hand[8], middleTip = hand[12], ringTip = hand[16], pinkyTip = hand[20];

  const palmW = dist(indexMcp, pinkyMcp);
  const palmLen = dist(wrist, middleMcp);
  if (palmW < 0.015 || palmLen < 0.015) return null;

  // How extended the fingers are (open hand ≈ 1+, curled grip ≈ 0.4–1.0).
  const curls = [
    dist(indexMcp, indexTip),
    dist(middleMcp, middleTip),
    dist(ringMcp, ringTip),
    dist(pinkyMcp, pinkyTip)
  ].map((d) => d / palmLen);
  const avgCurl = curls.reduce((a, b) => a + b, 0) / 4;
  const thumbSep = dist(thumbTip, indexMcp) / palmW;
  const span = dist(indexTip, pinkyTip) / palmW;

  let conf = 0;
  if (avgCurl > 0.35 && avgCurl < 1.05) conf += 0.42;
  else if (avgCurl > 0.28 && avgCurl < 1.2) conf += 0.22;
  if (thumbSep > 0.55 && thumbSep < 2.9) conf += 0.32;
  if (span > 0.7 && span < 3.6) conf += 0.18;
  // Slight bonus when the thumb is on the opposite side of the palm from the pinky
  // (classic phone edge grip), vs tucked in.
  const thumbPalm = dist(thumbTip, middleMcp) / palmW;
  if (thumbPalm > 0.7 && thumbPalm < 2.5) conf += 0.12;

  if (conf < 0.4) return null;

  // Oriented phone rectangle: long axis along the palm width (landscape hold),
  // short axis along wrist→fingertips; fall back AABB for touch tests.
  const longX = pinkyMcp.x - indexMcp.x;
  const longY = pinkyMcp.y - indexMcp.y;
  const longN = Math.hypot(longX, longY) || 1e-6;
  const lx = longX / longN, ly = longY / longN;

  const shortX = middleTip.x - wrist.x;
  const shortY = middleTip.y - wrist.y;
  const shortN = Math.hypot(shortX, shortY) || 1e-6;
  const sx = shortX / shortN, sy = shortY / shortN;

  const center = {
    x: middleMcp.x + sx * palmLen * 0.4,
    y: middleMcp.y + sy * palmLen * 0.4
  };
  const halfLong = Math.max(palmW * 1.35, palmLen * 0.55);
  const halfShort = Math.max(palmLen * 0.75, palmW * 0.7);

  const corners = [
    [center.x + lx * halfLong + sx * halfShort, center.y + ly * halfLong + sy * halfShort],
    [center.x + lx * halfLong - sx * halfShort, center.y + ly * halfLong - sy * halfShort],
    [center.x - lx * halfLong + sx * halfShort, center.y - ly * halfLong + sy * halfShort],
    [center.x - lx * halfLong - sx * halfShort, center.y - ly * halfLong - sy * halfShort]
  ];
  const xs = corners.map((c) => c[0]), ys = corners.map((c) => c[1]);
  const b = [
    Math.max(0, Math.min(...xs)),
    Math.max(0, Math.min(...ys)),
    Math.min(1, Math.max(...xs)),
    Math.min(1, Math.max(...ys))
  ];
  if ((b[2] - b[0]) < 0.03 || (b[3] - b[1]) < 0.03) return null;
  return { b, s: Math.min(0.95, conf) };
}

// box = [x0, y0, x1, y1], normalised
function touches(hand, box) {
  const [bx0, by0, bx1, by1] = box;
  const pw = bx1 - bx0, ph = by1 - by0;
  // Generous margin: grip boxes and sideways OD boxes often sit slightly off the palm.
  const mx = Math.max(pw * 0.25, 0.03);
  const my = Math.max(ph * 0.25, 0.03);
  const inside = CONTACT_POINTS.some((i) => {
    const p = hand[i];
    return p.x > bx0 - mx && p.x < bx1 + mx && p.y > by0 - my && p.y < by1 + my;
  });
  if (inside) return true;

  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const p of hand) {
    x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y);
    x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y);
  }
  const pad = 0.15 * Math.max(x1 - x0, y1 - y0);
  x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
  const ix = Math.max(0, Math.min(x1, bx1) - Math.max(x0, bx0));
  const iy = Math.max(0, Math.min(y1, by1) - Math.max(y0, by0));
  return (ix * iy) / (pw * ph || 1) > 0.18;
}

export const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;
