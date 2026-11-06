// On-device hand + phone detection with MediaPipe Tasks Vision.
// The library, its WebAssembly runtime and the models are served from this
// site (vendor/ and models/), so no frame or request ever leaves the device.
//
// How a phone in a hand is found (each step measured on labelled Open Images
// photos; see "How accurate is it?" in the README):
// - Hand landmarks every frame, with low thresholds: a hand gripping a phone
//   hides most of the palm the hand model looks for.
// - The phone detector sees its input at only 320x320 px, so a phone held at
//   desk distance is tiny in the whole frame. It also looks at square crops
//   around each hand, where the phone is several times larger.
// - When a phone is visible but no hand landmarks touch it, a body-pose model
//   checks whether a wrist/finger point is on the phone: it finds hands from
//   the arm even when the hand model cannot.
// - Phones are tracked between passes with hysteresis (a higher score to
//   appear than to stay), so a box near the threshold does not flicker.

import {
  FilesetResolver,
  HandLandmarker,
  ObjectDetector,
  PoseLandmarker
} from '../vendor/mediapipe/vision_bundle.mjs';

const WASM_PATH = new URL('../vendor/mediapipe/wasm', import.meta.url).href;
const HAND_MODEL = new URL('../models/hand_landmarker.task', import.meta.url).href;
const OBJECT_MODEL = new URL('../models/efficientdet_lite0.tflite', import.meta.url).href;
const POSE_MODEL = new URL('../models/pose_landmarker_lite.task', import.meta.url).href;

const OBJECT_INPUT = 320;   // EfficientDet-Lite0 input size
const HAND_EVERY_MS = 30;   // hand landmarks at most ~33 times a second
const PHONE_EVERY_MS = 90;  // phone pass at most ~11 times a second (slower on slow devices)
const TRACK_TTL_MS = 450;   // forget a phone that has not been seen for this long
const POSE_TTL_MS = 500;    // pose hand points stay valid this long
const KEEP_RATIO = 0.65;    // a tracked phone stays while its score >= start threshold x this
const CROP_SCALE = 2.6;     // crop side = hand size x this
const POSE_MARGIN = 0.3;    // pose hand point may be this x phone size outside the phone box
const POSE_SIDES = [[15, 17, 19, 21], [16, 18, 20, 22]]; // wrist, pinky, index, thumb (left, right)

// Score a new phone needs, per source: whole frame / crop around a hand.
// Crops get a higher bar because zooming in on a hand makes any object in it
// look more phone-like. In a hand crop a "remote" also counts: zoomed in, the
// model often calls a held phone a remote (a mouse almost never is).
export const SENSITIVITY_THRESHOLD = {
  low: { full: 0.6, crop: 0.7 },
  normal: { full: 0.5, crop: 0.6 },
  high: { full: 0.4, crop: 0.5 }
};

// Fingertips + palm landmarks used to decide whether a hand touches the phone.
const CONTACT_POINTS = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];

export class PhoneDetector {
  constructor() {
    this.hands = null;
    this.objects = null;
    this.pose = null;
    this.lastTs = -1;
    this.handAt = -Infinity;
    this.phoneAt = -Infinity;
    this.phoneEvery = PHONE_EVERY_MS;
    this.pass = 0;
    this.lastHands = [];
    this.tracks = [];
    this.posePoints = [];
    this.poseAt = -Infinity;
    this.failures = 0;
    this.broken = false; // set when detection cannot run in this browser at all
    this.crop = document.createElement('canvas');
    this.crop.width = this.crop.height = OBJECT_INPUT;
    this.cropCtx = this.crop.getContext('2d');
  }

  async init(threshold, { delegate } = {}) {
    this.threshold = threshold;
    this.fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
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
    // The pose model is only a fallback, so it loads in the background and
    // detection starts without it.
    this.poseReady = this.createPose().catch((e) => console.warn('Pose model unavailable', e));
  }

  async create(delegate) {
    const baseOptions = (modelAssetPath) => ({ modelAssetPath, delegate });
    const [hands, objects] = await Promise.all([
      HandLandmarker.createFromOptions(this.fileset, {
        baseOptions: baseOptions(HAND_MODEL),
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.2,
        minHandPresenceConfidence: 0.2,
        minTrackingConfidence: 0.2
      }),
      // IMAGE mode: the detector is stateless and runs several times per pass.
      ObjectDetector.createFromOptions(this.fileset, {
        baseOptions: baseOptions(OBJECT_MODEL),
        runningMode: 'IMAGE',
        scoreThreshold: 0.25, // the real thresholds (with hysteresis) are applied below
        maxResults: 6,
        categoryAllowlist: ['cell phone', 'remote']
      })
    ]);
    this.close();
    this.hands = hands;
    this.objects = objects;
    this.delegate = delegate;
    if (this.pose) await this.createPose();
  }

  async createPose() {
    const pose = await PoseLandmarker.createFromOptions(this.fileset, {
      baseOptions: { modelAssetPath: POSE_MODEL, delegate: this.delegate },
      runningMode: 'IMAGE',
      numPoses: 2,
      minPoseDetectionConfidence: 0.4,
      minPosePresenceConfidence: 0.4
    });
    this.pose?.close();
    this.pose = pose;
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
      // Repeated failures on CPU mean no usable WebGL at all. Stop calling
      // into the runtime: hundreds of failed calls eventually hang the page.
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
      // On slow devices keep the phone pass to about half of the time.
      this.phoneEvery = Math.min(600, Math.max(PHONE_EVERY_MS, 2 * (performance.now() - t0)));
    }
    this.tracks = this.tracks.filter((tr) => now - tr.seen <= TRACK_TTL_MS);
    if (now - this.poseAt > POSE_TTL_MS) this.posePoints = [];

    const phones = this.tracks.map((tr) => ({
      x: tr.b[0], y: tr.b[1], w: tr.b[2] - tr.b[0], h: tr.b[3] - tr.b[1], score: tr.s,
      inHand: this.held(tr.b, w, h)
    }));
    return { hands: this.lastHands, phones, phoneVisible: phones.length > 0, inHand: phones.some((p) => p.inHand) };
  }

  held(box, w, h) {
    return this.lastHands.some((hand) => touches(hand, box)) || poseHolds(this.posePoints, box, w, h);
  }

  phonePass(source, w, h, now) {
    const found = [];
    const rects = cropsFor(this.lastHands, w, h);
    for (const r of rects) {
      this.cropCtx.drawImage(source, r.x, r.y, r.s, r.s, 0, 0, OBJECT_INPUT, OBJECT_INPUT);
      const k = r.s / OBJECT_INPUT;
      collect(this.objects.detect(this.crop), found, 'crop',
        (x, y, bw, bh) => [(r.x + x * k) / w, (r.y + y * k) / h, (r.x + (x + bw) * k) / w, (r.y + (y + bh) * k) / h]);
    }
    // The whole frame every other pass (every pass when there is no hand): it
    // finds phones away from hands, for "any phone" mode and "out of sight".
    if (!rects.length || this.pass++ % 2 === 0) {
      collect(this.objects.detect(source), found, 'full',
        (x, y, bw, bh) => [x / w, y / h, (x + bw) / w, (y + bh) / h]);
    }
    this.updateTracks(found, now);

    // A phone that no detected hand touches: ask the pose model where the hands are.
    if (this.pose && this.tracks.some((tr) => !this.lastHands.some((hand) => touches(hand, tr.b)))) {
      this.posePoints = poseHandPoints(this.pose.detect(source).landmarks || []);
      this.poseAt = now;
    }
  }

  updateTracks(found, now) {
    const phones = found
      .filter((d) => d.cls === 'cell phone' || (d.cls === 'remote' && d.src === 'crop'))
      .sort((a, b) => b.s - a.s);
    for (const d of phones) {
      const start = this.threshold[d.src];
      const tr = this.tracks.find((t) => iou(t.b, d.b) > 0.3);
      if (tr) {
        if (tr.seen !== now && d.s >= start * KEEP_RATIO) {
          tr.b = tr.b.map((v, i) => v + 0.6 * (d.b[i] - v));
          tr.s = d.s;
          tr.seen = now;
        }
      } else if (d.s >= start) {
        this.tracks.push({ b: d.b, s: d.s, seen: now });
      }
    }
  }

  close() {
    this.hands?.close();
    this.objects?.close();
    this.hands = this.objects = null;
  }
}

function hasWebGL2() {
  try {
    return !!document.createElement('canvas').getContext('webgl2');
  } catch {
    return false;
  }
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

// Square crops (in px) around each hand; two overlapping hands share one crop,
// and a crop that would cover most of the frame is skipped (the full frame
// pass already sees that).
function cropsFor(hands, w, h) {
  const frame = Math.min(w, h);
  let rects = hands.map((hand) => {
    const b = handBox(hand, w, h);
    const s = Math.min(frame, Math.max(0.2 * frame, CROP_SCALE * Math.max(b.x1 - b.x0, b.y1 - b.y0)));
    return { x: (b.x0 + b.x1 - s) / 2, y: (b.y0 + b.y1 - s) / 2, s };
  });
  if (rects.length === 2) {
    const [a, b] = rects;
    const x0 = Math.min(a.x, b.x), y0 = Math.min(a.y, b.y);
    const s = Math.max(Math.max(a.x + a.s, b.x + b.s) - x0, Math.max(a.y + a.s, b.y + b.s) - y0);
    if (s <= 1.3 * Math.max(a.s, b.s) && s <= frame) rects = [{ x: x0, y: y0, s }];
  }
  return rects
    .map((r) => ({ x: Math.min(Math.max(0, r.x), w - r.s), y: Math.min(Math.max(0, r.y), h - r.s), s: r.s }))
    .filter((r) => r.s < 0.9 * frame);
}

// box = [x0, y0, x1, y1], normalised
function touches(hand, box) {
  const [bx0, by0, bx1, by1] = box;
  const pw = bx1 - bx0, ph = by1 - by0;
  // 1) any contact landmark inside the phone box (slightly enlarged)
  const mx = pw * 0.15;
  const my = ph * 0.15;
  const inside = CONTACT_POINTS.some((i) => {
    const p = hand[i];
    return p.x > bx0 - mx && p.x < bx1 + mx && p.y > by0 - my && p.y < by1 + my;
  });
  if (inside) return true;

  // 2) or the hand's bounding box overlaps a good part of the phone
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
  for (const p of hand) {
    x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y);
    x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y);
  }
  const pad = 0.1 * Math.max(x1 - x0, y1 - y0);
  x0 -= pad; y0 -= pad; x1 += pad; y1 += pad;
  const ix = Math.max(0, Math.min(x1, bx1) - Math.max(x0, bx0));
  const iy = Math.max(0, Math.min(y1, by1) - Math.max(y0, by0));
  return (ix * iy) / (pw * ph || 1) > 0.25;
}

// One point per visible pose hand: the mean of its wrist/pinky/index/thumb points.
function poseHandPoints(poses) {
  const points = [];
  for (const pose of poses) {
    for (const side of POSE_SIDES) {
      const vis = side.map((i) => pose[i]).filter((p) => p && (p.visibility ?? 0) >= 0.5);
      if (vis.length >= 2) {
        points.push({ x: vis.reduce((s, p) => s + p.x, 0) / vis.length, y: vis.reduce((s, p) => s + p.y, 0) / vis.length });
      }
    }
  }
  return points;
}

// The margin is small on purpose: a phone lying next to typing hands must not count.
function poseHolds(points, box, w, h) {
  const [x0, y0, x1, y1] = box;
  const m = POSE_MARGIN * Math.max((x1 - x0) * w, (y1 - y0) * h);
  const mx = m / w, my = m / h;
  return points.some((p) => p.x >= x0 - mx && p.x <= x1 + mx && p.y >= y0 - my && p.y <= y1 + my);
}

export const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;
