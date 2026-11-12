// On-device hand + phone detection with MediaPipe Tasks Vision.
// The library, its WebAssembly runtime and both models are served from this
// site (vendor/ and models/), so no frame or request ever leaves the device.

import {
  FilesetResolver,
  HandLandmarker,
  ObjectDetector
} from '../vendor/mediapipe/vision_bundle.mjs';

const WASM_PATH = new URL('../vendor/mediapipe/wasm', import.meta.url).href;
const HAND_MODEL = new URL('../models/hand_landmarker.task', import.meta.url).href;
const OBJECT_MODEL = new URL('../models/efficientdet_lite0.tflite', import.meta.url).href;

export const SENSITIVITY_THRESHOLD = { low: 0.5, normal: 0.36, high: 0.24 };

// Fingertips + palm landmarks used to decide whether a hand touches the phone.
const CONTACT_POINTS = [0, 4, 5, 8, 9, 12, 13, 16, 17, 20];

// Map a normalized box seen in a 90° CW rotated canvas back to the original frame.
const mapBackCW = ({ x, y, w, h }) => ({
  x: y,
  y: 1 - x - w,
  w: h,
  h: w
});

// Map a normalized box seen in a 90° CCW rotated canvas back to the original frame.
const mapBackCCW = ({ x, y, w, h }) => ({
  x: 1 - y - h,
  y: x,
  w: h,
  h: w
});

const clamp01 = (v) => Math.min(1, Math.max(0, v));
const clampBox = (b) => ({
  x: clamp01(b.x),
  y: clamp01(b.y),
  w: clamp01(b.w),
  h: clamp01(b.h),
  score: b.score
});

function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
  const inter = ix * iy;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

function mergeAndDedupe(boxes) {
  const kept = [];
  for (const box of boxes.slice().sort((a, b) => b.score - a.score)) {
    const dup = kept.some((k) => iou(k, box) > 0.45);
    if (!dup) kept.push(clampBox(box));
  }
  return kept;
}

export class PhoneDetector {
  constructor() {
    this.hands = null;
    this.objects = null;
    this.frame = 0;
    this.lastTs = -1;
    this.lastHands = [];
    this.lastPhones = [];
    this.rotCache = {};
  }

  async init(threshold) {
    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
    const make = async (Task, opts) => {
      try {
        return await Task.createFromOptions(fileset, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'GPU' } });
      } catch (e) {
        console.warn('GPU delegate unavailable, falling back to CPU', e);
        return Task.createFromOptions(fileset, { ...opts, baseOptions: { ...opts.baseOptions, delegate: 'CPU' } });
      }
    };
    [this.hands, this.objects] = await Promise.all([
      make(HandLandmarker, {
        baseOptions: { modelAssetPath: HAND_MODEL },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5
      }),
      make(ObjectDetector, {
        baseOptions: { modelAssetPath: OBJECT_MODEL },
        runningMode: 'VIDEO',
        scoreThreshold: threshold,
        maxResults: 3,
        categoryAllowlist: ['cell phone']
      })
    ]);
  }

  async setThreshold(threshold) {
    if (this.objects) await this.objects.setOptions({ scoreThreshold: threshold });
  }

  /**
   * Runs detection on the current video frame.
   * Hands run every frame; the heavier object detector every 2nd frame.
   * Coordinates are normalised to 0..1.
   *
   * When the phone is sideways the upright object detector can miss it, so on
   * empty results we also scan 90° CW / CCW rotated copies of the frame and map
   * any found boxes back to the original orientation.
   */
  detect(video) {
    if (!this.hands || video.readyState < 2 || !video.videoWidth) return null;
    let ts = performance.now();
    if (ts <= this.lastTs) ts = this.lastTs + 1;
    this.lastTs = ts;
    this.frame++;

    const handRes = this.hands.detectForVideo(video, ts);
    this.lastHands = handRes.landmarks || [];

    if (this.frame % 2 === 0) {
      this.lastPhones = this.detectPhones(video, ts);
    }

    const phoneVisible = this.lastPhones.length > 0;
    const inHand = this.lastPhones.some((p) => this.lastHands.some((hand) => touches(hand, p)));
    return { hands: this.lastHands, phones: this.lastPhones, phoneVisible, inHand };
  }

  detectPhones(video, ts) {
    const runOn = (target) => {
      const res = this.objects.detectForVideo(target, ts);
      const dw = target.videoWidth != null ? target.videoWidth : target.width;
      const dh = target.videoHeight != null ? target.videoHeight : target.height;
      return (res.detections || []).map((d) => ({
        x: d.boundingBox.originX / dw,
        y: d.boundingBox.originY / dh,
        w: d.boundingBox.width / dw,
        h: d.boundingBox.height / dh,
        score: d.categories[0]?.score ?? 0
      }));
    };

    // Upright frame first — fast path for the common case.
    const found = runOn(video);
    if (found.length > 0) return found;

    // Sideways phone: scan rotated copies so EfficientDet sees it upright.
    const cw = runOn(this.rotateVideo(video, 90)).map(mapBackCW);
    const ccw = runOn(this.rotateVideo(video, 270)).map(mapBackCCW);

    return mergeAndDedupe([...cw, ...ccw]);
  }

  rotateVideo(video, deg) {
    const w = video.videoWidth;
    const h = video.videoHeight;
    const is90 = deg === 90 || deg === 270;
    let slot = this.rotCache[deg];
    if (!slot || slot.w !== w || slot.h !== h) {
      const c = document.createElement('canvas');
      c.width = is90 ? h : w;
      c.height = is90 ? w : h;
      slot = { canvas: c, ctx: c.getContext('2d'), w, h };
      this.rotCache[deg] = slot;
    }
    const { canvas, ctx } = slot;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate((deg * Math.PI) / 180);
    ctx.drawImage(video, -w / 2, -h / 2);
    return canvas;
  }

  close() {
    this.hands?.close();
    this.objects?.close();
    this.hands = this.objects = null;
    this.rotCache = {};
  }
}

function touches(hand, phone) {
  // 1) any contact landmark inside the phone box (slightly enlarged)
  const mx = phone.w * 0.15;
  const my = phone.h * 0.15;
  const inside = CONTACT_POINTS.some((i) => {
    const p = hand[i];
    return p.x > phone.x - mx && p.x < phone.x + phone.w + mx &&
           p.y > phone.y - my && p.y < phone.y + phone.h + my;
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
  const ix = Math.max(0, Math.min(x1, phone.x + phone.w) - Math.max(x0, phone.x));
  const iy = Math.max(0, Math.min(y1, phone.y + phone.h) - Math.max(y0, phone.y));
  return (ix * iy) / (phone.w * phone.h || 1) > 0.25;
}

export const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;
