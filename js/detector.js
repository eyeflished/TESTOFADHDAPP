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

export class PhoneDetector {
  constructor() {
    this.hands = null;
    this.objects = null;
    this.frame = 0;
    this.lastTs = -1;
    this.lastHands = [];
    this.lastPhones = [];
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
      const w = video.videoWidth;
      const h = video.videoHeight;
      const objRes = this.objects.detectForVideo(video, ts);
      this.lastPhones = (objRes.detections || []).map((d) => ({
        x: d.boundingBox.originX / w,
        y: d.boundingBox.originY / h,
        w: d.boundingBox.width / w,
        h: d.boundingBox.height / h,
        score: d.categories[0]?.score ?? 0
      }));
    }

    const phoneVisible = this.lastPhones.length > 0;
    const inHand = this.lastPhones.some((p) => this.lastHands.some((hand) => touches(hand, p)));
    return { hands: this.lastHands, phones: this.lastPhones, phoneVisible, inHand };
  }

  close() {
    this.hands?.close();
    this.objects?.close();
    this.hands = this.objects = null;
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
