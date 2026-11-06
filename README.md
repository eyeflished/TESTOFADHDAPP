# FocusGuard

**EN** · [RU](#ru)

A browser-only focus helper for people with concentration problems and ADHD.
Allow camera access, and FocusGuard watches your hands in real time. When it sees
a phone in your hand, a siren turns on. Put the phone down and the siren stops.
There's nothing to install: it's a static web page.

## Features

- **Real-time hand + phone tracking** in the browser (MediaPipe Hand Landmarker,
  the EfficientDet-Lite0 object detector and, as a fallback, the Pose Landmarker):
  - the phone detector also looks at zoomed-in crops around each hand, where a
    phone held at desk distance is several times larger than in the whole frame
  - when a phone is visible but the hand model misses the gripping hand (the
    phone hides the palm), the body-pose model checks whether a wrist or finger
    is on the phone
  - phones are tracked between frames with hysteresis, so the outline does not
    flicker between red and yellow, and the drawn hand skeleton is smoothed
- **Siren** made with the Web Audio API: soft chime, beeping alarm, siren or harsh buzzer.
- **Configurable punishment severity**: Gentle / Medium / Strict presets, or set each option yourself:
  - grace period before the siren (0–15 s)
  - sound type and volume, with optional escalation (gets louder over time)
  - screen effect: none / soft glow / red pulse / intense flash (kept under 3 flashes per second for photosensitive safety)
  - minimum siren duration
  - penalty minutes added to the focus timer
  - stop rule: "phone released" or the stricter "phone out of sight"
- **ADHD adaptation**:
  - calm, low-stimulation palette, larger text, reduced motion, no harsh flashing
  - gentle warnings before a real siren (if you ignore a warning, the siren starts)
  - focus timer with breaks (Pomodoro style); the phone is allowed during breaks
  - a "What am I focusing on?" field that the reminders show back to you
  - encouragement for phone-free streaks and for putting the phone down quickly
  - camera preview can be hidden (a calm breathing orb shows instead)
  - one-click "recommended ADHD setup"
- **Works in the background**: once a session is running, you can switch to
  other sites, your IDE, email or any other app, and tracking and the siren
  keep going. Just keep the FocusGuard tab open (it can be in the background or minimised).
  - a Web Worker clock drives detection while the tab is hidden, because browsers
    pause or throttle normal page timers there
  - **Mini window** (Chrome, Edge, Opera, Yandex Browser): a small always-on-top
    camera window that floats over other apps
  - optional **system notification** when the phone is detected, and the tab
    title switches to "🔴 SIREN"
- **English / Russian** toggle (picked automatically from the browser language).
- **Fully anonymous**:
  - video is processed on the device only and is never uploaded or recorded
  - no accounts, cookies, analytics or third-party requests
  - the library, its WebAssembly runtime and the models are all served from this
    site, and a strict Content-Security-Policy blocks connections to any other origin
  - session stats are kept in memory only; settings are saved in `localStorage`
    only if you allow it, and one button erases them

## Running it

The camera only works on `https://` or `localhost`.

**Locally**

```bash
python3 -m http.server 8000
# open http://localhost:8000
```

**GitHub Pages (free hosting)**: in the repo, go to *Settings → Pages → Build
and deployment → Deploy from a branch*, then pick the `focusguard` branch and `/ (root)`.
The site will appear at `https://<user>.github.io/<repo>/`.

Any other static host (Netlify, Cloudflare Pages, Vercel, nginx) works too. Just
upload the folder.

## How accurate is it?

Measured on labelled photos from [Open Images](https://storage.googleapis.com/openimages/web/index.html)
(people holding phones, and hands without phones), run through the app's own
detection code in Chromium. Each photo was also played as about one second of
slightly shaky 15 fps video, so frame-to-frame tracking counts too. Settings
were tuned on one set of photos and scored on photos never used for tuning.

Webcam-like photos (people holding phones, 136 clips; hands only, 200 clips):

| | Before | Now |
|---|---|---|
| Phone in hand caught within about a second | 65% | **82%** |
| Outline red while the phone is held | 57% | **73%** |
| Outline yellow although the phone is in a hand | 26% | **12%** |
| Hands only: lasting false alarm | 4.5% | 5.5% |

The false-alarm difference is within noise. Combined over both held-out sets
(400 hand-only clips) it is 13 vs 14 clips, mostly microphones, an ID badge,
and devices the labels missed. Extreme close-ups, where only fingers are visible,
are harder: 14% → 51% caught. The same photos shrunk into a busy scene: 21% → 45%.

These are internet photos, not real webcam footage, so the absolute numbers will
differ with your light and camera angle. The improvement is the reliable part.

## Tips

- Use a laptop or desktop webcam that can see your hands and desk.
- Good lighting makes a big difference.
- Don't close the FocusGuard tab. It can stay in the background.
- Background mode works best in Chromium browsers (Chrome, Edge, Opera, Yandex).
  Firefox works too, but has no mini window. Safari may slow down hidden tabs more.

## Structure

```
index.html          page markup + CSP
css/style.css       styles (normal and ADHD themes)
js/app.js           session state machine, settings, UI
js/detector.js      on-device hand/phone detection
js/audio.js         synthesised siren and chimes
js/ticker.js        worker clock that keeps detection running in background tabs
js/i18n.js          English and Russian strings
vendor/mediapipe/   @mediapipe/tasks-vision 0.10.21 (Apache-2.0)
models/             hand_landmarker.task, efficientdet_lite0.tflite,
                    pose_landmarker_lite.task (Apache-2.0, Google)
```

---

<a id="ru"></a>
## RU — кратко

FocusGuard — сайт, который помогает удерживать внимание. Устанавливать ничего не нужно.
Разрешите доступ к камере, и программа будет отслеживать руки в реальном времени.
Как только в руке появляется телефон, включается сирена. Положите телефон, и сирена выключится.

- Гибкая настройка строгости наказания: отсрочка, звук, громкость, нарастание,
  эффект на экране, минимальная длительность, штрафное время, условие выключения.
- Адаптация для СДВГ: спокойные цвета, крупный текст, мягкие предупреждения перед сиреной,
  таймер фокуса с перерывами, поддержка и похвала, можно скрыть изображение камеры.
- Работа в фоне: можно переключаться на другие сайты и программы, и
  отслеживание с сиреной продолжат работать. Есть мини-окно поверх всех окон и
  системные уведомления.
- Точнее находит телефон в руке: смотрит отдельно на область вокруг каждой руки,
  находит руку по позе тела, если ладонь закрыта телефоном, и не мигает рамкой.
  На тестовых фото: 82% вместо 65% телефонов в руке замечено примерно за секунду,
  при том же уровне ложных срабатываний.
- Переключение языка: русский и английский.
- Полная анонимность: всё обрабатывается на устройстве, без серверов, аккаунтов,
  cookies и аналитики.

Запуск: `python3 -m http.server 8000` и откройте `http://localhost:8000`,
или опубликуйте папку через GitHub Pages.
