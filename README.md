# FocusGuard

**EN** · [RU](#ru)

A browser-only focus helper for people with concentration problems and ADHD.
Allow camera access, and FocusGuard watches your hands in real time. When it sees
a phone in your hand, a siren turns on. Put the phone down and the siren stops.
There's nothing to install: it's a static web page.

## Features

- **Real-time hand + phone tracking** in the browser (MediaPipe Hand Landmarker
  plus the EfficientDet-Lite0 object detector, "cell phone" class).
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
and deployment → Deploy from a branch*, then pick the branch and `/ (root)`.
The site will appear at `https://<user>.github.io/<repo>/`.

Any other static host (Netlify, Cloudflare Pages, Vercel, nginx) works too. Just
upload the folder.

## Tips

- Use a laptop or desktop webcam that can see your hands and desk.
- Good lighting makes a big difference.
- Keep the tab visible. Browsers slow down hidden tabs.

## Structure

```
index.html          page markup + CSP
css/style.css       styles (normal and ADHD themes)
js/app.js           session state machine, settings, UI
js/detector.js      on-device hand/phone detection
js/audio.js         synthesised siren and chimes
js/i18n.js          English and Russian strings
vendor/mediapipe/   @mediapipe/tasks-vision 0.10.21 (Apache-2.0)
models/             hand_landmarker.task, efficientdet_lite0.tflite (Apache-2.0, Google)
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
- Переключение языка: русский и английский.
- Полная анонимность: всё обрабатывается на устройстве, без серверов, аккаунтов,
  cookies и аналитики.

Запуск: `python3 -m http.server 8000` и откройте `http://localhost:8000`,
или опубликуйте папку через GitHub Pages.
