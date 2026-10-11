// Background clock. Browsers throttle timers on hidden tabs (to 1/s, then
// 1/min after a few minutes) and stop requestAnimationFrame entirely, but
// dedicated-worker timers keep their rate. One-shot: the page posts a delay
// in ms (0 = cancel), gets one message back, runs a detection step and asks
// again, so slow devices never build up a queue of pending ticks.

let id = null;

self.onmessage = (e) => {
  clearTimeout(id);
  id = e.data > 0 ? setTimeout(() => self.postMessage(0), e.data) : null;
};
