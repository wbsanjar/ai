import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const video = document.getElementById("video");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const errorEl = document.getElementById("error");
const btnStart = document.getElementById("btnStart");

const chkShield = document.getElementById("chkShield");
const chkBeam = document.getElementById("chkBeam");
const chkParticles = document.getElementById("chkParticles");

let handLandmarker = null;
let running = false;

// Frames are often Y-flipped? MediaPipe Tasks expects a single video stream.
// We render from the video element directly using requestVideoFrameCallback.
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

// Effect state
const state = {
  trail: [],
  particles: [],
  shieldAlpha: 0,
  beamAlpha: 0,
  lastCenters: [],
};

const COLORS = {
  handA: "#00ffff",
  handB: "#ff00ff",
  shield: "rgba(0,140,255,",
  beam: "#ffd26e",
  particle: "#ffcc66",
};

function setStatus(text, on = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("on", on);
}

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}

function hideError() {
  errorEl.classList.add("hidden");
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Shared: draw a glowing (layered) line
function glowLine(x1, y1, x2, y2, color, width, layers) {
  for (let i = layers; i >= 1; i--) {
    ctx.strokeStyle = color;
    ctx.globalAlpha = i === 1 ? 0.9 : 0.1 * (1 - (i - 1) / layers);
    ctx.lineWidth = width + (i - 1) * 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  }
}

function glowDot(x, y, radius, color, alpha) {
  for (let i = 3; i >= 1; i--) {
    const r = radius * (0.5 + 0.5 * (i / 3));
    ctx.globalAlpha = alpha * (i === 1 ? 0.9 : 0.15);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function palmCenter(lm) {
  const idxs = [0, 5, 9, 13, 17];
  let x = 0, y = 0;
  for (const i of idxs) { x += lm[i].x; y += lm[i].y; }
  return { x: x / idxs.length, y: y / idxs.length };
}

function isOpenPalm(lm) {
  const wrist = lm[0];
  let ok = true;
  for (const [tip, pip] of [[8, 6], [12, 10], [16, 14], [20, 18]]) {
    if (dist(lm[tip], wrist) < dist(lm[pip], wrist)) { ok = false; break; }
  }
  return ok;
}

function isPointing(lm) {
  return dist(lm[8], lm[0]) > dist(lm[6], lm[0]) * 1.05 &&
         dist(lm[12], lm[0]) <= dist(lm[10], lm[0]) * 1.05 &&
         dist(lm[16], lm[0]) <= dist(lm[14], lm[0]) * 1.05 &&
         dist(lm[20], lm[0]) <= dist(lm[18], lm[0]) * 1.05;
}

function isFist(lm) {
  return dist(lm[8], lm[0]) <= dist(lm[6], lm[0]) * 1.05 &&
         dist(lm[12], lm[0]) <= dist(lm[10], lm[0]) * 1.05 &&
         dist(lm[16], lm[0]) <= dist(lm[14], lm[0]) * 1.05 &&
         dist(lm[20], lm[0]) <= dist(lm[18], lm[0]) * 1.05;
}

// Fist size to grow particles — approximate via pinky-to-thumb spread
function handSize(lm) {
  return Math.max(0.05, dist(lm[4], lm[17]));
}

// Draw neon skeleton for one hand
function drawSkeleton(lm, color) {
  for (const [a, b] of HAND_CONNECTIONS) {
    const p1 = { x: lm[a].x * canvas.width, y: lm[a].y * canvas.height };
    const p2 = { x: lm[b].x * canvas.width, y: lm[b].y * canvas.height };
    glowLine(p1.x, p1.y, p2.x, p2.y, color, 2, 5);
  }
  // Joint dots
  for (const p of lm) {
    ctx.globalAlpha = 0.8;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(p.x * canvas.width, p.y * canvas.height, 2, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---- Effects ----

function updateShield(hands, dt) {
  const target = chkShield.checked && hands.some(isOpenPalm) ? 1 : 0;
  state.shieldAlpha += (target - state.shieldAlpha) * Math.min(1, dt * 8);
  state.shieldAlpha = Math.max(0, Math.min(1, state.shieldAlpha));
}

function updateBeam(hands, dt) {
  const pointing = hands.find(isPointing);
  const target = chkBeam.checked && pointing ? 1 : 0;
  state.beamAlpha += (target - state.beamAlpha) * Math.min(1, dt * 8);
  state.beamAlpha = Math.max(0, Math.min(1, state.beamAlpha));
  if (pointing && state.beamAlpha > 0.01) {
    // remember fingertip + direction
    const tip = pointing[8];
    const pip = pointing[6];
    const dx = tip.x - pip.x;
    const dy = tip.y - pip.y;
    const len = Math.hypot(dx, dy) || 1;
    state.beam = {
      x: tip.x * canvas.width,
      y: tip.y * canvas.height,
      dx: dx / len,
      dy: dy / len,
    };
  }
}

function drawBeam(t) {
  if (state.beamAlpha <= 0.01 || !state.beam) return;
  const { x, y, dx, dy } = state.beam;
  const lengthState = Math.min(canvas.width, canvas.height);
  const ex = x + dx * lengthState;
  const ey = y + dy * lengthState;
  ctx.globalAlpha = state.beamAlpha;
  glowLine(x, y, ex, ey, COLORS.beam, 3, 6);
  // racing pulse
  const pulse = ((t * 2.2) % 1);
  const px = x + dx * lengthState * pulse;
  const py = y + dy * lengthState * pulse;
  glowDot(px, py, 8, "#ffffff", state.beamAlpha * (1 - pulse));
  glowDot(x, y, 14, COLORS.beam, state.beamAlpha);
  glowDot(x, y, 5, "#ffffff", state.beamAlpha);
}

function updateParticles(hands, dt) {
  const fist = hands.find(isFist);
  let center = null;
  if (fist && chkParticles.checked) {
    const c = palmCenter(fist);
    center = { x: c.x * canvas.width, y: c.y * canvas.height };
    // emit a few particles each frame
    for (let i = 0; i < 3; i++) {
      const ang = Math.random() * Math.PI * 2;
      const sp = 60 + Math.random() * 120;
      state.particles.push({
        x: center.x, y: center.y,
        vx: Math.cos(ang) * sp, vy: Math.sin(ang) * sp,
        life: 0, max: 0.6 + Math.random() * 0.5,
        size: 2 + Math.random() * 4,
      });
    }
  }
  for (let i = state.particles.length - 1; i >= 0; i--) {
    const p = state.particles[i];
    p.life += dt;
    if (p.life >= p.max) { state.particles.splice(i, 1); continue; }
    p.x += p.vx * dt;
    p.y += p.vy * dt;
    p.vx *= (1 - 2 * dt);
    p.vy *= (1 - 2 * dt);
  }
}

function drawParticles() {
  for (const p of state.particles) {
    const a = Math.max(0, 1 - p.life / p.max);
    glowDot(p.x, p.y, p.size * a, COLORS.particle, a);
  }
}

function updateTrail(hands, dt) {
  const centers = hands.map((h) => palmCenter(h));
  if (hands.length) {
    state.trail.push(centers[0]);
    if (state.trail.length > 22) state.trail.shift();
  } else {
    state.trail = [];
  }
}

function drawTrail() {
  if (state.trail.length < 2) return;
  for (let i = 0; i < state.trail.length - 1; i++) {
    const p1 = state.trail[i], p2 = state.trail[i + 1];
    const a = (i / state.trail.length) * 0.7;
    glowLine(p1.x * canvas.width, p1.y * canvas.height,
             p2.x * canvas.width, p2.y * canvas.height,
             "#c8f0ff", 2, 3);
  }
}

// ---- Main loop ----

async function initLandmarker() {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );
  handLandmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = canvas.clientWidth * dpr;
  canvas.height = canvas.clientHeight * dpr;
  const scale = dpr;
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}

let lastVideoTime = -1;
let lastTimestamp = 0;

function drawLoop(now) {
  if (!running) return;
  const dt = Math.min(0.1, (now - lastTimestamp) / 1000);
  lastTimestamp = now;
  const t = now / 1000;

  resize();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  let hands = [];
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    try {
      const res = handLandmarker.detectForVideo(video, now);
      hands = res.landmarks || [];
    } catch (e) {
      setStatus("Tracking error");
    }
  }

  // Update + draw skeleton
  const colors = [COLORS.handA, COLORS.handB];
  hands.forEach((h, i) => drawSkeleton(h, colors[i % colors.length]));

  updateTrail(hands, dt);
  drawTrail();

  updateShield(hands, dt);
  // shield uses palm centers of open palms
  if (state.shieldAlpha > 0.01) {
    hands.forEach((h) => {
      if (isOpenPalm(h)) {
        const c = palmCenter(h);
        drawShieldAt(c.x * canvas.width, c.y * canvas.height, t, state.shieldAlpha);
      }
    });
  }

  updateBeam(hands, dt);
  drawBeam(t);

  updateParticles(hands, dt);
  drawParticles();

  setStatus(`Hands: ${hands.length}`, hands.length > 0);

  requestAnimationFrame(drawLoop);
}

function drawShieldAt(cx, cy, t, alpha) {
  const radius = 90 + Math.sin(t * 2) * 6;
  // main ring glow
  utilCircle(cx, cy, radius, alpha * 0.06, radius + 6);
  utilCircle(cx, cy, radius, alpha * 0.4, radius - 6);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = "#008cff";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.stroke();
  // inner rotating octagon
  ctx.save();
  ctx.translate(cx, cy);
  ctx.rotate(t * 2);
  ctx.strokeStyle = "#66c2ff";
  ctx.lineWidth = 2;
  ctx.globalAlpha = alpha * 0.7;
  ctx.beginPath();
  for (let i = 0; i < 8; i++) {
    const ang = (i / 8) * Math.PI * 2;
    const x = (radius - 20) * Math.cos(ang);
    const y = (radius - 20) * Math.sin(ang);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
  ctx.globalAlpha = 1;
}

function utilCircle(cx, cy, radius, alpha, fillRadius) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = "#008cff";
  ctx.beginPath();
  ctx.arc(cx, cy, fillRadius, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalAlpha = 1;
}

// ---- Start / permission ----

btnStart.addEventListener("click", async () => {
  if (running) {
    stopAll();
    return;
  }
  hideError();
  setStatus("Setting up model…");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    // wait for metadata
    await new Promise((resolve) => {
      if (video.readyState >= 1) resolve();
      else video.onloadedmetadata = resolve;
    });
    await video.play();
    await initLandmarker();
    running = true;
    lastTimestamp = performance.now();
    btnStart.textContent = "■ Stop";
    btnStart.classList.remove("primary");
    setStatus("Starting…");
    requestAnimationFrame(drawLoop);
  } catch (err) {
    console.error(err);
    showError(`Could not start: ${err.name}. Please allow camera access and ensure you're on HTTPS.`);
    setStatus("Error");
  }
});

function stopAll() {
  running = false;
  if (video.srcObject) {
    video.srcObject.getTracks().forEach((tr) => tr.stop());
    video.srcObject = null;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  btnStart.textContent = "▶ Start Camera";
  btnStart.classList.add("primary");
  setStatus("Stopped");
}
