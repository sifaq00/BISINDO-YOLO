import { useEffect, useRef, useState } from "react";
import { MdCameraswitch } from "react-icons/md";
import labels from "../utils/labels.json";

// ====== Konfigurasi backend ======
const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const DETECT_URL = `${API_BASE}/detect`;

// ====== Tuning & tampilan ======
const DETECT_INTERVAL_MS = 80;     // jeda antar request deteksi (≈12.5 Hz)
const MAX_CANVAS_DPR = 1.5;        // clamp DPR biar overlay hemat
const FONT_SCALE = 4.0;            // font relatif ketebalan bbox

// ====== Smoothing / tracking ======
const MATCH_IOU = 0.3;             // ambang IoU untuk match track
const TRACK_TTL_MS = 400;          // track dihapus jika tak terlihat selama ini
const LERP_SPEED_PER_SEC = 8;      // makin besar makin cepat ngejar target (8-12 enak)

// ====== Logging ======
const DEBUG = (import.meta.env.VITE_DEBUG_CONSOLE ?? "true") !== "false";
let DETECT_SEQ = 0;
const dlog = (...a) => { if (DEBUG) console.log(...a); };
const dwarn = (...a) => { if (DEBUG) console.warn(...a); };
const derr = (...a) => { if (DEBUG) console.error(...a); };
const group = (name, collapsed = true) => {
  if (!DEBUG) return { end: () => {} };
  const fn = collapsed ? console.groupCollapsed : console.group;
  fn.call(console, name);
  return { end: () => console.groupEnd() };
};

// Warna deterministik per classId (golden-angle)
function classColorFromId(id) {
  const h = (Math.abs(id) * 137.508) % 360;
  return `hsl(${h}deg 90% 55%)`;
}

// ====== Util IoU ======
function iou(a, b) {
  // a,b dalam {x,y,w,h} (sumber coords: piksel video)
  const ax2 = a.x + a.w, ay2 = a.y + a.h;
  const bx2 = b.x + b.w, by2 = b.y + b.h;
  const ix = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
  const iy = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
  const inter = ix * iy;
  const ua = a.w * a.h + b.w * b.h - inter;
  return ua <= 0 ? 0 : inter / ua;
}

export default function WebcamDetection() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // stream/runtime
  const streamRef = useRef(null);
  const cameraOnRef = useRef(false);
  const rafId = useRef(null);

  // network/runtime
  const sendingRef = useRef(false);
  const abortRef = useRef(null);
  const detectTimerRef = useRef(null);

  // FPS berbasis respons server (deteksi)
  const fpsCount = useRef(0);
  const fpsLastT = useRef(performance.now());
  const [fps, setFps] = useState(0);

  // ====== FPS kamera (baru) ======
  const [camFps, setCamFps] = useState(0);
  const camLastT = useRef(0);

  // UI
  const [cameraOn, setCameraOn] = useState(false);
  const [facing, setFacing] = useState("user"); // "user" | "environment"
  const [switching, setSwitching] = useState(false);

  // ====== Tracks (smoothing) ======
  // Track: { id, classId, className, score, lastSeen, display:{x,y,w,h}, target:{x,y,w,h} }
  const tracksRef = useRef([]);
  const nextTrackId = useRef(1);
  const lastOverlayT = useRef(performance.now());

  useEffect(() => {
    dlog("[CFG] API_BASE:", API_BASE);
    dlog("[CFG] DETECT_URL:", DETECT_URL);
  }, []);

  // ====== Overlay (tanpa draw video) ======
  function drawOverlay() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    const ctx = canvas.getContext("2d", { alpha: true });
    const rect = canvas.getBoundingClientRect();
    const wCss = rect.width;
    const hCss = rect.height;

    const dpr = Math.min(MAX_CANVAS_DPR, window.devicePixelRatio || 1);
    canvas.width = Math.round(wCss * dpr);
    canvas.height = Math.round(hCss * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, wCss, hCss);

    // letterbox karena object-contain
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const scale = Math.min(wCss / vw, hCss / vh);
    const drawW = vw * scale;
    const drawH = vh * scale;
    const offX = (wCss - drawW) / 2;
    const offY = (hCss - drawH) / 2;

    // ketebalan & font adaptif (berdasar area gambar)
    const minSide = Math.min(drawW, drawH);
    const lineW = Math.max(6, Math.round(minSide / 110));
    const fontPx = Math.max(20, Math.round(lineW * FONT_SCALE));
    const padX = Math.max(10, Math.round(lineW * 2.0));
    const padY = Math.max(6, Math.round(lineW * 1.2));

    ctx.textBaseline = "top";
    ctx.lineJoin = "round";
    ctx.miterLimit = 2;
    ctx.font = `600 ${fontPx}px Inter, Arial, sans-serif`;

    const now = performance.now();
    const dt = Math.min(100, now - lastOverlayT.current) / 1000; // dt detik, capped 100ms
    lastOverlayT.current = now;

    // bersihkan track kadaluwarsa + animasikan display → target
    const tracks = tracksRef.current.filter(tr => (now - tr.lastSeen) <= TRACK_TTL_MS);

    const k = 1 - Math.exp(-LERP_SPEED_PER_SEC * dt); // koefisien lerp independen fps
    for (const tr of tracks) {
      // lerp posisi & ukuran
      tr.display.x += (tr.target.x - tr.display.x) * k;
      tr.display.y += (tr.target.y - tr.display.y) * k;
      tr.display.w += (tr.target.w - tr.display.w) * k;
      tr.display.h += (tr.target.h - tr.display.h) * k;
    }
    tracksRef.current = tracks; // simpan kembali setelah pruning & lerp

    for (const tr of tracks) {
      const { x, y, w, h } = tr.display;

      // skala ke layar
      const sx = offX + x * scale;
      const sy = offY + y * scale;
      const sw = w * scale;
      const sh = h * scale;

      const color = classColorFromId(tr.classId);

      // bbox
      ctx.lineWidth = lineW;
      ctx.strokeStyle = color;
      ctx.strokeRect(sx, sy, sw, sh);

      // label
      const label = tr.className ?? labels[tr.classId] ?? `cls ${tr.classId}`;
      const text = `${label} (${tr.score.toFixed(2)})`;

      const tw = ctx.measureText(text).width;
      const th = fontPx + padY;

      let lx = sx - Math.floor(lineW / 2);
      let ly = sy - th - lineW;
      if (ly < offY + lineW) ly = sy + lineW;

      ctx.fillStyle = color;
      ctx.fillRect(lx, ly, tw + padX, th);

      ctx.fillStyle = "#000";
      ctx.fillText(text, lx + Math.round(padX / 2), ly + Math.round((th - fontPx) / 2));
    }
  }

  // rAF overlay
  function overlayLoop() {
    if (!cameraOnRef.current) return;
    drawOverlay();
    rafId.current = requestAnimationFrame(overlayLoop);
  }

  // ====== Tracking: update tracks dari detections baru ======
  function updateTracksFromDetections(dets) {
    const now = performance.now();
    const tracks = tracksRef.current.slice();

    // tandai semua belum dipakai di putaran ini
    for (const tr of tracks) tr._matched = false;

    // bentuk kandidat deteksi dalam {x,y,w,h}
    const detBoxes = dets.map(d => ({
      x: d.x1,
      y: d.y1,
      w: d.x2 - d.x1,
      h: d.y2 - d.y1,
      score: d.score ?? 0,
      classId: Math.round(d.classId ?? -1),
      className: d.className
    }));

    // greedy matching per deteksi → track (kelas harus sama, IoU tertinggi)
    for (const db of detBoxes) {
      let best = null;
      let bestIoU = 0;
      for (const tr of tracks) {
        if (tr._matched) continue;
        if (tr.classId !== db.classId) continue;
        const i = iou(tr.target, db);
        if (i > bestIoU) { bestIoU = i; best = tr; }
      }
      if (best && bestIoU >= MATCH_IOU) {
        // update target + score + lastSeen
        best.target = { x: db.x, y: db.y, w: db.w, h: db.h };
        best.score = best.score * 0.7 + db.score * 0.3;
        best.lastSeen = now;
        best._matched = true;
      } else {
        // buat track baru
        const id = nextTrackId.current++;
        tracks.push({
          id,
          classId: db.classId,
          className: db.className,
          score: db.score,
          lastSeen: now,
          display: { x: db.x, y: db.y, w: db.w, h: db.h }, // mulai dari posisi saat ini
          target:  { x: db.x, y: db.y, w: db.w, h: db.h },
          _matched: true
        });
      }
    }

    // buang track yang sudah terlalu lama tak terlihat (biar nggak nge-freeze)
    const kept = tracks.filter(tr => (now - tr.lastSeen) <= TRACK_TTL_MS);
    tracksRef.current = kept;
  }

  // ====== Snapshot & kirim ke backend ======
  function captureDataUrlFromVideo(quality = 0.75) {
    const v = videoRef.current;
    if (!v) return null;
    const w = v.videoWidth || 0;
    const h = v.videoHeight || 0;
    if (!w || !h) return null;

    const off = document.createElement("canvas");
    off.width = w;
    off.height = h;
    const octx = off.getContext("2d");
    octx.drawImage(v, 0, 0, w, h);
    return { dataUrl: off.toDataURL("image/jpeg", quality), w, h };
  }

  async function sendDetect() {
    if (sendingRef.current || !cameraOnRef.current) return;
    const snap = captureDataUrlFromVideo(0.75);
    if (!snap) return;

    const { dataUrl, w, h } = snap;
    const approxBytes = Math.round((dataUrl.length * 3) / 4);
    const approxKB = (approxBytes / 1024).toFixed(1);

    sendingRef.current = true;
    abortRef.current = new AbortController();
    const t0 = performance.now();
    const seq = ++DETECT_SEQ;

    const g = group(`[DETECT #${seq}] → sending (${w}x${h}, ~${approxKB} KB)`);
    try {
      const res = await fetch(DETECT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
        signal: abortRef.current.signal,
      });
      const t1 = performance.now();
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const detections = await res.json();
      const dets = Array.isArray(detections) ? detections : [];

      // ===> inti smoothing:
      updateTracksFromDetections(dets);

      // log ringkasan
      const clsCount = {};
      for (const d of dets) {
        const name = d.className ?? labels[Math.round(d.classId ?? -1)] ?? `cls ${d.classId}`;
        clsCount[name] = (clsCount[name] || 0) + 1;
      }
      console.log("✔ response in", (t1 - t0).toFixed(1), "ms | count:", dets.length);
      if (Object.keys(clsCount).length) console.log("class summary:", clsCount);

      // FPS deteksi (server)
      fpsCount.current += 1;
      const now = performance.now();
      const dt = now - fpsLastT.current;
      if (dt >= 500) {
        setFps((fpsCount.current * 1000) / dt);
        fpsCount.current = 0;
        fpsLastT.current = now;
      }
    } catch (e) {
      if (e?.name === "AbortError") {
        dwarn(`✖ request aborted (#${seq})`);
      } else {
        derr(`✖ detect error (#${seq}):`, e);
      }
    } finally {
      g.end?.();
      sendingRef.current = false;
    }
  }

  function scheduleDetect(delay = DETECT_INTERVAL_MS) {
    clearTimeout(detectTimerRef.current);
    detectTimerRef.current = setTimeout(async () => {
      if (!cameraOnRef.current) return;
      await sendDetect();
      scheduleDetect(); // loop
    }, delay);
  }

  // ====== Start/Stop kamera ======
  function resetTracks(reason = "") {
    tracksRef.current = [];
    nextTrackId.current = 1;
    lastOverlayT.current = performance.now();
    if (reason && DEBUG) console.info("[TRACK] reset:", reason);
  }

  function clearAllTimersAndRequests() {
    clearTimeout(detectTimerRef.current);
    detectTimerRef.current = null;
    try { abortRef.current?.abort(); } catch {}
    sendingRef.current = false;
  }

  function stopWebcam() {
    const g = group("[CAMERA] stop");
    cameraOnRef.current = false;
    setCameraOn(false);
    clearAllTimersAndRequests();

    if (rafId.current) {
      cancelAnimationFrame(rafId.current);
      rafId.current = null;
    }

    const v = videoRef.current;
    const s = streamRef.current || (v ? v.srcObject : null);
    try {
      s?.getTracks?.().forEach(t => {
        dlog("  · stopping track:", t.kind, t.label);
        t.stop();
      });
    } catch (e) { dwarn("  · stop tracks warn:", e); }
    if (v) {
      try { v.pause(); } catch {}
      v.srcObject = null;
    }
    streamRef.current = null;

    resetTracks("camera stopped");
    const ctx = canvasRef.current?.getContext("2d");
    if (ctx && canvasRef.current) ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    g.end?.();
  }

  async function startWithConstraints(constraints) {
    stopWebcam(); // bersih
    const g = group("[CAMERA] start", false);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { ...constraints, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30, max: 30 } },
        audio: false,
      });
      streamRef.current = stream;
      const v = videoRef.current;
      v.srcObject = stream;

      return new Promise((resolve) => {
        v.onloadedmetadata = () => {
          v.play();
          const track = stream.getVideoTracks?.()[0];
          dlog("  · facing:", constraints.facingMode ?? "deviceId");
          dlog("  · track settings:", track?.getSettings?.());
          dlog("  · track label:", track?.label);

          // ====== (BARU) hitung FPS kamera ======
          if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
            const step = (now /* DOMHighResTimeStamp */) => {
              if (!cameraOnRef.current) return;
              if (camLastT.current) {
                const dt = now - camLastT.current;
                const fpsNow = dt > 0 ? 1000 / dt : 0;
                setCamFps(prev => prev ? prev * 0.8 + fpsNow * 0.2 : fpsNow);
              }
              camLastT.current = now;
              v.requestVideoFrameCallback(step);
            };
            v.requestVideoFrameCallback(step);
          } else {
            // fallback: estimasi via rAF
            let last = performance.now();
            const loop = () => {
              if (!cameraOnRef.current) return;
              const now = performance.now();
              const dt = now - last;
              last = now;
              const fpsNow = dt > 0 ? 1000 / dt : 0;
              setCamFps(prev => prev ? prev * 0.8 + fpsNow * 0.2 : fpsNow);
              requestAnimationFrame(loop);
            };
            requestAnimationFrame(loop);
          }
          // ====== END (BARU) ======

          resetTracks("camera started / size change");
          cameraOnRef.current = true;
          setCameraOn(true);

          rafId.current = requestAnimationFrame(overlayLoop);
          scheduleDetect(0);
          resolve();
          g.end?.();
        };
      });
    } catch (e) {
      g.end?.();
      throw e;
    }
  }

  async function startWebcamWithFacing(targetFacing = facing) {
    setSwitching(true);
    const g = group(`[CAMERA] switch → ${targetFacing}`);
    try {
      try {
        await startWithConstraints({ facingMode: { exact: targetFacing } });
      } catch {
        try {
          await startWithConstraints({ facingMode: targetFacing });
        } catch {
          const devices = await navigator.mediaDevices.enumerateDevices();
          const cams = devices.filter((d) => d.kind === "videoinput");
          let deviceId = null;
          for (const cam of cams) {
            const name = (cam.label || "").toLowerCase();
            if (targetFacing === "environment" && (name.includes("back") || name.includes("rear") || name.includes("environment"))) {
              deviceId = cam.deviceId; break;
            }
            if (targetFacing === "user" && (name.includes("front") || name.includes("user"))) {
              deviceId = cam.deviceId; break;
            }
          }
          if (!deviceId && cams.length) deviceId = cams[0].deviceId;
          await startWithConstraints({ deviceId });
        }
      }
      setFacing(targetFacing);
    } catch (e) {
      derr("Tidak dapat mengakses kamera:", e);
      alert("Tidak dapat mengakses kamera. Pastikan izin kamera aktif.");
    } finally {
      g.end?.();
      setSwitching(false);
    }
  }

  // ====== Auto-off saat pindah/refresh/tab sembunyi ======
  useEffect(() => {
    const off = () => { dlog("[LIFECYCLE] pagehide/beforeunload/popstate/hashchange"); stopWebcam(); };
    const onVis = () => { dlog("[LIFECYCLE] visibilitychange:", document.visibilityState); if (document.visibilityState === "hidden") stopWebcam(); };

    window.addEventListener("pagehide", off);
    window.addEventListener("beforeunload", off);
    window.addEventListener("popstate", off);
    window.addEventListener("hashchange", off);
    document.addEventListener("visibilitychange", onVis);
    dlog("[LIFECYCLE] listeners attached");

    return () => {
      window.removeEventListener("pagehide", off);
      window.removeEventListener("beforeunload", off);
      window.removeEventListener("popstate", off);
      window.removeEventListener("hashchange", off);
      document.removeEventListener("visibilitychange", onVis);
      dlog("[LIFECYCLE] listeners removed");
      stopWebcam();
    };
  }, []);

  return (
    <div className="relative w-full">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        {!cameraOn ? (
          <button
            onClick={() => startWebcamWithFacing(facing)}
            className="px-3 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500"
          >
            Nyalakan Kamera
          </button>
        ) : (
          <button
            onClick={stopWebcam}
            className="px-3 py-2 rounded-lg bg-rose-600 hover:bg-rose-500"
          >
            Matikan Kamera
          </button>
        )}

        <span className="text-sm opacity-70">
          Mode: {facing === "user" ? "Depan" : "Belakang"} {switching ? "(beralih…)" : ""}
        </span>
      </div>

      <div className="relative w-full aspect-video rounded-lg overflow-hidden shadow-lg">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="absolute inset-0 w-full h-full object-contain bg-black"
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* Switch camera icon */}
        <button
          type="button"
          onClick={() => {
            if (!cameraOnRef.current || switching) return;
            startWebcamWithFacing(facing === "user" ? "environment" : "user");
          }}
          title="Ganti kamera"
          className="absolute right-3 bottom-3 z-10 grid place-items-center w-12 h-12 rounded-full bg-black/60 hover:bg-black/70 backdrop-blur text-white disabled:opacity-50"
          disabled={!cameraOn || switching}
        >
          <MdCameraswitch className={`text-2xl ${switching ? "animate-spin" : ""}`} />
        </button>

        {/* FPS badge (deteksi server + kamera) */}
        <div className="absolute left-3 top-3 z-10 px-2 py-1 rounded bg-black/60 text-white text-xs">
          FPS: {fps.toFixed(1)} | Cam: {camFps.toFixed(1)}
        </div>
      </div>
    </div>
  );
}
