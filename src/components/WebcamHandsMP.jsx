import { useEffect, useRef, useState } from "react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { MdCameraswitch } from "react-icons/md";

// ======= Tuning tampilan (samakan rasa dengan komponen kamu) =======
const FONT_SCALE = 4.0;  // skala font relatif ketebalan bbox
const PAD_X_SCALE = 2.0; // padding horizontal label relatif lineW
const PAD_Y_SCALE = 1.2; // padding vertikal label relatif lineW

// Warna konsisten untuk Left/Right (boleh ganti sesuka hati)
const COLOR_LEFT  = "hsl(140deg 90% 55%)"; // hijau
const COLOR_RIGHT = "hsl(210deg 90% 55%)"; // biru

export default function WebcamHandsMP() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // MediaPipe refs
  const mpRef = useRef({ vision: null, hand: null });
  const modelLoadingRef = useRef(true);

  // runtime refs
  const rafId = useRef(null);
  const cameraOnRef = useRef(false);
  const streamRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  // UI state
  const [loading, setLoading] = useState("Memuat model MediaPipe…");
  const [fps, setFps] = useState(0);
  const [cameraOn, setCameraOn] = useState(false);
  const [facing, setFacing] = useState("user"); // "user" | "environment"
  const [switching, setSwitching] = useState(false);

  // FPS
  const fpsFrames = useRef(0);
  const fpsLast = useRef(performance.now());

  // ====== LOAD MEDIAPIPE HAND LANDMARKER ======
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          // WASM path (pakai CDN). Ganti ke host lokal kalau perlu.
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        const hand = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            // Model dari CDN. Untuk offline, taruh file .task di public/ dan ubah pathnya.
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU",
          },
          numHands: 2,
          runningMode: "VIDEO",
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (cancelled) return;
        mpRef.current = { vision, hand };
        modelLoadingRef.current = false;
        setLoading(null);
      } catch (e) {
        console.error(e);
        setLoading("Gagal memuat MediaPipe Hand Landmarker.");
      }
    })();

    return () => {
      cancelled = true;
      try { mpRef.current.hand?.close?.(); } catch {}
    };
  }, []);

  // ====== START/STOP CAMERA (mirip punyamu) ======
  function clearCanvas() {
    const c = canvasRef.current;
    if (c) {
      const ctx = c.getContext("2d");
      ctx?.clearRect(0, 0, c.width, c.height);
    }
  }

  function stopWebcam() {
    cameraOnRef.current = false;
    setCameraOn(false);

    if (rafId.current) {
      cancelAnimationFrame(rafId.current);
      rafId.current = null;
    }

    const v = videoRef.current;
    const stream = streamRef.current || (v ? v.srcObject : null);
    try { if (stream) stream.getTracks().forEach((t) => t.stop()); } catch {}
    if (v) v.srcObject = null;
    streamRef.current = null;

    lastVideoTimeRef.current = -1;
    clearCanvas();
  }

  async function startWithConstraints(constraints) {
    stopWebcam();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { ...constraints, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    streamRef.current = stream;

    const v = videoRef.current;
    v.srcObject = stream;

    return new Promise((resolve) => {
      v.onloadedmetadata = () => {
        v.play();
        cameraOnRef.current = true;
        setCameraOn(true);

        // reset FPS
        fpsFrames.current = 0;
        fpsLast.current = performance.now();

        rafId.current = requestAnimationFrame(loop);
        resolve();
      };
    });
  }

  async function startWebcamWithFacing(targetFacing = facing) {
    if (modelLoadingRef.current) {
      alert("Model belum siap. Tunggu sebentar…");
      return;
    }
    setSwitching(true);
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
            if (
              targetFacing === "environment" &&
              (name.includes("back") || name.includes("rear") || name.includes("environment"))
            ) { deviceId = cam.deviceId; break; }
            if (targetFacing === "user" &&
                (name.includes("front") || name.includes("user"))) { deviceId = cam.deviceId; break; }
          }
          if (!deviceId && cams.length) deviceId = cams[0].deviceId;
          await startWithConstraints({ deviceId });
        }
      }
      setFacing(targetFacing);
    } catch (e) {
      console.error("Tidak dapat mengakses kamera:", e);
      alert("Tidak dapat mengakses kamera. Pastikan izin kamera aktif.");
    } finally {
      setSwitching(false);
    }
  }

  // ====== LOOP (panggil MediaPipe detectForVideo) ======
  function loop() {
    if (!cameraOnRef.current) return;

    const hand = mpRef.current.hand;
    const v = videoRef.current;
    if (!hand || !v) return;

    // Hindari inferensi berulang di frame yang sama
    if (v.currentTime === lastVideoTimeRef.current) {
      rafId.current = requestAnimationFrame(loop);
      return;
    }
    lastVideoTimeRef.current = v.currentTime;

    // Jalankan MP (koordinat landmark x,y ter-normalisasi [0..1]) :contentReference[oaicite:2]{index=2}
    const nowMs = performance.now();
    const res = hand.detectForVideo(v, nowMs);

    drawResults(res);

    // FPS kalkulasi
    fpsFrames.current += 1;
    const now = performance.now();
    const dt = now - fpsLast.current;
    if (dt >= 500) {
      setFps((fpsFrames.current * 1000) / dt);
      fpsFrames.current = 0;
      fpsLast.current = now;
    }

    rafId.current = requestAnimationFrame(loop);
  }

  // ====== DRAW (bbox + label background warna sesuai tangan) ======
  function drawResults(res) {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d");

    // Hi-DPI
    const dpr = window.devicePixelRatio || 1;
    const wCss = video.videoWidth || 640;
    const hCss = video.videoHeight || 480;
    canvas.width  = Math.round(wCss * dpr);
    canvas.height = Math.round(hCss * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, wCss, hCss);
    ctx.drawImage(video, 0, 0, wCss, hCss);

    const minSide = Math.min(wCss, hCss);
    const lineW  = Math.max(6, Math.round(minSide / 110));
    const fontPx = Math.max(20, Math.round(lineW * FONT_SCALE));
    const padX   = Math.max(10, Math.round(lineW * PAD_X_SCALE));
    const padY   = Math.max(6,  Math.round(lineW * PAD_Y_SCALE));

    ctx.textBaseline = "top";
    ctx.lineJoin = "round";
    ctx.miterLimit = 2;
    ctx.font = `600 ${fontPx}px Inter, Arial, sans-serif`;

    if (!res) return;

    const hands = res.landmarks || [];           // array of 21-pts per hand
    const handsLR = res.handedness || [];        // Left/Right per hand, dengan score :contentReference[oaicite:3]{index=3}

    for (let i = 0; i < hands.length; i++) {
      const lm = hands[i]; // [{x,y,z}...], x,y normalized [0..1]
      if (!lm || lm.length === 0) continue;

      // bbox dari min/max landmark
      let minX = +Infinity, minY = +Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const p of lm) {
        const px = p.x * wCss;
        const py = p.y * hCss;
        if (px < minX) minX = px;
        if (py < minY) minY = py;
        if (px > maxX) maxX = px;
        if (py > maxY) maxY = py;
      }

      // padding kecil biar tidak mepet
      const pad = Math.max(4, Math.round(lineW * 1.2));
      const x = Math.max(0, minX - pad);
      const y = Math.max(0, minY - pad);
      const w = Math.min(wCss - x, (maxX - minX) + pad * 2);
      const h = Math.min(hCss - y, (maxY - minY) + pad * 2);

      // Label: Left/Right + skor
      let which = "Hand";
      let score = 0;
      if (handsLR[i]?.[0]) {
        which = handsLR[i][0].categoryName || which; // "Left" / "Right"
        score = handsLR[i][0].score ?? 0;           // 0..1
      }

      const color = which.toLowerCase() === "right" ? COLOR_RIGHT : COLOR_LEFT;

      // BBOX
      ctx.lineWidth   = lineW;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      // Label background = warna bbox, teks hitam
      const text = `${which} (${score.toFixed(2)})`;
      const tw = ctx.measureText(text).width;
      const th = fontPx + padY;

      let lx = x - Math.floor(lineW / 2);
      let ly = y - th - lineW;
      if (ly < lineW) ly = y + lineW;

      ctx.fillStyle = color;
      ctx.fillRect(lx, ly, tw + padX, th);
      ctx.fillStyle = "#000";
      ctx.fillText(text, lx + Math.round(padX / 2), ly + Math.round((th - fontPx) / 2));
    }
  }

  // ====== Auto OFF saat back/refresh/tab hidden (mirip punyamu) ======
  useEffect(() => {
    const off = () => stopWebcam();
    const onVis = () => { if (document.visibilityState === "hidden") stopWebcam(); };

    window.addEventListener("pagehide", off);
    window.addEventListener("beforeunload", off);
    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("popstate", off);
    window.addEventListener("hashchange", off);

    return () => {
      window.removeEventListener("pagehide", off);
      window.removeEventListener("beforeunload", off);
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("popstate", off);
      window.removeEventListener("hashchange", off);
      stopWebcam();
      try { mpRef.current.hand?.close?.(); } catch {}
    };
  }, []);

  return (
    <div className="relative w-full">
      {loading && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-gray-900/60 rounded-lg">
          <p className="text-2xl animate-pulse">{loading}</p>
        </div>
      )}

      <div className="mb-3 flex flex-wrap items-center gap-2">
        {!cameraOn ? (
          <button
            onClick={() => startWebcamWithFacing(facing)}
            disabled={!!loading || switching}
            className="px-3 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50"
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
          className="absolute inset-0 w-full h-full object-contain"
        />

        {/* Switch icon */}
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

        {/* FPS badge */}
        <div className="absolute left-3 top-3 z-10 px-2 py-1 rounded bg-black/60 text-white text-xs">
          FPS: {fps.toFixed(1)}
        </div>
      </div>
    </div>
  );
}
