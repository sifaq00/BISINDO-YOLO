import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgpu"; // pastikan package ini terinstall
import labels from "../utils/labels.json";
import { MdCameraswitch } from "react-icons/md";

const MODEL_SIZE = 640;
const MAX_CANVAS_DPR = 1.5; // clamp DPR biar nggak boros VRAM

// ======== Tuning tampilan ========
const FONT_SCALE = 4.0;           // skala font relatif terhadap ketebalan bbox
const TEXT_THICKNESS_RATIO = 0.35; // (tidak dipakai untuk label bg)

// Warna deterministik per classId (golden-angle)
function classColorFromId(id) {
  const h = (Math.abs(id) * 137.508) % 360;
  const s = 90, l = 55;
  return `hsl(${h}deg ${s}% ${l}%)`;
}

export default function WebcamDetection() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const modelRef = useRef(null);

  // runtime refs
  const rafId = useRef(null);
  const lastInferT = useRef(0);
  const cameraOnRef = useRef(false);
  const streamRef = useRef(null);
  const detRef = useRef([]);
  const inferBusyRef = useRef(false);

  // FPS
  const fpsFrames = useRef(0);
  const fpsLast = useRef(performance.now());
  const [fps, setFps] = useState(0);

  const [loadingModel, setLoadingModel] = useState("Memuat model…");
  const [cameraOn, setCameraOn] = useState(false);
  const [facing, setFacing] = useState("user");    // "user" | "environment"
  const [switching, setSwitching] = useState(false);

  // ====== MODEL LOAD (prefer WebGPU) ======
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // coba webgpu → fallback webgl → wasm
        try {
          await tf.setBackend("webgpu");
        } catch {
          try {
            await tf.setBackend("webgl");
          } catch {
            await tf.setBackend("wasm");
          }
        }
        await tf.ready();

        const model = await tf.loadGraphModel("/bestlasttrain_web_model/model.json");

        // Warm-up (pakai executeAsync karena model kamu punya NMS V4)
        const dummy = tf.zeros([1, MODEL_SIZE, MODEL_SIZE, 3]);
        const warm = await model.executeAsync(dummy);
        if (Array.isArray(warm)) warm.forEach((t) => t.dispose());
        else warm?.dispose();
        dummy.dispose();

        if (!cancelled) {
          modelRef.current = model;
          setLoadingModel(null);
        }
      } catch (e) {
        console.error(e);
        setLoadingModel("Gagal memuat model.");
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // ====== PREPROCESS (letterbox 114/255) ======
  function preprocess(source) {
    const sw = source.videoWidth || source.naturalWidth || source.width;
    const sh = source.videoHeight || source.naturalHeight || source.height;

    const r0 = Math.min(MODEL_SIZE / sw, MODEL_SIZE / sh);
    const nw = Math.round(sw * r0);
    const nh = Math.round(sh * r0);
    const padX0 = (MODEL_SIZE - nw) / 2;
    const padY0 = (MODEL_SIZE - nh) / 2;

    const img = tf.browser.fromPixels(source);
    const imgFloat = img.toFloat().div(255);
    const resized = tf.image.resizeBilinear(imgFloat, [nh, nw], true);

    const padValue = 114 / 255;
    const top = Math.floor(padY0), bottom = Math.ceil(padY0);
    const left = Math.floor(padX0), right = Math.ceil(padX0);
    const padded = tf.pad(resized, [[top, bottom], [left, right], [0, 0]], padValue);

    const tensor = padded.expandDims(0);
    img.dispose(); imgFloat.dispose(); resized.dispose(); padded.dispose();
    return { tensor, r0, padX0, padY0 };
  }

  // ====== INFERENCE (executeAsync + dataSync) ======
  async function runOnce(source) {
    const model = modelRef.current;
    if (!model) return [];
    const { tensor, r0, padX0, padY0 } = preprocess(source);
    try {
      const outputs = await model.executeAsync(tensor);
      tensor.dispose();

      // cari tensor [1, N, 6]
      let detTensor = Array.isArray(outputs)
        ? outputs.find((t) => t.shape.length === 3 && t.shape.at(-1) === 6)
        : outputs;
      if (!detTensor) {
        if (Array.isArray(outputs)) outputs.forEach((t) => t.dispose());
        else outputs?.dispose();
        return [];
      }

      const shape = detTensor.shape;   // [1, N, 6]
      const N = shape[1] ?? 0;
      const buf = detTensor.dataSync(); // Float32Array N*6 (lebih hemat dari arraySync)
      const dets = [];

      for (let i = 0; i < N; i++) {
        const off = i * 6;
        const x1 = buf[off + 0];
        const y1 = buf[off + 1];
        const x2 = buf[off + 2];
        const y2 = buf[off + 3];
        const conf = buf[off + 4];
        const classId = buf[off + 5];
        if (conf >= 0.4) {
          dets.push({
            x: x1, y: y1, w: x2 - x1, h: y2 - y1,
            score: conf, classId, r0, padX0, padY0,
          });
        }
      }

      if (Array.isArray(outputs)) outputs.forEach((t) => t.dispose());
      else outputs?.dispose();

      return dets;
    } catch (e) {
      console.error(e);
      tensor.dispose();
      return [];
    }
  }

  // ====== DRAW (overlay-only: canvas TIDAK menggambar video) ======
  function drawFrame(dets) {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const ctx = canvas.getContext("2d", { alpha: true });

    // Pakai ukuran CSS elemen (overlay mengikuti <video> object-contain)
    const rect = canvas.getBoundingClientRect();
    const wCss = rect.width;
    const hCss = rect.height;

    const rawDpr = window.devicePixelRatio || 1;
    const dpr = Math.min(MAX_CANVAS_DPR, rawDpr);
    canvas.width  = Math.round(wCss * dpr);
    canvas.height = Math.round(hCss * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, wCss, hCss);

    // Hitung letterbox yang muncul di <video class="object-contain">
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    const scale = Math.min(wCss / vw, hCss / vh);
    const drawW = vw * scale;
    const drawH = vh * scale;
    const offX = (wCss - drawW) / 2;
    const offY = (hCss - drawH) / 2;

    // Tebal & besar adaptif (berdasar area gambar yang terlihat)
    const minSide = Math.min(drawW, drawH);
    const lineW  = Math.max(6, Math.round(minSide / 110));
    const fontPx = Math.max(20, Math.round(lineW * 4.0));
    const padX   = Math.max(10, Math.round(lineW * 2.0));
    const padY   = Math.max(6,  Math.round(lineW * 1.2));

    ctx.textBaseline = "top";
    ctx.lineJoin = "round";
    ctx.miterLimit = 2;
    ctx.font = `600 ${fontPx}px Inter, Arial, sans-serif`;

    for (const det of dets) {
      // Koordinat balik ke ruang video asli (setelah letterbox preprocess)
      const x = (det.x - det.padX0) / det.r0;
      const y = (det.y - det.padY0) / det.r0;
      const w = det.w / det.r0;
      const h = det.h / det.r0;

      // Skala ke layar (yang sudah di-letterbox object-contain)
      const sx = offX + x * scale;
      const sy = offY + y * scale;
      const sw = w * scale;
      const sh = h * scale;

      const classId = Math.round(det.classId);
      const color   = classColorFromId(classId);

      // BBOX
      ctx.lineWidth   = lineW;
      ctx.strokeStyle = color;
      ctx.strokeRect(sx, sy, sw, sh);

      // Label dengan background = warna bbox, teks hitam
      const label = labels[classId] ?? `cls ${classId}`;
      const text  = `${label} (${det.score.toFixed(2)})`;

      const tw = ctx.measureText(text).width;
      const th = fontPx + padY;

      let lx = sx - Math.floor(lineW / 2);
      let ly = sy - th - lineW;
      if (ly < offY + lineW) ly = sy + lineW;

      ctx.fillStyle = color;
      ctx.fillRect(lx, ly, tw + padX, th);

      ctx.fillStyle = "#000000";
      ctx.fillText(text, lx + Math.round(padX / 2), ly + Math.round((th - fontPx) / 2));
    }
  }

  // ====== LOOP (gunakan ref, bukan state) ======
  function loop(t = 0) {
    if (!cameraOnRef.current) return;

    // throttle & cegah overlap
    if (t - lastInferT.current > 70 && !inferBusyRef.current) {
      lastInferT.current = t;
      inferBusyRef.current = true;
      runOnce(videoRef.current)
        .then((d) => { detRef.current = d; })
        .finally(() => { inferBusyRef.current = false; });
    }

    // FPS hitung tiap ~0.5s
    fpsFrames.current += 1;
    const now = performance.now();
    const dt = now - fpsLast.current;
    if (dt >= 500) {
      setFps((fpsFrames.current * 1000) / dt);
      fpsFrames.current = 0;
      fpsLast.current = now;
    }

    // gambar selalu pakai deteksi terbaru
    drawFrame(detRef.current || []);
    rafId.current = requestAnimationFrame(loop);
  }

  // ====== START/STOP STREAM ======
  function clearCanvas() {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    ctx?.clearRect(0, 0, c.width, c.height);
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

    detRef.current = [];
    clearCanvas();
  }

  async function startWithConstraints(constraints) {
    stopWebcam(); // pastikan bersih
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
        cameraOnRef.current = true; // set ref dulu agar loop langsung jalan
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

  // ====== Matikan kamera saat back/refresh/tab hidden/SPA nav ======
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
      modelRef.current?.dispose?.();
    };
  }, []);

  return (
    <div className="relative w-full">
      {loadingModel && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-gray-900/60 rounded-lg">
          <p className="text-2xl animate-pulse">{loadingModel}</p>
        </div>
      )}

      <div className="mb-3 flex flex-wrap items-center gap-2">
        {!cameraOn ? (
          <button
            onClick={() => startWebcamWithFacing(facing)}
            disabled={!!loadingModel || switching}
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

        {/* Floating switch icon */}
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

        {/* FPS badge DOM */}
        <div className="absolute left-3 top-3 z-10 px-2 py-1 rounded bg-black/60 text-white text-xs">
          FPS: {fps.toFixed(1)}
        </div>
      </div>
    </div>
  );
}
