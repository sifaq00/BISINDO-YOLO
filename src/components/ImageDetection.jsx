import { useRef, useState } from "react";
import { FaUpload } from "react-icons/fa";
import labels from "../utils/labels.json";

// ====== Konfigurasi backend ======
const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const DETECT_URL = `${API_BASE}/detect`;

// ====== Logging ======
const DEBUG = (import.meta.env.VITE_DEBUG_CONSOLE ?? "true") !== "false";
function dlog(...args) { if (DEBUG) console.log(...args); }
function derr(...args) { if (DEBUG) console.error(...args); }
function group(name, collapsed = true) {
  if (!DEBUG) return { end: () => {} };
  const fn = collapsed ? console.groupCollapsed : console.group;
  fn.call(console, name);
  return { end: () => console.groupEnd() };
}

// Tuning tampilan
const FONT_SCALE = 4.0;
function classColorFromId(id) {
  const h = (Math.abs(id) * 137.508) % 360;
  return `hsl(${h}deg 90% 55%)`;
}

export default function ImageDetection() {
  const canvasRef = useRef(null);
  const [loading, setLoading] = useState(null);

  const handleImageUpload = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const g = group(`[UPLOAD] ${f.name} (${(f.size/1024).toFixed(1)} KB, ${f.type})`, false);
    const img = new Image();
    img.src = URL.createObjectURL(f);
    img.onload = async () => {
      dlog("  · natural size:", img.naturalWidth, "x", img.naturalHeight);
      URL.revokeObjectURL(img.src);
      await runDetectionOnImage(img);
      g.end?.();
    };
  };

  async function runDetectionOnImage(imgEl) {
    const g = group("[DETECT Image] send");
    try {
      setLoading("Memproses…");

      // Kirim gambar sebagai dataURL
      const off = document.createElement("canvas");
      off.width = imgEl.naturalWidth;
      off.height = imgEl.naturalHeight;
      const octx = off.getContext("2d");
      octx.drawImage(imgEl, 0, 0);
      const dataUrl = off.toDataURL("image/jpeg", 0.9);

      const approxBytes = Math.round((dataUrl.length * 3) / 4);
      dlog("  · payload ~", (approxBytes/1024).toFixed(1), "KB");

      const t0 = performance.now();
      const res = await fetch(DETECT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataUrl }),
      });
      const t1 = performance.now();
      if (!res.ok) throw new Error(`HTTP ${res.status}`);

      const detections = await res.json();
      dlog("  ✔ response in", (t1 - t0).toFixed(1), "ms | count:", Array.isArray(detections) ? detections.length : 0);
      if (Array.isArray(detections)) {
        console.table(detections.map(d => ({
          class: d.className ?? labels[Math.round(d.classId ?? -1)] ?? d.classId,
          score: (d.score ?? 0).toFixed(3),
          x1: Math.round(d.x1), y1: Math.round(d.y1),
          x2: Math.round(d.x2), y2: Math.round(d.y2),
        })));
      }

      drawResult(imgEl, Array.isArray(detections) ? detections : []);
    } catch (e) {
      derr(e);
      alert("Gagal mendeteksi gambar.");
    } finally {
      setLoading(null);
      g.end?.();
    }
  }

  function drawResult(img, dets) {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const w = img.naturalWidth;
    const h = img.naturalHeight;

    // Hi-DPI
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // gambar asli
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(img, 0, 0, w, h);

    // tebal & font
    const minSide = Math.min(w, h);
    const lineW = Math.max(6, Math.round(minSide / 110));
    const fontPx = Math.max(20, Math.round(lineW * FONT_SCALE));
    const padX = Math.max(10, Math.round(lineW * 2.0));
    const padY = Math.max(6, Math.round(lineW * 1.2));

    ctx.textBaseline = "top";
    ctx.lineJoin = "round";
    ctx.miterLimit = 2;
    ctx.font = `600 ${fontPx}px Inter, Arial, sans-serif`;

    const g = group("[DRAW] results");
    for (const d of dets) {
      const x = d.x1, y = d.y1, bw = d.x2 - d.x1, bh = d.y2 - d.y1;
      const classId = Math.round(d.classId ?? -1);
      const color = classColorFromId(classId);

      // bbox
      ctx.lineWidth = lineW;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, bw, bh);

      // label
      const label = d.className ?? labels[classId] ?? `cls ${classId}`;
      const text = `${label} (${(d.score ?? 0).toFixed(2)})`;

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
    g.end?.();
  }

  return (
    <div className="w-full max-w-4xl mx-auto flex flex-col items-center">
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900/70 rounded-lg z-20">
          <p className="text-2xl animate-pulse">{loading}</p>
        </div>
      )}

      <div className="w-full mb-4">
        <label
          htmlFor="image-upload"
          className="cursor-pointer w-full flex flex-col items-center justify-center border-2 border-dashed border-gray-500 hover:border-purple-500 rounded-lg p-6 transition-colors duration-300"
        >
          <FaUpload className="text-4xl text-gray-400 mb-2" />
          <span className="text-lg font-semibold">Pilih atau jatuhkan gambar di sini</span>
        </label>
        <input id="image-upload" type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
      </div>

      <div className="w-full">
        <canvas ref={canvasRef} className="w-full h-auto rounded-lg" />
      </div>
    </div>
  );
}
