import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import labels from "../utils/labels.json";
import { FaUpload } from "react-icons/fa";

const MODEL_SIZE = 640;

// ======== Tuning tampilan (samakan dengan webcam) ========
const FONT_SCALE = 4.0;            // skala font relatif ketebalan bbox
const PAD_X_SCALE = 2.0;           // padding horizontal label relatif lineW
const PAD_Y_SCALE = 1.2;           // padding vertikal label relatif lineW

// Warna deterministik per classId (golden angle supaya merata & kontras)
function classColorFromId(id) {
  const h = (Math.abs(id) * 137.508) % 360;
  const s = 90;
  const l = 55;
  return `hsl(${h}deg ${s}% ${l}%)`;
}

export default function ImageDetection() {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState("Memuat model...");
  const canvasRef = useRef(null);

  useEffect(() => {
    (async () => {
      try {
        await tf.setBackend("webgl");
        await tf.ready();

        const m = await tf.loadGraphModel("/bestlasttrain_web_model/model.json");
        // Warm-up
        const dummy = tf.zeros([1, MODEL_SIZE, MODEL_SIZE, 3]);
        const warm = await m.executeAsync(dummy);
        if (Array.isArray(warm)) warm.forEach((t) => t.dispose());
        else warm?.dispose();
        dummy.dispose();

        setModel(m);
        setLoading(null);
      } catch (e) {
        console.error(e);
        setLoading("Gagal memuat model.");
      }
    })();
  }, []);

  // === Letterbox preprocess (114/255) + return r0 & pad untuk mapping balik
  function preprocess(source) {
    const sw = source.naturalWidth || source.width;
    const sh = source.naturalHeight || source.height;

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

  async function detect(img) {
    if (!model) return [];
    const { tensor, r0, padX0, padY0 } = preprocess(img);
    try {
      const outputs = await model.executeAsync(tensor);
      tensor.dispose();

      let detTensor = Array.isArray(outputs)
        ? outputs.find((t) => t.shape.length === 3 && t.shape.at(-1) === 6)
        : outputs;
      if (!detTensor) {
        if (Array.isArray(outputs)) outputs.forEach((t) => t.dispose());
        else outputs?.dispose();
        return [];
      }

      const raw = detTensor.arraySync()[0]; // [N,6] -> [x1,y1,x2,y2,conf,cls]
      const dets = [];
      for (let i = 0; i < raw.length; i++) {
        const [x1, y1, x2, y2, conf, classId] = raw[i];
        if (conf >= 0.4) {
          dets.push({
            x: x1, y: y1, w: x2 - x1, h: y2 - y1,
            score: conf, classId, r0, padX0, padY0
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

  const handleImageUpload = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const img = new Image();
    img.src = URL.createObjectURL(f);
    img.onload = async () => {
      URL.revokeObjectURL(img.src);
      const dets = await detect(img);
      draw(img, dets);
    };
  };

  function draw(img, dets) {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const imgW = img.naturalWidth;
    const imgH = img.naturalHeight;

    // Hi-DPI scaling
    const dpr = window.devicePixelRatio || 1;
    canvas.width  = Math.round(imgW * dpr);
    canvas.height = Math.round(imgH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, imgW, imgH);
    ctx.drawImage(img, 0, 0, imgW, imgH);

    // Ketebalan & ukuran font adaptif (tebal & besar)
    const minSide = Math.min(imgW, imgH);
    const lineW  = Math.max(6, Math.round(minSide / 110));                // tebal bbox
    const fontPx = Math.max(20, Math.round(lineW * FONT_SCALE));          // besar font
    const padX   = Math.max(10, Math.round(lineW * PAD_X_SCALE));         // padding label X
    const padY   = Math.max(6,  Math.round(lineW * PAD_Y_SCALE));         // padding label Y

    ctx.textBaseline = "top";
    ctx.lineJoin = "round";
    ctx.miterLimit = 2;

    dets.forEach((d) => {
      const x = (d.x - d.padX0) / d.r0;
      const y = (d.y - d.padY0) / d.r0;
      const w = d.w / d.r0;
      const h = d.h / d.r0;

      const classId = Math.round(d.classId);
      const color = classColorFromId(classId);

      // BBOX (warna per-kelas)
      ctx.lineWidth   = lineW;
      ctx.strokeStyle = color;
      ctx.strokeRect(x, y, w, h);

      // Label dengan background = warna bbox, teks hitam
      const label = labels[classId] ?? `cls ${classId}`;
      const text  = `${label} (${d.score.toFixed(2)})`;

      ctx.font = `600 ${fontPx}px Inter, Arial, sans-serif`; // semi-bold biar rapi
      const tw = ctx.measureText(text).width;
      const th = fontPx + padY;

      // posisi default: di atas-kiri bbox; kalau mepet atas, taruh di dalam
      let lx = x - Math.floor(lineW / 2);
      let ly = y - th - lineW;
      if (ly < lineW) ly = y + lineW;

      // background label warna kelas
      ctx.fillStyle = color;
      ctx.fillRect(lx, ly, tw + padX, th);

      // teks hitam
      ctx.fillStyle = "#000000";
      ctx.fillText(
        text,
        lx + Math.round(padX / 2),
        ly + Math.round((th - fontPx) / 2)
      );
    });
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
