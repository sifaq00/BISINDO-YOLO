import { useEffect, useRef, useState } from 'react';
import * as ort from 'onnxruntime-web';
import cv from '@techstark/opencv-js'; // Untuk preprocess
import labels from '../utils/labels.json'; // Label kelas BISINDO-mu

const MODEL_PATH = '/model/modelfix.onnx'; // Ganti jika nama berbeda
const NMS_PATH = '/model/nms-yolov8.onnx';
const INPUT_SHAPE = [1, 3, 640, 640]; // Default YOLOv8
const IOU_THRESHOLD = 0.4;
const SCORE_THRESHOLD = 0.2;
const TOPK = 100;

const Detection = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isRunning, setIsRunning] = useState(false);
  const [session, setSession] = useState(null);
  const [nmsSession, setNmsSession] = useState(null);

  // Load models saat component mount
  useEffect(() => {
    const loadModels = async () => {
      try {
        ort.InferenceSession.create(MODEL_PATH, { executionProviders: ['wasm'] }).then(setSession);
        ort.InferenceSession.create(NMS_PATH, { executionProviders: ['wasm'] }).then(setNmsSession);
      } catch (error) {
        console.error('Error loading models:', error);
      }
    };
    loadModels();
  }, []);

  // Start webcam
  const startWebcam = async () => {
    try {
      setIsRunning(true);
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
    } catch (error) {
      console.error('Error accessing webcam:', error);
    }
  };

  // Stop webcam
  const stopWebcam = () => {
    if (videoRef.current) {
      const stream = videoRef.current.srcObject;
      stream?.getTracks().forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setIsRunning(false);
    }
  };

  // Process frame secara kontinu
  useEffect(() => {
    let intervalId;
    if (isRunning && session && nmsSession) {
      intervalId = setInterval(async () => {
        if (videoRef.current && canvasRef.current) {
          const ctx = canvasRef.current.getContext('2d');
          ctx.drawImage(videoRef.current, 0, 0, 640, 480); // Draw frame ke canvas

          // Preprocess: Gunakan OpenCV untuk resize dan normalize
          const img = cv.imread(canvasRef.current);
          cv.cvtColor(img, img, cv.COLOR_RGBA2RGB);
          cv.resize(img, img, new cv.Size(640, 640));
          const input = new Float32Array(1 * 3 * 640 * 640);
          for (let i = 0; i < 640 * 640; i++) {
            input[i] = img.data[i * 3] / 255.0; // R
            input[i + 640*640] = img.data[i * 3 + 1] / 255.0; // G
            input[i + 2*640*640] = img.data[i * 3 + 2] / 255.0; // B
          }
          const tensor = new ort.Tensor('float32', input, INPUT_SHAPE);

          // Inference
          const { output } = await session.run({ input: tensor });
          const boxes = output.data; // Output YOLOv8: [batch, num_boxes, 4+num_classes]

          // Postprocess dengan NMS
          const nmsInput = { boxes: new ort.Tensor('float32', boxes, [1, boxes.length / (4 + labels.length), 4 + labels.length]), scores: SCORE_THRESHOLD, iou_threshold: IOU_THRESHOLD, top_k: TOPK };
          const nmsOutput = await nmsSession.run(nmsInput);
          const selectedBoxes = nmsOutput.selected_indices.data; // Filtered boxes

          // Draw boxes
          ctx.clearRect(0, 0, 640, 480);
          selectedBoxes.forEach(box => {
            const [classId, x, y, w, h, score] = box; // Parse sesuai output
            if (score > SCORE_THRESHOLD) {
              ctx.strokeStyle = 'red';
              ctx.lineWidth = 2;
              ctx.strokeRect(x, y, w, h);
              ctx.fillStyle = 'red';
              ctx.fillText(`${labels[classId]} (${score.toFixed(2)})`, x, y - 5);
            }
          });
        }
      }, 100); // Jalankan setiap 100ms untuk ~10 FPS, sesuaikan untuk performa
    }

    return () => clearInterval(intervalId);
  }, [isRunning, session, nmsSession]);

  return (
    <div className="w-full max-w-2xl">
      <video ref={videoRef} className="hidden" width="640" height="480" autoPlay muted />
      <canvas ref={canvasRef} className="w-full h-auto border-2 border-gray-300" width="640" height="480" />
      <div className="mt-4 flex justify-center space-x-4">
        <button onClick={isRunning ? stopWebcam : startWebcam} className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
          {isRunning ? 'Stop' : 'Start'} Webcam
        </button>
      </div>
    </div>
  );
};

export default Detection;