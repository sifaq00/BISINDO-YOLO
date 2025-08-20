import { useEffect, useRef, useState } from 'react';
// Import TensorFlow.js.  This library is loaded as an ES module from node_modules
// and provides the primitives to execute the exported YOLOv8 model in the browser.
import * as tf from '@tensorflow/tfjs';
// Import label definitions for your sign language classes.  The file
// `labels.json` should map class indices to their string names (e.g. A–Z).
import labels from '../utils/labels.json';

/*
 * DetectionTFJS is a React component that performs real‑time sign language
 * detection in the browser using a TensorFlow.js model exported from
 * Ultralytics.  Unlike the earlier ONNX version that relied on a Web
 * Worker, this implementation runs directly in the main thread using
 * TensorFlow.js.  It performs the following high‑level steps:
 *
 * 1. Load the model (a GraphModel) once on mount.
 * 2. Start the webcam stream when the user clicks the “Start” button.
 * 3. On a regular interval, capture a frame from the webcam, letterbox it
 *    to 640×640, normalise pixel values to [0,1], and perform inference.
 * 4. Parse the output tensor of shape (1, N, 6) where each entry is
 *    [x1, y1, x2, y2, confidence, class].  Filter by confidence threshold
 *    and save detections to a ref.
 * 5. Draw the video frame and bounding boxes on a canvas at the browser’s
 *    resolution, taking into account letterboxing and optional horizontal
 *    mirroring for user convenience.
 */

const Detection = () => {
  // Refs to the <video> and <canvas> elements.  These are used to draw
  // frames and detections without re-rendering the component on every
  // animation frame.
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Ref to hold the loaded TensorFlow.js model.  We don’t store this in
  // state because it’s not part of the UI state – mutating the ref will
  // not trigger a re-render.
  const modelRef = useRef(null);

  // Ref to store the current list of detections.  Each detection is
  // {x, y, w, h, conf, classId} with coordinates in the 640×640 letterbox
  // coordinate space expected by the model.  We draw these in the
  // `mainLoop` after converting them back to the display coordinate space.
  const detectionsRef = useRef([]);

  // State for uploaded image detection
  const [uploadImage, setUploadImage] = useState(null);
  const [uploadDetections, setUploadDetections] = useState([]);
  // Canvas ref for displaying uploaded image and its detections
  const uploadCanvasRef = useRef(null);

  // Boolean state indicating whether inference is running.  When true the
  // component captures frames, performs inference and draws results; when
  // false the webcam stream is stopped.
  const [isRunning, setIsRunning] = useState(false);

  // State to indicate whether the model is still loading.  While loading
  // the start button is disabled and shows “Memuat Model...”.
  const [isLoading, setIsLoading] = useState(true);

  // Control horizontal mirroring of the display.  For this simplified
  // version we disable flipping entirely, so the webcam view and
  // detections are displayed in their natural orientation.

  // Load the TensorFlow.js model on component mount.  The exported model
  // resides in the public folder under MODELLAST_web_model/model.json.
  useEffect(() => {
    let cancelled = false;
    async function loadModel() {
      try {
        // Adjust the model URL to match where the TF.js files are served.
        // Because CRA/Next.js will copy everything in public/ to the root
        // of the site, the path here should be relative to public.
        const modelUrl = '/MODELLAST_web_model/model.json';
        const model = await tf.loadGraphModel(modelUrl);
        if (!cancelled) {
          modelRef.current = model;
          setIsLoading(false);
        }
      } catch (err) {
        console.error('Gagal memuat model TF.js:', err);
      }
    }
    loadModel();
    return () => {
      cancelled = true;
    };
  }, []);

  /**
   * Preprocess a video frame for the YOLOv8 model.
   *
   * The model expects a 4D tensor [1,640,640,3] with pixel values
   * normalised to [0,1] and padded using Ultralytics’ letterbox method.
   * Given the video dimensions, this function computes the resize ratio
   * r0 and paddings padX0, padY0 such that the aspect ratio is preserved.
   * It returns the tensor along with r0, padX0 and padY0 which are
   * needed later to map predictions back to the source frame.
   */
  const preprocess = (source) => {
    // Determine the source dimensions.  We support both HTMLVideoElement
    // (video) and HTMLImageElement (uploaded image).  Fallback to
    // `naturalWidth`/`naturalHeight` for images.
    const imgW = source.videoWidth || source.naturalWidth;
    const imgH = source.videoHeight || source.naturalHeight;
    // Compute the letterbox scale for a 640×640 square
    const r0 = Math.min(640 / imgW, 640 / imgH);
    const newW0 = Math.round(imgW * r0);
    const newH0 = Math.round(imgH * r0);
    const padX0 = (640 - newW0) / 2;
    const padY0 = (640 - newH0) / 2;

    // Convert the frame into a float32 Tensor and normalise to [0,1]
    const img = tf.browser.fromPixels(source);
    const imgFloat = img.toFloat().div(255.0);
    // Resize using bilinear interpolation
    const resized = tf.image.resizeBilinear(imgFloat, [newH0, newW0]);
    // Pad with the value 114/255 for each channel.  Ultralytics pads
    // letterbox regions with RGB(114,114,114).  Because the image has
    // already been normalised to [0,1], the pad value becomes 114/255.
    const padValue = 114.0 / 255.0;
    const top = Math.floor(padY0);
    const bottom = Math.ceil(padY0);
    const left = Math.floor(padX0);
    const right = Math.ceil(padX0);
    const padded = tf.pad(resized, [[top, bottom], [left, right], [0, 0]], padValue);
    // Create the final input tensor by adding a batch dimension.  The
    // combination of resize + padding above yields exactly 640×640 so we
    // do not need resizeWithCropOrPad (which does not exist in tfjs).
    const inputTensor = padded.expandDims(0);
    // Dispose intermediate tensors to free GPU memory.  The caller is
    // responsible for disposing `inputTensor` after use.
    img.dispose();
    imgFloat.dispose();
    resized.dispose();
    padded.dispose();
    return { tensor: inputTensor, r0, padX0, padY0 };
  };

  // Main draw and inference loop.  This useEffect is responsible for:
  //  - Running inference on a fixed interval (e.g. every 200 ms) to
  //    minimise computational load while maintaining responsiveness.
  //  - Drawing the video and detections to the canvas on every
  //    animation frame, applying horizontal flip if requested.
  useEffect(() => {
    if (!isRunning) return undefined;
    let animationFrameId;
    let inferenceIntervalId;

    // Function to perform inference on the current video frame
    const runInference = async () => {
      const video = videoRef.current;
      const model = modelRef.current;
      if (!video || !model) return;
      // Only run inference when the video metadata is available
      if (video.readyState < 4) return;
      // Preprocess the frame
      const { tensor, r0, padX0, padY0 } = preprocess(video);
      try {
        // Execute the model.  Some models may return a single tensor,
        // others return an array.  We assume the first tensor with
        // shape (?, *, 6) contains the detections.
        const outputs = await model.executeAsync(tensor);
        // Determine which output tensor contains the detection results
        let detTensor = null;
        if (Array.isArray(outputs)) {
          detTensor = outputs.find((t) => t.shape.length === 3 && t.shape[t.shape.length - 1] === 6);
        } else {
          detTensor = outputs;
        }
        // If no appropriate output found, dispose and return
        if (!detTensor) {
          tf.dispose(outputs);
          tensor.dispose();
          return;
        }
        // Convert tensor to JavaScript array
        const raw = detTensor.arraySync();
        // raw should be shape [1, N, 6]
        const detections = [];
        const threshold = 0.4;
        const boxes = raw[0];
        for (let i = 0; i < boxes.length; i++) {
          const [x1, y1, x2, y2, conf, classId] = boxes[i];
          if (conf >= threshold) {
            const w = x2 - x1;
            const h = y2 - y1;
            detections.push({
              x: x1,
              y: y1,
              w,
              h,
              score: conf,
              classId,
              r0,
              padX0,
              padY0,
            });
          }
        }
        detectionsRef.current = detections;
        // Dispose tensors to free GPU memory
        if (Array.isArray(outputs)) {
          outputs.forEach((t) => t.dispose());
        } else {
          outputs.dispose();
        }
        tensor.dispose();
      } catch (err) {
        console.error('Kesalahan saat inferensi:', err);
        tensor.dispose();
      }
    };

    // Function to draw the video frame and detections
    const mainLoop = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) {
        animationFrameId = requestAnimationFrame(mainLoop);
        return;
      }
      const ctx = canvas.getContext('2d');
      const imgW = video.videoWidth;
      const imgH = video.videoHeight;
      if (!imgW || !imgH) {
        animationFrameId = requestAnimationFrame(mainLoop);
        return;
      }
      // Compute display letterbox parameters
      const rDisplay = Math.min(canvas.width / imgW, canvas.height / imgH);
      const newWDisplay = Math.round(imgW * rDisplay);
      const newHDisplay = Math.round(imgH * rDisplay);
      const padXDisplay = (canvas.width - newWDisplay) / 2;
      const padYDisplay = (canvas.height - newHDisplay) / 2;
      // Clear the canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      // Horizontal flip disabled: no translate/scale transformation
      // Draw the current video frame with letterbox
      ctx.drawImage(video, 0, 0, imgW, imgH, padXDisplay, padYDisplay, newWDisplay, newHDisplay);
      // Draw detections
      ctx.lineWidth = 3;
      detectionsRef.current.forEach((det) => {
        const { x, y, w, h, score, classId, r0, padX0, padY0 } = det;
        // Convert to original image coordinates
        const origX = (x - padX0) / r0;
        const origY = (y - padY0) / r0;
        const origW = w / r0;
        const origH = h / r0;
        // Scale to display coordinates
        const drawX = origX * rDisplay + padXDisplay;
        const drawY = origY * rDisplay + padYDisplay;
        const drawW = origW * rDisplay;
        const drawH = origH * rDisplay;
        // Draw bounding box
        ctx.strokeStyle = '#00FF00';
        ctx.strokeRect(drawX, drawY, drawW, drawH);
        // Draw label background
        const label = labels[Math.round(classId)] || `cls ${classId}`;
        const text = `${label} (${score.toFixed(2)})`;
        ctx.font = '18px Arial';
        const textWidth = ctx.measureText(text).width;
        const textHeight = 18;
        const textY = Math.max(textHeight, drawY - 5);
        ctx.fillStyle = '#00FF00';
        ctx.fillRect(drawX - 1, textY - textHeight, textWidth + 10, textHeight + 4);
        ctx.fillStyle = '#000000';
        ctx.fillText(text, drawX + 5, textY);
      });
      ctx.restore();
      animationFrameId = requestAnimationFrame(mainLoop);
    };

    // Start loops
    animationFrameId = requestAnimationFrame(mainLoop);
    inferenceIntervalId = setInterval(runInference, 200);

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      if (inferenceIntervalId) clearInterval(inferenceIntervalId);
    };
  }, [isRunning]);

  // Start the webcam stream.  When the user clicks the start button,
  // request access to the camera.  Once the stream metadata is loaded,
  // set the canvas dimensions and begin inference by setting isRunning.
  const startWebcam = async () => {
    if (isLoading || isRunning) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } } });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          canvasRef.current.width = videoRef.current.videoWidth;
          canvasRef.current.height = videoRef.current.videoHeight;
          setIsRunning(true);
        };
      }
    } catch (error) {
      console.error('Gagal mengakses webcam:', error);
    }
  };

  // Stop the webcam stream and halt inference/drawing
  const stopWebcam = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach((track) => track.stop());
      setIsRunning(false);
      detectionsRef.current = [];
    }
  };

  /**
   * Handle image upload.  When the user selects an image file this
   * function reads it into an HTMLImageElement, performs detection using
   * the same model and preprocessing as the webcam, and stores the
   * resulting detections so they can be drawn on the upload canvas.
   */
  const handleUpload = async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    // Create an image element
    const imgEl = new Image();
    imgEl.src = URL.createObjectURL(file);
    imgEl.onload = async () => {
      // Clean up any previous object URL
      URL.revokeObjectURL(imgEl.src);
      setUploadImage(imgEl);
      // Preprocess the image and run inference
      const { tensor, r0, padX0, padY0 } = preprocess(imgEl);
      try {
        const model = modelRef.current;
        if (!model) return;
        const outputs = await model.executeAsync(tensor);
        let detTensor = null;
        if (Array.isArray(outputs)) {
          detTensor = outputs.find((t) => t.shape.length === 3 && t.shape[t.shape.length - 1] === 6);
        } else {
          detTensor = outputs;
        }
        if (!detTensor) {
          tf.dispose(outputs);
          tensor.dispose();
          return;
        }
        const raw = detTensor.arraySync();
        const detections = [];
        const threshold = 0.4;
        const boxes = raw[0];
        for (let i = 0; i < boxes.length; i++) {
          const [x1, y1, x2, y2, conf, classId] = boxes[i];
          if (conf >= threshold) {
            detections.push({
              x: x1,
              y: y1,
              w: x2 - x1,
              h: y2 - y1,
              score: conf,
              classId,
              r0,
              padX0,
              padY0,
            });
          }
        }
        setUploadDetections(detections);
        // Dispose tensors
        if (Array.isArray(outputs)) {
          outputs.forEach((t) => t.dispose());
        } else {
          outputs.dispose();
        }
        tensor.dispose();
      } catch (error) {
        console.error('Kesalahan saat deteksi gambar:', error);
        tensor.dispose();
      }
    };
  };

  // Draw uploaded image and its detections whenever either changes
  useEffect(() => {
    const canvas = uploadCanvasRef.current;
    if (!canvas || !uploadImage) return;
    const ctx = canvas.getContext('2d');
    const imgW = uploadImage.naturalWidth;
    const imgH = uploadImage.naturalHeight;
    // Resize canvas to match image dimensions
    canvas.width = imgW;
    canvas.height = imgH;
    // Clear and draw the image
    ctx.clearRect(0, 0, imgW, imgH);
    ctx.drawImage(uploadImage, 0, 0, imgW, imgH);
    // Draw detections on top
    // Use a thicker line for uploaded images so bounding boxes remain visible
    ctx.lineWidth = 4;
    uploadDetections.forEach((det) => {
      const { x, y, w, h, score, classId, r0, padX0, padY0 } = det;
      // Convert back to original image coordinates
      const origX = (x - padX0) / r0;
      const origY = (y - padY0) / r0;
      const origW = w / r0;
      const origH = h / r0;
      // Draw bounding box
      ctx.strokeStyle = '#00FF00';
      ctx.strokeRect(origX, origY, origW, origH);
      const label = labels[Math.round(classId)] || `cls ${classId}`;
      const text = `${label} (${score.toFixed(2)})`;
      // Use a larger font for uploaded images
      ctx.font = '20px Arial';
      const textWidth = ctx.measureText(text).width;
      const textHeight = 22;
      const textY = Math.max(textHeight, origY - 5);
      ctx.fillStyle = '#00FF00';
      ctx.fillRect(origX - 1, textY - textHeight, textWidth + 12, textHeight + 6);
      ctx.fillStyle = '#000000';
      ctx.fillText(text, origX + 6, textY);
    });
  }, [uploadImage, uploadDetections]);

  return (
    <div className="w-full max-w-2xl mx-auto bg-white rounded-lg shadow-lg p-6">
      {/* Hide the video element as we draw frames onto the canvas instead */}
      <video ref={videoRef} className="hidden" autoPlay muted playsInline />
      <canvas ref={canvasRef} className="w-full h-auto border-2 border-gray-300" width="640" height="480" />
      <div className="mt-4 flex justify-center space-x-4">
        <button
          onClick={isRunning ? stopWebcam : startWebcam}
          disabled={isLoading}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-400"
        >
          {isLoading ? 'Memuat Model...' : (isRunning ? 'Stop' : 'Start')} Webcam
        </button>
        {/* Tombol flip dihapus karena tampilan selalu dalam orientasi normal */}
      </div>
      {/* Section for image upload detection */}
      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-2">Unggah Gambar untuk Deteksi</h2>
        <input type="file" accept="image/*" onChange={handleUpload} className="mb-4" />
        {/* Canvas untuk menampilkan gambar yang diunggah dan kotak deteksinya */}
        <canvas ref={uploadCanvasRef} className="w-full h-auto border-2 border-dashed border-gray-400" />
      </div>
    </div>
  );
};

export default Detection;