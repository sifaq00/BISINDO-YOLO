import { useState } from "react";
import WebcamDetection from "./components/WebcamDetection";
import ImageDetection from "./components/ImageDetection";
import { FaVideo, FaImage, FaArrowLeft } from "react-icons/fa";

function App() {
  const [detectionMode, setDetectionMode] = useState(null);

  const renderContent = () => {
    if (detectionMode === "webcam") {
      return <WebcamDetection />;
    }
    if (detectionMode === "image") {
      return <ImageDetection />;
    }
    return null;
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center p-4 bg-gray-900 text-white">
      <header className="w-full max-w-6xl text-center mb-8">
        <h1 className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
          BISINDO Live Detection
        </h1>
        <p className="text-lg text-gray-400 mt-2">
          Pilih mode deteksi untuk memulai terjemahan Bahasa Isyarat Indonesia secara real-time.
        </p>
      </header>

      {detectionMode ? (
        <div className="w-full max-w-6xl">
          <button
            onClick={() => setDetectionMode(null)}
            className="flex items-center gap-2 mb-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors duration-300"
          >
            <FaArrowLeft />
            Kembali
          </button>
          <div className="w-full bg-gray-800 rounded-xl shadow-2xl p-6">
            {renderContent()}
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl">
          <div
            className="group cursor-pointer p-8 bg-gray-800 rounded-xl shadow-lg hover:shadow-2xl hover:bg-blue-600 transition-all duration-300 transform hover:-translate-y-2"
            onClick={() => setDetectionMode("webcam")}
          >
            <FaVideo className="text-6xl text-blue-400 group-hover:text-white mx-auto mb-4 transition-colors duration-300" />
            <h2 className="text-2xl font-bold text-center">Deteksi via Webcam</h2>
            <p className="text-gray-400 group-hover:text-gray-200 text-center mt-2 transition-colors duration-300">
              Gunakan kamera Anda untuk mendeteksi isyarat secara langsung.
            </p>
          </div>
          <div
            className="group cursor-pointer p-8 bg-gray-800 rounded-xl shadow-lg hover:shadow-2xl hover:bg-purple-600 transition-all duration-300 transform hover:-translate-y-2"
            onClick={() => setDetectionMode("image")}
          >
            <FaImage className="text-6xl text-purple-400 group-hover:text-white mx-auto mb-4 transition-colors duration-300" />
            <h2 className="text-2xl font-bold text-center">Deteksi via Gambar</h2>
            <p className="text-gray-400 group-hover:text-gray-200 text-center mt-2 transition-colors duration-300">
              Unggah gambar untuk mendeteksi isyarat yang ada di dalamnya.
            </p>
          </div>
        </div>
      )}

      <footer className="w-full max-w-6xl text-center mt-12 text-gray-500">
        <p>&copy; 2025 - Dibuat dengan Kecerdasan Buatan Oleh Muhammad Asifaq</p>
      </footer>
    </div>
  );
}

export default App;