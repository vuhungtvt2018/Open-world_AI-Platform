import React, { useState } from 'react';
import { X, Camera as CameraIcon, Video, Globe, Settings, PlusCircle, RefreshCw } from 'lucide-react';
import { Camera, SystemMode } from '../types/vision';
import './ModalAddCamera.css';

interface ModalAddCameraProps {
  isOpen: boolean;
  onClose: () => void;
  onAddCamera: (newCam: Camera) => void;
}

type CameraSourceType = 'rtsp' | 'basler';

export const ModalAddCamera: React.FC<ModalAddCameraProps> = ({
  isOpen,
  onClose,
  onAddCamera
}) => {
  const [cameraId, setCameraId] = useState('');
  const [name, setName] = useState('');
  const [width, setWidth] = useState('1920');
  const [height, setHeight] = useState('1080');
  const [fps, setFps] = useState('30');
  const [sourceType, setSourceType] = useState<CameraSourceType>('rtsp');
  const [sourceUrl, setSourceUrl] = useState('');
  const [serialNumber, setSerialNumber] = useState('');
  const [mode, setMode] = useState<SystemMode>('INSPECTION');
  const [location, setLocation] = useState('');
  const [productLine, setProductLine] = useState('Dây chuyền Đúc Vỏ A1');

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !cameraId.trim()) return;

    const finalCamId = cameraId.trim();
    const finalName = name.trim();
    const rtsp = sourceType === 'rtsp' ? sourceUrl.trim() : `basler://${serialNumber.trim()}`;

    const newCam: Camera = {
      id: finalCamId,
      name: finalName,
      location: location || 'Băng tải sản xuất',
      productLine: productLine,
      status: 'active',
      mode: mode,
      rtspUrl: rtsp,
      ipAddress: `192.168.1.${Math.floor(Math.random()*150)+10}`,
      resolution: `${width || 1920}x${height || 1080}`,
      targetFps: Number(fps) || 30,
      currentFps: Number(fps) || 30.0,
      sensorTemp: 39.5,
      previewImage: 'https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=1200&q=80',
      liveUrl: `${import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000'}/video-feed?cam=${encodeURIComponent(finalCamId)}&fps=15`,
      inspectionType: mode === 'INSPECTION' ? 'Kiểm tra lỗi bề mặt AI' : 'Đếm sản phẩm tự động',
      roiEnabled: true,
      exposureTime: 12.0,
      gain: 3.5
    };

    onAddCamera(newCam);
    onClose();
  };

  return (
    <div className="modal-backdrop">
      <div className="modal-container add-camera-modal">
        <div className="modal-header">
          <div className="modal-title">
            <PlusCircle size={18} className="text-cyan" />
            <span>Thêm & Khởi tạo Camera Mới (Live Stream Hub)</span>
          </div>
          <button className="modal-close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="modal-body">
          {/* Row 1: Camera ID & Camera Name */}
          <div className="form-row">
            <div className="form-group">
              <label>Camera ID (*):</label>
              <input
                type="text"
                placeholder="VD: webcam_01 hoặc CAM07"
                value={cameraId}
                onChange={(e) => setCameraId(e.target.value)}
                required
                className="modal-input font-mono"
              />
            </div>

            <div className="form-group">
              <label>Tên Camera (Camera Name) (*):</label>
              <input
                type="text"
                placeholder="VD: Camera Line 1 - Trạm QC"
                value={name}
                onChange={(e) => setName(e.target.value)}
                required
                className="modal-input"
              />
            </div>
          </div>

          {/* Row 2: Resolution (Width x Height) & FPS */}
          <div className="form-row three-col">
            <div className="form-group">
              <label>Chiều rộng (Width):</label>
              <input
                type="number"
                min="1"
                placeholder="1920"
                value={width}
                onChange={(e) => setWidth(e.target.value)}
                className="modal-input font-mono"
              />
            </div>

            <div className="form-group">
              <label>Chiều cao (Height):</label>
              <input
                type="number"
                min="1"
                placeholder="1080"
                value={height}
                onChange={(e) => setHeight(e.target.value)}
                className="modal-input font-mono"
              />
            </div>

            <div className="form-group">
              <label>Tốc độ khung hình (FPS):</label>
              <input
                type="number"
                min="1"
                step="0.1"
                placeholder="30"
                value={fps}
                onChange={(e) => setFps(e.target.value)}
                className="modal-input font-mono"
              />
            </div>
          </div>

          {/* Row 3: Source Type */}
          <div className="form-group">
            <label>Loại nguồn Cảm biến (Source Type):</label>
            <select
              value={sourceType}
              onChange={(e) => setSourceType(e.target.value as CameraSourceType)}
              className="modal-select"
            >
              <option value="rtsp">RTSP Stream / Webcam Index</option>
              <option value="basler">Basler Industrial Camera (GigE / USB3)</option>
            </select>
          </div>

          {/* Dynamic Source Input depending on Source Type */}
          {sourceType === 'rtsp' ? (
            <div className="form-group">
              <label>RTSP URL / Webcam Index (*):</label>
              <input
                type="text"
                placeholder="VD: 0 hoặc rtsp://admin:pass@192.168.1.107:554/live/stream1"
                value={sourceUrl}
                onChange={(e) => setSourceUrl(e.target.value)}
                required
                className="modal-input font-mono"
              />
            </div>
          ) : (
            <div className="form-group">
              <label>Basler Serial Number / Device ID (*):</label>
              <input
                type="text"
                placeholder="VD: 24158912 hoặc basler_acA1920-40gc"
                value={serialNumber}
                onChange={(e) => setSerialNumber(e.target.value)}
                required
                className="modal-input font-mono"
              />
            </div>
          )}

          {/* Row 4: Assigned Task & Location */}
          <div className="form-row">
            <div className="form-group">
              <label>Nhiệm vụ AI Phân công (Assigned Task):</label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value as SystemMode)}
                className="modal-select"
              >
                <option value="INSPECTION">Inspection (Kiểm tra lỗi QC)</option>
                <option value="COUNTING">Counting (Đếm số lượng sản phẩm)</option>
              </select>
            </div>

            <div className="form-group">
              <label>Vị trí & Dây chuyền:</label>
              <input
                type="text"
                placeholder="VD: Băng tải A - Trạm 01"
                value={location}
                onChange={(e) => setLocation(e.target.value)}
                className="modal-input"
              />
            </div>
          </div>

          <div className="modal-footer">
            <button type="button" className="modal-btn cancel" onClick={onClose}>
              Hủy bỏ
            </button>
            <button type="submit" className="modal-btn submit">
              Save Camera
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
