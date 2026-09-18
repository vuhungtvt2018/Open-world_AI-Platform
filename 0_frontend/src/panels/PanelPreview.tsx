import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Maximize2, 
  Camera as CameraIcon, 
  Pause, 
  Play, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Move, 
  Upload as UploadIcon,
  Video,
  Monitor,
  FolderOpen,
  Zap,
  CheckCircle2,
  AlertTriangle,
  Boxes,
  Flame,
  Crop,
  Search,
  Target
} from 'lucide-react';
import { Camera, SystemMode, DetectionBox, InputSourceMode, InferenceResult } from '../types/vision';
import DetectionResultImage from '../components/DetectionResultImage';
import './PanelPreview.css';

const SAMPLE_UPLOAD_IMAGES = [
  { id: 'img1', label: 'Bề mặt sản phẩm mẫu A', url: 'https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?auto=format&fit=crop&w=1200&q=80' },
  { id: 'img2', label: 'Thân máy ốc vít đếm mẫu', url: 'https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=1200&q=80' },
  { id: 'img3', label: 'Linh kiện vi mạch B3', url: 'https://images.unsplash.com/photo-1581092335397-9583fe92d232?auto=format&fit=crop&w=1200&q=80' }
];

const SAMPLE_HEATMAP_IMAGE = 'https://images.unsplash.com/photo-1504328345606-18bbc8c9d7d1?auto=format&fit=crop&w=1200&q=80';
const SAMPLE_CROPS_IMAGE = 'https://images.unsplash.com/photo-1581092580497-e0d23cbdf1dc?auto=format&fit=crop&w=1200&q=80';

interface PanelPreviewProps {
  camera: Camera | null;
  cameras: Camera[];
  selectedCameraId: string;
  onSelectCamera: (id: string) => void;
  mode: SystemMode;
  confThreshold: number;
  sourceMode: InputSourceMode;
  onSourceModeChange: (mode: InputSourceMode) => void;
  isRunning: boolean;
  onToggleRun: () => void;
  uploadedImage: string | null;
  onUploadImage: (imageUrl: string) => void;
  inferenceResult?: InferenceResult | null;
}

export const PanelPreview: React.FC<PanelPreviewProps> = ({
  camera,
  cameras,
  selectedCameraId,
  onSelectCamera,
  mode,
  confThreshold,
  sourceMode,
  onSourceModeChange,
  isRunning,
  onToggleRun,
  uploadedImage,
  onUploadImage,
  inferenceResult
}) => {
  const [isPaused, setIsPaused] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);
  const [boxes, setBoxes] = useState<DetectionBox[]>([]);
  const [frameCount, setFrameCount] = useState(0);

  // Zoom & Pan state for single preview view
  const [zoomScale, setZoomScale] = useState<number>(1.0);
  const [panOffset, setPanOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });
  const viewportRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Modal zoom preview state
  const [activeModal, setActiveModal] = useState<{ title: string; image: string } | null>(null);

  // Reset zoom & pan when camera or sourceMode changes
  useEffect(() => {
    setZoomScale(1.0);
    setPanOffset({ x: 0, y: 0 });
  }, [camera?.id, sourceMode]);

  const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

  // Handle local image file upload
  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const formData = new FormData();
      formData.append('file', file);
      try {
        const response = await fetch(`${API_BASE_URL}/upload`, {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();
        if (response.ok) {
          if (!data.filename) {
            throw new Error('Upload response did not include an image filename');
          }
          onUploadImage(data.filename);
          setToastMessage(`Đã tải ảnh lên: ${file.name}`);
        } else {
          setToastMessage(`Lỗi tải ảnh: ${data.detail || 'Upload failed'}`);
        }
      } catch (error) {
        setToastMessage('Upload failed');
      }
      setTimeout(() => setToastMessage(null), 3000);
    }
  };


  // Zoom handlers
  const handleZoomIn = () => {
    setZoomScale((prev) => Math.min(5.0, parseFloat((prev + 0.25).toFixed(2))));
  };

  const handleZoomOut = () => {
    setZoomScale((prev) => {
      const next = Math.max(0.5, parseFloat((prev - 0.25).toFixed(2)));
      if (next <= 1.0) {
        setPanOffset({ x: 0, y: 0 });
      }
      return next;
    });
  };

  const handleResetZoom = () => {
    setZoomScale(1.0);
    setPanOffset({ x: 0, y: 0 });
  };

  // Wheel Zoom Listener
  const handleWheel = (e: React.WheelEvent<HTMLDivElement>) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 0.15 : -0.15;
    setZoomScale((prev) => {
      const newScale = Math.min(5.0, Math.max(0.5, parseFloat((prev + zoomFactor).toFixed(2))));
      if (newScale <= 1.0) {
        setPanOffset({ x: 0, y: 0 });
      }
      return newScale;
    });
  };

  // Pan (Drag to move) handlers
  const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    setIsPanning(true);
    dragStartRef.current = {
      x: e.clientX - panOffset.x,
      y: e.clientY - panOffset.y
    };
  };

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!isPanning) return;
    const newX = e.clientX - dragStartRef.current.x;
    const newY = e.clientY - dragStartRef.current.y;
    setPanOffset({ x: newX, y: newY });
  }, [isPanning]);

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  const handleTakeSnapshot = () => {
    const camName = camera?.name || 'Preview Feed';
    setToastMessage(`Đã chụp ảnh & lưu snapshot từ ${camName} thành công!`);
    setTimeout(() => setToastMessage(null), 3000);
  };

  const handleFullscreen = () => {
    if (!viewportRef.current) return;
    if (document.fullscreenElement) {
      document.exitFullscreen();
    } else {
      viewportRef.current.requestFullscreen().catch((err) => {
        console.error('Error attempting to enable fullscreen:', err);
      });
    }
  };

  const getImageUrl = (urlPath?: string | null) => {
    if (!urlPath) return SAMPLE_UPLOAD_IMAGES[0].url;
    if (urlPath.startsWith('http://') || urlPath.startsWith('https://') || urlPath.startsWith('blob:')) return urlPath;
    if (urlPath.startsWith('in_memory_image')) return `${API_BASE_URL}/images/${urlPath}`;
    if (urlPath.startsWith('/')) return `${API_BASE_URL}${urlPath}`;
    return `${API_BASE_URL}/${urlPath}`;
  };

  // Current display image based on sourceMode
  const cameraLiveImage = camera?.status === 'active' && camera.liveUrl
    ? getImageUrl(camera.liveUrl)
    : getImageUrl(camera?.previewImage || SAMPLE_UPLOAD_IMAGES[0].url);

  const currentDisplayImage = sourceMode === 'upload'
    ? getImageUrl(uploadedImage)
    : cameraLiveImage;

  const displayHeatmap = inferenceResult?.visUrls?.heatmap 
    ? getImageUrl(inferenceResult.visUrls.heatmap)
    : SAMPLE_HEATMAP_IMAGE;
  
  const displayCrops = inferenceResult?.visUrls?.crops 
    ? getImageUrl(inferenceResult.visUrls.crops)
    : SAMPLE_CROPS_IMAGE;
    
  const overallResultImage = inferenceResult?.visUrls?.overall
    ? getImageUrl(inferenceResult.visUrls.overall)
    : (inferenceResult?.original_image_url ? getImageUrl(inferenceResult.original_image_url) : currentDisplayImage);

  const isMultiScreenMode = sourceMode === 'camera' || sourceMode === 'upload';

  return (
    <div className="panel-container panel-preview">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="preview-toast">
          <CameraIcon size={16} />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Hidden File Input for Upload mode */}
      <input 
        type="file" 
        ref={fileInputRef} 
        onChange={handleFileChange} 
        accept="image/*,video/*" 
        style={{ display: 'none' }} 
      />

      {/* Header bar with Source Dropdown */}
      <div className="preview-header">
        <div className="preview-header-left">
          <span className="preview-tag">Preview</span>

          {/* Source Mode Dropdown */}
          <div className="source-mode-dropdown-wrapper">
            <select
              className="source-mode-select"
              value={sourceMode}
              onChange={(e) => onSourceModeChange(e.target.value as InputSourceMode)}
            >
              <option value="preview">⚡ Mode: Preview (Mô phỏng)</option>
              <option value="camera">📹 Mode: Camera (Live System)</option>
              <option value="upload">📁 Mode: Upload (Tải tệp lên)</option>
            </select>
          </div>
        </div>
        
        {/* Dynamic Center selector depending on sourceMode */}
        <div className="preview-header-center">
          {sourceMode === 'camera' && (
            <div className="camera-select-wrapper">
              <span className="select-label">Chọn Camera:</span>
              <select 
                className="camera-dropdown"
                value={selectedCameraId}
                onChange={(e) => onSelectCamera(e.target.value)}
              >
                {cameras.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name} ({c.ipAddress})
                  </option>
                ))}
              </select>
              {camera && <span className={`status-pill ${camera.status}`}>{camera.status}</span>}
            </div>
          )}

          {sourceMode === 'upload' && (
            <div className="upload-header-controls">
              <button 
                className="upload-btn-sm"
                onClick={() => fileInputRef.current?.click()}
              >
                <FolderOpen size={13} /> Chọn tệp ảnh/video
              </button>
              
            </div>
          )}

          {sourceMode === 'preview' && camera && (
            <div className="preview-camera-info">
              <span className="camera-display-title">{camera.name}</span>
              <span className={`status-pill ${camera.status}`}>{camera.status}</span>
            </div>
          )}
        </div>

        {/* Header Right Zoom & Fullscreen Toolbar */}
        <div className="preview-header-right">
          {!isMultiScreenMode && (
            <div className="zoom-toolbar">
              <button className="icon-action-btn" title="Thu nhỏ (-)" onClick={handleZoomOut}>
                <ZoomOut size={15} />
              </button>
              <span className="zoom-level-badge" title="Tỉ lệ Zoom hiện tại">
                {Math.round(zoomScale * 100)}%
              </span>
              <button className="icon-action-btn" title="Phóng to (+)" onClick={handleZoomIn}>
                <ZoomIn size={15} />
              </button>
              {zoomScale !== 1.0 && (
                <button className="icon-action-btn reset-zoom-btn" title="Đặt lại 100%" onClick={handleResetZoom}>
                  <RotateCcw size={14} />
                </button>
              )}
            </div>
          )}

          <div className="header-divider" />

          <button className="icon-action-btn" title="Toàn màn hình" onClick={handleFullscreen}>
            <Maximize2 size={16} />
          </button>
        </div>
      </div>

      {/* VIEWPORT AREA: Multi-screen Grid (Camera & Upload mode) vs Single Zoom View (Preview mode) */}
      <div className="preview-viewport-container" ref={viewportRef}>
        {isMultiScreenMode ? (
          /* MULTI-SCREEN GRID LAYOUT */
          <div className={`multi-screen-grid ${mode === 'COUNTING' ? 'counting-grid-2' : 'inspection-grid-4'}`}>
            {/* SCREEN 1: Input (Camera RTSP feed or Uploaded image) */}
            <div className="grid-screen-card">
              <div className="screen-card-header">
                <span className="step-badge">1</span>
                <span className="screen-title">Input (Src từ {sourceMode === 'camera' ? 'Camera' : 'Upload'})</span>
                <button 
                  className="screen-expand-btn"
                  onClick={() => setActiveModal({ title: '1. Input Source', image: currentDisplayImage })}
                >
                  <Maximize2 size={12} /> View
                </button>
              </div>
              <div className="screen-card-media">
                <img src={currentDisplayImage} alt="Input" className="screen-img" />
                <span className="media-badge-bl">INPUT SOURCE ({sourceMode.toUpperCase()})</span>
              </div>
            </div>

            {/* COUNTING MODE -> SCREEN 2: Result */}
            {mode === 'COUNTING' && (
              <div className="grid-screen-card">
                <div className="screen-card-header">
                  <span className="step-badge">2</span>
                  <span className="screen-title">Result (Kết quả Counting)</span>
                  <button 
                    className="screen-expand-btn"
                    onClick={() => setActiveModal({ title: '2. Result (Counting)', image: overallResultImage })}
                  >
                    <Maximize2 size={12} /> View
                  </button>
                </div>
                <div className="screen-card-media">
                  <img src={overallResultImage} alt="Counting Result" className="screen-img" />
                  
                  {/* Overlays */}
                  <svg className="screen-overlay-svg">

                    {boxes.map((box) => (
                      <g key={box.id} className="bbox-group count">
                        <rect x={`${box.x}%`} y={`${box.y}%`} width={`${box.width}%`} height={`${box.height}%`} className="bbox-rect count" />
                        <g className="bbox-label-bg">
                          <rect x={`${box.x}%`} y={`${box.y - 6}%`} width="130" height="20" className="label-bg count" />
                          <text x={`${box.x + 1}%`} y={`${box.y - 2}%`} className="label-text">{box.label}</text>
                        </g>
                      </g>
                    ))}
                  </svg>
                  <span className="media-badge-bl text-cyan">COUNTING RESULT: {inferenceResult ? `${inferenceResult.totalObjects} OBJECTS` : 'READY'}</span>
                </div>
              </div>
            )}

            {/* INSPECTION MODE -> SCREENS 2, 3, 4 */}
            {mode === 'INSPECTION' && (
              <>
                {/* SCREEN 2: Object Heatmaps */}
                <div className="grid-screen-card">
                  <div className="screen-card-header">
                    <span className="step-badge">2</span>
                    <span className="screen-title">Object Heatmaps</span>
                    <button 
                      className="screen-expand-btn"
                      onClick={() => setActiveModal({ title: '2. Object Heatmaps', image: displayHeatmap })}
                    >
                      <Maximize2 size={12} /> View
                    </button>
                  </div>
                  <div className="screen-card-media">
                    {inferenceResult || isRunning ? (
                      <img src={displayHeatmap} alt="Heatmap" className="screen-img heatmap-effect" />
                    ) : (
                      <div className="media-placeholder">
                        <Flame size={36} className="placeholder-icon" />
                        <span>Nhấn "Chạy" để phân tích Heatmap</span>
                      </div>
                    )}
                    <span className="media-badge-bl">GRAD-CAM HEATMAP</span>
                  </div>
                </div>

                {/* SCREEN 3: Object Outputs (Crops) */}
                <div className="grid-screen-card">
                  <div className="screen-card-header">
                    <span className="step-badge">3</span>
                    <span className="screen-title">Object Outputs (Crops)</span>
                    <button 
                      className="screen-expand-btn"
                      onClick={() => setActiveModal({ title: '3. Object Outputs (Crops)', image: displayCrops })}
                    >
                      <Maximize2 size={12} /> View
                    </button>
                  </div>
                  <div className="screen-card-media">
                    {inferenceResult || isRunning ? (
                      <img src={displayCrops} alt="Crops" className="screen-img" />
                    ) : (
                      <div className="media-placeholder">
                        <Crop size={36} className="placeholder-icon" />
                        <span>Nhấn "Chạy" để trích xuất Object Crops</span>
                      </div>
                    )}
                    <span className="media-badge-bl">DEFECT CROPS</span>
                  </div>
                </div>

                {/* SCREEN 4: Overall Result */}
                <div className="grid-screen-card">
                  <div className="screen-card-header">
                    <span className="step-badge">4</span>
                    <span className="screen-title">Overall Result</span>
                    {inferenceResult?.status === 'NG' && (
                      <span className="result-tag ng">
                        NG DETECTED
                      </span>
                    )}
                    <button 
                      className="screen-expand-btn"
                      onClick={() => setActiveModal({ title: '4. Overall Result', image: overallResultImage })}
                    >
                      <Maximize2 size={12} /> View
                    </button>
                  </div>
                  <div className="screen-card-media">
                    <img src={overallResultImage} alt="Overall Result" className="screen-img" />
                    {/* Bounding box overlays */}
                    {!inferenceResult?.visUrls?.overall && (
                      <svg className="screen-overlay-svg">
                        {boxes.map((box) => (
                          <g key={box.id} className={`bbox-group ${box.status.toLowerCase()}`}>
                            <rect x={`${box.x}%`} y={`${box.y}%`} width={`${box.width}%`} height={`${box.height}%`} className={`bbox-rect ${box.status.toLowerCase()}`} />
                            <g className="bbox-label-bg">
                              <rect x={`${box.x}%`} y={`${box.y - 6}%`} width="130" height="20" className={`label-bg ${box.status.toLowerCase()}`} />
                              <text x={`${box.x + 1}%`} y={`${box.y - 2}%`} className="label-text">{box.label}</text>
                            </g>
                          </g>
                        ))}
                      </svg>
                    )}
                    <span className={`media-badge-bl ${inferenceResult?.status === 'NG' ? 'text-red' : ''}`}>
                      {inferenceResult ? `QC RESULT: ${inferenceResult.status}` : 'OVERALL RESULT'}
                    </span>
                  </div>
                </div>
              </>
            )}
          </div>
        ) : (
          /* SINGLE VIEWPORT FOR PREVIEW MODE (Simulated Stream) */
          <div 
            className={`preview-viewport ${isPanning ? 'panning' : ''} ${zoomScale > 1.0 ? 'can-pan' : ''}`}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            <div 
              className="preview-zoom-canvas"
              style={{
                transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${zoomScale})`,
                transformOrigin: 'center center'
              }}
            >
              <img 
                src={currentDisplayImage} 
                alt="Inspection Preview" 
                className={`stream-image ${isPaused ? 'paused' : ''}`}
                draggable={false}
              />

              <svg className="stream-overlay">

                {boxes.map((box) => (
                  <g key={box.id} className={`bbox-group ${box.status.toLowerCase()}`}>
                    <rect x={`${box.x}%`} y={`${box.y}%`} width={`${box.width}%`} height={`${box.height}%`} className={`bbox-rect ${box.status.toLowerCase()}`} />
                    <g className="bbox-label-bg">
                      <rect x={`${box.x}%`} y={`${box.y - 5}%`} width="140" height="22" className={`label-bg ${box.status.toLowerCase()}`} />
                      <text x={`${box.x + 1}%`} y={`${box.y - 1.5}%`} className="label-text">{box.label}</text>
                    </g>
                  </g>
                ))}
              </svg>
            </div>

            <div className="stream-badge-tl">
              <span className="live-dot red pulse"></span>
              <span>PREVIEW MODE</span>
              <span className="divider">|</span>
              <span>TASK: {mode}</span>
            </div>

            <div className="stream-badge-tr">
              <span>SIMULATED STREAM</span>
              <span className="divider">|</span>
              <span>30.0 FPS</span>
            </div>

            <div className="stream-badge-bl">
              <span>{new Date().toLocaleTimeString()}</span>
              <span className="divider">|</span>
              <span>STATUS: LIVE STREAMING</span>
            </div>

            {zoomScale > 1.0 && (
              <div className="zoom-pan-hint">
                <Move size={13} />
                <span>Kéo chuột để di chuyển ({Math.round(zoomScale * 100)}%)</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Control Buttons Footer */}
      <div className="preview-controls-footer">
        <button className="preview-btn btn-capture" onClick={handleTakeSnapshot}>
          <CameraIcon size={16} />
          <span>Chụp ảnh</span>
        </button>

        {/* Footer Main Action Button (Behavior differs by SourceMode) */}
        {sourceMode === 'upload' && (
          <button 
            className={`preview-btn ${isRunning ? 'btn-running' : 'btn-run-action'}`}
            onClick={onToggleRun}
          >
            <Zap size={16} />
            <span>{isRunning ? 'Đang xử lý Inference...' : 'Chạy (Run Inference)'}</span>
          </button>
        )}

        {sourceMode === 'camera' && (
          <button 
            className={`preview-btn ${isRunning ? 'btn-stop-action' : 'btn-run-action'}`}
            onClick={onToggleRun}
          >
            {isRunning ? <Pause size={16} /> : <Play size={16} />}
            <span>{isRunning ? 'Dừng Inference' : 'Chạy (Start Inference)'}</span>
          </button>
        )}

        {sourceMode === 'preview' && (
          <button 
            className={`preview-btn ${isPaused ? 'btn-resume' : 'btn-pause'}`}
            onClick={() => setIsPaused(!isPaused)}
          >
            {isPaused ? <Play size={16} /> : <Pause size={16} />}
            <span>{isPaused ? 'Tiếp tục' : 'Tạm dừng'}</span>
          </button>
        )}

        {/* Footer Zoom status summary */}
        <div className="footer-zoom-info">
          <span>Mode: <strong className="text-cyan">{sourceMode.toUpperCase()}</strong> ({mode})</span>
        </div>
      </div>

      {/* Modal Popup for Zoomed Image View */}
      {activeModal && (
        <div className="preview-modal-backdrop" onClick={() => setActiveModal(null)}>
          <div className="preview-modal-box" onClick={(e) => e.stopPropagation()}>
            <div className="preview-modal-header">
              <span>{activeModal.title}</span>
              <button onClick={() => setActiveModal(null)} className="icon-action-btn">
                <RotateCcw size={16} />
              </button>
            </div>
            <div className="preview-modal-body">
              <img src={activeModal.image} alt={activeModal.title} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
