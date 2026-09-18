import React, { useCallback, useEffect, useState } from 'react';
import {
  BarChart3 as BarChart,
  Boxes,
  Clock,
  Folder,
  Image as ImageIcon,
  Maximize2,
  Play,
  RotateCcw,
  Search,
  ShieldAlert,
  Target,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import DetectionResultImage, {
  type DetectionObject,
} from '../components/DetectionResultImage';
import './VisionInspection.css';

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

type InputMode = 'folder' | 'camera';
type TaskMode = 'inspection' | 'detection';

interface ConfiguredCamera {
  camera_id: string;
  name: string;
  source_type: 'rtsp' | 'basler';
  assigned_task: TaskMode | null;
  status: 'disconnected' | 'connecting' | 'online' | 'error';
}

interface InferenceMetrics {
  latency_ms?: number;
  total_objects?: number;
  counts_by_class?: Record<string, number>;
  ng_count?: number;
  max_score?: number;
}

interface InferenceResult {
  status?: string;
  message?: string;
  detail?: string;
  timestamp?: string;
  task?: TaskMode;
  ng_detected?: boolean;
  original_image_url?: string;
  image_width?: number;
  image_height?: number;
  metrics?: InferenceMetrics;
  objects?: DetectionObject[];
  vis_urls?: {
    heatmap?: string;
    crops?: string;
    overall?: string;
  };
}

interface HistoryRecord {
  id?: number;
  timestamp: string;
  task_type?: TaskMode;
  ng_detected?: boolean;
  total_objects?: number;
  counts_by_class?: Record<string, number>;
}

const DETECTION_CLASSES = ['screw', 'washer', 'wood_screw']; //har nữa chỉnh sau

// 19082026 - PHUC - Update UI
// 12082026 - KIET - Hiển thị màn hình Inspection và Object Detection trên Edge UI.
export default function VisionInspection() {
  const [isInspecting, setIsInspecting] = useState(false);
  const [latestResult, setLatestResult] = useState<InferenceResult | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [inputMode, setInputMode] = useState<InputMode>('folder');
  const [taskMode, setTaskMode] = useState<TaskMode>('detection');
  const [availableImages, setAvailableImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState('');
  const [configuredCameras, setConfiguredCameras] = useState<ConfiguredCamera[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [modalImage, setModalImage] = useState<{
    title: string;
    type: 'image' | 'detection';
    imageUrl?: string;
    imageWidth?: number;
    imageHeight?: number;
    objects?: DetectionObject[];
  } | null>(null);
  const [zoomLevel, setZoomLevel] = useState(1);

  const handleOpenModal = (data: {
    title: string;
    type: 'image' | 'detection';
    imageUrl?: string;
    imageWidth?: number;
    imageHeight?: number;
    objects?: DetectionObject[];
  }) => {
    setModalImage(data);
    setZoomLevel(1);
  };

  const handleCloseModal = useCallback(() => {
    setModalImage(null);
    setZoomLevel(1);
  }, []);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        handleCloseModal();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleCloseModal]);

  // 12082026 - KIET - Tải lịch sử inference để hiển thị theo AI task mode.
  const fetchHistory = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/history`, { cache: 'no-store' });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch history');
      }

      setHistory([...(data.results || [])].reverse());
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  }, []);

  // 12082026 - KIET - Tải danh sách ảnh input hiện có trên Edge API.
  const fetchImages = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/images`);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch images');
      }

      // chỉ nhận ảnh in-memory trong phiên đang mở, reload trang thì list để trống.
      const persistedImages: string[] = (data.images || []).filter(
        (name: string) => !name.startsWith('in_memory_image'),
      );

      setAvailableImages(persistedImages);
      if (persistedImages.length > 0) {
        setSelectedImage((currentImage) => currentImage || persistedImages[0]);
      }
    } catch (error) {
      console.error('Failed to fetch images:', error);
    }
  }, []);

  // 19082026 - KIET - Tải camera đã config ở Live Stream để dùng làm input inference.
  const fetchConfiguredCameras = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/cameras`, { cache: 'no-store' });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch configured cameras');
      }

      const cameras: ConfiguredCamera[] = data.cameras || [];
      setConfiguredCameras(cameras);
      setSelectedCameraId((currentCameraId) => {
        if (cameras.some((camera) => camera.camera_id === currentCameraId)) {
          return currentCameraId;
        }

        return cameras.find((camera) => camera.assigned_task === 'detection')?.camera_id
          || cameras[0]?.camera_id
          || '';
      });
    } catch (error) {
      console.error('Failed to fetch configured cameras:', error);
    }
  }, []);

  useEffect(() => {
    // 12082026 - KIET - Đồng bộ dữ liệu Edge API khi màn hình được mở lần đầu.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    fetchHistory();
    fetchImages();
    fetchConfiguredCameras();
  }, [fetchConfiguredCameras, fetchHistory, fetchImages]);

  // 19082026 - KIET - Đổi input giữa ảnh upload và camera đã config ở Live Stream.
  const handleInputModeChange = async (mode: InputMode) => {
    if (mode === 'camera') {
      // 19082026 - KIET - Refresh trạng thái camera mỗi lần chọn Camera nhưng không polling nền.
      await fetchConfiguredCameras();
    }

    setInputMode(mode);
    setLatestResult(null);
  };

  // 12082026 - KIET - Đổi AI task mode và xóa kết quả cũ trên giao diện.
  const handleTaskModeChange = (mode: TaskMode) => {
    setTaskMode(mode);
    setLatestResult(null);

    // 19082026 - KIET - Ưu tiên camera đã được gán đúng task từ Live Stream config.
    const assignedCamera = configuredCameras.find((camera) => camera.assigned_task === mode);
    if (assignedCamera) {
      setSelectedCameraId(assignedCamera.camera_id);
    }
  };

  // 12082026 - KIET - Chọn ảnh input đang được giữ trên RAM của Edge API.
  const handleImageSelect = async (filename: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/set-image/${filename}`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error('Failed to select image');
      }

      setSelectedImage(filename);
      setLatestResult(null);
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Failed to select image');
    }
  };

  // 12082026 - KIET - Upload ảnh vào RAM của Edge API để phục vụ inference.
  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || 'Upload failed');
      }

      setAvailableImages(data.images || []);
      setSelectedImage(data.filename || '');
      setLatestResult(null);
    } catch (error) {
      alert(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      event.target.value = '';
    }
  };

  // 12082026 - KIET - Gọi endpoint tương ứng với Object Detection hoặc Inspection.
  const handleInference = async () => {
    setIsInspecting(true);

    try {
      const targetCam = inputMode === 'camera' ? selectedCameraId : 'default';
      const endpoint = taskMode === 'detection' ? '/detect' : '/inspect';
      const response = await fetch(
        `${API_BASE_URL}${endpoint}?cam=${encodeURIComponent(targetCam)}`,
        { method: 'POST' },
      );
      const result: InferenceResult = await response.json();

      if (!response.ok || result.status === 'error') {
        throw new Error(result.detail || result.message || 'Inference failed');
      }

      setLatestResult(result);
      await fetchHistory();
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unknown inference error';
      alert(`Inference failed: ${message}`);
    } finally {
      setIsInspecting(false);
    }
  };

  const visibleHistory = history
    .filter((item) => (item.task_type || 'inspection') === taskMode)
    .slice(0, 5);

  const selectedConfiguredCamera = configuredCameras.find(
    (camera) => camera.camera_id === selectedCameraId,
  );

  // 19082026 - KIET - Chỉ mở MJPEG stream khi camera config đang online hoặc connecting.
  // 22082026 - PHUC - Camera mode mà camera offline thì hiển thị placeholder,
  // không fallback sang ảnh folder để tránh nhầm ảnh cũ là feed camera.
  const inputImage =
    inputMode === 'camera'
      ? selectedCameraId
        && ['online', 'connecting'].includes(selectedConfiguredCamera?.status || '')
        ? `${API_BASE_URL}/video-feed?cam=${encodeURIComponent(selectedCameraId)}&fps=15`
        : null
      : selectedImage
        ? `${API_BASE_URL}/images/${selectedImage}`
        : null;

  return (
    <div className="inspection-container">
      <header className="dashboard-header">
        <div>
          <h1 className="text-primary">Vision Inspection</h1>

          <div className="mode-group-row mt-2">
            <div>
              <span className="mode-group-label">AI Task</span>
              <div className="mode-selector">
                <button
                  className={`mode-btn ${taskMode === 'detection' ? 'active' : ''}`}
                  onClick={() => handleTaskModeChange('detection')}
                >
                  <Boxes size={14} /> Counting
                </button>
                <button
                  className={`mode-btn ${taskMode === 'inspection' ? 'active' : ''}`}
                  onClick={() => handleTaskModeChange('inspection')}
                // disabled
                // title="Inspection đang tắt vì chưa load Anomaly checkpoint"
                >
                  <ShieldAlert size={14} /> Inspection
                </button>
              </div>
            </div>

            <div>
              <span className="mode-group-label">Input Source</span>
              <div className="mode-selector">
                <button
                  className={`mode-btn ${inputMode === 'folder' ? 'active' : ''}`}
                  onClick={() => handleInputModeChange('folder')}
                >
                  <ImageIcon size={14} /> Folder Mode
                </button>
                <button
                  className={`mode-btn ${inputMode === 'camera' ? 'active' : ''}`}
                  onClick={() => handleInputModeChange('camera')}
                >
                  <Play size={14} /> Camera
                </button>
              </div>
            </div>
          </div>
        </div>

        <button
          className={`inspect-btn ${isInspecting ? 'loading' : ''}`}
          onClick={handleInference}
          disabled={
            isInspecting
            || (inputMode === 'folder' && !selectedImage)
            || (
              inputMode === 'camera'
              && (!selectedCameraId || selectedConfiguredCamera?.status !== 'online')
            )
          }
        >
          {isInspecting ? (
            'Analyzing...'
          ) : (
            <>
              <Play size={18} fill="currentColor" />
              {taskMode === 'detection' ? 'Run Detection' : 'Run Inspection'}
            </>
          )}
        </button>
      </header>

      <div className="inspection-grid">
        <div className="main-viewer">
          <div className={`quad-viewer ${taskMode === 'detection' ? 'detection-mode' : ''}`}>
            <div className="view-panel glass-panel">
              <div className="view-header">
                <div className="view-header-title">
                  <span className="view-step-badge">1</span>
                  <span className="view-header-label">Input Source</span>
                </div>
                {inputImage && (
                  <button
                    className="view-expand-btn"
                    title="Phóng to xem ảnh"
                    onClick={() => handleOpenModal({
                      title: '1. Input Source',
                      type: 'image',
                      imageUrl: inputImage,
                    })}
                  >
                    <Maximize2 size={13} /> View
                  </button>
                )}
              </div>
              <div
                className={`image-display ${inputImage ? 'cursor-pointer' : ''}`}
                onClick={() => {
                  if (inputImage) {
                    handleOpenModal({
                      title: '1. Input Source',
                      type: 'image',
                      imageUrl: inputImage,
                    });
                  }
                }}
              >
                {inputImage ? (
                  <img src={inputImage} alt="Selected input" className="result-image" />
                ) : (
                  <div className="placeholder-image">
                    <ImageIcon size={48} className="text-muted" />
                    <p className="text-muted mt-4">Select an image</p>
                  </div>
                )}
              </div>
            </div>

            {taskMode === 'detection' ? (
              <>
                <div className="view-panel glass-panel">
                  <div className="view-header">
                    <div className="view-header-title">
                      <span className="view-step-badge">2</span>
                      <span className="view-header-label">Detection Result</span>
                    </div>
                    {latestResult?.original_image_url && (
                      <button
                        className="view-expand-btn"
                        title="Phóng to xem kết quả"
                        onClick={() => handleOpenModal({
                          title: '2. Detection Result',
                          type: 'detection',
                          imageUrl: `${API_BASE_URL}${latestResult.original_image_url}`,
                          imageWidth: latestResult.image_width,
                          imageHeight: latestResult.image_height,
                          objects: latestResult.objects,
                        })}
                      >
                        <Maximize2 size={13} /> View
                      </button>
                    )}
                  </div>
                  <div
                    className={`image-display ${latestResult?.original_image_url ? 'cursor-pointer' : ''}`}
                    onClick={() => {
                      if (latestResult?.original_image_url) {
                        handleOpenModal({
                          title: '2. Detection Result',
                          type: 'detection',
                          imageUrl: `${API_BASE_URL}${latestResult.original_image_url}`,
                          imageWidth: latestResult.image_width,
                          imageHeight: latestResult.image_height,
                          objects: latestResult.objects,
                        });
                      }
                    }}
                  >
                    {latestResult?.original_image_url
                      && latestResult.image_width
                      && latestResult.image_height
                      && latestResult.objects ? (
                      <DetectionResultImage
                        imageUrl={`${API_BASE_URL}${latestResult.original_image_url}`}
                        imageWidth={latestResult.image_width}
                        imageHeight={latestResult.image_height}
                        objects={latestResult.objects}
                      />
                    ) : (
                      <div className="placeholder-image text-muted">
                        <Search size={48} />
                        <p className="mt-4">Run detection to view bounding boxes</p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="view-panel glass-panel counting-summary-panel">
                  <div className="view-header">
                    <div className="view-header-title">
                      <span className="view-step-badge">3</span>
                      <span className="view-header-label">Class Counting Summary</span>
                    </div>
                  </div>
                  <div className="counting-summary-body">
                    <div className="total-objects-box">
                      <div className="total-objects-icon">
                        <Target size={24} />
                      </div>
                      <div className="total-objects-info">
                        <span className="total-objects-label">Tổng Objects</span>
                        <span className="total-objects-val">
                          {latestResult?.metrics?.total_objects || 0}
                        </span>
                      </div>
                    </div>

                    <div className="counting-divider" />

                    <div className="class-counts-wrapper">
                      <span className="class-counts-header">Số lượng của mỗi class</span>
                      <div className="class-counts-grid">
                        {DETECTION_CLASSES.map((className) => (
                          <div className="class-count-card" key={className}>
                            <span className="class-name">{className}</span>
                            <span className="class-val">
                              {latestResult?.metrics?.counts_by_class?.[className] ?? 0}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <>
                <div className="view-panel glass-panel">
                  <div className="view-header">
                    <div className="view-header-title">
                      <span className="view-step-badge">2</span>
                      <span className="view-header-label">Object Heatmaps</span>
                    </div>
                  </div>
                  <div className="image-display">
                    {latestResult?.vis_urls?.heatmap ? (
                      <img src={`${API_BASE_URL}${latestResult.vis_urls.heatmap}`} alt="Heatmaps" className="result-image" />
                    ) : (
                      <div className="placeholder-image text-muted">
                        <Search size={48} />
                        <p className="mt-4">Analysis pending</p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="view-panel glass-panel">
                  <div className="view-header">
                    <div className="view-header-title">
                      <span className="view-step-badge">3</span>
                      <span className="view-header-label">Object Outputs (Crops)</span>
                    </div>
                  </div>
                  <div className="image-display">
                    {latestResult?.vis_urls?.crops ? (
                      <img src={`${API_BASE_URL}${latestResult.vis_urls.crops}`} alt="Crops" className="result-image" />
                    ) : (
                      <div className="placeholder-image text-muted">
                        <Search size={48} />
                        <p className="mt-4">Analysis pending</p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="view-panel glass-panel">
                  <div className="view-header">
                    <div className="view-header-title">
                      <span className="view-step-badge">4</span>
                      <span className="view-header-label">Overall Result</span>
                    </div>
                    {latestResult && (
                      <span className={`result-tag ${latestResult.ng_detected ? 'ng' : 'ok'}`}>
                        {latestResult.ng_detected ? 'NG DETECTED' : 'QUALITY OK'}
                      </span>
                    )}
                  </div>
                  <div className="image-display">
                    {latestResult?.vis_urls?.overall ? (
                      <img src={`${API_BASE_URL}${latestResult.vis_urls.overall}`} alt="Overall result" className="result-image" />
                    ) : (
                      <div className="placeholder-image text-muted">
                        <Search size={48} />
                        <p className="mt-4">Analysis pending</p>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        </div>

        <div className="side-panels">
          {inputMode === 'folder' ? (
            <div className="panel glass-panel">
              <div className="panel-header">
                <h3 className="panel-title">Source Images</h3>
                <label className="upload-btn-label">
                  <Folder size={14} /> Open Image
                  <input type="file" hidden onChange={handleUpload} accept="image/*" />
                </label>
              </div>
              <div className="image-list">
                {availableImages.map((image) => (
                  <div
                    key={image}
                    className={`image-item ${selectedImage === image ? 'active' : ''}`}
                    onClick={() => handleImageSelect(image)}
                  >
                    <div className="item-thumbnail-wrapper">
                      <img src={`${API_BASE_URL}/images/${image}`} alt={image} className="item-thumbnail" />
                    </div>
                    <span className="text-xs truncate">{image}</span>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="panel glass-panel">
              <div className="panel-header">
                <h3 className="panel-title">Configured Cameras</h3>
              </div>
              <div className="image-list">
                {configuredCameras.length > 0 ? (
                  configuredCameras.map((cam) => (
                    <div
                      key={cam.camera_id}
                      className={`image-item ${selectedCameraId === cam.camera_id ? 'active' : ''}`}
                      onClick={() => setSelectedCameraId(cam.camera_id)}
                    >
                      <div className="flex items-center justify-between w-full">
                        <div className="flex items-center gap-2">
                          <span
                            className={`status-dot ${
                              cam.status === 'online'
                                ? 'bg-success'
                                : cam.status === 'connecting'
                                ? 'bg-warning'
                                : 'bg-secondary'
                            }`}
                          />
                          <span className="text-xs font-semibold">{cam.name || cam.camera_id}</span>
                        </div>
                        <span className="text-xs text-muted uppercase">{cam.status}</span>
                      </div>
                    </div>
                  ))
                ) : (
                  <p className="text-muted text-center py-4 text-xs">No cameras configured</p>
                )}
              </div>
            </div>
          )}

          <div className="panel glass-panel">
            <h3 className="panel-title">Inference Results</h3>
            {latestResult ? (
              <div className="result-details">
                <div className="metrics-grid-compact">
                  <div className="metric-item-small">
                    <Clock size={16} className="text-muted" />
                    <div className="metric-info-small">
                      <span className="metric-label">Latency</span>
                      <span className="metric-val">
                        {latestResult.metrics?.latency_ms || 0} ms
                      </span>
                    </div>
                  </div>
                  <div className="metric-item-small">
                    <Target size={16} className="text-muted" />
                    <div className="metric-info-small">
                      <span className="metric-label">Objects</span>
                      <span className="metric-val">
                        {latestResult.metrics?.total_objects || 0}
                      </span>
                    </div>
                  </div>

                  {taskMode === 'detection' ? (
                    <div className="metric-item-small">
                      <Boxes size={16} className="text-muted" />
                      <div className="metric-info-small">
                        <span className="metric-label">Classes</span>
                        <span className="metric-val">
                          {Object.keys(latestResult.metrics?.counts_by_class || {}).length}
                        </span>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="metric-item-small">
                        <ShieldAlert size={16} className={latestResult.metrics?.ng_count ? 'text-secondary' : 'text-muted'} />
                        <div className="metric-info-small">
                          <span className="metric-label">NG Count</span>
                          <span className="metric-val">{latestResult.metrics?.ng_count || 0}</span>
                        </div>
                      </div>
                      <div className="metric-item-small">
                        <BarChart size={16} className="text-muted" />
                        <div className="metric-info-small">
                          <span className="metric-label">Max Score</span>
                          <span className="metric-val">{latestResult.metrics?.max_score || 0}</span>
                        </div>
                      </div>
                    </>
                  )}
                </div>

                <div className="divider-line" />
                <div className="result-item mt-2">
                  <span className="text-muted text-xs">Timestamp:</span>
                  <span className="font-mono text-xs">{latestResult.timestamp}</span>
                </div>
              </div>
            ) : (
              <p className="text-muted text-center py-8">No data available</p>
            )}
          </div>

          <div className="panel glass-panel">
            <h3 className="panel-title">Recent History</h3>
            <div className="history-list">
              {visibleHistory.length ? (
                visibleHistory.map((item, index) => (
                  <div key={item.id ?? `${item.timestamp}-${index}`} className="history-item">
                    <div
                      className={`status-dot ${taskMode === 'detection'
                        ? 'bg-primary'
                        : item.ng_detected
                          ? 'bg-secondary'
                          : 'bg-success'
                        }`}
                    />
                    <div className="history-info">
                      <span className="text-xs font-mono">{item.timestamp}</span>
                      <span className="text-sm">
                        {taskMode === 'detection'
                          ? `${item.total_objects || 0} objects`
                          : item.ng_detected
                            ? 'NG'
                            : 'OK'}
                      </span>
                    </div>
                    <Search size={14} className="text-muted" />
                  </div>
                ))
              ) : (
                <p className="text-muted text-center py-8">No history available</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {modalImage && (
        <div className="image-modal-backdrop" onClick={handleCloseModal}>
          <div className="image-modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="image-modal-header">
              <div className="image-modal-title">
                <ImageIcon size={16} />
                <span>{modalImage.title}</span>
              </div>

              <div className="image-modal-actions">
                <button
                  className="modal-icon-btn"
                  title="Zoom In"
                  onClick={() => setZoomLevel((prev) => Math.min(prev + 0.25, 3))}
                >
                  <ZoomIn size={16} />
                </button>
                <button
                  className="modal-icon-btn"
                  title="Zoom Out"
                  onClick={() => setZoomLevel((prev) => Math.max(prev - 0.25, 0.5))}
                >
                  <ZoomOut size={16} />
                </button>
                <button
                  className="modal-icon-btn"
                  title="Reset Zoom"
                  onClick={() => setZoomLevel(1)}
                >
                  <RotateCcw size={16} />
                </button>
                <span className="zoom-value">{Math.round(zoomLevel * 100)}%</span>
                <button className="modal-close-btn" title="Close" onClick={handleCloseModal}>
                  <X size={18} />
                </button>
              </div>
            </div>

            <div className="image-modal-body">
              <div
                className="image-modal-viewport"
                style={{ transform: `scale(${zoomLevel})` }}
              >
                {modalImage.type === 'detection' && modalImage.imageUrl && modalImage.imageWidth && modalImage.imageHeight && modalImage.objects ? (
                  <DetectionResultImage
                    imageUrl={modalImage.imageUrl}
                    imageWidth={modalImage.imageWidth}
                    imageHeight={modalImage.imageHeight}
                    objects={modalImage.objects}
                  />
                ) : modalImage.imageUrl ? (
                  <img
                    src={modalImage.imageUrl}
                    alt={modalImage.title}
                    className="modal-full-image"
                  />
                ) : null}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
