import React, { useCallback, useEffect, useState } from 'react';
import {
  BarChart3 as BarChart,
  Boxes,
  Clock,
  Folder,
  Image as ImageIcon,
  Play,
  Search,
  ShieldAlert,
  Target,
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
  const [selectedCameraId, setSelectedCameraId] = useState('');

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

      setAvailableImages(data.images || []);
      if (data.images?.length > 0) {
        setSelectedImage((currentImage) => currentImage || data.images[0]);
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
  const inputImage = inputMode === 'camera'
    && selectedCameraId
    && ['online', 'connecting'].includes(selectedConfiguredCamera?.status || '')
    ? `${API_BASE_URL}/video-feed?cam=${encodeURIComponent(selectedCameraId)}&fps=15`
    : selectedImage
      ? `${API_BASE_URL}/images/${selectedImage}`
      : null;

  const averageConfidence = latestResult?.objects?.length
    ? latestResult.objects.reduce((total, item) => total + item.score, 0)
      / latestResult.objects.length
    : 0;

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
                  <Boxes size={14} /> Object Counting
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
          <div className="quad-viewer">
            <div className="view-panel glass-panel">
              <div className="view-header">
                <span className="text-xs font-bold uppercase tracking-wider text-muted">
                  1. Input Source
                </span>
              </div>
              <div className="image-display">
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
                    <span className="text-xs font-bold uppercase tracking-wider text-muted">
                      2. Detection Frame
                    </span>
                    {latestResult && (
                      <span className="result-tag detected">
                        {latestResult.metrics?.total_objects || 0} objects
                      </span>
                    )}
                  </div>
                  <div className="image-display">
                    {latestResult?.original_image_url ? (
                      <img
                        src={`${API_BASE_URL}${latestResult.original_image_url}`}
                        alt="Detection frame"
                        className="result-image"
                      />
                    ) : (
                      <div className="placeholder-image text-muted">
                        <Target size={48} />
                        <p className="mt-4">Run detection to view the saved frame</p>
                      </div>
                    )}
                  </div>
                </div>

                <div className="view-panel glass-panel detection-data-panel">
                  <div className="view-header">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted">
                      3. Class Counting
                    </span>
                  </div>
                  <div className="detection-count-grid">
                    {DETECTION_CLASSES.map((className) => (
                      <div className="detection-count-card" key={className}>
                        <span>{className}</span>
                        <strong>
                          {latestResult?.metrics?.counts_by_class?.[className] ?? 0}
                        </strong>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="view-panel glass-panel">
                  <div className="view-header">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted">
                      4. Detection Result
                    </span>
                  </div>
                  <div className="image-display">
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
              </>
            ) : (
              <>
                <div className="view-panel glass-panel">
                  <div className="view-header">
                    <span className="text-xs font-bold uppercase tracking-wider text-muted">
                      2. Object Heatmaps
                    </span>
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
                    <span className="text-xs font-bold uppercase tracking-wider text-muted">
                      3. Object Outputs (Crops)
                    </span>
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
                    <span className="text-xs font-bold uppercase tracking-wider text-muted">
                      4. Overall Result
                    </span>
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
          {inputMode === 'folder' && (
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
                    <>
                      <div className="metric-item-small">
                        <Boxes size={16} className="text-muted" />
                        <div className="metric-info-small">
                          <span className="metric-label">Classes</span>
                          <span className="metric-val">
                            {Object.keys(latestResult.metrics?.counts_by_class || {}).length}
                          </span>
                        </div>
                      </div>
                      <div className="metric-item-small">
                        <BarChart size={16} className="text-muted" />
                        <div className="metric-info-small">
                          <span className="metric-label">Avg Confidence</span>
                          <span className="metric-val">{(averageConfidence * 100).toFixed(1)}%</span>
                        </div>
                      </div>
                    </>
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
                      className={`status-dot ${
                        taskMode === 'detection'
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
    </div>
  );
}
