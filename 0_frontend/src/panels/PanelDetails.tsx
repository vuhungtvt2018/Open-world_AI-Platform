import React, { useState } from 'react';
import { 
  Camera as CameraIcon, 
  Clock, 
  Sliders, 
  AlertTriangle,
  Flame,
  Activity,
  CheckCircle,
  Eye,
  Boxes,
  Layers,
  History,
  Inbox,
  Sparkles,
  Trash2,
  Plug,
  Unplug,
  LoaderCircle
} from 'lucide-react';
import { Camera, SystemMode, DetectionLog, InputSourceMode, InferenceResult, HistoryRecord } from '../types/vision';
import { MOCK_DETECTION_LOGS } from '../mock/visionData';
import './PanelDetails.css';

interface PanelDetailsProps {
  camera: Camera | null;
  mode: SystemMode;
  sourceMode: InputSourceMode;
  isRunning: boolean;
  inferenceResult: InferenceResult | null;
  history: HistoryRecord[];
  onDeleteCamera: (cameraId: string) => void;
  onToggleCameraConnection: (cameraId: string, shouldConnect: boolean) => Promise<void> | void;
}

export const PanelDetails: React.FC<PanelDetailsProps> = ({
  camera,
  mode,
  sourceMode,
  isRunning,
  inferenceResult,
  history,
  onDeleteCamera,
  onToggleCameraConnection
}) => {
  const [activeTab, setActiveTab] = useState<'telemetry' | 'config'>('telemetry');
  const [exposure, setExposure] = useState<number>(camera?.exposureTime || 12.5);
  const [gain, setGain] = useState<number>(camera?.gain || 4.2);
  const [roiEnabled, setRoiEnabled] = useState<boolean>(camera?.roiEnabled ?? true);
  const [isTogglingConnection, setIsTogglingConnection] = useState(false);
  const [logs] = useState<DetectionLog[]>(MOCK_DETECTION_LOGS);

  // If sourceMode is 'camera' or 'upload', and not running & no active result => render empty state
  const isIdleMode = (sourceMode === 'camera' || sourceMode === 'upload') && !isRunning && !inferenceResult;

  if (isIdleMode) {
    return (
      <div className="panel-container panel-details empty-idle">
        <div className="details-header">
          <span className="details-tag">Details</span>
          <span className="idle-header-badge">IDLE - CHỜ CHẠY</span>
        </div>
        <div className="details-idle-container">
          <div className="idle-icon-wrapper">
            <Inbox size={42} className="idle-icon" />
          </div>
          <h3 className="idle-title">Nội dung Details chưa có dữ liệu</h3>
          <p className="idle-description">
            Ở chế độ <strong>{sourceMode.toUpperCase()}</strong>, thông tin chi tiết (Inference Results, Recent History, Tổng Objects, Số lượng mỗi class) sẽ hiển thị tại đây khi hệ thống đang chạy.
          </p>
          <div className="idle-hint-box">
            <Sparkles size={14} className="text-primary" />
            <span>Vui lòng nhấp nút <strong>"Chạy"</strong> ở bảng Preview phía trên để khởi chạy.</span>
          </div>
        </div>
      </div>
    );
  }

  // Active result state for Camera & Upload mode (or Preview mode when results active)
  if ((sourceMode === 'camera' || sourceMode === 'upload') && (isRunning || inferenceResult)) {
    const currentRes = inferenceResult || {
      timestamp: new Date().toLocaleTimeString(),
      status: mode === 'COUNTING' ? 'COUNT' : 'NG',
      mode: mode,
      totalObjects: mode === 'COUNTING' ? 25 : 2,
      countsByClass: {
        screw: 12,
        washer: 8,
        wood_screw: 5
      },
      defectType: 'Vết xước (Scratch) - 94.8%',
      confidence: 94.8
    };

    return (
      <div className="panel-container panel-details">
        <div className="details-header">
          <span className="details-tag">Details</span>
          <span className="live-run-badge pulse">
            <Activity size={12} /> INFERENCE RESULTS ({sourceMode.toUpperCase()})
          </span>
        </div>

        <div className="details-body scrollable">
          {/* Main Inference Summary Card */}
          <div className="inference-result-card">
            <div className="result-card-header">
              <span className="result-card-title">Kết quả Phân tích AI thời gian thực</span>
              <span className="result-timestamp">{currentRes.timestamp}</span>
            </div>

            {/* COUNTING MODE DETAILS */}
            {mode === 'COUNTING' && (
              <div className="counting-result-group">
                <div className="total-objects-box">
                  <div className="total-icon-wrapper">
                    <Boxes size={24} />
                  </div>
                  <div className="total-info">
                    <span className="total-label">Tổng Objects đếm được:</span>
                    <span className="total-value font-mono">{currentRes.totalObjects} <small>sản phẩm</small></span>
                  </div>
                </div>

                <div className="class-breakdown-section">
                  <div className="section-subtitle">
                    <Layers size={13} />
                    <span>Số lượng phân loại theo từng Class (Counting):</span>
                  </div>

                  <div className="class-grid">
                    {Object.entries(currentRes.countsByClass || {}).map(([className, count]) => (
                      <div key={className} className="class-item-card">
                        <span className="class-name">{className}</span>
                        <span className="class-count font-mono">{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* INSPECTION MODE DETAILS */}
            {mode === 'INSPECTION' && (
              <div className="inspection-result-group">
                <div className={`qc-status-banner ${currentRes.status.toLowerCase()}`}>
                  {currentRes.status === 'OK' ? (
                    <>
                      <CheckCircle size={22} />
                      <div>
                        <strong>KẾT QUẢ QC: PASS (OK)</strong>
                        <span>Tất cả sản phẩm đạt tiêu chuẩn chất lượng.</span>
                      </div>
                    </>
                  ) : (
                    <>
                      <AlertTriangle size={22} />
                      <div>
                        <strong>KẾT QUẢ QC: FAIL (NG)</strong>
                        <span>Phát hiện phế phẩm cần xử lý!</span>
                      </div>
                    </>
                  )}
                </div>

                <div className="inspection-metrics-row">
                  <div className="insp-metric-card">
                    <span className="insp-label">Tổng kiểm tra</span>
                    <span className="insp-val font-mono">2 SP</span>
                  </div>
                  <div className="insp-metric-card ok">
                    <span className="insp-label">Đạt (OK)</span>
                    <span className="insp-val font-mono">1 SP</span>
                  </div>
                  <div className="insp-metric-card ng">
                    <span className="insp-label">Lỗi (NG)</span>
                    <span className="insp-val font-mono">1 SP</span>
                  </div>
                </div>

                <div className="defect-detail-box">
                  <span className="defect-detail-label">Chi tiết lỗi chính:</span>
                  <span className="defect-detail-val text-red">
                    <AlertTriangle size={13} /> {currentRes.defectType || 'Vết xước (Scratch) - 94.8%'}
                  </span>
                </div>
              </div>
            )}

            {/* Common Latency Metrics */}
            <div className="metrics-footer-row">
              <div className="metric-chip">
                <span>Độ trễ Processing:</span>
                <strong className="font-mono">6.4 ms</strong>
              </div>
              <div className="metric-chip">
                <span>Độ chính xác (Score):</span>
                <strong className="font-mono">99.2%</strong>
              </div>
            </div>
          </div>

          {/* Recent History Section */}
          <div className="history-section">
            <div className="section-subtitle">
              <History size={13} />
              <span>Recent History (Lịch sử chạy gần đây)</span>
            </div>

            <div className="history-list">
              {history.map((rec) => (
                <div key={rec.id} className="history-item">
                  <div className="history-time-col">
                    <Clock size={11} />
                    <span className="font-mono">{rec.timestamp}</span>
                  </div>
                  <div className="history-mode-col">
                    <span className="mode-badge">{rec.mode}</span>
                  </div>
                  <div className="history-status-col">
                    <span className={`status-tag ${(rec.status || '').toLowerCase()}`}>
                      {rec.status === 'COUNT' ? `Objects: ${rec.totalObjects}` : (rec.status || 'UNKNOWN')}
                    </span>
                  </div>
                  <div className="history-detail-col">
                    <span>{rec.defectType || `Total: ${rec.totalObjects} items`}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // DEFAULT PREVIEW MODE DETAILS (Live Config & Telemetry)
  if (!camera) {
    return (
      <div className="panel-container panel-details empty">
        <div className="details-header">
          <span className="details-tag">Details</span>
        </div>
        <div className="details-placeholder">Chưa chọn Camera</div>
      </div>
    );
  }

  return (
    <div className="panel-container panel-details">
      {/* Header bar matching sketch */}
      <div className="details-header">
        <span className="details-tag">Details</span>
        <div className="details-subtabs">
          <button
            className={`details-subtab-btn ${activeTab === 'telemetry' ? 'active' : ''}`}
            onClick={() => setActiveTab('telemetry')}
          >
            <Activity size={12} /> Live Log
          </button>
          <button
            className={`details-subtab-btn ${activeTab === 'config' ? 'active' : ''}`}
            onClick={() => setActiveTab('config')}
          >
            <Sliders size={12} /> Cấu hình Cam
          </button>
        </div>
      </div>

      <div className="details-body">
        {/* Selected Camera Specs Card */}
        <div className="camera-spec-card">
          <div className="spec-card-title">
            <CameraIcon size={14} className="text-cyan" />
            <span>{camera.name}</span>
          </div>
          
          <div className="spec-grid">
            <div className="spec-item">
              <span className="spec-label">Vị trí:</span>
              <span className="spec-val">{camera.location}</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Dây chuyền:</span>
              <span className="spec-val">{camera.productLine}</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Độ phân giải:</span>
              <span className="spec-val font-mono">{camera.resolution}</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">IP Address:</span>
              <span className="spec-val font-mono">{camera.ipAddress}</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Nhiệt độ Cam:</span>
              <span className="spec-val font-mono text-orange">
                <Flame size={12} /> {camera.sensorTemp}°C
              </span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Khung hình:</span>
              <span className="spec-val font-mono text-green">{camera.currentFps} FPS</span>
            </div>
            <div className="spec-item">
              <span className="spec-label">Kết nối:</span>
              <span className={`spec-val connection-status ${camera.status}`}>
                {camera.status === 'active'
                  ? 'TRỰC TUYẾN'
                  : camera.status === 'warning'
                    ? 'ĐANG KẾT NỐI'
                    : 'ĐÃ NGẮT KẾT NỐI'}
              </span>
            </div>
          </div>
        </div>

        {/* TAB 1: TELEMETRY DETECTIONS LOG */}
        {activeTab === 'telemetry' && (
          <div className="details-tab-content">
            <div className="section-subtitle">
              <span>Lịch sử phát hiện thời gian thực</span>
              <span className="live-log-badge">REALTIME</span>
            </div>

            <div className="telemetry-logs-list">
              {logs.map((log) => (
                <div key={log.id} className="log-item-card">
                  <div className="log-thumbnail-wrapper">
                    <img src={log.imageUrl} alt="crop" className="log-thumb" />
                    <span className={`log-badge ${log.status.toLowerCase()}`}>
                      {log.status}
                    </span>
                  </div>

                  <div className="log-details">
                    <div className="log-top">
                      <span className="log-time">
                        <Clock size={11} /> {log.timestamp}
                      </span>
                      <span className="log-conf font-mono">{log.confidence}%</span>
                    </div>

                    <div className="log-desc">
                      {log.status === 'NG' ? (
                        <span className="defect-name text-red">
                          <AlertTriangle size={12} /> {log.defectType}
                        </span>
                      ) : log.status === 'OK' ? (
                        <span className="defect-name text-green">
                          <CheckCircle size={12} /> Sản phẩm Đạt standard
                        </span>
                      ) : (
                        <span className="defect-name text-cyan">
                          <Eye size={12} /> SP #{log.countNumber}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* TAB 2: FAST CAMERA CONFIGURATION */}
        {activeTab === 'config' && (
          <div className="details-tab-content">
            <div className="section-subtitle">Thiết lập nhanh cảm biến Camera</div>

            <div className="camera-config-box">
              <div className="config-row connection-row">
                <div className="config-info">
                  <span className="config-title">Trạng thái kết nối Camera</span>
                  <span className={`config-desc connection-status ${camera.status}`}>
                    {camera.status === 'active'
                      ? 'TRỰC TUYẾN — camera đang phát livestream'
                      : camera.status === 'warning'
                        ? 'ĐANG KẾT NỐI — chờ tín hiệu từ camera'
                        : 'ĐÃ NGẮT KẾT NỐI — livestream đang tắt'}
                  </span>
                </div>
                <button
                  type="button"
                  className={`connection-toggle-btn ${camera.status === 'inactive' ? 'connect' : 'disconnect'}`}
                  disabled={isTogglingConnection}
                  onClick={async () => {
                    const shouldConnect = camera.status === 'inactive';
                    setIsTogglingConnection(true);
                    try {
                      await onToggleCameraConnection(camera.id, shouldConnect);
                    } finally {
                      setIsTogglingConnection(false);
                    }
                  }}
                >
                  {isTogglingConnection ? (
                    <LoaderCircle size={14} className="spin" />
                  ) : camera.status === 'inactive' ? (
                    <Plug size={14} />
                  ) : (
                    <Unplug size={14} />
                  )}
                  {isTogglingConnection
                    ? (camera.status === 'inactive' ? 'Đang kết nối...' : 'Đang ngắt...')
                    : camera.status === 'inactive'
                      ? 'Kết nối'
                      : 'Ngắt kết nối'}
                </button>
              </div>

              <div className="config-row">
                <div className="config-info">
                  <span className="config-title">Thời gian phơi sáng (Exposure Time)</span>
                  <span className="config-val font-mono">{exposure} ms</span>
                </div>
                <input
                  type="range"
                  min="1"
                  max="50"
                  step="0.5"
                  value={exposure}
                  onChange={(e) => setExposure(parseFloat(e.target.value))}
                  className="ops-slider"
                />
              </div>

              <div className="config-row">
                <div className="config-info">
                  <span className="config-title">Độ khuếch đại (Sensor Gain)</span>
                  <span className="config-val font-mono">{gain} dB</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="12"
                  step="0.1"
                  value={gain}
                  onChange={(e) => setGain(parseFloat(e.target.value))}
                  className="ops-slider"
                />
              </div>

              <div className="config-row toggle-row">
                <div className="config-info">
                  <span className="config-title">Vùng quan tâm (ROI Filter)</span>
                  <span className="config-desc">Giới hạn vùng quét để tối ưu FPS</span>
                </div>
                <button
                  className={`toggle-btn ${roiEnabled ? 'on' : 'off'}`}
                  onClick={() => setRoiEnabled(!roiEnabled)}
                >
                  {roiEnabled ? 'BẬT' : 'TẮT'}
                </button>
              </div>

              <div className="rtsp-box">
                <span className="rtsp-label">RTSP Stream URL:</span>
                <span className="rtsp-url font-mono">{camera.rtspUrl}</span>
              </div>

              <button className="save-cam-btn">Lưu cấu hình Camera</button>
              <button
                type="button"
                className="delete-cam-btn"
                onClick={() => onDeleteCamera(camera.id)}
              >
                <Trash2 size={14} /> Xóa Camera
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
