import { useCallback, useEffect, useState } from 'react';
import type { FormEvent } from 'react';
import { 
  Maximize2, 
  Settings, 
  Circle, 
  Video, 
  VideoOff, 
  Activity,
  Zap,
  Plus,
  RefreshCw,
  Link,
  Unplug,
  Play,
} from 'lucide-react';
import './LiveStream.css';

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

type CameraTask = 'detection' | 'inspection' | null;
type CameraSourceType = 'rtsp' | 'basler';

interface CameraItem {
  id: string;
  name: string;
  status: 'disconnected' | 'connecting' | 'online' | 'error';
  resolution: string;
  fps: number | null;
  bitrate: string;
  streamType: string;
  sourceType: CameraSourceType;
  sourceUrl: string;
  serialNumber: string;
  assignedTask: CameraTask;
  enabled: boolean;
  error: string | null;
}

interface CameraApiItem {
  camera_id: string;
  name: string;
  status: CameraItem['status'];
  width: number | null;
  height: number | null;
  fps: number | null;
  source_type: CameraSourceType;
  source_url: string | null;
  serial_number: string | null;
  assigned_task: CameraTask;
  enabled: boolean;
  error: string | null;
}

interface CameraFormState {
  cameraId: string;
  name: string;
  sourceType: CameraSourceType;
  sourceUrl: string;
  serialNumber: string;
  assignedTask: CameraTask;
}

const EMPTY_CAMERA_FORM: CameraFormState = {
  cameraId: '',
  name: '',
  sourceType: 'rtsp',
  sourceUrl: '',
  serialNumber: '',
  assignedTask: null,
};

const EMPTY_CAMERA: CameraItem = {
  id: '',
  name: 'No camera configured',
  status: 'disconnected',
  resolution: '-',
  fps: null,
  bitrate: '-',
  streamType: 'None',
  sourceType: 'rtsp',
  sourceUrl: '',
  serialNumber: '',
  assignedTask: null,
  enabled: false,
  error: null,
};

// 19082026 - KIET - Gắn multi-camera API vào đúng layout Live Stream hiện tại.
export default function LiveStream() {
  const [cameras, setCameras] = useState<CameraItem[]>([]);
  const [selectedCamId, setSelectedCamId] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [editingCameraId, setEditingCameraId] = useState<string | null>(null);
  const [cameraForm, setCameraForm] = useState<CameraFormState>(EMPTY_CAMERA_FORM);
  const [baslerDevices, setBaslerDevices] = useState<Array<{ serial_number: string; display_name: string }>>([]);
  const [isBusy, setIsBusy] = useState(false);
  const [message, setMessage] = useState('');

  const selectedCam = cameras.find((camera) => camera.id === selectedCamId) || cameras[0] || EMPTY_CAMERA;

  // 19082026 - KIET - Load camera config và runtime status từ backend một lần khi mở page.
  const loadCameras = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/cameras`, { cache: 'no-store' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Failed to load cameras');

      const loadedCameras: CameraItem[] = (data.cameras || []).map((camera: CameraApiItem) => ({
        id: camera.camera_id,
        name: camera.name,
        status: camera.status,
        resolution: camera.width && camera.height ? `${camera.width}x${camera.height}` : '-',
        fps: camera.fps,
        bitrate: '-',
        streamType: String(camera.source_type || 'none').toUpperCase(),
        sourceType: camera.source_type,
        sourceUrl: camera.source_url || '',
        serialNumber: camera.serial_number || '',
        assignedTask: camera.assigned_task,
        enabled: camera.enabled,
        error: camera.error,
      }));
      setCameras(loadedCameras);
      setSelectedCamId((currentId) => loadedCameras.some((camera) => camera.id === currentId)
        ? currentId
        : loadedCameras[0]?.id || '');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to load cameras');
    }
  }, []);

  useEffect(() => {
    // 19082026 - KIET - Không polling liên tục để tránh page quay/loading mãi.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadCameras();
  }, [loadCameras]);

  // 19082026 - KIET - Mở form thêm camera RTSP, webcam index hoặc Basler.
  const openAddCamera = () => {
    setEditingCameraId(null);
    setCameraForm(EMPTY_CAMERA_FORM);
    setMessage('');
    setShowConfig(true);
  };

  // 19082026 - KIET - Load camera đang chọn vào form để sửa cấu hình cũ.
  const openEditCamera = (camera: CameraItem) => {
    setEditingCameraId(camera.id);
    setCameraForm({
      cameraId: camera.id,
      name: camera.name,
      sourceType: camera.sourceType,
      sourceUrl: camera.sourceUrl,
      serialNumber: camera.serialNumber,
      assignedTask: camera.assignedTask,
    });
    setMessage('');
    setShowConfig(true);
  };

  // 19082026 - KIET - Kiểm tra RTSP URL hoặc webcam index như 0, 1 trước khi lưu.
  const testCameraSource = async () => {
    if (!cameraForm.sourceUrl.trim()) {
      setMessage('Nhập RTSP URL hoặc webcam index, ví dụ 0');
      return;
    }

    setIsBusy(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/cameras/test-rtsp`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ source_url: cameraForm.sourceUrl.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Camera test failed');
      setMessage(`Camera available: ${data.width}x${data.height}`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Camera test failed');
    } finally {
      setIsBusy(false);
    }
  };

  // 19082026 - KIET - Quét danh sách Basler camera khả dụng trên Edge node.
  const scanBaslerCameras = async () => {
    setIsBusy(true);
    try {
      const response = await fetch(`${API_BASE_URL}/api/cameras/discover/basler`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Basler discovery failed');
      setBaslerDevices(data.cameras || []);
      setMessage(data.cameras?.length ? `Found ${data.cameras.length} Basler camera(s)` : 'No Basler camera found');
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Basler discovery failed');
    } finally {
      setIsBusy(false);
    }
  };

  // 19082026 - KIET - Lưu mới hoặc cập nhật camera bằng camera registry API.
  const saveCamera = async (event: FormEvent) => {
    event.preventDefault();
    if (!cameraForm.cameraId.trim() || !cameraForm.name.trim()) {
      setMessage('Camera ID và Camera Name là bắt buộc');
      return;
    }

    setIsBusy(true);
    try {
      const oldCamera = cameras.find((camera) => camera.id === editingCameraId);
      const response = await fetch(`${API_BASE_URL}/api/cameras`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_id: cameraForm.cameraId.trim(),
          name: cameraForm.name.trim(),
          source_type: cameraForm.sourceType,
          source_url: cameraForm.sourceType === 'rtsp' ? cameraForm.sourceUrl.trim() : null,
          serial_number: cameraForm.sourceType === 'basler' ? cameraForm.serialNumber : null,
          assigned_task: cameraForm.assignedTask,
          enabled: oldCamera?.enabled || false,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Failed to save camera');
      setMessage(`Saved camera: ${cameraForm.name}`);
      setSelectedCamId(cameraForm.cameraId.trim());
      await loadCameras();
      setShowConfig(false);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Failed to save camera');
    } finally {
      setIsBusy(false);
    }
  };

  // 19082026 - KIET - Connect hoặc disconnect riêng camera đang chọn.
  const setCameraConnection = async (camera: CameraItem) => {
    if (!camera.id) return;
    const shouldConnect = !['online', 'connecting'].includes(camera.status);
    setIsBusy(true);
    try {
      const action = shouldConnect ? 'connect' : 'disconnect';
      const response = await fetch(`${API_BASE_URL}/api/cameras/${encodeURIComponent(camera.id)}/${action}`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Failed to ${action} camera`);
      setMessage(`${camera.name}: ${shouldConnect ? 'connection started' : 'disconnected'}`);
      if (shouldConnect) await new Promise((resolve) => window.setTimeout(resolve, 1000));
      await loadCameras();
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Camera connection failed');
    } finally {
      setIsBusy(false);
    }
  };

  // 19082026 - KIET - Gọi đúng Detection hoặc Inspection API theo task đã gán.
  const runAssignedTask = async (camera: CameraItem) => {
    if (!camera.assignedTask || camera.status !== 'online') {
      setMessage('Camera phải online và được gán task trước khi chạy');
      return;
    }

    setIsBusy(true);
    try {
      const endpoint = camera.assignedTask === 'detection' ? '/detect' : '/inspect';
      const response = await fetch(`${API_BASE_URL}${endpoint}?cam=${encodeURIComponent(camera.id)}`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok || data.status === 'error') throw new Error(data.detail || data.message || 'Inference failed');
      setMessage(`${camera.name}: ${camera.assignedTask} completed`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Inference failed');
    } finally {
      setIsBusy(false);
    }
  };

  // 19082026 - KIET - Gọi song song task API của các camera online đã được assignment.
  const runAllAssignedTasks = async () => {
    const runnableCameras = cameras.filter((camera) => camera.status === 'online' && camera.assignedTask);
    if (!runnableCameras.length) {
      setMessage('Không có camera online nào đã được gán task');
      return;
    }

    setIsBusy(true);
    try {
      const results = await Promise.allSettled(runnableCameras.map(async (camera) => {
        const endpoint = camera.assignedTask === 'detection' ? '/detect' : '/inspect';
        const response = await fetch(`${API_BASE_URL}${endpoint}?cam=${encodeURIComponent(camera.id)}`, { method: 'POST' });
        const data = await response.json();
        if (!response.ok || data.status === 'error') throw new Error(data.detail || data.message || `${camera.name} failed`);
        return data;
      }));
      const successCount = results.filter((result) => result.status === 'fulfilled').length;
      setMessage(`${successCount}/${results.length} camera task(s) completed`);
    } finally {
      setIsBusy(false);
    }
  };

  return (
    <div className="livestream-container">
      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Live Stream Center</h1>
          <p className="text-muted">Real-time multi-camera monitoring with AI metadata overlay</p>
        </div>
        <div className="header-stats">
          <div className="stream-stat">
            <Activity size={16} className="text-success" />
            <span>{cameras.filter((camera) => camera.status === 'online').length}/{cameras.length} Cameras Online</span>
          </div>
          <div className="stream-stat">
            <Zap size={16} className="text-primary" />
            <span>{cameras.filter((camera) => camera.assignedTask).length} AI Tasks Assigned</span>
          </div>
          <button className="camera-primary-btn" disabled={isBusy} onClick={runAllAssignedTasks}>
            <Play size={14} /> Run All
          </button>
        </div>
      </header>

      {message && <div className="camera-feedback">{message}</div>}

      <div className="stream-layout">
        <div className="stream-main">
          <div className="active-stream-window glass-panel">
            <div className="stream-viewport">
                <div className="stream-placeholder" style={{ padding: 0, position: 'relative', overflow: 'hidden' }}>
                  {selectedCam.id && ['online', 'connecting'].includes(selectedCam.status) ? (
                    <>
                      <div className="stream-watermark" style={{ zIndex: 10 }}>
                        <span className="live-badge">LIVE - {selectedCam.streamType}</span>
                        <span className="cam-name">{selectedCam.name}</span>
                      </div>
                      <img
                        key={`main-${selectedCam.id}-${selectedCam.status}`}
                        src={`${API_BASE_URL}/video-feed?cam=${encodeURIComponent(selectedCam.id)}&fps=15`}
                        alt="Live Stream"
                        className="result-image"
                        style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
                      />
                    </>
                  ) : (
                    <div className="stream-empty-state">
                      <VideoOff size={54} />
                      <strong>{selectedCam.status}</strong>
                      <span>{selectedCam.error || 'Add or connect a camera from Camera Configuration'}</span>
                    </div>
                  )}
                </div>
              
              <div className="stream-controls">
                <div className="control-group">
                  <button className={`control-btn ${isRecording ? 'recording' : ''}`} onClick={() => setIsRecording(!isRecording)}>
                    <Circle size={16} fill={isRecording ? '#f43f5e' : 'none'} />
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                  </button>
                  {selectedCam.id && (
                    <button className="control-btn" disabled={isBusy || !selectedCam.assignedTask} onClick={() => runAssignedTask(selectedCam)}>
                      <Play size={16} /> Run {selectedCam.assignedTask || 'Task'}
                    </button>
                  )}
                </div>
                <div className="control-group">
                  {selectedCam.id && (
                    <button className="icon-btn" title="Connect / Disconnect" disabled={isBusy} onClick={() => setCameraConnection(selectedCam)}>
                      {['online', 'connecting'].includes(selectedCam.status) ? <Unplug size={18} /> : <Link size={18} />}
                    </button>
                  )}
                  <button className="icon-btn" title="Settings" onClick={() => selectedCam.id ? openEditCamera(selectedCam) : openAddCamera()}><Settings size={18} /></button>
                  <button className="icon-btn" title="Fullscreen"><Maximize2 size={18} /></button>
                </div>
              </div>
            </div>
            
            <div className="stream-metadata">
              <div className="meta-item">
                <span className="label">Resolution</span>
                <span className="value">{camStats[selectedCam.id]?.resolution || selectedCam.resolution}</span>
              </div>
              <div className="meta-item">
                <span className="label">Source</span>
                <span className="value">{selectedCam.streamType}</span>
              </div>
              <div className="meta-item">
                <span className="label">Task</span>
                <span className="value">{selectedCam.assignedTask || '-'}</span>
              </div>
              <div className="meta-item">
                <span className="label">Status</span>
                <span className={`value status-text ${selectedCam.status}`}>{selectedCam.status}</span>
              </div>
            </div>
          </div>

          <div className="stream-grid">
            {cameras.map((cam) => (
              <div 
                key={cam.id} 
                className={`grid-item glass-panel ${selectedCam.id === cam.id ? 'active' : ''}`}
                onClick={() => setSelectedCamId(cam.id)}
              >
                <div className="grid-preview" style={{ padding: 0, position: 'relative', overflow: 'hidden' }}>
                  {['online', 'connecting'].includes(cam.status) ? (
                    <img
                      src={`${API_BASE_URL}/video-feed?cam=${encodeURIComponent(cam.id)}&fps=3`}
                      alt={cam.name}
                      style={{ width: '100%', height: '100%', objectFit: 'cover', background: '#000' }}
                    />
                  ) : (
                    <VideoOff size={30} className="text-muted" />
                  )}
                  <div className="grid-status-badge">
                    <div className={`status-dot ${cam.status}`}></div>
                    <span>{cam.status}</span>
                  </div>
                </div>
                <div className="grid-info">
                  <span className="name">{cam.name}</span>
                  <span className="meta">{cam.resolution} • {cam.fps} FPS • {cam.streamType}</span>
                </div>
              </div>
            ))}
            {!cameras.length && (
              <button className="empty-camera-card glass-panel" onClick={openAddCamera}>
                <Plus size={24} /> Add first camera
              </button>
            )}
          </div>
        </div>

        <aside className="stream-sidebar glass-panel">
          {showConfig ? (
            <>
              <div className="sidebar-header-sm">
                <Settings size={18} className="text-primary" />
                <h3>{editingCameraId ? 'Edit Camera' : 'Add Camera'}</h3>
              </div>
              <form className="camera-config-form" onSubmit={saveCamera}>
                <label>Camera ID</label>
                <input
                  value={cameraForm.cameraId}
                  disabled={Boolean(editingCameraId)}
                  placeholder="webcam_01"
                  onChange={(event) => setCameraForm((current) => ({ ...current, cameraId: event.target.value }))}
                />

                <label>Camera Name</label>
                <input
                  value={cameraForm.name}
                  placeholder="Camera Line 1"
                  onChange={(event) => setCameraForm((current) => ({ ...current, name: event.target.value }))}
                />

                <label>Source Type</label>
                <select
                  value={cameraForm.sourceType}
                  onChange={(event) => setCameraForm((current) => ({ ...current, sourceType: event.target.value as CameraSourceType }))}
                >
                  <option value="rtsp">RTSP / Webcam</option>
                  <option value="basler">Basler</option>
                </select>

                {cameraForm.sourceType === 'rtsp' ? (
                  <>
                    <label>RTSP URL / Webcam Index</label>
                    <input
                      value={cameraForm.sourceUrl}
                      placeholder="0 or rtsp://user:password@host/stream"
                      onChange={(event) => setCameraForm((current) => ({ ...current, sourceUrl: event.target.value }))}
                    />
                    <button type="button" className="camera-secondary-btn" disabled={isBusy} onClick={testCameraSource}>
                      <Video size={14} /> Test Source
                    </button>
                  </>
                ) : (
                  <>
                    <label>Basler Device</label>
                    <button type="button" className="camera-secondary-btn" disabled={isBusy} onClick={scanBaslerCameras}>
                      <RefreshCw size={14} /> Scan Devices
                    </button>
                    <select
                      value={cameraForm.serialNumber}
                      onChange={(event) => {
                        const device = baslerDevices.find((item) => item.serial_number === event.target.value);
                        const safeSerial = event.target.value.replace(/[^a-zA-Z0-9_-]/g, '_');
                        setCameraForm((current) => ({
                          ...current,
                          serialNumber: event.target.value,
                          cameraId: editingCameraId ? current.cameraId : `basler_${safeSerial}`,
                          name: editingCameraId ? current.name : device?.display_name || current.name,
                        }));
                      }}
                    >
                      <option value="">Select Basler camera</option>
                      {baslerDevices.map((device) => (
                        <option key={device.serial_number} value={device.serial_number}>{device.display_name} - {device.serial_number}</option>
                      ))}
                    </select>
                  </>
                )}

                <label>Assigned Task</label>
                <select
                  value={cameraForm.assignedTask || ''}
                  onChange={(event) => setCameraForm((current) => ({ ...current, assignedTask: (event.target.value || null) as CameraTask }))}
                >
                  <option value="">No Task</option>
                  <option value="detection">Object Detection</option>
                  <option value="inspection">Inspection</option>
                </select>

                <div className="camera-form-actions">
                  <button type="button" className="camera-secondary-btn" onClick={() => setShowConfig(false)}>Cancel</button>
                  <button type="submit" className="camera-primary-btn" disabled={isBusy}>{isBusy ? 'Saving...' : 'Save Camera'}</button>
                </div>
              </form>
            </>
          ) : (
            <>
              <div className="sidebar-header-sm">
                <Activity size={18} className="text-primary" />
                <h3>Camera Status</h3>
                <button className="sidebar-icon-btn" title="Refresh" onClick={loadCameras}><RefreshCw size={14} /></button>
              </div>
              <div className="event-timeline">
                {cameras.map((camera) => (
                  <div key={camera.id} className={`timeline-item ${camera.status === 'error' ? 'ng' : ''}`}>
                    <div className={`status-dot ${camera.status}`} />
                    <div className="desc">
                      <strong>{camera.name}</strong>
                      <span>{camera.streamType} • {camera.assignedTask || 'No task'}</span>
                    </div>
                    <button className="sidebar-icon-btn" title="Edit" onClick={() => openEditCamera(camera)}><Settings size={14} /></button>
                  </div>
                ))}
                {!cameras.length && <div className="sidebar-empty">No configured camera</div>}
              </div>
              <button className="view-history-btn camera-add-btn" onClick={openAddCamera}><Plus size={14} /> Add Camera</button>
            </>
          )}
        </aside>
      </div>
    </div>
  );
}
