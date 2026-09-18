import React, { useState, useRef, useCallback } from 'react';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import { PanelPreview } from './panels/PanelPreview';
import { PanelDetails } from './panels/PanelDetails';
import { PanelLivestream } from './panels/PanelLivestream';
import { PanelSplitter } from './components/PanelSplitter';
import { ModalAddCamera } from './components/ModalAddCamera';
import { ModalSettings } from './components/ModalSettings';

import DashboardSystemInfo from './pages/DashboardSystemInfo';
import DefectSearch from './pages/DefectSearch';
import Reports from './pages/Reports';

import { 
  Camera, 
  SystemMode, 
  LayoutVisibility, 
  SystemKPIs, 
  EdgeTelemetry, 
  MainViewMode, 
  InputSourceMode,
  InferenceResult,
  HistoryRecord
} from './types/vision';
import { MOCK_CAMERAS, INITIAL_KPIS, MOCK_TELEMETRY } from './mock/visionData';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

const getCameraLiveUrl = (cameraId: string) =>
  `${API_BASE_URL}/video-feed?cam=${encodeURIComponent(cameraId)}&fps=15`;

// const INITIAL_HISTORY: HistoryRecord[] = [
//   {
//     id: 'H-101',
//     timestamp: '13:40:12',
//     mode: 'COUNTING',
//     status: 'COUNT',
//     totalObjects: 25,
//     countsByClass: { screw: 12, washer: 8, wood_screw: 5 }
//   },
//   {
//     id: 'H-100',
//     timestamp: '13:35:45',
//     mode: 'INSPECTION',
//     status: 'NG',
//     totalObjects: 2,
//     defectType: 'Vết xước (Scratch) - 94.8%'
//   },
//   {
//     id: 'H-099',
//     timestamp: '13:30:10',
//     mode: 'INSPECTION',
//     status: 'OK',
//     totalObjects: 1
//   }
// ];

export function App() {
  // Global View Mode State (4 main choices)
  const [activeViewMode, setActiveViewMode] = useState<MainViewMode>('CAMERA_OPS');

  // Global State
  const [mode, setMode] = useState<SystemMode>('INSPECTION');
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string>('');
  const [confThreshold] = useState<number>(0.75);
  const [kpis] = useState<SystemKPIs>(INITIAL_KPIS);
  const [telemetry] = useState<EdgeTelemetry>(MOCK_TELEMETRY);

  // New Camera Operations Source Mode & Inference State
  const [sourceMode, setSourceMode] = useState<InputSourceMode>('preview');
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [availableImages, setAvailableImages] = useState<string[]>([]);

  // Fetch logic
  const fetchHistory = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/history`, { cache: 'no-store' });
      const data = await response.json();
      if (response.ok) {
        setHistory([...(data.results || [])].reverse());
      }
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  }, []);

  const fetchImages = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/images`);
      const data = await response.json();
      if (response.ok) {
        const persistedImages: string[] = (data.images || []).filter(
          (name: string) => !name.startsWith('in_memory_image'),
        );
        setAvailableImages(persistedImages);
      }
    } catch (error) {
      console.error('Failed to fetch images:', error);
    }
  }, []);

  const fetchConfiguredCameras = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/cameras`, { cache: 'no-store' });
      const data = await response.json();
      if (response.ok) {
        const cams: Camera[] = data.cameras.map((c: any) => {
          const liveUrl = getCameraLiveUrl(c.camera_id);
          return {
            id: c.camera_id,
            name: c.name,
            location: c.location || 'Station',
            productLine: c.product_line || '',
            status: c.status === 'online' ? 'active' : (c.status === 'connecting' || c.status === 'error') ? 'warning' : 'inactive',
            mode: c.assigned_task === 'detection' ? 'COUNTING' : 'INSPECTION',
            rtspUrl: c.source_url || '',
            ipAddress: c.source_type || '',
            resolution: `${c.width || 1920}x${c.height || 1080}`,
            targetFps: Number(c.fps) || 15,
            currentFps: Number(c.current_fps ?? c.fps) || 15,
            sensorTemp: Number(c.sensor_temp) || 0,
            previewImage: c.preview_image || liveUrl,
            liveUrl,
            inspectionType: c.assigned_task || 'inspection',
            roiEnabled: true,
            exposureTime: 0,
            gain: 0,
          };
        });
        setCameras(cams);
        setSelectedCameraId((prev) => {
          if (cams.some((cam: any) => cam.id === prev)) return prev;
          return cams.find((cam: any) => cam.assigned_task === 'detection')?.id || cams[0]?.id || '';
        });
      }
    } catch (error) {
      console.error('Failed to fetch configured cameras:', error);
    }
  }, []);

  React.useEffect(() => {
    fetchHistory();
    fetchImages();
    fetchConfiguredCameras();
  }, [fetchHistory, fetchImages, fetchConfiguredCameras]);

  // Modals state
  const [isAddCameraOpen, setIsAddCameraOpen] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);

  // Camera Operations Panels Visibility (Preview, Detail, Livestream)
  const [layout, setLayout] = useState<LayoutVisibility>({
    preview: true,
    details: true,
    livestream: true
  });

  // Dynamic Resizable Dimensions (70% Top Row Preview/Details, 30% Bottom Row Livestream)
  const [topRowHeight, setTopRowHeight] = useState<number>(70);
  const [previewWidth, setPreviewWidth] = useState<number>(65); // 65% Preview, 35% Details

  const workspaceMainRef = useRef<HTMLDivElement>(null);
  const topRowRef = useRef<HTMLDivElement>(null);

  const selectedCamera = cameras.find((c) => c.id === selectedCameraId) || cameras[0];

  const handleSourceModeChange = (newMode: InputSourceMode) => {
    setSourceMode(newMode);
    setIsRunning(false);
    setInferenceResult(null);
  };

  const handleToggleRun = async () => {
    if (isRunning) {
      setIsRunning(false);
    } else {
      setIsRunning(true);
      try {
        const targetCam = sourceMode === 'camera' ? selectedCameraId : 'default';
        const endpoint = mode === 'COUNTING' ? '/detect' : '/inspect';
        const response = await fetch(
          `${API_BASE_URL}${endpoint}?cam=${encodeURIComponent(targetCam)}`,
          { method: 'POST' },
        );
        const result = await response.json();

        if (!response.ok || result.status === 'error') {
          throw new Error(result.detail || result.message || 'Inference failed');
        }

        const newTime = new Date().toLocaleTimeString();
        const newRes: InferenceResult = {
          timestamp: newTime,
          status: result.ng_detected ? 'NG' : (mode === 'COUNTING' ? 'COUNT' : 'OK'),
          mode: mode,
          totalObjects: result.metrics?.total_objects || (result.objects ? result.objects.length : 0),
          countsByClass: result.metrics?.counts_by_class,
          defectType: result.ng_detected ? 'Defect Detected' : undefined,
          confidence: result.metrics?.max_score || 0,
          original_image_url: result.original_image_url,
          image_width: result.image_width,
          image_height: result.image_height,
          objects: result.objects,
          visUrls: {
            heatmap: result.vis_urls?.heatmap,
            crops: result.vis_urls?.crops,
            overall: result.vis_urls?.overall || result.original_image_url
          }
        };
        setInferenceResult(newRes);
        await fetchHistory();
      } catch (error) {
        alert(`Inference failed: ${error instanceof Error ? error.message : 'Unknown'}`);
      } finally {
        setIsRunning(false);
      }
    }
  };

  const handleTogglePanel = (panelKey: keyof LayoutVisibility) => {
    setLayout((prev) => ({
      ...prev,
      [panelKey]: !prev[panelKey]
    }));
  };

  const handleResetLayout = () => {
    setLayout({
      preview: true,
      details: true,
      livestream: true
    });
    setTopRowHeight(60);
    setPreviewWidth(65);
  };

  const handleToggleCameraConnection = async (cameraId: string, shouldConnect: boolean) => {
    const endpoint = shouldConnect ? 'connect' : 'disconnect';
    setCameras((prev) =>
      prev.map((cam) =>
        cam.id === cameraId
          ? { ...cam, status: shouldConnect ? 'warning' : 'inactive' }
          : cam,
      ),
    );

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/cameras/${encodeURIComponent(cameraId)}/${endpoint}`,
        { method: 'POST' },
      );
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Failed to ${endpoint} camera`);
      }
      await fetchConfiguredCameras();

      if (shouldConnect) {
        window.setTimeout(() => { void fetchConfiguredCameras(); }, 1500);
        window.setTimeout(() => { void fetchConfiguredCameras(); }, 4000);
      }
    } catch (error) {
      console.error(`Failed to ${endpoint} camera:`, error);
      alert(error instanceof Error ? error.message : `Không thể ${shouldConnect ? 'kết nối' : 'ngắt kết nối'} camera`);
      await fetchConfiguredCameras();
    }
  };

  const handleDeleteCamera = async (cameraId: string) => {
    if (!window.confirm(`Delete camera ${cameraId}?`)) return;

    try {
      const response = await fetch(`${API_BASE_URL}/api/cameras/${encodeURIComponent(cameraId)}`, {
        method: 'DELETE',
      });
      if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || 'Failed to delete camera');
      }

      const remainingCameras = cameras.filter((cam) => cam.id !== cameraId);
      setCameras(remainingCameras);
      setSelectedCameraId((currentId) =>
        currentId === cameraId ? remainingCameras[0]?.id || '' : currentId,
      );
    } catch (error) {
      console.error('Failed to delete camera:', error);
      alert(error instanceof Error ? error.message : 'Network error when deleting camera');
    }
  };

  const handleAddCamera = async (newCam: Camera) => {
    try {
      const [widthStr, heightStr] = (newCam.resolution || '1920x1080').split('x');
      const isBasler = newCam.rtspUrl?.startsWith('basler');
      
      const response = await fetch(`${API_BASE_URL}/api/cameras`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          camera_id: newCam.id,
          name: newCam.name,
          width: parseInt(widthStr, 10) || 1920,
          height: parseInt(heightStr, 10) || 1080,
          fps: newCam.targetFps || 30,
          source_type: isBasler ? 'basler' : 'rtsp',
          source_url: isBasler ? null : newCam.rtspUrl,
          serial_number: isBasler ? newCam.rtspUrl?.replace('basler://', '') : null,
          assigned_task: newCam.mode?.toLowerCase() === 'counting' ? 'detection' : 'inspection',
          enabled: true,
        }),
      });
      if (response.ok) {
        try {
          await fetch(`${API_BASE_URL}/api/cameras/${encodeURIComponent(newCam.id)}/connect`, { method: 'POST' });
        } catch (err) {
          console.log('Connect camera error:', err);
        }
        await fetchConfiguredCameras();
        setSelectedCameraId(newCam.id);
      } else {
        const err = await response.json();
        alert(`Failed to add camera: ${err.detail || 'Unknown error'}`);
      }
    } catch (e) {
      console.error(e);
      alert('Network error when adding camera');
    }
  };

  // Horizontal Resize handler (between Top Row & Bottom Row)
  const handleResizeHorizontal = useCallback((deltaY: number) => {
    if (!workspaceMainRef.current) return;
    const totalHeight = workspaceMainRef.current.clientHeight;
    if (totalHeight <= 0) return;

    const deltaPercent = (deltaY / totalHeight) * 100;
    setTopRowHeight((prev) => Math.min(85, Math.max(15, parseFloat((prev + deltaPercent).toFixed(2)))));
  }, []);

  // Top Row Vertical Resize Handler (Between Preview & Details)
  const handleResizeTopRow = useCallback((deltaX: number) => {
    if (!topRowRef.current) return;
    const totalWidth = topRowRef.current.clientWidth;
    if (totalWidth <= 0) return;

    const deltaPercent = (deltaX / totalWidth) * 100;
    setPreviewWidth((prev) => Math.min(85, Math.max(15, parseFloat((prev + deltaPercent).toFixed(2)))));
  }, []);

  const isTopRowVisible = layout.preview || layout.details;
  const isBottomRowVisible = layout.livestream;

  return (
    <div className="workspace-app">
      {/* Header Bar with 4 main view choices */}
      <Header
        activeViewMode={activeViewMode}
        onViewModeChange={setActiveViewMode}
        mode={mode}
        onModeChange={setMode}
        onOpenSettings={() => setIsSettingsOpen(true)}
      />

      {/* View 1: Dashboard and System Information */}
      {activeViewMode === 'DASHBOARD' && (
        <div className="view-page-container">
          <DashboardSystemInfo mode={mode} />
        </div>
      )}

      {/* View 2: Camera Operation (Preview, Detail, Livestream) */}
      {activeViewMode === 'CAMERA_OPS' && (
        <main className="workspace-main" ref={workspaceMainRef}>
          {/* Top Row: Panel 1 (Preview), Panel 2 (Details) */}
          {isTopRowVisible && (
            <div 
              className="workspace-row top-row" 
              ref={topRowRef}
              style={{
                height: isBottomRowVisible ? `${topRowHeight}%` : '100%'
              }}
            >
              {layout.preview && (
                <div 
                  className="workspace-col col-preview"
                  style={{
                    width: layout.details ? `${previewWidth}%` : '100%'
                  }}
                >
                  <PanelPreview
                    camera={selectedCamera}
                    cameras={cameras}
                    selectedCameraId={selectedCameraId}
                    onSelectCamera={setSelectedCameraId}
                    mode={mode}
                    confThreshold={confThreshold}
                    sourceMode={sourceMode}
                    onSourceModeChange={handleSourceModeChange}
                    isRunning={isRunning}
                    onToggleRun={handleToggleRun}
                    uploadedImage={uploadedImage}
                    onUploadImage={setUploadedImage}
                    inferenceResult={inferenceResult}
                  />
                </div>
              )}

              {layout.preview && layout.details && (
                <PanelSplitter
                  direction="vertical"
                  onDrag={handleResizeTopRow}
                />
              )}

              {layout.details && (
                <div 
                  className="workspace-col col-details"
                  style={{
                    width: layout.preview ? `${100 - previewWidth}%` : '100%'
                  }}
                >
                  <PanelDetails
                    camera={selectedCamera}
                    mode={mode}
                    sourceMode={sourceMode}
                    isRunning={isRunning}
                    inferenceResult={inferenceResult}
                    history={history}
                    onDeleteCamera={handleDeleteCamera}
                    onToggleCameraConnection={handleToggleCameraConnection}
                  />
                </div>
              )}
            </div>
          )}

          {/* Horizontal Splitter */}
          {isTopRowVisible && isBottomRowVisible && (
            <PanelSplitter
              direction="horizontal"
              onDrag={handleResizeHorizontal}
            />
          )}

          {/* Bottom Row: Livestream */}
          {isBottomRowVisible && (
            <div 
              className="workspace-row bottom-row" 
              style={{
                height: isTopRowVisible ? `${100 - topRowHeight}%` : '100%'
              }}
            >
              <div className="workspace-col col-livestream" style={{ width: '100%' }}>
                <PanelLivestream
                  cameras={cameras}
                  selectedCameraId={selectedCameraId}
                  onSelectCamera={setSelectedCameraId}
                  onOpenAddCamera={() => setIsAddCameraOpen(true)}
                  onDeleteCamera={handleDeleteCamera}
                />
              </div>
            </div>
          )}
        </main>
      )}

      {/* View 3: Search (Defect Search) */}
      {activeViewMode === 'SEARCH' && (
        <div className="view-page-container">
          <DefectSearch />
        </div>
      )}

      {/* View 4: Report (Reports System) */}
      {activeViewMode === 'REPORT' && (
        <div className="view-page-container">
          <Reports />
        </div>
      )}

      {/* Modals */}
      <ModalAddCamera
        isOpen={isAddCameraOpen}
        onClose={() => setIsAddCameraOpen(false)}
        onAddCamera={handleAddCamera}
      />

      <ModalSettings
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
      />

      {/* Global Footer */}
      <Footer />
    </div>
  );
}

export default App;
