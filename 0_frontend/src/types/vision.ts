export type SystemMode = 'INSPECTION' | 'COUNTING';

export type MainViewMode = 'DASHBOARD' | 'CAMERA_OPS' | 'SEARCH' | 'REPORT';

export interface Camera {
  id: string;
  name: string;
  location: string;
  productLine: string;
  status: 'active' | 'inactive' | 'warning';
  mode: SystemMode;
  rtspUrl: string;
  ipAddress: string;
  resolution: string;
  targetFps: number;
  currentFps: number;
  sensorTemp: number;
  previewImage: string;
  liveUrl: string;
  inspectionType: string;
  roiEnabled: boolean;
  exposureTime: number; // in ms
  gain: number;
}

export interface DetectionBox {
  id: string;
  label: string;
  confidence: number;
  x: number; // percentage 0-100
  y: number; // percentage 0-100
  width: number;
  height: number;
  status: 'OK' | 'NG' | 'COUNT';
}

export interface DetectionLog {
  id: string;
  timestamp: string;
  cameraId: string;
  cameraName: string;
  productLine: string;
  status: 'OK' | 'NG' | 'COUNT';
  defectType?: string;
  confidence: number;
  imageUrl: string;
  countNumber?: number;
}

export interface AIModel {
  id: string;
  name: string;
  version: string;
  type: SystemMode;
  accuracy: number;
  status: 'active' | 'ready' | 'training';
  fps: number;
  vramUsed: string;
  lastUpdated: string;
}

export interface DefectCategory {
  id: string;
  name: string;
  code: string;
  color: string;
  count: number;
}

export interface SystemKPIs {
  totalInspected: number;
  okCount: number;
  ngCount: number;
  yieldRate: number;
  avgFps: number;
  processingLatencyMs: number;
  totalCountedProducts: number;
  activeCamerasCount: number;
}

export interface EdgeTelemetry {
  cpuUsage: number;
  gpuUsage: number;
  ramUsage: number;
  gpuTemp: number;
  npuLoad: number;
}

export type InputSourceMode = 'upload' | 'camera' | 'preview';

export interface InferenceMetrics {
  latency_ms?: number;
  total_objects?: number;
  counts_by_class?: Record<string, number>;
  ok_count?: number;
  ng_count?: number;
  max_score?: number;
}

export interface InferenceResult {
  timestamp: string;
  status: 'OK' | 'NG' | 'COUNT';
  mode: SystemMode;
  ngDetected?: boolean;
  totalObjects: number;
  countsByClass?: Record<string, number>;
  defectType?: string;
  confidence?: number;
  imageUrl?: string;
  original_image_url?: string;
  image_width?: number;
  image_height?: number;
  metrics?: InferenceMetrics;
  objects?: any[];
  visUrls?: {
    heatmap?: string;
    crops?: string;
    overall?: string;
  };
}

export interface HistoryRecord {
  id: string;
  timestamp: string;
  mode: SystemMode;
  status: 'OK' | 'NG' | 'COUNT';
  totalObjects: number;
  countsByClass?: Record<string, number>;
  defectType?: string;
}

export interface LayoutVisibility {
  preview: boolean;    // Preview panel
  details: boolean;    // Details panel
  livestream: boolean; // Livestream panel
}
