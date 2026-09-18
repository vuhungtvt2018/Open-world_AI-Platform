import { Camera, AIModel, DefectCategory, SystemKPIs, EdgeTelemetry, DetectionLog } from '../types/vision';

// High resolution industrial inspection sample images
const CAM_IMAGES = {
  cam1: 'https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?auto=format&fit=crop&w=1200&q=80',
  cam2: 'https://images.unsplash.com/photo-1581092160607-ee22621dd758?auto=format&fit=crop&w=1200&q=80',
  cam3: 'https://images.unsplash.com/photo-1581092335397-9583fe92d232?auto=format&fit=crop&w=1200&q=80',
  cam4: 'https://images.unsplash.com/photo-1581092580497-e0d23cbdf1dc?auto=format&fit=crop&w=1200&q=80',
  cam5: 'https://images.unsplash.com/photo-1581091870621-0d32d03a273b?auto=format&fit=crop&w=1200&q=80',
  cam6: 'https://images.unsplash.com/photo-1504328345606-18bbc8c9d7d1?auto=format&fit=crop&w=1200&q=80'
};

export const MOCK_CAMERAS: Camera[] = [
  {
    id: 'CAM01',
    name: 'CAM01 - Sản phẩm 01',
    location: 'Băng tải A - Trạm 01',
    productLine: 'Dây chuyền Đúc Vỏ',
    status: 'active',
    mode: 'INSPECTION',
    rtspUrl: 'rtsp://192.168.1.101:554/live/cam01',
    ipAddress: '192.168.1.101',
    resolution: '1920x1080',
    targetFps: 30,
    currentFps: 29.8,
    sensorTemp: 42.5,
    liveUrl: 'http://localhost:8000/video-feed?cam=CAM-01&fps=15',
  previewImage: CAM_IMAGES.cam1,
    inspectionType: 'Kiểm tra nứt bề mặt & Vết xước',
    roiEnabled: true,
    exposureTime: 12.5,
    gain: 4.2
  },
  {
    id: 'CAM02',
    name: 'CAM02 - Thân máy 02',
    location: 'Băng tải A - Trạm 02',
    productLine: 'Dây chuyền Lắp ráp',
    status: 'active',
    mode: 'COUNTING',
    rtspUrl: 'rtsp://192.168.1.102:554/live/cam02',
    ipAddress: '192.168.1.102',
    resolution: '1920x1080',
    targetFps: 30,
    currentFps: 30.0,
    sensorTemp: 44.1,
  liveUrl: `http://localhost:8000/video-feed?cam=CAM02&fps=15`,
    previewImage: CAM_IMAGES.cam2,
    inspectionType: 'Đếm số lượng ổ trục băng tải',
    roiEnabled: true,
    exposureTime: 10.0,
    gain: 3.5
  },
  {
    id: 'CAM03',
    name: 'CAM03 - Linh kiện B3',
    location: 'Dây chuyền B - Kiểm lỗi cuối',
    productLine: 'Dây chuyền Đóng gói',
    status: 'warning',
    mode: 'INSPECTION',
    rtspUrl: 'rtsp://192.168.1.103:554/live/cam03',
    ipAddress: '192.168.1.103',
    resolution: '2560x1440',
    targetFps: 60,
    currentFps: 54.2,
    sensorTemp: 51.8,
  liveUrl: `http://localhost:8000/video-feed?cam=CAM03&fps=15`,
    previewImage: CAM_IMAGES.cam3,
    inspectionType: 'Độ biến dạng & Thiếu ốc vít',
    roiEnabled: true,
    exposureTime: 8.0,
    gain: 5.0
  },
  {
    id: 'CAM04',
    name: 'CAM04 - Vỏ hợp kim',
    location: 'Phòng kiểm định QC-02',
    productLine: 'Dây chuyền CNC Precision',
    status: 'active',
    mode: 'INSPECTION',
    rtspUrl: 'rtsp://192.168.1.104:554/live/cam04',
    ipAddress: '192.168.1.104',
    resolution: '1920x1080',
    targetFps: 30,
    currentFps: 29.9,
    sensorTemp: 41.0,
  liveUrl: `http://localhost:8000/video-feed?cam=CAM04&fps=15`,
    previewImage: CAM_IMAGES.cam4,
    inspectionType: 'Bề mặt bavia & Vết bẩn',
    roiEnabled: false,
    exposureTime: 15.0,
    gain: 2.8
  },
  {
    id: 'CAM05',
    name: 'CAM05 - Đếm vỏ chai',
    location: 'Băng tải Đóng lọ C1',
    productLine: 'Dây chuyền Đóng chai',
    status: 'active',
    mode: 'COUNTING',
    rtspUrl: 'rtsp://192.168.1.105:554/live/cam05',
    ipAddress: '192.168.1.105',
    resolution: '1920x1080',
    targetFps: 30,
    currentFps: 30.1,
    sensorTemp: 39.4,
  liveUrl: `http://localhost:8000/video-feed?cam=CAM05&fps=15`,
    previewImage: CAM_IMAGES.cam5,
    inspectionType: 'Đếm sản phẩm đầu ra',
    roiEnabled: true,
    exposureTime: 11.2,
    gain: 3.0
  },
  {
    id: 'CAM06',
    name: 'CAM06 - Băng tải D',
    location: 'Kho xuất hàng K-01',
    productLine: 'Kho vận Logistics',
    status: 'inactive',
    mode: 'COUNTING',
    rtspUrl: 'rtsp://192.168.1.106:554/live/cam06',
    ipAddress: '192.168.1.106',
    resolution: '1280x720',
    targetFps: 25,
    currentFps: 0.0,
    sensorTemp: 32.0,
  liveUrl: `http://localhost:8000/video-feed?cam=CAM06&fps=15`,
    previewImage: CAM_IMAGES.cam6,
    inspectionType: 'Đếm thùng hàng xuất kho',
    roiEnabled: false,
    exposureTime: 20.0,
    gain: 1.0
  }
];

export const MOCK_MODELS: AIModel[] = [
  {
    id: 'MOD-01',
    name: 'Edgify-YOLOv8-SurfaceDefect',
    version: 'v3.4.2 (TensorRT FP16)',
    type: 'INSPECTION',
    accuracy: 98.7,
    status: 'active',
    fps: 142.5,
    vramUsed: '2.1 GB',
    lastUpdated: '12/09/2026'
  },
  {
    id: 'MOD-02',
    name: 'Edgify-DeepCount-V2',
    version: 'v2.1.0 (ONNX DirectML)',
    type: 'COUNTING',
    accuracy: 99.4,
    status: 'active',
    fps: 210.0,
    vramUsed: '1.4 GB',
    lastUpdated: '10/09/2026'
  },
  {
    id: 'MOD-03',
    name: 'Edgify-QC-OCR-Reader',
    version: 'v1.8.0',
    type: 'INSPECTION',
    accuracy: 96.5,
    status: 'ready',
    fps: 85.0,
    vramUsed: '1.8 GB',
    lastUpdated: '05/09/2026'
  }
];

export const MOCK_DEFECT_CATEGORIES: DefectCategory[] = [
  { id: 'DEF-1', name: 'Nứt bề mặt (Crack)', code: 'CRACK', color: '#EF4444', count: 142 },
  { id: 'DEF-2', name: 'Vết xước (Scratch)', code: 'SCRATCH', color: '#F59E0B', count: 289 },
  { id: 'DEF-3', name: 'Biến dạng (Deform)', code: 'DEFORM', color: '#EC4899', count: 64 },
  { id: 'DEF-4', name: 'Vết bẩn (Stain)', code: 'STAIN', color: '#8B5CF6', count: 91 },
  { id: 'DEF-5', name: 'Thiếu chi tiết (Missing Part)', code: 'MISSING', color: '#3B82F6', count: 37 }
];

export const INITIAL_KPIS: SystemKPIs = {
  totalInspected: 148590,
  okCount: 147967,
  ngCount: 623,
  yieldRate: 99.58,
  avgFps: 29.8,
  processingLatencyMs: 6.4,
  totalCountedProducts: 342810,
  activeCamerasCount: 5
};

export const MOCK_TELEMETRY: EdgeTelemetry = {
  cpuUsage: 38,
  gpuUsage: 64,
  ramUsage: 52,
  gpuTemp: 58,
  npuLoad: 78
};

export const MOCK_DETECTION_LOGS: DetectionLog[] = [
  {
    id: 'LOG-1009',
    timestamp: '21:44:12',
    cameraId: 'CAM01',
    cameraName: 'CAM01 - Sản phẩm 01',
    productLine: 'Dây chuyền Đúc Vỏ',
    status: 'OK',
    confidence: 99.2,
    imageUrl: CAM_IMAGES.cam1
  },
  {
    id: 'LOG-1008',
    timestamp: '21:44:08',
    cameraId: 'CAM01',
    cameraName: 'CAM01 - Sản phẩm 01',
    productLine: 'Dây chuyền Đúc Vỏ',
    status: 'NG',
    defectType: 'Vết xước (Scratch)',
    confidence: 94.8,
    imageUrl: CAM_IMAGES.cam1
  },
  {
    id: 'LOG-1007',
    timestamp: '21:44:02',
    cameraId: 'CAM02',
    cameraName: 'CAM02 - Thân máy 02',
    productLine: 'Dây chuyền Lắp ráp',
    status: 'COUNT',
    confidence: 99.8,
    imageUrl: CAM_IMAGES.cam2,
    countNumber: 342810
  },
  {
    id: 'LOG-1006',
    timestamp: '21:43:55',
    cameraId: 'CAM03',
    cameraName: 'CAM03 - Linh kiện B3',
    productLine: 'Dây chuyền Đóng gói',
    status: 'NG',
    defectType: 'Nứt bề mặt (Crack)',
    confidence: 97.1,
    imageUrl: CAM_IMAGES.cam3
  },
  {
    id: 'LOG-1005',
    timestamp: '21:43:40',
    cameraId: 'CAM01',
    cameraName: 'CAM01 - Sản phẩm 01',
    productLine: 'Dây chuyền Đúc Vỏ',
    status: 'OK',
    confidence: 99.5,
    imageUrl: CAM_IMAGES.cam1
  }
];

export const HOURLY_TREND_DATA = [
  { time: '08:00', ok: 1250, ng: 6, yield: 99.5 },
  { time: '10:00', ok: 1840, ng: 12, yield: 99.3 },
  { time: '12:00', ok: 1420, ng: 4, yield: 99.7 },
  { time: '14:00', ok: 2100, ng: 9, yield: 99.6 },
  { time: '16:00', ok: 1950, ng: 15, yield: 99.2 },
  { time: '18:00', ok: 1780, ng: 7, yield: 99.6 },
  { time: '20:00', ok: 1650, ng: 5, yield: 99.7 }
];
