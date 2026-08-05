import React, { useState, useEffect } from 'react';
import { 
  Camera, 
  Maximize2, 
  Settings, 
  Circle, 
  Video, 
  VideoOff, 
  ChevronRight, 
  Activity,
  Zap,
  Monitor
} from 'lucide-react';
import './LiveStream.css';

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

const initialCameras = [
  { id: 1, name: 'Main Intake - CAM 01', status: 'online', resolution: '1920x1080', fps: 30, bitrate: '4.2 Mbps', streamType: 'Basler' },
  { id: 2, name: 'Quality Station - CAM 02', status: 'online', resolution: '1920x1080', fps: 28, bitrate: '3.8 Mbps', streamType: 'RTSP' },
  { id: 3, name: 'Packing Line - CAM 03', status: 'offline', resolution: '1280x720', fps: 25, bitrate: '2.1 Mbps', streamType: 'None' },
  { id: 4, name: 'Storage Exit - CAM 04', status: 'offline', resolution: '1280x720', fps: 25, bitrate: '1.9 Mbps', streamType: 'None' },
];

export default function LiveStream() {
  const [cameras, setCameras] = useState(initialCameras);
  const [selectedCamId, setSelectedCamId] = useState(1);
  const [isRecording, setIsRecording] = useState(false);

  const selectedCam = cameras.find(c => c.id === selectedCamId) || cameras[0];


  useEffect(() => {
    // Initialize all cameras concurrently
    const initCameras = async () => {
      for (const cam of cameras) {
        if (cam.streamType === 'None') continue;
        const mode = cam.streamType === 'Basler' ? 'basler' : 'stream';
        try {
          await fetch(`${API_BASE_URL}/set-mode/${mode}?cam=${cam.id}`, { method: 'POST' });
        } catch (err) {
          console.error(`Failed to initialize CAM ${cam.id}`, err);
        }
      }
    };
    initCameras();
  }, []); // Run once on mount

  const updateStreamType = async (type: string) => {
    // Update local state
    setCameras(cams => cams.map(c => 
      c.id === selectedCamId ? { ...c, streamType: type } : c
    ));
    
    // Update backend for this specific camera
    const mode = type === 'Basler' ? 'basler' : 'stream';
    try {
      await fetch(`${API_BASE_URL}/set-mode/${mode}?cam=${selectedCamId}`, { method: 'POST' });
    } catch (err) {
      console.error(`Failed to switch mode for CAM ${selectedCamId}`, err);
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
            <span>Total Bandwidth: 12.0 Mbps</span>
          </div>
          <div className="stream-stat">
            <Zap size={16} className="text-primary" />
            <span>Inference: Active</span>
          </div>
        </div>
      </header>

      <div className="stream-layout">
        <div className="stream-main">
          <div className="active-stream-window glass-panel">
            <div className="stream-viewport">
                <div className="stream-placeholder" style={{ padding: 0, position: 'relative', overflow: 'hidden' }}>
                  <div className="stream-watermark" style={{ zIndex: 10 }}>
                    <span className="live-badge">LIVE - {selectedCam.streamType}</span>
                    <span className="cam-name">{selectedCam.name}</span>
                  </div>
                  
                  <img 
                    key={`main-${selectedCam.id}-${selectedCam.streamType}`}
                    src={`${API_BASE_URL}/video-feed?cam=${selectedCam.id}&mode=${selectedCam.streamType}`} 
                    alt="Live Stream" 
                    className="result-image" 
                    style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }}
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiMwMDAiLz48dGV4dCB4PSI1MCUiIHk9IjUwJSIgZmlsbD0iI2ZmZiIgZmlsbC1vcGFjaXR5PSIwLjUiIGZvbnQtZmFtaWx5PSJzYW5zLXNlcmlmIiBmb250LXNpemU9IjI0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5ObyBTdHJlYW0gQ29ubmVjdGlvbjwvdGV4dD48L3N2Zz4=';
                    }}
                  />
                </div>
              
              <div className="stream-controls">
                <div className="control-group">
                  <button className={`control-btn ${isRecording ? 'recording' : ''}`} onClick={() => setIsRecording(!isRecording)}>
                    <Circle size={16} fill={isRecording ? '#f43f5e' : 'none'} />
                    {isRecording ? 'Stop Recording' : 'Start Recording'}
                  </button>
                  <button className="control-btn"><Video size={16} /> Capture Frame</button>
                </div>
                <div className="control-group">
                  <div className="mode-selector" style={{ background: 'rgba(0,0,0,0.5)', padding: '2px', borderRadius: '6px', display: 'flex', gap: '4px' }}>
                    <button 
                      className={`mode-btn ${selectedCam.streamType === 'RTSP' ? 'active' : ''}`}
                      onClick={() => updateStreamType('RTSP')}
                      style={{ padding: '4px 8px', fontSize: '11px', minWidth: '80px', border: 'none', background: selectedCam.streamType === 'RTSP' ? 'var(--primary)' : 'transparent', color: 'white', borderRadius: '4px', cursor: 'pointer' }}
                    >RTSP</button>
                    <button 
                      className={`mode-btn ${selectedCam.streamType === 'Basler' ? 'active' : ''}`}
                      onClick={() => updateStreamType('Basler')}
                      style={{ padding: '4px 8px', fontSize: '11px', minWidth: '80px', border: 'none', background: selectedCam.streamType === 'Basler' ? 'var(--primary)' : 'transparent', color: 'white', borderRadius: '4px', cursor: 'pointer' }}
                    >Basler</button>
                  </div>
                  <button className="icon-btn" title="Settings"><Settings size={18} /></button>
                  <button className="icon-btn" title="Fullscreen"><Maximize2 size={18} /></button>
                </div>
              </div>
            </div>
            
            <div className="stream-metadata">
              <div className="meta-item">
                <span className="label">Resolution</span>
                <span className="value">{selectedCam.resolution}</span>
              </div>
              <div className="meta-item">
                <span className="label">Bitrate</span>
                <span className="value">{selectedCam.bitrate}</span>
              </div>
              <div className="meta-item">
                <span className="label">Latency</span>
                <span className="value">42ms</span>
              </div>
              <div className="meta-item">
                <span className="label">Codec</span>
                <span className="value">H.264 / NVENC</span>
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
                  <img 
                    src={`${API_BASE_URL}/video-feed?cam=${cam.id}&mode=${cam.streamType}`}
                    alt={cam.name}
                    style={{ width: '100%', height: '100%', objectFit: 'cover', background: '#000' }}
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = 'data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxyZWN0IHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiIGZpbGw9IiMzMzMiLz48L3N2Zz4=';
                    }}
                  />
                  <div className="grid-status-badge" style={{ position: 'absolute', bottom: '8px', left: '8px', zIndex: 10 }}>
                    <div className="status-dot online"></div>
                    <span>CAM 0{cam.id}</span>
                  </div>
                </div>
                <div className="grid-info">
                  <span className="name">{cam.name}</span>
                  <span className="meta">{cam.resolution} • {cam.fps} FPS • {cam.streamType}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <aside className="stream-sidebar glass-panel">
          <div className="sidebar-header-sm">
            <Activity size={18} className="text-primary" />
            <h3>Live AI Events</h3>
          </div>
          <div className="event-timeline">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className={`timeline-item ${i === 2 ? 'ng' : ''}`}>
                <div className="time">22:45:1{i}</div>
                <div className="desc">
                  <strong>Object Detected</strong>
                  <span>Bolt_{i === 2 ? 'NG' : 'OK'} (Line 01)</span>
                </div>
                <ChevronRight size={14} className="text-muted" />
              </div>
            ))}
          </div>
          <button className="view-history-btn">View All Events</button>
        </aside>
      </div>
    </div>
  );
}
