import React, { useState } from 'react';
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

const cameras = [
  { id: 1, name: 'Main Intake - CAM 01', status: 'online', resolution: '1920x1080', fps: 30, bitrate: '4.2 Mbps' },
  { id: 2, name: 'Quality Station - CAM 02', status: 'online', resolution: '1920x1080', fps: 28, bitrate: '3.8 Mbps' },
  { id: 3, name: 'Packing Line - CAM 03', status: 'online', resolution: '1280x720', fps: 25, bitrate: '2.1 Mbps' },
  { id: 4, name: 'Storage Exit - CAM 04', status: 'online', resolution: '1280x720', fps: 25, bitrate: '1.9 Mbps' },
];

export default function LiveStream() {
  const [selectedCam, setSelectedCam] = useState(cameras[0]);
  const [isRecording, setIsRecording] = useState(false);

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
              <div className="stream-placeholder">
                <div className="stream-watermark">
                  <span className="live-badge">LIVE</span>
                  <span className="cam-name">{selectedCam.name}</span>
                </div>
                
                {/* Simulated AI Overlays */}
                <div className="ai-overlay">
                  <div className="detection-box" style={{ top: '20%', left: '30%', width: '150px', height: '100px' }}>
                    <span className="box-label">BOLT_OK 98%</span>
                  </div>
                  <div className="detection-box ng" style={{ top: '50%', left: '60%', width: '120px', height: '90px' }}>
                    <span className="box-label">BOLT_NG 92% - SURFACE</span>
                  </div>
                </div>

                <Monitor size={64} className="placeholder-icon opacity-10" />
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
                onClick={() => setSelectedCam(cam)}
              >
                <div className="grid-preview">
                  <Camera size={24} className="opacity-20" />
                  <div className="grid-status-badge">
                    <div className="status-dot online"></div>
                    <span>CAM 0{cam.id}</span>
                  </div>
                </div>
                <div className="grid-info">
                  <span className="name">{cam.name}</span>
                  <span className="meta">{cam.resolution} • {cam.fps} FPS</span>
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
