import React, { useState, useEffect } from 'react';
import { Play, Square, AlertCircle, CheckCircle2, Monitor, LogOut, Activity, LayoutDashboard, Camera, Bell, PlaySquare, Settings2 } from 'lucide-react';
import GlobalMonitoring from './GlobalMonitoring';
import VisionInspection from './VisionInspection';
import Alerts from './Alerts';
import LiveStream from './LiveStream';
import './EdgeDashboard.css';

interface EdgeDashboardProps {
  onExit: () => void;
}

export default function EdgeDashboard({ onExit }: EdgeDashboardProps) {
  const [activeTab, setActiveTab] = useState('control');
  const [isRunning, setIsRunning] = useState(false);
  const [stats, setStats] = useState({ ok: 1248, ng: 12, total: 1260 });
  const [latestResult, setLatestResult] = useState<'OK' | 'NG' | null>(null);
  const [time, setTime] = useState(new Date().toLocaleTimeString());

  // Clock
  useEffect(() => {
    const timer = setInterval(() => setTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Simulate live data when running
  useEffect(() => {
    let interval: any;
    if (isRunning) {
      interval = setInterval(() => {
        const isNg = Math.random() > 0.85;
        setLatestResult(isNg ? 'NG' : 'OK');
        setStats(prev => ({
          ok: prev.ok + (isNg ? 0 : 1),
          ng: prev.ng + (isNg ? 1 : 0),
          total: prev.total + 1
        }));
      }, 3000);
    } else {
      setLatestResult(null);
    }
    return () => clearInterval(interval);
  }, [isRunning]);

  // Try to request fullscreen on mount
  useEffect(() => {
    const elem = document.documentElement;
    if (elem.requestFullscreen) {
      elem.requestFullscreen().catch((err) => console.log("Fullscreen error:", err));
    }
    return () => {
      if (document.exitFullscreen && document.fullscreenElement) {
        document.exitFullscreen().catch(err => console.log(err));
      }
    };
  }, []);

  return (
    <div className="edge-hmi-container dark-theme">
      {/* Top Bar */}
      <header className="hmi-header">
        <div className="hmi-brand">
          <Monitor size={28} className="text-primary glow" />
          <h1>EDGE TERMINAL <span className="hmi-node-id">#NODE-01</span></h1>
        </div>
        
        <div className="hmi-status-bar">
          <div className="hmi-time font-mono">{time}</div>
          <div className="hmi-conn-status">
            <div className="status-dot-pulse-green"></div>
            <span>PLC SYNCED</span>
          </div>
          <button className="hmi-exit-btn" onClick={onExit}>
            <LogOut size={20} />
            <span>EXIT HMI</span>
          </button>
        </div>
      </header>

      <div className="hmi-content-area">
        {activeTab === 'control' && (
          <div className="hmi-main-grid">
            {/* Left Column: Video Feed & Result */}
            <div className="hmi-video-section">
              <div className="hmi-video-container">
                {isRunning ? (
                  <img 
                    src="http://localhost:8000/video-feed" 
                    alt="Live Inspection Feed" 
                    className="hmi-live-feed"
                    crossOrigin="anonymous"
                    onError={(e) => {
                      (e.target as HTMLImageElement).style.display = 'none';
                      document.getElementById('hmi-video-fallback')!.style.display = 'flex';
                    }}
                  />
                ) : (
                  <div className="hmi-standby">
                    <Activity size={64} className="opacity-20 mb-4" />
                    <h2>SYSTEM STANDBY</h2>
                    <p>Press START to begin inspection</p>
                  </div>
                )}
                <div id="hmi-video-fallback" className="hmi-fallback" style={{display: 'none'}}>
                  <AlertCircle size={48} className="text-danger mb-4" />
                  <h3>Camera Feed Offline</h3>
                  <p>Check connection to the Vision node.</p>
                </div>
                
                {/* Overlay Status */}
                {isRunning && latestResult && (
                  <div className={`hmi-result-overlay ${latestResult === 'OK' ? 'result-ok' : 'result-ng'}`}>
                    {latestResult === 'OK' ? <CheckCircle2 size={80} /> : <AlertCircle size={80} />}
                    <span>{latestResult}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column: Controls & Stats */}
            <div className="hmi-control-section">
              {/* Big Buttons */}
              <div className="hmi-controls">
                <button 
                  className={`hmi-btn hmi-btn-start ${isRunning ? 'active' : ''}`}
                  onClick={() => setIsRunning(true)}
                >
                  <Play size={40} fill="currentColor" />
                  <span>START LINE</span>
                </button>
                <button 
                  className={`hmi-btn hmi-btn-stop ${!isRunning ? 'active' : ''}`}
                  onClick={() => setIsRunning(false)}
                >
                  <Square size={40} fill="currentColor" />
                  <span>EMERGENCY STOP</span>
                </button>
              </div>

              {/* Huge Counters */}
              <div className="hmi-stats">
                <div className="hmi-stat-card bg-glass">
                  <h3>TOTAL INSPECTED</h3>
                  <div className="hmi-stat-value font-mono text-primary">{stats.total.toLocaleString()}</div>
                </div>
                <div className="hmi-stat-card bg-glass-ok">
                  <h3>TOTAL OK</h3>
                  <div className="hmi-stat-value font-mono text-success">{stats.ok.toLocaleString()}</div>
                </div>
                <div className="hmi-stat-card bg-glass-ng">
                  <h3>TOTAL NG</h3>
                  <div className="hmi-stat-value font-mono text-danger">{stats.ng.toLocaleString()}</div>
                </div>
                
                <div className="hmi-yield-card">
                  <h3>YIELD RATE</h3>
                  <div className="hmi-yield-value">
                    {((stats.ok / stats.total) * 100).toFixed(1)}%
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'overview' && <div className="hmi-embedded-view"><GlobalMonitoring /></div>}
        {activeTab === 'vision' && <div className="hmi-embedded-view"><VisionInspection /></div>}
        {activeTab === 'live' && <div className="hmi-embedded-view"><LiveStream /></div>}
        {activeTab === 'alerts' && <div className="hmi-embedded-view"><Alerts /></div>}
      </div>

      {/* Bottom Navigation Dock */}
      <nav className="hmi-bottom-nav">
        <button className={`hmi-nav-btn ${activeTab === 'control' ? 'active' : ''}`} onClick={() => setActiveTab('control')}>
          <Settings2 size={24} />
          <span>Operator Control</span>
        </button>
        <button className={`hmi-nav-btn ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
          <LayoutDashboard size={24} />
          <span>Overview</span>
        </button>
        <button className={`hmi-nav-btn ${activeTab === 'vision' ? 'active' : ''}`} onClick={() => setActiveTab('vision')}>
          <Camera size={24} />
          <span>Vision Details</span>
        </button>
        <button className={`hmi-nav-btn ${activeTab === 'live' ? 'active' : ''}`} onClick={() => setActiveTab('live')}>
          <PlaySquare size={24} />
          <span>Live Stream</span>
        </button>
        <button className={`hmi-nav-btn ${activeTab === 'alerts' ? 'active' : ''}`} onClick={() => setActiveTab('alerts')}>
          <Bell size={24} />
          <span>Alerts</span>
        </button>
      </nav>
    </div>
  );
}
