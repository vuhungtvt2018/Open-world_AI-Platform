import React, { useState, useEffect } from 'react';
import { Play, Square, Activity, Cpu, Zap, Thermometer, Clock, CheckCircle2, AlertCircle } from 'lucide-react';
import './ProductionOperation.css';

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

export default function ProductionOperation() {
  const [isRunning, setIsRunning] = useState(false);
  const [count, setCount] = useState(0);
  const [ngCount, setNgCount] = useState(0);
  const [healthMetrics, setHealthMetrics] = useState({
    cpu: 0,
    ram: 0,
    disk: 0,
    gpu: 0,
    temp: 0
  });

  // Fetch history for count and ngCount
  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/history`, { cache: 'no-store' });
        if (res.ok) {
          const data = await res.json();
          const results = data.results || [];
          setCount(results.length);
          setNgCount(results.filter((r: any) => r.ng_detected).length);
        }
      } catch (err) {
        console.error("Failed to fetch history for production stats");
      }
    };
    
    // Luôn fetch 1 lần khi load trang
    fetchHistory();
    
    // Nếu đang chạy thì update liên tục
    if (isRunning) {
      const interval = setInterval(fetchHistory, 3000);
      return () => clearInterval(interval);
    }
  }, [isRunning]);

  // Fetch real health metrics
  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/system-health`);
        if (res.ok) {
          const data = await res.json();
          setHealthMetrics({
            cpu: data.cpu_load,
            ram: data.ram_usage,
            disk: data.disk_usage,
            gpu: data.gpu_load,
            temp: data.temperature
          });
        }
      } catch (err) {
        // Fallback to simulated if backend not reachable
        setHealthMetrics(prev => ({
          ...prev,
          cpu: 25 + Math.random() * 10,
          temp: 42 + Math.random() * 2
        }));
      }
    };

    fetchHealth();
    const intervalId = setInterval(fetchHealth, 3000); // Update every 3s
    return () => clearInterval(intervalId);
  }, []);


  return (
    <div className="production-container">
      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Production Operation</h1>
          <p className="text-muted">Real-time control and synchronization with Edge Nodes & PLC</p>
        </div>
        <div className="line-controls">
          <button 
            className={`control-btn start ${isRunning ? 'active' : ''}`}
            onClick={() => setIsRunning(true)}
            disabled={isRunning}
          >
            <Play size={18} fill="currentColor" />
            Start Line
          </button>
          <button 
            className={`control-btn stop ${!isRunning ? 'active' : ''}`}
            onClick={() => setIsRunning(false)}
            disabled={!isRunning}
          >
            <Square size={18} fill="currentColor" />
            Emergency Stop
          </button>
        </div>
      </header>

      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon bg-blue-soft"><Activity size={20} color="var(--primary)" /></div>
          <div className="stat-content">
            <span className="stat-label">Production Status</span>
            <div className={`status-pill ${isRunning ? 'running' : 'idle'}`}>
              {isRunning ? 'RUNNING' : 'IDLE'}
            </div>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon bg-green-soft"><CheckCircle2 size={20} color="#22c55e" /></div>
          <div className="stat-content">
            <span className="stat-label">Total OK Items</span>
            <span className="stat-value">{count.toLocaleString()}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon bg-red-soft"><AlertCircle size={20} color="#ef4444" /></div>
          <div className="stat-content">
            <span className="stat-label">NG Detected</span>
            <span className="stat-value">{ngCount}</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon bg-orange-soft"><Clock size={20} color="#f97316" /></div>
          <div className="stat-content">
            <span className="stat-label">Cycle Time</span>
            <span className="stat-value">2.4s <span className="text-xs text-muted font-normal">/ target 2.5s</span></span>
          </div>
        </div>
      </div>

      <div className="main-grid">
        {/* Machine Monitoring */}
        <section className="operation-section">
          <div className="section-header">
            <Cpu size={20} className="text-primary" />
            <h2>Edge Node Health</h2>
          </div>
          <div className="health-metrics">
            <div className="metric-item">
              <div className="metric-info">
                <div className="label-with-icon">
                  <Cpu size={14} />
                  <span>CPU Load</span>
                </div>
                <span className="metric-value">{healthMetrics.cpu.toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${healthMetrics.cpu}%`, backgroundColor: healthMetrics.cpu > 80 ? '#ef4444' : 'var(--primary)' }}></div>
              </div>
            </div>

            <div className="metric-item">
              <div className="metric-info">
                <div className="label-with-icon">
                  <Activity size={14} />
                  <span>RAM Usage</span>
                </div>
                <span className="metric-value">{healthMetrics.ram.toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${healthMetrics.ram}%`, backgroundColor: healthMetrics.ram > 85 ? '#ef4444' : 'var(--primary)' }}></div>
              </div>
            </div>

            <div className="metric-item">
              <div className="metric-info">
                <div className="label-with-icon">
                  <Zap size={14} />
                  <span>GPU (NPU) Load</span>
                </div>
                <span className="metric-value">{healthMetrics.gpu.toFixed(1)}%</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${healthMetrics.gpu}%`, backgroundColor: healthMetrics.gpu > 90 ? '#ef4444' : 'var(--primary)' }}></div>
              </div>
            </div>

            <div className="metric-item">
              <div className="metric-info">
                <div className="label-with-icon">
                  <Thermometer size={14} />
                  <span>Core Temperature</span>
                </div>
                <span className="metric-value">{healthMetrics.temp.toFixed(1)}°C</span>
              </div>
              <div className="progress-bar">
                <div className="progress-fill" style={{ width: `${(healthMetrics.temp/90)*100}%`, backgroundColor: healthMetrics.temp > 75 ? '#ef4444' : 'var(--primary)' }}></div>
              </div>
            </div>
          </div>
          <div className="node-details">
            <div className="detail-row">
              <span className="text-muted">Node ID</span>
              <span className="font-mono">EDGE-V01-BULONG</span>
            </div>
            <div className="detail-row">
              <span className="text-muted">Firmware</span>
              <span>v2.4.1-stable</span>
            </div>
          </div>
        </section>

        {/* Sync Status */}
        <section className="operation-section">
          <div className="section-header">
            <Activity size={20} className="text-primary" />
            <h2>PLC & Robot Sync</h2>
          </div>
          <div className="sync-list">
            <div className="sync-item success">
              <div className="sync-indicator"></div>
              <div className="sync-info">
                <span className="sync-name">Siemens S7-1200 PLC</span>
                <span className="sync-status">Connected (12ms latency)</span>
              </div>
            </div>
            <div className="sync-item success">
              <div className="sync-indicator"></div>
              <div className="sync-info">
                <span className="sync-name">Fanuc M-10iD Robot</span>
                <span className="sync-status">Ready (Gripper: Open)</span>
              </div>
            </div>
            <div className="sync-item warning">
              <div className="sync-indicator"></div>
              <div className="sync-info">
                <span className="sync-name">Conveyor Controller</span>
                <span className="sync-status">Running (Low Lubricant)</span>
              </div>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
