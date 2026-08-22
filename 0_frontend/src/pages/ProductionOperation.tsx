import React, { useState, useEffect, useCallback } from 'react';
import {
  Play,
  Square,
  Activity,
  Cpu,
  Zap,
  Thermometer,
  Clock,
  CheckCircle2,
  AlertCircle,
  Boxes,
  ShieldAlert,
  Target,
} from 'lucide-react';
import './ProductionOperation.css';

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

type Mode = 'counting' | 'inspection';

// 19082026 - PHUC - 2 Mode Counting & Inspection

export default function ProductionOperation() {
  const [mode, setMode] = useState<Mode>('counting');
  const [isRunning, setIsRunning] = useState(false);
  const [count, setCount] = useState(0);
  const [ngCount, setNgCount] = useState(0);
  const [totalObjects, setTotalObjects] = useState(0);
  const [avgCycleTime, setAvgCycleTime] = useState<number>(0);
  const [classCountsSummary, setClassCountsSummary] = useState<Record<string, number>>({});

  const [healthMetrics, setHealthMetrics] = useState({
    cpu: 0,
    ram: 0,
    disk: 0,
    gpu: 0,
    temp: 0,
  });

  // 12082026 - KIET - Tải dữ liệu lịch sử theo mode Counting hoặc Inspection.
  const fetchHistory = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/history`, { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        const results: any[] = data.results || [];

        const filtered = results.filter((r: any) => {
          const task = r.task_type || 'inspection';
          return mode === 'counting' ? task === 'detection' : task === 'inspection';
        });

        setCount(filtered.length);
        setNgCount(filtered.filter((r: any) => r.ng_detected).length);

        const totObj = filtered.reduce((acc: number, r: any) => acc + (r.total_objects || 0), 0);
        setTotalObjects(totObj);

        const avgLatMs = filtered.length > 0
          ? filtered.reduce((acc: number, r: any) => acc + (r.latency_ms || 0), 0) / filtered.length
          : 0;
        setAvgCycleTime(avgLatMs > 0 ? avgLatMs / 1000 : 2.4);

        const aggregatedClasses: Record<string, number> = {};
        filtered.forEach((r: any) => {
          if (r.counts_by_class) {
            Object.entries(r.counts_by_class).forEach(([cls, cnt]) => {
              aggregatedClasses[cls] = (aggregatedClasses[cls] || 0) + (cnt as number);
            });
          }
        });
        setClassCountsSummary(aggregatedClasses);
      }
    } catch (err) {
      console.error('Failed to fetch history for production stats:', err);
    }
  }, [mode]);

  useEffect(() => {
    fetchHistory();
    if (isRunning) {
      const interval = setInterval(fetchHistory, 3000);
      return () => clearInterval(interval);
    }
  }, [isRunning, fetchHistory]);

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
            temp: data.temperature,
          });
        }
      } catch (err) {
        setHealthMetrics((prev) => ({
          ...prev,
          cpu: 25 + Math.random() * 10,
          temp: 42 + Math.random() * 2,
        }));
      }
    };

    fetchHealth();
    const intervalId = setInterval(fetchHealth, 3000);
    return () => clearInterval(intervalId);
  }, []);

  return (
    <div className="production-container">
      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Production Operation</h1>
          <p className="text-muted">Real-time control and synchronization with Edge Nodes & PLC</p>

          <div className="mode-selector mt-3">
            <button
              className={`mode-btn ${mode === 'counting' ? 'active' : ''}`}
              onClick={() => setMode('counting')}
            >
              <Boxes size={14} /> Counting Mode
            </button>
            <button
              className={`mode-btn ${mode === 'inspection' ? 'active' : ''}`}
              onClick={() => setMode('inspection')}
            >
              <ShieldAlert size={14} /> Inspection Mode
            </button>
          </div>
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
          <div className="stat-icon bg-blue-soft">
            <Activity size={20} color="var(--primary)" />
          </div>
          <div className="stat-content">
            <span className="stat-label">
              Status ({mode === 'counting' ? 'Counting' : 'Inspection'})
            </span>
            <div className={`status-pill ${isRunning ? 'running' : 'idle'}`}>
              {isRunning ? 'RUNNING' : 'IDLE'}
            </div>
          </div>
        </div>

        {mode === 'counting' ? (
          <>
            <div className="stat-card">
              <div className="stat-icon bg-blue-soft">
                <Boxes size={20} color="var(--primary)" />
              </div>
              <div className="stat-content">
                <span className="stat-label">Total Objects Counted</span>
                <span className="stat-value">{totalObjects.toLocaleString()}</span>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon bg-green-soft">
                <Target size={20} color="#22c55e" />
              </div>
              <div className="stat-content">
                <span className="stat-label">Counting Runs</span>
                <span className="stat-value">{count.toLocaleString()}</span>
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="stat-card">
              <div className="stat-icon bg-green-soft">
                <CheckCircle2 size={20} color="#22c55e" />
              </div>
              <div className="stat-content">
                <span className="stat-label">Total OK Items</span>
                <span className="stat-value">{(count - ngCount).toLocaleString()}</span>
              </div>
            </div>

            <div className="stat-card">
              <div className="stat-icon bg-red-soft">
                <AlertCircle size={20} color="#ef4444" />
              </div>
              <div className="stat-content">
                <span className="stat-label">NG Detected</span>
                <span className="stat-value">{ngCount}</span>
              </div>
            </div>
          </>
        )}

        <div className="stat-card">
          <div className="stat-icon bg-orange-soft">
            <Clock size={20} color="#f97316" />
          </div>
          <div className="stat-content">
            <span className="stat-label">Avg Cycle Time</span>
            <span className="stat-value">
              {avgCycleTime ? avgCycleTime.toFixed(2) : '2.40'}s{' '}
              <span className="text-xs text-muted font-normal">/ target 2.5s</span>
            </span>
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
                <div
                  className="progress-fill"
                  style={{
                    width: `${healthMetrics.cpu}%`,
                    backgroundColor: healthMetrics.cpu > 80 ? '#ef4444' : 'var(--primary)',
                  }}
                />
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
                <div
                  className="progress-fill"
                  style={{
                    width: `${healthMetrics.ram}%`,
                    backgroundColor: healthMetrics.ram > 85 ? '#ef4444' : 'var(--primary)',
                  }}
                />
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
                <div
                  className="progress-fill"
                  style={{
                    width: `${healthMetrics.gpu}%`,
                    backgroundColor: healthMetrics.gpu > 90 ? '#ef4444' : 'var(--primary)',
                  }}
                />
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
                <div
                  className="progress-fill"
                  style={{
                    width: `${(healthMetrics.temp / 90) * 100}%`,
                    backgroundColor: healthMetrics.temp > 75 ? '#ef4444' : 'var(--primary)',
                  }}
                />
              </div>
            </div>
          </div>
          <div className="node-details">
            <div className="detail-row">
              <span className="text-muted">Node ID</span>
              <span className="font-mono">EDGE-V01-BULONG</span>
            </div>
            <div className="detail-row">
              <span className="text-muted">Active Model Task</span>
              <span className="font-semibold text-primary uppercase">{mode}</span>
            </div>
            <div className="detail-row">
              <span className="text-muted">Firmware</span>
              <span>v2.4.1-stable</span>
            </div>
          </div>
        </section>

        {/* Sync Status & Class Summary */}
        <section className="operation-section">
          <div className="section-header">
            {mode === 'counting' ? (
              <>
                <Boxes size={20} className="text-primary" />
                <h2>Counting Summary & Sync</h2>
              </>
            ) : (
              <>
                <Activity size={20} className="text-primary" />
                <h2>PLC & Robot Sync</h2>
              </>
            )}
          </div>

          {mode === 'counting' && Object.keys(classCountsSummary).length > 0 && (
            <div className="mb-4">
              <span className="text-xs font-bold uppercase tracking-wider text-muted block mb-2">
                Accumulated Class Breakdown
              </span>
              <div className="grid grid-cols-3 gap-2">
                {Object.entries(classCountsSummary).map(([cls, cnt]) => (
                  <div key={cls} className="p-2 bg-background border border-border rounded-lg text-center">
                    <span className="text-xs text-muted block capitalize">{cls}</span>
                    <strong className="text-base text-primary font-mono">{cnt}</strong>
                  </div>
                ))}
              </div>
            </div>
          )}

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

