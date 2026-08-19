import React, { useState, useEffect } from 'react';
import { Activity, Server, Cpu, HardDrive, AlertTriangle, CheckCircle2 } from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area
} from 'recharts';
import './GlobalMonitoring.css';

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

export default function GlobalMonitoring() {
  const [history, setHistory] = useState<any[]>([]);
  const [metrics, setMetrics] = useState({
    throughput: 0,
    defectRate: 0,
    avgLatency: 0,
    latencyData: [] as any[]
  });

  const fetchData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/history`, { cache: 'no-store' });
      const data = await response.json();
      const results = data.results || [];
      setHistory(results.slice(-5).reverse());

      // Calculate stats
      const total = results.length;
      const ngCount = results.filter((r: any) => r.ng_detected).length;
      const dRate = total > 0 ? (ngCount / total) * 100 : 0;

      const latencyPoints = results.slice(-10).map((r: any) => ({
        time: r.timestamp.split('_')[1] || r.timestamp,
        value: r.latency_ms || 0
      }));

      const avgLat = results.length > 0
        ? results.reduce((acc: number, r: any) => acc + (r.latency_ms || 0), 0) / results.length
        : 0;

      setMetrics({
        throughput: total,
        defectRate: dRate,
        avgLatency: avgLat,
        latencyData: latencyPoints
      });
    } catch (error) {
      console.error('Failed to fetch monitoring data:', error);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000); // Update every 5s
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div>
          <h1 className="text-primary">Global Monitoring</h1>
          <p className="text-muted text-sm mt-1">Factory overview and edge node status</p>
        </div>
        <div className="header-actions">
          <div className="status-badge glass-panel glow-success">
            <CheckCircle2 size={16} color="var(--success)" />
            <span>AI Inference Node Online</span>
          </div>
        </div>
      </header>

      <div className="metrics-grid">
        <div className="metric-card glass-panel">
          <div className="metric-header">
            <h3 className="text-muted">Total Throughput</h3>
            <Activity size={20} color="var(--primary)" />
          </div>
          <div className="metric-value">{metrics.throughput} <span className="metric-unit">items</span></div>
          <div className="metric-chart">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={metrics.latencyData}>
                <defs>
                  <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="var(--primary)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <Area type="monotone" dataKey="value" stroke="var(--primary)" fillOpacity={1} fill="url(#colorValue)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-header">
            <h3 className="text-muted">Defect Rate (NG)</h3>
            <AlertTriangle size={20} color="var(--warning)" />
          </div>
          <div className="metric-value text-warning">{metrics.defectRate.toFixed(1)}<span className="metric-unit">%</span></div>
          <div className="metric-trend text-success">Live Analysis</div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-header">
            <h3 className="text-muted">Avg Latency</h3>
            <Cpu size={20} color="var(--secondary)" />
          </div>
          <div className="metric-value text-secondary">{metrics.avgLatency.toFixed(0)}<span className="metric-unit">ms</span></div>
          <div className="progress-bar-bg">
            <div className="progress-bar-fill glow-secondary" style={{ width: `${Math.min(100, metrics.avgLatency / 10)}%`, backgroundColor: 'var(--secondary)' }}></div>
          </div>
        </div>

        <div className="metric-card glass-panel">
          <div className="metric-header">
            <h3 className="text-muted">Active Node</h3>
            <Server size={20} color="var(--success)" />
          </div>
          <div className="metric-value">1<span className="text-muted text-xl">/1</span></div>
          <div className="metric-trend text-success">100% Uptime</div>
        </div>
      </div>

      <div className="dashboard-content">
        <div className="panel glass-panel col-span-2">
          <h3 className="panel-title">Inference Latency Trend (ms)</h3>
          <div className="panel-body" style={{ height: '300px' }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={metrics.latencyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" vertical={false} />
                <XAxis dataKey="time" stroke="var(--text-muted)" fontSize={12} tickLine={false} />
                <YAxis stroke="var(--text-muted)" fontSize={12} tickLine={false} axisLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: 'var(--surface)', borderColor: 'var(--border)' }}
                  itemStyle={{ color: 'var(--primary)' }}
                />
                <Line type="monotone" dataKey="value" stroke="var(--primary)" strokeWidth={2} dot={true} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="panel glass-panel">
          <h3 className="panel-title">Recent Inspection Events</h3>
          <div className="event-list">
            {history.map((event, idx) => (
              <div key={idx} className="event-item">
                <div className={`event-icon ${event.ng_detected ? 'bg-secondary' : 'bg-success'}`}>
                  {event.ng_detected ? <AlertTriangle size={14} color="#fff" /> : <CheckCircle2 size={14} color="#fff" />}
                </div>
                <div className="event-details">
                  <span className="event-time">
                    {(() => {
                      const ts = event.timestamp || '';
                      const match = ts.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})/);
                      if (match) {
                        const [_, year, month, day, hour, min, sec] = match;
                        return `${day}/${month}/${year} ${hour}:${min}:${sec}`;
                      }
                      return ts;
                    })()}
                  </span>
                  <span className="event-message">
                    {event.ng_detected ? `NG Detected (${event.ng_count}/${event.total_objects} sản phẩm)` : 'Sản phẩm OK'}
                  </span>
                </div>
              </div>
            ))}
            {history.length === 0 && <p className="text-muted text-center py-4">No events recorded</p>}
          </div>
        </div>
      </div>
    </div>
  );
}
