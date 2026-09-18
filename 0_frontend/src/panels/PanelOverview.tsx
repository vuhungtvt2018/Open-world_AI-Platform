import React from 'react';
import { 
  CheckCircle2, 
  XCircle, 
  Percent, 
  Zap, 
  Cpu, 
  HardDrive, 
  Activity,
  Layers,
  TrendingUp
} from 'lucide-react';
import { ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, PieChart, Pie, Cell } from 'recharts';
import { SystemKPIs, EdgeTelemetry, SystemMode } from '../types/vision';
import { MOCK_DEFECT_CATEGORIES, HOURLY_TREND_DATA } from '../mock/visionData';
import './PanelOverview.css';

interface PanelOverviewProps {
  kpis: SystemKPIs;
  telemetry: EdgeTelemetry;
  mode: SystemMode;
}

export const PanelOverview: React.FC<PanelOverviewProps> = ({
  kpis,
  telemetry,
  mode
}) => {
  return (
    <div className="panel-container panel-overview">
      {/* Header bar matching sketch */}
      <div className="overview-header">
        <span className="overview-tag">Overview</span>
        <span className="overview-subtitle">Tổng quan Hệ thống toàn ca</span>
      </div>

      <div className="overview-body">
        {/* KPI Cards Grid */}
        <div className="kpi-grid">
          {mode === 'INSPECTION' ? (
            <>
              <div className="kpi-card">
                <div className="kpi-icon-wrapper blue">
                  <Layers size={18} />
                </div>
                <div className="kpi-content">
                  <span className="kpi-label">Tổng kiểm tra</span>
                  <span className="kpi-value font-mono">{kpis.totalInspected.toLocaleString()}</span>
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-icon-wrapper green">
                  <CheckCircle2 size={18} />
                </div>
                <div className="kpi-content">
                  <span className="kpi-label">Sản phẩm Đạt (OK)</span>
                  <span className="kpi-value font-mono text-green">{kpis.okCount.toLocaleString()}</span>
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-icon-wrapper red">
                  <XCircle size={18} />
                </div>
                <div className="kpi-content">
                  <span className="kpi-label">Sản phẩm Lỗi (NG)</span>
                  <span className="kpi-value font-mono text-red">{kpis.ngCount.toLocaleString()}</span>
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-icon-wrapper cyan">
                  <Percent size={18} />
                </div>
                <div className="kpi-content">
                  <span className="kpi-label">Tỷ lệ Đạt (Yield)</span>
                  <span className="kpi-value font-mono text-cyan">{kpis.yieldRate}%</span>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="kpi-card">
                <div className="kpi-icon-wrapper cyan">
                  <TrendingUp size={18} />
                </div>
                <div className="kpi-content">
                  <span className="kpi-label">Tổng số lượng Đếm</span>
                  <span className="kpi-value font-mono text-cyan">{kpis.totalCountedProducts.toLocaleString()} SP</span>
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-icon-wrapper green">
                  <Zap size={18} />
                </div>
                <div className="kpi-content">
                  <span className="kpi-label">Tốc độ trung bình</span>
                  <span className="kpi-value font-mono text-green">142 SP/phút</span>
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-icon-wrapper blue">
                  <Activity size={18} />
                </div>
                <div className="kpi-content">
                  <span className="kpi-label">Độ chính xác Đếm</span>
                  <span className="kpi-value font-mono text-blue">99.8%</span>
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-icon-wrapper purple">
                  <Layers size={18} />
                </div>
                <div className="kpi-content">
                  <span className="kpi-label">Camera hoạt động</span>
                  <span className="kpi-value font-mono text-purple">{kpis.activeCamerasCount} / 6</span>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Charts & Hardware Section */}
        <div className="overview-charts-row">
          {/* Trend Chart */}
          <div className="chart-box">
            <div className="chart-title">Xu hướng sản lượng theo giờ</div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={120}>
                <AreaChart data={HOURLY_TREND_DATA}>
                  <defs>
                    <linearGradient id="colorOk" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" stroke="#475569" fontSize={10} />
                  <YAxis stroke="#475569" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ background: '#0f172a', borderColor: '#334155', borderRadius: '6px', fontSize: '11px' }}
                  />
                  <Area type="monotone" dataKey="ok" stroke="#10b981" fillOpacity={1} fill="url(#colorOk)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Edge Node Hardware Monitor */}
          <div className="hardware-box">
            <div className="chart-title">
              <Cpu size={14} className="text-cyan" />
              <span>Edge AI Hardware Telemetry</span>
            </div>

            <div className="hardware-metrics">
              <div className="hw-bar-group">
                <div className="hw-label font-mono">CPU Usage ({telemetry.cpuUsage}%)</div>
                <div className="hw-progress-track">
                  <div className="hw-progress-bar blue" style={{ width: `${telemetry.cpuUsage}%` }}></div>
                </div>
              </div>

              <div className="hw-bar-group">
                <div className="hw-label font-mono">GPU AI Compute ({telemetry.gpuUsage}%)</div>
                <div className="hw-progress-track">
                  <div className="hw-progress-bar cyan" style={{ width: `${telemetry.gpuUsage}%` }}></div>
                </div>
              </div>

              <div className="hw-bar-group">
                <div className="hw-label font-mono">RAM Memory ({telemetry.ramUsage}%)</div>
                <div className="hw-progress-track">
                  <div className="hw-progress-bar green" style={{ width: `${telemetry.ramUsage}%` }}></div>
                </div>
              </div>

              <div className="hw-bar-group">
                <div className="hw-label font-mono">NPU Neural Acceleration ({telemetry.npuLoad}%)</div>
                <div className="hw-progress-track">
                  <div className="hw-progress-bar purple" style={{ width: `${telemetry.npuLoad}%` }}></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
