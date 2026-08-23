import React, { useRef } from 'react';
import { 
  Activity,
  TrendingUp, 
  BarChart3, 
  Calendar, 
  Download,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Package
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import './Analytics.css';

const EDGE_API_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

type AnalyticsMode = 'counting' | 'defect';

interface DistributionItem {
  name: string;
  count: number;
  color: string;
}

interface TrendItem {
  time: string;
  rate?: number;
  count?: number;
}

interface HourlyItem {
  hour: string;
  ok?: number;
  ng?: number;
  count?: number;
}

// 23082026-KIET-Tạo ngày Analytics mặc định theo timezone local thay vì UTC
const getLocalDateString = () => {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
};

export default function Analytics() {
  const [mode, setMode] = React.useState<AnalyticsMode>('counting');
  const [selectedDate, setSelectedDate] = React.useState(getLocalDateString);
  const dateInputRef = useRef<HTMLInputElement>(null);
  const [stats, setStats] = React.useState({
    totalInspected: 0,
    overallYield: 100.0,
    totalNg: 0,
    totalRuns: 0,
    totalObjects: 0,
    avgObjectsPerRun: 0,
    avgCycleTime: 0,
    pareto: [] as DistributionItem[],
    classDistribution: [] as DistributionItem[],
  });
  const [yieldTrend, setYieldTrend] = React.useState<TrendItem[]>([]);
  const [objectTrend, setObjectTrend] = React.useState<TrendItem[]>([]);
  const [hourlyOutput, setHourlyOutput] = React.useState<HourlyItem[]>([]);
  const [hourlyCounting, setHourlyCounting] = React.useState<HourlyItem[]>([]);
  const [errorMessage, setErrorMessage] = React.useState('');
  const [isExporting, setIsExporting] = React.useState(false);

  React.useEffect(() => {
    let active = true;

    // 23082026-KIET-Tải đúng analytics schema theo Counting hoặc Defect mode
    const fetchAnalytics = async () => {
      try {
        const query = new URLSearchParams({ date: selectedDate, mode });
        const res = await fetch(`${EDGE_API_URL}/analytics?${query.toString()}`, { cache: 'no-store' });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Failed to fetch analytics');
        if (!active) return;

        setErrorMessage('');
        if (mode === 'counting') {
          setStats({
            totalInspected: 0,
            overallYield: 100,
            totalNg: 0,
            totalRuns: data.totalRuns || 0,
            totalObjects: data.totalObjects || 0,
            avgObjectsPerRun: data.avgObjectsPerRun || 0,
            avgCycleTime: data.avgCycleTime || 0,
            pareto: [],
            classDistribution: data.classDistribution || [],
          });
          setObjectTrend(data.objectTrendData || []);
          setHourlyCounting(data.hourlyCountingData || []);
          setYieldTrend([]);
          setHourlyOutput([]);
        } else {
          setStats({
            totalInspected: data.totalInspected || 0,
            overallYield: data.overallYield ?? 100,
            totalNg: data.totalNg || 0,
            totalRuns: 0,
            totalObjects: 0,
            avgObjectsPerRun: 0,
            avgCycleTime: data.avgCycleTime || 0,
            pareto: data.pareto || [],
            classDistribution: [],
          });
          setYieldTrend(data.yieldTrendData || []);
          setHourlyOutput(data.hourlyOutputData || []);
          setObjectTrend([]);
          setHourlyCounting([]);
        }
      } catch (err) {
        if (active) {
          setErrorMessage(err instanceof Error ? err.message : 'Failed to fetch analytics');
        }
      }
    };

    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 10000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [mode, selectedDate]);

  const handleExport = async () => {
    // 23082026-KIET-Tải file Excel hai sheet được query trực tiếp từ Edge database
    setIsExporting(true);
    setErrorMessage('');
    try {
      const query = new URLSearchParams({ date: selectedDate, mode });
      const response = await fetch(`${EDGE_API_URL}/analytics/export?${query.toString()}`);
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'Failed to export analytics');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `vision_analytics_${mode}_${selectedDate}.xlsx`;
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'Failed to export analytics');
    } finally {
      setIsExporting(false);
    }
  };



  return (
    <div className="analytics-container">
      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Vision Analytics</h1>
          <p className="text-muted">
            {mode === 'counting'
              ? 'Object counting performance and class distribution'
              : 'Production quality and defect pattern analysis'}
          </p>
        </div>
        <div className="header-actions">
          {/* 23082026-KIET-Chuyển nhanh giữa Counting Analytics và Defect Analytics */}
          <div className="analytics-mode-switch">
            <button className={mode === 'counting' ? 'active' : ''} onClick={() => setMode('counting')}>Counting</button>
            <button className={mode === 'defect' ? 'active' : ''} onClick={() => setMode('defect')}>Defect</button>
          </div>
          <div
            className="date-picker"
            style={{ position: 'relative', cursor: 'pointer' }}
            onClick={() => dateInputRef.current?.showPicker()}
          >
            <Calendar size={18} />
            <input
              ref={dateInputRef}
              type="date"
              value={selectedDate}
              onChange={(e) => setSelectedDate(e.target.value)}
              style={{
                position: 'absolute', opacity: 0, pointerEvents: 'none',
                width: 0, height: 0, top: 0, left: 0
              }}
            />
            <span>
              {new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
                .format(new Date(selectedDate + 'T00:00:00'))}
            </span>
          </div>
          <button className="export-btn" onClick={handleExport} disabled={isExporting}>
            <Download size={18} />
            {isExporting ? 'Exporting...' : 'Export Data'}
          </button>
        </div>
      </header>

      {errorMessage && <div className="analytics-feedback">{errorMessage}</div>}

      <div className="stats-row">
        {mode === 'counting' ? (
          <>
            <div className="mini-stat-card glass-panel">
              <div className="stat-icon bg-blue"><Activity size={20} /></div>
              <div className="stat-content">
                <span className="stat-label">Detection Runs</span>
                <span className="stat-value">{stats.totalRuns}</span>
              </div>
            </div>
            <div className="mini-stat-card glass-panel">
              <div className="stat-icon bg-green"><Package size={20} /></div>
              <div className="stat-content">
                <span className="stat-label">Total Objects</span>
                <span className="stat-value text-success">{stats.totalObjects}</span>
              </div>
            </div>
            <div className="mini-stat-card glass-panel">
              <div className="stat-icon bg-red"><BarChart3 size={20} /></div>
              <div className="stat-content">
                <span className="stat-label">Avg. Objects / Run</span>
                <span className="stat-value">{stats.avgObjectsPerRun}</span>
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="mini-stat-card glass-panel">
              <div className="stat-icon bg-blue"><Package size={20} /></div>
              <div className="stat-content">
                <span className="stat-label">Total Inspected</span>
                <span className="stat-value">{stats.totalInspected}</span>
              </div>
            </div>
            <div className="mini-stat-card glass-panel">
              <div className="stat-icon bg-green"><CheckCircle2 size={20} /></div>
              <div className="stat-content">
                <span className="stat-label">Overall Yield</span>
                <span className="stat-value text-success">{stats.overallYield}%</span>
              </div>
            </div>
            <div className="mini-stat-card glass-panel">
              <div className="stat-icon bg-red"><AlertTriangle size={20} /></div>
              <div className="stat-content">
                <span className="stat-label">Total NG</span>
                <span className="stat-value text-danger">{stats.totalNg}</span>
              </div>
            </div>
          </>
        )}
        <div className="mini-stat-card glass-panel">
          <div className="stat-icon bg-amber"><Clock size={20} /></div>
          <div className="stat-content">
            <span className="stat-label">Avg. Cycle Time</span>
            <span className="stat-value">{stats.avgCycleTime}s</span>
          </div>
        </div>
      </div>

      <div className="analytics-grid">
        {/* 23082026-KIET-Dùng chung chart trend nhưng đổi metric theo Analytics mode */}
        <section className="chart-section glass-panel">
          <div className="section-header">
            <TrendingUp size={20} className="text-primary" />
            <h2>{mode === 'counting' ? 'Object Count Trend' : 'Yield Rate Trend (%)'}</h2>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={mode === 'counting' ? objectTrend : yieldTrend} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorAnalyticsRate" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="time" axisLine={false} tickLine={false} fontSize={12} tick={{fill: '#64748b'}} />
                <YAxis domain={mode === 'defect' ? [0, 100] : undefined} axisLine={false} tickLine={false} fontSize={12} tick={{fill: '#64748b'}} />
                <Tooltip 
                  contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)'}}
                />
                <Area type="monotone" dataKey={mode === 'counting' ? 'count' : 'rate'} stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorAnalyticsRate)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* 23082026-KIET-Hiển thị class distribution cho Counting hoặc Pareto cho Defect */}
        <section className="chart-section glass-panel">
          <div className="section-header">
            <BarChart3 size={20} className="text-primary" />
            <h2>{mode === 'counting' ? 'Object Class Distribution' : 'Defect Pareto Analysis'}</h2>
          </div>
          <div className="chart-wrapper">
            {(mode === 'counting' ? stats.classDistribution : stats.pareto).length ? (
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={mode === 'counting' ? stats.classDistribution : stats.pareto} layout="vertical" margin={{ left: 40 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" hide />
                  <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} fontSize={11} width={100} />
                  <Tooltip cursor={{fill: '#f8fafc'}} />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={20}>
                    {(mode === 'counting' ? stats.classDistribution : stats.pareto).map((entry, index) => (
                      <Cell key={`${entry.name}-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="analytics-empty-chart">No data for the selected date</div>
            )}
          </div>
        </section>

        {/* 23082026-KIET-Đổi hourly chart giữa object count và OK/NG output */}
        <section className="chart-section full-width glass-panel">
          <div className="section-header">
            <Clock size={20} className="text-primary" />
            <h2>{mode === 'counting' ? 'Hourly Counting Output' : 'Hourly Production Output'}</h2>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={mode === 'counting' ? hourlyCounting : hourlyOutput}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="hour" axisLine={false} tickLine={false} fontSize={12} />
                <YAxis axisLine={false} tickLine={false} fontSize={12} />
                <Tooltip 
                   contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)'}}
                />
                {mode === 'counting' ? (
                  <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} barSize={40} name="Objects" />
                ) : (
                  <>
                    <Bar dataKey="ok" stackId="result" fill="#3b82f6" barSize={40} name="OK" />
                    <Bar dataKey="ng" stackId="result" fill="#f43f5e" radius={[4, 4, 0, 0]} barSize={40} name="NG" />
                  </>
                )}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
    </div>
  );
}
