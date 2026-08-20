import React, { useState, useEffect, useRef } from 'react';
import { 
  TrendingUp, 
  BarChart3, 
  PieChart as PieIcon, 
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
  Pie
} from 'recharts';
import './Analytics.css';

const yieldTrendData = [
  { time: '08:00', rate: 98.2 },
  { time: '09:00', rate: 97.5 },
  { time: '10:00', rate: 94.2 },
  { time: '11:00', rate: 96.8 },
  { time: '12:00', rate: 98.5 },
  { time: '13:00', rate: 97.2 },
  { time: '14:00', rate: 95.8 },
  { time: '15:00', rate: 93.5 },
  { time: '16:00', rate: 97.8 },
];

const defectParetoData = [
  { name: 'Missing Thread', count: 124, color: '#f43f5e' },
  { name: 'Surface Scratch', count: 86, color: '#fb923c' },
  { name: 'Dent / Deformation', count: 42, color: '#facc15' },
  { name: 'Dimension Error', count: 18, color: '#38bdf8' },
  { name: 'Other', count: 12, color: '#94a3b8' },
];

const hourlyOutputData = [
  { hour: '08h', ok: 420, ng: 5 },
  { hour: '09h', ok: 445, ng: 8 },
  { hour: '10h', ok: 390, ng: 15 },
  { hour: '11h', ok: 430, ng: 6 },
  { hour: '12h', ok: 450, ng: 3 },
  { hour: '13h', ok: 410, ng: 12 },
  { hour: '14h', ok: 425, ng: 9 },
  { hour: '15h', ok: 380, ng: 22 },
];

const EDGE_API_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

export default function Analytics() {
  const todayStr = new Date().toISOString().split('T')[0];
  const [selectedDate, setSelectedDate] = React.useState(todayStr);
  const dateInputRef = useRef<HTMLInputElement>(null);
  const [stats, setStats] = React.useState({
    totalInspected: 0,
    overallYield: 100.0,
    totalNg: 0,
    avgCycleTime: 1.2,
    pareto: defectParetoData
  });
  const [yieldTrend, setYieldTrend] = React.useState(yieldTrendData);
  const [hourlyOutput, setHourlyOutput] = React.useState(hourlyOutputData);

  React.useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const res = await fetch(`${EDGE_API_URL}/analytics?date=${selectedDate}`);
        if (!res.ok) return;
        const data = await res.json();
        
        setStats({
          totalInspected: data.totalInspected,
          overallYield: data.overallYield,
          totalNg: data.totalNg,
          avgCycleTime: data.avgCycleTime,
          pareto: data.pareto
        });
        
        setYieldTrend(data.yieldTrendData);
        setHourlyOutput(data.hourlyOutputData);
      } catch (err) {
        console.error("Failed to fetch analytics", err);
      }
    };

    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 10000);
    return () => clearInterval(interval);
  }, [selectedDate]);

  const handleExport = () => {
    const lines: string[] = [];

    // === SECTION 1: SUMMARY ===
    lines.push(`Vision Analytics Report`);
    lines.push(`Date,${selectedDate}`);
    lines.push(`Total Inspected,${stats.totalInspected}`);
    lines.push(`Overall Yield (%),${stats.overallYield}`);
    lines.push(`Total NG,${stats.totalNg}`);
    lines.push(`Avg Cycle Time (s),${stats.avgCycleTime}`);
    lines.push(``);

    // === SECTION 2: HOURLY PRODUCTION OUTPUT ===
    lines.push(`Hourly Production Output`);
    lines.push(`Hour,OK,NG,Total,Yield (%)`);
    for (const row of hourlyOutput) {
      const total = row.ok + row.ng;
      const rate = total > 0 ? ((row.ok / total) * 100).toFixed(1) : '0.0';
      lines.push(`${row.hour},${row.ok},${row.ng},${total},${rate}`);
    }
    lines.push(``);

    // === SECTION 3: DEFECT PARETO ===
    lines.push(`Defect Pareto Analysis`);
    lines.push(`Defect Type,Count`);
    for (const item of stats.pareto) {
      lines.push(`${item.name},${item.count}`);
    }

    const csvContent = lines.join('\n');
    const blob = new Blob(['\uFEFF' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vision_analytics_${selectedDate}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };



  return (
    <div className="analytics-container">
      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Vision Analytics</h1>
          <p className="text-muted">Deep analysis of production quality and defect patterns</p>
        </div>
        <div className="header-actions">
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
          <button className="export-btn" onClick={handleExport}>
            <Download size={18} />
            Export Data
          </button>
        </div>
      </header>

      <div className="stats-row">
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
        <div className="mini-stat-card glass-panel">
          <div className="stat-icon bg-amber"><Clock size={20} /></div>
          <div className="stat-content">
            <span className="stat-label">Avg. Cycle Time</span>
            <span className="stat-value">{stats.avgCycleTime}s</span>
          </div>
        </div>
      </div>

      <div className="analytics-grid">
        {/* Yield Trend */}
        <section className="chart-section glass-panel">
          <div className="section-header">
            <TrendingUp size={20} className="text-primary" />
            <h2>Yield Rate Trend (%)</h2>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={yieldTrend} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="colorRate" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="time" axisLine={false} tickLine={false} fontSize={12} tick={{fill: '#64748b'}} />
                <YAxis domain={[90, 100]} axisLine={false} tickLine={false} fontSize={12} tick={{fill: '#64748b'}} />
                <Tooltip 
                  contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)'}}
                />
                <Area type="monotone" dataKey="rate" stroke="#3b82f6" strokeWidth={3} fillOpacity={1} fill="url(#colorRate)" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Defect Pareto */}
        <section className="chart-section glass-panel">
          <div className="section-header">
            <BarChart3 size={20} className="text-primary" />
            <h2>Defect Pareto Analysis</h2>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={stats.pareto} layout="vertical" margin={{ left: 40 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} fontSize={11} width={100} />
                <Tooltip cursor={{fill: '#f8fafc'}} />
                <Bar dataKey="count" radius={[0, 4, 4, 0]} barSize={20}>
                  {stats.pareto.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>

        {/* Hourly Production */}
        <section className="chart-section full-width glass-panel">
          <div className="section-header">
            <Clock size={20} className="text-primary" />
            <h2>Hourly Production Output</h2>
          </div>
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={hourlyOutput}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f1f5f9" />
                <XAxis dataKey="hour" axisLine={false} tickLine={false} fontSize={12} />
                <YAxis axisLine={false} tickLine={false} fontSize={12} />
                <Tooltip 
                   contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)'}}
                />
                <Bar dataKey="ok" stackId="a" fill="#3b82f6" radius={[0, 0, 0, 0]} barSize={40} name="OK" />
                <Bar dataKey="ng" stackId="a" fill="#f43f5e" radius={[4, 4, 0, 0]} barSize={40} name="NG" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>
    </div>
  );
}
