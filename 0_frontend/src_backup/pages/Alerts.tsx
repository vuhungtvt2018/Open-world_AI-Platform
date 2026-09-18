import React, { useState } from 'react';
import { 
  AlertCircle, 
  Bell, 
  ShieldAlert, 
  CheckCircle, 
  Settings, 
  Trash2, 
  BellOff, 
  MessageSquare,
  Clock
} from 'lucide-react';
import './Alerts.css';

const initialAlerts = [
  { id: 1, severity: 'critical', source: 'Line 01 - Camera 2', message: 'Consecutive NG detected (5 items)', time: '2 mins ago', status: 'active' },
  { id: 2, severity: 'warning', source: 'Edge Node A', message: 'High GPU Temperature detected (82°C)', time: '12 mins ago', status: 'active' },
  { id: 3, severity: 'info', source: 'System', message: 'Backup completed successfully', time: '1 hour ago', status: 'acknowledged' },
  { id: 4, severity: 'critical', source: 'PLC Bridge', message: 'Connection timeout - Retrying...', time: '2 hours ago', status: 'active' },
  { id: 5, severity: 'warning', source: 'Storage', message: 'Disk space low (85% full)', time: '4 hours ago', status: 'acknowledged' },
];

export default function Alerts() {
  const [alerts, setAlerts] = useState(initialAlerts);

  const getSeverityIcon = (sev: string) => {
    switch (sev) {
      case 'critical': return <ShieldAlert size={20} className="text-danger" />;
      case 'warning': return <AlertCircle size={20} className="text-warning" />;
      default: return <Bell size={20} className="text-primary" />;
    }
  };

  return (
    <div className="alerts-container">
      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Alert Management</h1>
          <p className="text-muted">Real-time system events and quality notifications</p>
        </div>
        <div className="header-actions">
          <button className="btn-secondary"><BellOff size={18} /> Mute All</button>
          <button className="btn-primary"><CheckCircle size={18} /> Acknowledge All</button>
        </div>
      </header>

      <div className="alerts-layout">
        <div className="alerts-main">
          <div className="alerts-list glass-panel">
            <div className="list-header">
              <h2>Recent Events</h2>
              <div className="filter-tabs">
                <button className="active">All</button>
                <button>Critical</button>
                <button>Warnings</button>
              </div>
            </div>
            <div className="alert-items">
              {alerts.map((alert) => (
                <div key={alert.id} className={`alert-item ${alert.status} ${alert.severity}`}>
                  <div className="alert-severity-icon">
                    {getSeverityIcon(alert.severity)}
                  </div>
                  <div className="alert-content">
                    <div className="alert-top">
                      <span className="alert-source">{alert.source}</span>
                      <span className="alert-time"><Clock size={12} /> {alert.time}</span>
                    </div>
                    <div className="alert-message">{alert.message}</div>
                  </div>
                  <div className="alert-actions">
                    {alert.status === 'active' ? (
                      <button className="ack-btn">Acknowledge</button>
                    ) : (
                      <span className="ack-badge"><CheckCircle size={14} /> Resolved</span>
                    )}
                    <button className="delete-btn"><Trash2 size={16} /></button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="alerts-sidebar">
          <section className="config-section glass-panel">
            <div className="section-header">
              <Settings size={20} className="text-primary" />
              <h2>Alert Configuration</h2>
            </div>
            <div className="config-item">
              <div className="config-info">
                <span className="label">Consecutive NG Limit</span>
                <span className="sub">Trigger alert after X NG items</span>
              </div>
              <input type="number" defaultValue={3} className="config-input" />
            </div>
            <div className="config-item">
              <div className="config-info">
                <span className="label">Email Notifications</span>
                <span className="sub">Send report after each shift</span>
              </div>
              <div className="toggle-switch active"></div>
            </div>
            <div className="config-item">
              <div className="config-info">
                <span className="label">Telegram Bot</span>
                <span className="sub">Real-time alerts for critical events</span>
              </div>
              <div className="toggle-switch"></div>
            </div>
          </section>

          <section className="help-section glass-panel">
            <div className="section-header">
              <MessageSquare size={20} className="text-primary" />
              <h2>Alert Severity Guide</h2>
            </div>
            <div className="severity-guide">
              <div className="guide-item">
                <div className="dot critical"></div>
                <span><strong>Critical:</strong> Immediate line stop or hardware failure.</span>
              </div>
              <div className="guide-item">
                <div className="dot warning"></div>
                <span><strong>Warning:</strong> Quality degradation or resource limits.</span>
              </div>
              <div className="guide-item">
                <div className="dot info"></div>
                <span><strong>Info:</strong> System logs and routine events.</span>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
