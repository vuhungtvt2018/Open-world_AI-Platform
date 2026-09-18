import React from 'react';
import { ShieldCheck, User, Wrench, Settings, Database, Activity, Lock } from 'lucide-react';
import './Administration.css';

interface AdministrationProps {
  userRole: 'USER' | 'ENGINEER' | 'ADMIN';
  onRoleChange: (role: 'USER' | 'ENGINEER' | 'ADMIN') => void;
}

export default function Administration({ userRole, onRoleChange }: AdministrationProps) {
  const isAdmin = userRole === 'ADMIN';

  return (
    <div className="admin-container">
      <header className="page-header">
        <h1 className="text-primary">Administration</h1>
        <p className="text-muted">System configuration and user access management</p>
      </header>

      <div className="admin-grid">
        {/* Role Management Section */}
        <section className="admin-section glass-panel">
          <div className="section-header">
            <ShieldCheck className="text-primary" />
            <h2>Access Control & Role Switching</h2>
          </div>
          <p className="text-sm text-muted mb-6">
            Switch between different user perspectives to test permissions and interface layouts.
          </p>
          
          <div className="role-management-cards">
            <div 
              className={`role-card ${userRole === 'USER' ? 'active' : ''}`}
              onClick={() => onRoleChange('USER')}
            >
              <div className="role-icon-wrapper">
                <User />
              </div>
              <div className="role-info">
                <h3>Operator (User)</h3>
                <p>View-only access to monitoring and inspection feeds.</p>
              </div>
              <div className="role-status">Active</div>
            </div>

            <div 
              className={`role-card ${userRole === 'ENGINEER' ? 'active' : ''}`}
              onClick={() => onRoleChange('ENGINEER')}
            >
              <div className="role-icon-wrapper">
                <Wrench />
              </div>
              <div className="role-info">
                <h3>Process Engineer</h3>
                <p>Access to production parameters, datasets, and analytics.</p>
              </div>
              <div className="role-status">Active</div>
            </div>

            <div 
              className={`role-card ${userRole === 'ADMIN' ? 'active' : ''}`}
              onClick={() => onRoleChange('ADMIN')}
            >
              <div className="role-icon-wrapper">
                <ShieldCheck />
              </div>
              <div className="role-info">
                <h3>System Administrator</h3>
                <p>Full ROOT access to all system configurations and node settings.</p>
              </div>
              <div className="role-status">Active</div>
            </div>
          </div>
        </section>

        {/* System Configuration (Restricted) */}
        <section className={`admin-section glass-panel ${!isAdmin ? 'restricted' : ''}`}>
          {!isAdmin && (
            <div className="restriction-overlay">
              <Lock size={32} />
              <p>Admin Access Required</p>
            </div>
          )}
          <div className="section-header">
            <Settings className="text-primary" />
            <h2>Edge Node Configuration</h2>
          </div>
          <div className="settings-list">
            <div className="setting-item">
              <div className="setting-info">
                <span className="setting-label">Inference Threshold</span>
                <span className="setting-desc">Minimum confidence score for NG detection</span>
              </div>
              <input type="range" disabled={!isAdmin} defaultValue={85} />
            </div>
            <div className="setting-item">
              <div className="setting-info">
                <span className="setting-label">Hardware Acceleration</span>
                <span className="setting-desc">Toggle TensorRT / OpenVINO optimization</span>
              </div>
              <div className="toggle-switch disabled"></div>
            </div>
          </div>
        </section>

        {/* Database & Maintenance (Restricted) */}
        <section className={`admin-section glass-panel ${!isAdmin ? 'restricted' : ''}`}>
          {!isAdmin && (
            <div className="restriction-overlay">
              <Lock size={32} />
              <p>Admin Access Required</p>
            </div>
          )}
          <div className="section-header">
            <Database className="text-primary" />
            <h2>System Maintenance</h2>
          </div>
          <div className="action-buttons">
            <button className="btn-secondary" disabled={!isAdmin}>Backup Database</button>
            <button className="btn-secondary" disabled={!isAdmin}>Clear Cache</button>
            <button className="btn-danger" disabled={!isAdmin}>Purge History</button>
          </div>
        </section>
      </div>
    </div>
  );
}
