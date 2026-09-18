import React from 'react';
import { 
  Settings, 
  Cpu, 
  Activity,
  Layers,
  Sparkles,
  Camera,
  Search,
  FileText,
  LayoutDashboard
} from 'lucide-react';
import { SystemMode, MainViewMode } from '../types/vision';
import './Header.css';

interface HeaderProps {
  activeViewMode: MainViewMode;
  onViewModeChange: (viewMode: MainViewMode) => void;
  mode: SystemMode;
  onModeChange: (newMode: SystemMode) => void;
  onOpenSettings: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  activeViewMode,
  onViewModeChange,
  mode,
  onModeChange,
  onOpenSettings
}) => {
  return (
    <header className="main-header">
      <div className="header-left">
        <div className="brand-logo">
          <div className="brand-icon">
            <Cpu size={22} className="cpu-icon" />
            <Sparkles size={12} className="sparkle-icon" />
          </div>
          <div className="brand-text-group">
            <span className="brand-title">Smart Counting & Inspection</span>
          </div>
        </div>

        {/* 4 Main View Choices */}
        <nav className="main-views-nav">
          <button 
            className={`view-tab-btn ${activeViewMode === 'DASHBOARD' ? 'active' : ''}`}
            onClick={() => onViewModeChange('DASHBOARD')}
            title="Dashboard & System Information"
          >
            <LayoutDashboard size={15} />
            <span>Dashboard & System Info</span>
          </button>

          <button 
            className={`view-tab-btn ${activeViewMode === 'CAMERA_OPS' ? 'active' : ''}`}
            onClick={() => onViewModeChange('CAMERA_OPS')}
            title="Camera Operation (Preview - Detail - Livestream)"
          >
            <Camera size={15} />
            <span>Camera Operation</span>
          </button>

          <button 
            className={`view-tab-btn ${activeViewMode === 'SEARCH' ? 'active' : ''}`}
            onClick={() => onViewModeChange('SEARCH')}
            title="Defect Search"
          >
            <Search size={15} />
            <span>Search</span>
          </button>

          <button 
            className={`view-tab-btn ${activeViewMode === 'REPORT' ? 'active' : ''}`}
            onClick={() => onViewModeChange('REPORT')}
            title="Reports"
          >
            <FileText size={15} />
            <span>Report</span>
          </button>
        </nav>
      </div>

      <div className="header-right">
        {/* Mode Selector */}
        <div className="mode-selector">
          <span className="mode-label">Mode:</span>
          <div className="mode-switch-group">
            <button
              className={`mode-btn ${mode === 'COUNTING' ? 'active counting' : ''}`}
              onClick={() => onModeChange('COUNTING')}
            >
              <Activity size={14} />
              <span>Counting</span>
            </button>
            <button
              className={`mode-btn ${mode === 'INSPECTION' ? 'active inspection' : ''}`}
              onClick={() => onModeChange('INSPECTION')}
            >
              <Layers size={14} />
              <span>Inspection</span>
            </button>
          </div>
        </div>

        {/* Settings button */}
        <button 
          className="header-icon-btn" 
          title="Thiết lập hệ thống"
          onClick={onOpenSettings}
        >
          <Settings size={18} />
        </button>
      </div>
    </header>
  );
};
