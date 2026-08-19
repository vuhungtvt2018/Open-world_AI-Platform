import { useState, useMemo } from 'react';
import {
  LayoutDashboard,
  Activity,
  Camera,
  PlaySquare,
  Database,
  Cpu,
  BarChart3,
  FileText,
  Bell,
  Settings,
  PanelLeftClose,
  PanelLeftOpen,
  Sun,
  Moon,
  Lock,
  Monitor,
  Search
} from 'lucide-react';
import './Sidebar.css';

const navItems = [
  { icon: LayoutDashboard, label: 'Overview', layer: 'L1' },
  { icon: Activity, label: 'Production', layer: 'L2' },
  { icon: Camera, label: 'Vision Inspection', layer: 'L3' },
  { icon: PlaySquare, label: 'Live Stream', layer: 'L3' },
  { icon: Database, label: 'Dataset', layer: 'L4' },
  { icon: Cpu, label: 'Model Operation', layer: 'L4' },
  { icon: BarChart3, label: 'Analytics', layer: 'L5' },
  { icon: Search, label: 'Defect Search', layer: 'L5' },
  { icon: FileText, label: 'Reports', layer: 'L5' },
  { icon: Bell, label: 'Alerts', layer: 'L6' },
  { icon: Settings, label: 'Administration', layer: 'L6' },
];

interface SidebarProps {
  activeTab: string;
  onTabChange: (tabId: string) => void;
  userRole: 'USER' | 'ENGINEER' | 'ADMIN';
  onRoleChange: (role: 'USER' | 'ENGINEER' | 'ADMIN') => void;
}

const ROLE_PERMISSIONS: Record<string, string[]> = {
  USER: ['overview', 'vision-inspection', 'live-stream', 'alerts', 'production'],
  ENGINEER: ['overview', 'vision-inspection', 'live-stream', 'alerts', 'production', 'dataset', 'model-operation', 'analytics', 'reports', 'defect-search'],
  ADMIN: ['overview', 'vision-inspection', 'live-stream', 'alerts', 'production', 'dataset', 'model-operation', 'analytics', 'reports', 'administration', 'defect-search'],
};

export default function Sidebar({ activeTab, onTabChange, userRole, onRoleChange }: SidebarProps) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [theme, setTheme] = useState('light');

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

  const visibleNavItems = navItems; // Always show all items, but disable restricted ones


  return (
    <aside className={`sidebar glass-panel ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="sidebar-header">
        {!isCollapsed && (
          <div className="logo-section">
            <div className="logo-text">
              <h2 className="text-primary">VISUAL INSPECTION</h2>
            </div>
          </div>
        )}
        <button 
          className={`collapse-toggle ${isCollapsed ? 'is-collapsed' : ''}`} 
          onClick={() => setIsCollapsed(!isCollapsed)}
          title={isCollapsed ? "Expand Sidebar" : "Collapse Sidebar"}
        >
          {isCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
        </button>
      </div>

      <nav className="sidebar-nav">
        {visibleNavItems.map((item, index) => {
          const tabId = item.label.toLowerCase().replace(' ', '-');
          const hasAccess = ROLE_PERMISSIONS[userRole].includes(tabId);
          
          return (
            <div
              key={index}
              className={`nav-item ${activeTab === tabId ? 'active' : ''} ${!hasAccess ? 'disabled' : ''}`}
              onClick={() => {
                if (hasAccess) onTabChange(tabId);
              }}
              title={!hasAccess ? "Restricted Access" : (isCollapsed ? item.label : '')}
            >
              <item.icon size={20} className="nav-icon" />
              {!isCollapsed && <span className="nav-label">{item.label}</span>}
              {!hasAccess && !isCollapsed && (
                <Lock size={14} className="nav-lock-icon" style={{ marginLeft: 'auto', opacity: 0.5 }} />
              )}
            </div>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <button 
          className="theme-toggle edge-mode-btn mb-2 glow-primary-hover" 
          onClick={() => window.open('/edge', '_blank')} 
          title="Open Edge HMI Terminal"
          style={{ borderColor: 'var(--primary)', color: 'var(--primary)' }}
        >
          <Monitor size={20} />
          {!isCollapsed && <span className="nav-label font-bold">Launch Edge HMI</span>}
        </button>

        <button className="theme-toggle" onClick={toggleTheme} title="Switch Theme">
          {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
          {!isCollapsed && <span className="nav-label">{theme === 'light' ? 'Dark Mode' : 'Light Mode'}</span>}
        </button>

        <div className="system-status">
          <div className="status-indicator status-online"></div>
          {!isCollapsed && (
            <div className="status-info">
              <span className="text-xs text-muted font-bold">Edge Node Online</span>
              <span 
                className="text-[10px] text-muted block cursor-pointer hover:text-primary transition-colors"
                onClick={() => {
                  const roles: ('USER' | 'ENGINEER' | 'ADMIN')[] = ['USER', 'ENGINEER', 'ADMIN'];
                  const currentIndex = roles.indexOf(userRole);
                  onRoleChange(roles[(currentIndex + 1) % roles.length]);
                }}
                title="Click to quickly switch roles for testing"
              >
                Role: {userRole}
              </span>
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
