import React, { useEffect, useRef, useState } from 'react';
import GlobalMonitoring from './GlobalMonitoring';
import Analytics from './Analytics';
import ModelOperation from './ModelOperation';
import DatasetManagement from './DatasetManagement';
import ProductionOperation from './ProductionOperation';
import { SystemMode } from '../types/vision';
import './DashboardSystemInfo.css';

interface DashboardSystemInfoProps {
  mode?: SystemMode;
}

const SECTIONS = [
  { id: 'sec-1-monitoring', label: 'Global Monitoring' },
  { id: 'sec-2-analytics', label: 'Vision Analytics' },
  { id: 'sec-3-model', label: 'Model Operation' },
  { id: 'sec-4-dataset', label: 'Dataset Management' },
  { id: 'sec-5-production', label: 'Production Operation' },
];

export default function DashboardSystemInfo({ mode = 'INSPECTION' }: DashboardSystemInfoProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [activeSection, setActiveSection] = useState<string>('sec-1-monitoring');

  // Track active visible section on scroll
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const scrollPos = container.scrollTop + 140;
      for (let i = SECTIONS.length - 1; i >= 0; i--) {
        const el = document.getElementById(SECTIONS[i].id);
        if (el && el.offsetTop <= scrollPos) {
          setActiveSection(SECTIONS[i].id);
          break;
        }
      }
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    setActiveSection(id);
    const targetEl = document.getElementById(id);
    if (targetEl) {
      targetEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className="dashboard-system-info-page" ref={containerRef}>
      {/* Sticky Header with Quick Nav Pills */}
      <div className="dashboard-page-header">
        <div className="page-header-title-group">
          <h1>Dashboard & System Information</h1>
          <span className="mode-tag">{mode} Mode</span>
        </div>

        {/* Quick Section Navigation Pills */}
        <nav className="section-nav-pills">
          {SECTIONS.map((sec) => (
            <button
              key={sec.id}
              className={`nav-pill ${activeSection === sec.id ? 'active' : ''}`}
              onClick={() => scrollToSection(sec.id)}
            >
              {sec.label}
            </button>
          ))}
        </nav>
      </div>

      {/* 1. Global Monitoring */}
      <div id="sec-1-monitoring" className="dashboard-section-block">
        <GlobalMonitoring mode={mode} />
      </div>

      {/* 2. Vision Analytics */}
      <div id="sec-2-analytics" className="dashboard-section-block">
        <Analytics mode={mode} />
      </div>

      {/* 3. Model Operation */}
      <div id="sec-3-model" className="dashboard-section-block">
        <ModelOperation />
      </div>

      {/* 4. Dataset Management */}
      <div id="sec-4-dataset" className="dashboard-section-block">
        <DatasetManagement />
      </div>

      {/* 5. Production Operation */}
      <div id="sec-5-production" className="dashboard-section-block">
        <ProductionOperation mode={mode} />
      </div>
    </div>
  );
}
