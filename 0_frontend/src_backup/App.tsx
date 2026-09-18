import { useState, type ReactNode } from 'react';
import { Header } from './components/Header';
import { Footer } from './components/Footer';
import DashboardSystemInfo from './pages/DashboardSystemInfo';
import VisionInspection from './pages/VisionInspection';
import DefectSearch from './pages/DefectSearch';
import Reports from './pages/Reports';
import EdgeDashboard from './pages/EdgeDashboard';
import type { MainViewMode, SystemMode } from './types/vision';
import './App.css';

const TabPanel = ({
  id,
  activeTab,
  children,
}: {
  id: string;
  activeTab: string;
  children: ReactNode;
}) => {
  if (activeTab !== id) return null;

  return (
    <div className="tab-panel">
      {children}
    </div>
  );
};

function App() {
  const [activeViewMode, setActiveViewMode] = useState<MainViewMode>('DASHBOARD');
  const [mode, setMode] = useState<SystemMode>('INSPECTION');
  const isEdgeMode = window.location.pathname === '/edge';

  if (isEdgeMode) {
    return <EdgeDashboard onExit={() => { window.location.href = '/'; }} />;
  }

  return (
    <div className="workspace-app">
      <Header
        activeViewMode={activeViewMode}
        onViewModeChange={setActiveViewMode}
        mode={mode}
        onModeChange={setMode}
      />

      <TabPanel id="DASHBOARD" activeTab={activeViewMode}>
        <div className="view-page-container">
          <DashboardSystemInfo mode={mode} />
        </div>
      </TabPanel>

      <TabPanel id="CAMERA_OPS" activeTab={activeViewMode}>
        <div className="view-page-container view-page-fill">
          <VisionInspection />
        </div>
      </TabPanel>

      <TabPanel id="SEARCH" activeTab={activeViewMode}>
        <div className="view-page-container">
          <DefectSearch />
        </div>
      </TabPanel>

      <TabPanel id="REPORT" activeTab={activeViewMode}>
        <div className="view-page-container">
          <Reports />
        </div>
      </TabPanel>

      <Footer />
    </div>
  );
}

export default App;
