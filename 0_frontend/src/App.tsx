import { useState, useEffect, useRef } from 'react';
import { ChevronUp } from 'lucide-react';
import Sidebar from './components/Sidebar';
import GlobalMonitoring from './pages/GlobalMonitoring';
import ProductionOperation from './pages/ProductionOperation';
import VisionInspection from './pages/VisionInspection';
import ModelOperation from './pages/ModelOperation';
import Administration from './pages/Administration';
import DatasetManagement from './pages/DatasetManagement';
import Analytics from './pages/Analytics';
import Reports from './pages/Reports';
import Alerts from './pages/Alerts';
import LiveStream from './pages/LiveStream';
import EdgeDashboard from './pages/EdgeDashboard';
import DefectSearch from './pages/DefectSearch';
import './App.css';

const TabPanel = ({ id, activeTab, children }: { id: string, activeTab: string, children: React.ReactNode }) => (
  <div 
    className="tab-panel" 
    style={{ 
      display: activeTab === id ? 'block' : 'none',
      height: '100%' 
    }}
  >
    {children}
  </div>
);

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [userRole, setUserRole] = useState<'USER' | 'ENGINEER' | 'ADMIN'>('ADMIN');
  const [showScrollTop, setShowScrollTop] = useState(false);
  const [isEdgeMode, setIsEdgeMode] = useState(window.location.pathname === '/edge');
  const mainContentRef = useRef<HTMLDivElement>(null);

  // If active tab becomes hidden for the current role, switch to overview
  useEffect(() => {
    const ROLE_PERMISSIONS: Record<string, string[]> = {
      USER: ['overview', 'vision-inspection', 'live-stream', 'alerts', 'production'],
      ENGINEER: ['overview', 'vision-inspection', 'live-stream', 'alerts', 'production', 'dataset', 'model-operation', 'analytics', 'reports', 'defect-search'],
      ADMIN: ['overview', 'vision-inspection', 'live-stream', 'alerts', 'production', 'dataset', 'model-operation', 'analytics', 'reports', 'administration', 'defect-search'],
    };

    if (!ROLE_PERMISSIONS[userRole].includes(activeTab)) {
      setActiveTab('overview');
    }
  }, [userRole, activeTab]);

  useEffect(() => {
    const handleScroll = () => {
      if (mainContentRef.current) {
        setShowScrollTop(mainContentRef.current.scrollTop > 300);
      }
    };

    const mainContent = mainContentRef.current;
    if (mainContent) {
      mainContent.addEventListener('scroll', handleScroll);
    }
    return () => {
      if (mainContent) {
        mainContent.removeEventListener('scroll', handleScroll);
      }
    };
  }, []);

  const scrollToTop = () => {
    if (mainContentRef.current) {
      mainContentRef.current.scrollTo({
        top: 0,
        behavior: 'smooth'
      });
    }
  };

  if (isEdgeMode) {
    return <EdgeDashboard onExit={() => window.location.href = '/'} />;
  }

  return (
    <div className="app-container">
      <Sidebar 
        activeTab={activeTab} 
        onTabChange={setActiveTab} 
        userRole={userRole}
        onRoleChange={setUserRole}
      />
      <main className="main-content" ref={mainContentRef}>
        <TabPanel id="overview" activeTab={activeTab}>
          <GlobalMonitoring />
        </TabPanel>

        <TabPanel id="production" activeTab={activeTab}>
          <ProductionOperation />
        </TabPanel>
        
        <TabPanel id="vision-inspection" activeTab={activeTab}>
          <VisionInspection />
        </TabPanel>

        <TabPanel id="dataset" activeTab={activeTab}>
          <DatasetManagement />
        </TabPanel>

        <TabPanel id="model-operation" activeTab={activeTab}>
          <ModelOperation />
        </TabPanel>

        <TabPanel id="analytics" activeTab={activeTab}>
          <Analytics />
        </TabPanel>

        <TabPanel id="defect-search" activeTab={activeTab}>
          <DefectSearch />
        </TabPanel>

        <TabPanel id="reports" activeTab={activeTab}>
          <Reports />
        </TabPanel>

        <TabPanel id="alerts" activeTab={activeTab}>
          <Alerts />
        </TabPanel>

        <TabPanel id="administration" activeTab={activeTab}>
          <Administration userRole={userRole} onRoleChange={setUserRole} />
        </TabPanel>

        <TabPanel id="live-stream" activeTab={activeTab}>
          <LiveStream />
        </TabPanel>
      </main>

      <button 
        className={`back-to-top ${showScrollTop ? 'visible' : ''}`}
        onClick={scrollToTop}
        aria-label="Back to top"
      >
        <ChevronUp size={24} />
      </button>
    </div>
  );
}

export default App;
