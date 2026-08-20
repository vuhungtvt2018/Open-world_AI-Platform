import React, { useState, useEffect } from 'react';
import { Play, Search, AlertCircle, CheckCircle, Image as ImageIcon, Folder, Clock, Target, ShieldAlert, BarChart3 as BarChart } from 'lucide-react';
import './VisionInspection.css';

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

export default function VisionInspection() {
  const [isInspecting, setIsInspecting] = useState(false);
  const [latestResult, setLatestResult] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [inputMode, setInputMode] = useState<'folder' | 'stream' | 'basler'>('folder');
  const [availableImages, setAvailableImages] = useState<string[]>([]);
  const [selectedImage, setSelectedImage] = useState<string>('');

  const fetchHistory = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/history`);
      const data = await response.json();
      setHistory(data.results.reverse().slice(0, 5));
    } catch (error) {
      console.error('Failed to fetch history:', error);
    }
  };

  const fetchImages = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/images`);
      const data = await response.json();
      setAvailableImages(data.images);
      if (data.images.length > 0 && !selectedImage) {
        setSelectedImage(data.images[0]);
      }
    } catch (error) {
      console.error('Failed to fetch images:', error);
    }
  };

  useEffect(() => {
    fetchHistory();
    fetchImages();
  }, []);

  const handleModeChange = async (mode: 'folder' | 'stream' | 'basler') => {
    try {
      const targetCam = mode === 'basler' ? '1' : 'default';
      const response = await fetch(`${API_BASE_URL}/set-mode/${mode}?cam=${targetCam}`, { method: 'POST' });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed');
      }
      setInputMode(mode);
    } catch (error: any) {
      alert(`Failed to change input mode: ${error.message}`);
    }
  };

  const handleImageSelect = async (filename: string) => {
    try {
      await fetch(`${API_BASE_URL}/set-image/${filename}`, { method: 'POST' });
      setSelectedImage(filename);
      setLatestResult(null); // Clear previous result when changing image
    } catch (error) {
      alert('Failed to select image');
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setAvailableImages(data.images);
      setSelectedImage(data.filename);
      setLatestResult(null);
    } catch (error) {
      alert('Upload failed');
    }
  };

  const handleInspect = async () => {
    setIsInspecting(true);
    try {
      const targetCam = inputMode === 'basler' ? '1' : 'default';
      const response = await fetch(`${API_BASE_URL}/inspect?cam=${targetCam}`, { method: 'POST' });
      const result = await response.json();
      setLatestResult(result);
      fetchHistory();
    } catch (error) {
      alert('Inference failed. Make sure the backend is running.');
    } finally {
      setIsInspecting(false);
    }
  };

  return (
    <div className="inspection-container">
      <header className="dashboard-header">
        <div>
          <h1 className="text-primary">Vision Inspection</h1>
          <div className="mode-selector mt-2">
            <button 
              className={`mode-btn ${inputMode === 'folder' ? 'active' : ''}`}
              onClick={() => handleModeChange('folder')}
            >
              <ImageIcon size={14} /> Folder Mode
            </button>
            <button 
              className={`mode-btn ${inputMode === 'basler' ? 'active' : ''}`}
              onClick={() => handleModeChange('basler')}
            >
              <Play size={14} /> Basler Camera
            </button>
          </div>
        </div>
        <button 
          className={`inspect-btn ${isInspecting ? 'loading' : ''}`}
          onClick={handleInspect}
          disabled={isInspecting}
        >
          {isInspecting ? 'Analyzing...' : <><Play size={18} fill="currentColor" /> Run Inspection</>}
        </button>
      </header>

      <div className="inspection-grid">
        <div className="main-viewer">
          <div className="quad-viewer">
            {/* 1. Input Source */}
            <div className="view-panel glass-panel">
              <div className="view-header">
                <span className="text-xs font-bold uppercase tracking-wider text-muted">1. Input Source</span>
              </div>
              <div className="image-display">
                {inputMode === 'stream' || inputMode === 'basler' ? (
                  <img src={`${API_BASE_URL}/video-feed?cam=${inputMode === 'basler' ? '1' : 'default'}&mode=${inputMode}`} alt="Live stream" className="result-image" />
                ) : selectedImage ? (
                  <img src={`${API_BASE_URL}/images/${selectedImage}`} alt="Selected input" className="result-image" />
                ) : (
                  <div className="placeholder-image">
                    <ImageIcon size={48} className="text-muted" />
                    <p className="text-muted mt-4">Select an image</p>
                  </div>
                )}
              </div>
            </div>
            
            {/* 2. Heatmaps */}
            <div className="view-panel glass-panel">
              <div className="view-header">
                <span className="text-xs font-bold uppercase tracking-wider text-muted">2. Object Heatmaps</span>
              </div>
              <div className="image-display">
                {latestResult?.vis_urls?.heatmap ? (
                  <img src={`${API_BASE_URL}${latestResult.vis_urls.heatmap}`} alt="Heatmaps" className="result-image" />
                ) : (
                  <div className="placeholder-image text-muted">
                    <Search size={48} />
                    <p className="mt-4">Analysis pending</p>
                  </div>
                )}
              </div>
            </div>

            {/* 3. Object Crops */}
            <div className="view-panel glass-panel">
              <div className="view-header">
                <span className="text-xs font-bold uppercase tracking-wider text-muted">3. Object Outputs (Crops)</span>
              </div>
              <div className="image-display">
                {latestResult?.vis_urls?.crops ? (
                  <img src={`${API_BASE_URL}${latestResult.vis_urls.crops}`} alt="Crops" className="result-image" />
                ) : (
                  <div className="placeholder-image text-muted">
                    <Search size={48} />
                    <p className="mt-4">Analysis pending</p>
                  </div>
                )}
              </div>
            </div>

            {/* 4. Overall Result */}
            <div className="view-panel glass-panel">
              <div className="view-header">
                <span className="text-xs font-bold uppercase tracking-wider text-muted">4. Overall Result</span>
                {latestResult && (
                  <span className={`result-tag ${latestResult.ng_detected ? 'ng' : 'ok'}`}>
                    {latestResult.ng_detected ? 'NG DETECTED' : 'QUALITY OK'}
                  </span>
                )}
              </div>
              <div className="image-display">
                {latestResult?.vis_urls?.overall ? (
                  <img src={`${API_BASE_URL}${latestResult.vis_urls.overall}`} alt="Overall result" className="result-image" />
                ) : (
                  <div className="placeholder-image text-muted">
                    <Search size={48} />
                    <p className="mt-4">Analysis pending</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="side-panels">
          {inputMode === 'folder' && (
            <div className="panel glass-panel">
              <div className="panel-header">
                <h3 className="panel-title">Source Images</h3>
                <label className="upload-btn-label">
                  <Folder size={14} /> Open Image
                  <input type="file" hidden onChange={handleUpload} accept="image/*" />
                </label>
              </div>
              <div className="image-list">
                {availableImages.map((img) => (
                  <div 
                    key={img} 
                    className={`image-item ${selectedImage === img ? 'active' : ''}`}
                    onClick={() => handleImageSelect(img)}
                  >
                    <div className="item-thumbnail-wrapper">
                      <img src={`${API_BASE_URL}/images/${img}`} alt={img} className="item-thumbnail" />
                    </div>
                    <span className="text-xs truncate">{img}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="panel glass-panel">
            <h3 className="panel-title">Inference Results</h3>
            {latestResult ? (
              <div className="result-details">
                <div className="metrics-grid-compact">
                  <div className="metric-item-small">
                    <Clock size={16} className="text-muted" />
                    <div className="metric-info-small">
                      <span className="metric-label">Latency</span>
                      <span className="metric-val">{latestResult.metrics?.latency_ms || 0} ms</span>
                    </div>
                  </div>
                  <div className="metric-item-small">
                    <Target size={16} className="text-muted" />
                    <div className="metric-info-small">
                      <span className="metric-label">Objects</span>
                      <span className="metric-val">{latestResult.metrics?.total_objects || 0}</span>
                    </div>
                  </div>
                  <div className="metric-item-small">
                    <ShieldAlert size={16} className={latestResult.metrics?.ng_count > 0 ? 'text-secondary' : 'text-muted'} />
                    <div className="metric-info-small">
                      <span className="metric-label">NG Count</span>
                      <span className={`metric-val ${latestResult.metrics?.ng_count > 0 ? 'text-secondary' : ''}`}>
                        {latestResult.metrics?.ng_count || 0}
                      </span>
                    </div>
                  </div>
                  <div className="metric-item-small">
                    <BarChart size={16} className="text-muted" />
                    <div className="metric-info-small">
                      <span className="metric-label">Max Score</span>
                      <span className="metric-val">{latestResult.metrics?.max_score || 0}</span>
                    </div>
                  </div>
                </div>
                <div className="divider-line"></div>
                <div className="result-item mt-2">
                  <span className="text-muted text-xs">Timestamp:</span>
                  <span className="font-mono text-xs">{latestResult.timestamp}</span>
                </div>
              </div>
            ) : (
              <p className="text-muted text-center py-8">No data available</p>
            )}
          </div>

          <div className="panel glass-panel">
            <h3 className="panel-title">Recent History</h3>
            <div className="history-list">
              {history.map((item, idx) => (
                <div key={idx} className="history-item">
                  <div className={`status-dot ${item.ng_detected ? 'bg-secondary' : 'bg-success'}`}></div>
                  <div className="history-info">
                    <span className="text-xs font-mono">{item.timestamp}</span>
                    <span className="text-sm">{item.ng_detected ? 'NG' : 'OK'}</span>
                  </div>
                  <Search size={14} className="text-muted cursor-pointer hover:text-primary" />
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
