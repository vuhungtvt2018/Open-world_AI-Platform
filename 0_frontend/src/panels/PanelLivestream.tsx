import React, { useState } from 'react';
import { Camera as CameraIcon, Plus, Filter, Video, CheckCircle2, AlertTriangle, Power, Trash2 } from 'lucide-react';
import { Camera } from '../types/vision';
import './PanelLivestream.css';

interface PanelLivestreamProps {
  cameras: Camera[];
  selectedCameraId: string;
  onSelectCamera: (camId: string) => void;
  onOpenAddCamera: () => void;
  onDeleteCamera: (cameraId: string) => void;
}

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

const normalizePreviewUrl = (url: string | undefined, cameraId: string) => {
  const fallback = `${API_BASE_URL}/video-feed?cam=${encodeURIComponent(cameraId)}&fps=15`;
  if (!url) return fallback;
  if (/^(https?:|data:|blob:)/i.test(url)) return url;
  if (url.startsWith('/')) return `${API_BASE_URL}${url}`;
  return `${API_BASE_URL}/${url}`;
};

export const PanelLivestream: React.FC<PanelLivestreamProps> = ({
  cameras,
  selectedCameraId,
  onSelectCamera,
  onOpenAddCamera,
  onDeleteCamera
}) => {
  const [filterStatus, setFilterStatus] = useState<'all' | 'active' | 'warning' | 'inactive'>('all');

  const filteredCameras = cameras.filter((cam) => {
    if (filterStatus === 'all') return true;
    return cam.status === filterStatus;
  });

  return (
    <div className="panel-container panel-livestream">
      {/* Header matching sketch */}
      <div className="livestream-header">
        <div className="livestream-header-left">
          <span className="livestream-tag">Livestream</span>
          <span className="cam-count-badge font-mono">{cameras.length} CAMERAS</span>
        </div>

        <div className="livestream-header-right">
          {/* Status Filter */}
          <div className="filter-group">
            <button
              className={`filter-btn ${filterStatus === 'all' ? 'active' : ''}`}
              onClick={() => setFilterStatus('all')}
            >
              Tất cả
            </button>
            <button
              className={`filter-btn ${filterStatus === 'active' ? 'active' : ''}`}
              onClick={() => setFilterStatus('active')}
            >
              Hoạt động
            </button>
            <button
              className={`filter-btn ${filterStatus === 'warning' ? 'active' : ''}`}
              onClick={() => setFilterStatus('warning')}
            >
              Đang kết nối
            </button>
          </div>

          {/* Add Camera Button */}
          <button className="add-cam-btn" onClick={onOpenAddCamera}>
            <Plus size={14} />
            <span>Thêm Camera</span>
          </button>
        </div>
      </div>

      {/* Grid of registered cameras matching sketch */}
      <div className="livestream-body">
        <div className="cameras-grid">
          {filteredCameras.map((cam) => {
            const isSelected = selectedCameraId === cam.id;
            const previewUrl = normalizePreviewUrl(
              cam.status === 'active' ? cam.liveUrl : cam.previewImage,
              cam.id,
            );

            return (
              <div
                key={cam.id}
                className={`camera-grid-card ${isSelected ? 'selected' : ''} ${cam.status}`}
                onClick={() => onSelectCamera(cam.id)}
              >
                <div className="cam-card-media">
                  <img src={previewUrl} alt={cam.name} className="cam-card-img" />
                  
                  {/* Status Overlay Badge */}
                  <div className="cam-card-badge-tl">
                    {cam.status === 'active' ? (
                      <span className="live-status-pill green">
                        <span className="dot pulse"></span> TRỰC TUYẾN
                      </span>
                    ) : cam.status === 'warning' ? (
                      <span className="live-status-pill warning">
                        <AlertTriangle size={10} /> ĐANG KẾT NỐI
                      </span>
                    ) : (
                      <span className="live-status-pill inactive">
                        <Power size={10} /> ĐÃ NGẮT KẾT NỐI
                      </span>
                    )}
                  </div>

                  <div className="cam-card-badge-tr font-mono">
                    {cam.id}
                  </div>
                  <button
                    type="button"
                    className="cam-card-delete-btn"
                    title={`Delete ${cam.name}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      onDeleteCamera(cam.id);
                    }}
                  >
                    <Trash2 size={13} />
                  </button>

                  {/* Bottom overlay text */}
                  <div className="cam-card-bottom-bar">
                    <span className="cam-card-name">{cam.name}</span>
                    <span className="cam-card-line">{cam.location}</span>
                  </div>
                </div>

                <div className="cam-card-meta-bar">
                  <span className="mode-badge">{cam.mode}</span>
                  <span className="fps-val font-mono">{cam.currentFps} FPS</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
