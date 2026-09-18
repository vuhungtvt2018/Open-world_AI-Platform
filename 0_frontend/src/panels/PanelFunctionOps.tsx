import React, { useState } from 'react';
import { 
  Sliders, 
  Database, 
  Factory, 
  CheckCircle2, 
  UploadCloud, 
  Search, 
  SlidersHorizontal,
  Plus,
  Play,
  FileSpreadsheet
} from 'lucide-react';
import { AIModel, DefectCategory } from '../types/vision';
import { MOCK_MODELS, MOCK_DEFECT_CATEGORIES } from '../mock/visionData';
import './PanelFunctionOps.css';

interface PanelFunctionOpsProps {
  confThreshold: number;
  onConfChange: (val: number) => void;
}

export const PanelFunctionOps: React.FC<PanelFunctionOpsProps> = ({
  confThreshold,
  onConfChange
}) => {
  const [activeSubTab, setActiveSubTab] = useState<'model' | 'data' | 'production'>('model');
  const [selectedModel, setSelectedModel] = useState<string>('MOD-01');
  const [iouThreshold, setIouThreshold] = useState<number>(0.45);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [defectList, setDefectList] = useState<DefectCategory[]>(MOCK_DEFECT_CATEGORIES);
  const [selectedLine, setSelectedLine] = useState<string>('Line-01');

  return (
    <div className="panel-container panel-function-ops">
      {/* Top Menu Tabs matching sketch */}
      <div className="panel-ops-header">
        <button
          className={`ops-tab-btn ${activeSubTab === 'model' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('model')}
        >
          <Sliders size={14} />
          <span>Model Operation</span>
        </button>
        <button
          className={`ops-tab-btn ${activeSubTab === 'data' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('data')}
        >
          <Database size={14} />
          <span>Data</span>
        </button>
        <button
          className={`ops-tab-btn ${activeSubTab === 'production' ? 'active' : ''}`}
          onClick={() => setActiveSubTab('production')}
        >
          <Factory size={14} />
          <span>Production</span>
        </button>
      </div>

      {/* Panel Content Area */}
      <div className="panel-ops-content">
        {/* SUBTAB 1: MODEL OPERATION */}
        {activeSubTab === 'model' && (
          <div className="subtab-content model-subtab">
            <div className="ops-section-title">Mô hình AI & Tham số suy luận</div>

            {/* Model Select Cards */}
            <div className="model-cards-list">
              {MOCK_MODELS.map((model: AIModel) => (
                <div
                  key={model.id}
                  className={`model-card ${selectedModel === model.id ? 'selected' : ''}`}
                  onClick={() => setSelectedModel(model.id)}
                >
                  <div className="model-card-header">
                    <span className="model-name">{model.name}</span>
                    <span className={`model-status-badge ${model.status}`}>
                      {model.status === 'active' ? 'ĐANG CHẠY' : 'SẴN SÀNG'}
                    </span>
                  </div>
                  <div className="model-card-meta">
                    <span>Phiên bản: {model.version}</span>
                    <span className="text-cyan">Độ chính xác: {model.accuracy}%</span>
                  </div>
                  <div className="model-card-metrics">
                    <span>Tốc độ: <strong>{model.fps} FPS</strong></span>
                    <span>VRAM: <strong>{model.vramUsed}</strong></span>
                  </div>
                </div>
              ))}
            </div>

            {/* Slider Controls */}
            <div className="ops-controls-box">
              <div className="control-group">
                <div className="control-header">
                  <span className="control-label">N ngưỡng Tin cậy (Confidence Threshold):</span>
                  <span className="control-val">{Math.round(confThreshold * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.99"
                  step="0.01"
                  value={confThreshold}
                  onChange={(e) => onConfChange(parseFloat(e.target.value))}
                  className="ops-slider"
                />
              </div>

              <div className="control-group">
                <div className="control-header">
                  <span className="control-label">Ngưỡng Trùng lấp (IoU Threshold):</span>
                  <span className="control-val">{Math.round(iouThreshold * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="0.9"
                  step="0.05"
                  value={iouThreshold}
                  onChange={(e) => setIouThreshold(parseFloat(e.target.value))}
                  className="ops-slider"
                />
              </div>
            </div>

            <div className="ops-action-buttons">
              <button className="ops-primary-btn">
                <Play size={14} /> Deploy Model lên Edge
              </button>
              <button className="ops-secondary-btn">
                <SlidersHorizontal size={14} /> Calibrate Camera
              </button>
            </div>
          </div>
        )}

        {/* SUBTAB 2: DATA MANAGEMENT */}
        {activeSubTab === 'data' && (
          <div className="subtab-content data-subtab">
            <div className="ops-section-title">Quản lý Dữ liệu Lỗi (Defect Classes)</div>

            <div className="ops-search-bar">
              <Search size={14} className="search-icon" />
              <input
                type="text"
                placeholder="Tìm kiếm loại lỗi..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>

            <div className="defect-classes-grid">
              {defectList
                .filter((d) => d.name.toLowerCase().includes(searchQuery.toLowerCase()))
                .map((defect) => (
                  <div key={defect.id} className="defect-item-card">
                    <div className="defect-color-indicator" style={{ backgroundColor: defect.color }}></div>
                    <div className="defect-info">
                      <span className="defect-name">{defect.name}</span>
                      <span className="defect-code">{defect.code}</span>
                    </div>
                    <span className="defect-count">{defect.count} ảnh</span>
                  </div>
                ))}
            </div>

            <div className="ops-action-buttons">
              <button className="ops-primary-btn">
                <UploadCloud size={14} /> Tải lên Dataset Mới
              </button>
              <button className="ops-secondary-btn">
                <FileSpreadsheet size={14} /> Xuất File YOLO/Pascal
              </button>
            </div>
          </div>
        )}

        {/* SUBTAB 3: PRODUCTION OPERATION */}
        {activeSubTab === 'production' && (
          <div className="subtab-content production-subtab">
            <div className="ops-section-title">Thiết lập Dây chuyền & Ca sản xuất</div>

            <div className="form-group">
              <label>Dây chuyền Sản xuất:</label>
              <select
                value={selectedLine}
                onChange={(e) => setSelectedLine(e.target.value)}
                className="ops-select"
              >
                <option value="Line-01">Dây chuyền Đúc Vỏ A1 (Chính)</option>
                <option value="Line-02">Dây chuyền Lắp ráp B2</option>
                <option value="Line-03">Dây chuyền Đóng gói C1</option>
                <option value="Line-04">Dây chuyền CNC Precision D1</option>
              </select>
            </div>

            <div className="production-metrics-box">
              <div className="prod-metric">
                <span className="label">Mục tiêu ca (Target):</span>
                <span className="val text-cyan">50,000 SP</span>
              </div>
              <div className="prod-metric">
                <span className="label">Tốc độ băng tải:</span>
                <span className="val text-green">1.2 m/s</span>
              </div>
              <div className="prod-metric">
                <span className="label">Trạng thái dây chuyền:</span>
                <span className="val badge-running">
                  <CheckCircle2 size={12} /> ĐANG HOẠT ĐỘNG
                </span>
              </div>
            </div>

            <div className="ops-controls-box">
              <label>Ghi chú ca sản xuất / Batch ID:</label>
              <input
                type="text"
                defaultValue="BATCH-2026-09-13-A"
                className="ops-input"
              />
            </div>

            <div className="ops-action-buttons">
              <button className="ops-primary-btn">
                <Plus size={14} /> Tạo Batch Mới
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
