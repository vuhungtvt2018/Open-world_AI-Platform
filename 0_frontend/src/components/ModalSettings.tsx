import React from 'react';
import { X, Settings, Server, Cpu, Bell, ShieldCheck } from 'lucide-react';
import './ModalAddCamera.css';

interface ModalSettingsProps {
  isOpen: boolean;
  onClose: () => void;
}

export const ModalSettings: React.FC<ModalSettingsProps> = ({ isOpen, onClose }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop">
      <div className="modal-container settings-modal">
        <div className="modal-header">
          <div className="modal-title">
            <Settings size={18} className="text-cyan" />
            <span>Thiết lập Cấu hình Hệ thống Edgify Vision AI</span>
          </div>
          <button className="modal-close-btn" onClick={onClose}>
            <X size={18} />
          </button>
        </div>

        <div className="modal-body">
          <div className="form-group">
            <label>Máy chủ Edge Inference (AI Node IP):</label>
            <input type="text" defaultValue="127.0.0.1:8080 (Local Jetson Orin AGX)" className="modal-input font-mono" />
          </div>

          <div className="form-group">
            <label>RTSP Relay Buffer Size (Frames):</label>
            <input type="number" defaultValue={60} className="modal-input font-mono" />
          </div>

          <div className="form-group">
            <label>Email / Webhook nhận Cảnh báo Lỗi NG liên tục:</label>
            <input type="text" defaultValue="qc-alerts@factory.internal" className="modal-input" />
          </div>

          <div className="modal-footer">
            <button className="modal-btn submit" onClick={onClose}>
              Lưu Cấu Hình
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
