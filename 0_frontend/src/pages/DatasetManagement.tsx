import React, { useState, useEffect } from 'react';
import { 
  Database, 
  Upload, 
  Tag, 
  Layers, 
  Image as ImageIcon, 
  CheckCircle, 
  Clock, 
  Plus,
  Filter,
  BrainCircuit,
  Eye
} from 'lucide-react';
import DetectionResultImage, {
  type DetectionObject,
} from '../components/DetectionResultImage';
import './DatasetManagement.css';

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

interface DatasetStat {
  label: string;
  value: string;
  color: string;
}

interface DatasetImage {
  id: number;
  name: string;
  status: string;
  type: 'Detection' | 'OK' | 'NG';
  task_type: 'detection' | 'inspection';
  total_objects: number;
  counts_by_class: Record<string, number>;
  date: string;
  product: string;
}

interface DatasetRecordDetail {
  id: number;
  task_type: 'detection' | 'inspection';
  image_url: string;
  image_width: number;
  image_height: number;
  total_objects: number;
  counts_by_class: Record<string, number>;
  objects: DetectionObject[];
}

interface DatasetDetectionResultProps {
  recordId: number;
}

// 12082026 - KIET - Tải object từ database và dựng ảnh annotation cho một record OD.
function DatasetDetectionResult({ recordId }: DatasetDetectionResultProps) {
  const [detail, setDetail] = useState<DatasetRecordDetail | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    let isMounted = true;

    fetch(`${API_BASE_URL}/api/dataset-records/${recordId}`, { cache: 'no-store' })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error('Detection result not found');
        }
        return response.json() as Promise<DatasetRecordDetail>;
      })
      .then((data) => {
        if (isMounted) setDetail(data);
      })
      .catch((fetchError: unknown) => {
        if (!isMounted) return;
        setError(
          fetchError instanceof Error
            ? fetchError.message
            : 'Failed to load detection result',
        );
      });

    return () => {
      isMounted = false;
    };
  }, [recordId]);

  if (error) {
    return <span className="dataset-result-message">{error}</span>;
  }

  if (!detail) {
    return <span className="dataset-result-message">Loading result...</span>;
  }

  if (!detail.image_width || !detail.image_height) {
    return <span className="dataset-result-message">Original image not found</span>;
  }

  return (
    <DetectionResultImage
      imageUrl={`${API_BASE_URL}${detail.image_url}`}
      imageWidth={detail.image_width}
      imageHeight={detail.image_height}
      objects={detail.objects}
    />
  );
}

// 12082026 - KIET - Hiển thị và lọc dữ liệu Inspection, Object Detection từ Edge database.
export default function DatasetManagement() {
  const [datasetStats, setDatasetStats] = useState<DatasetStat[]>([
    { label: 'Total Images', value: '...', color: '#3b82f6' },
    { label: 'Labeled', value: '...', color: '#10b981' },
    { label: 'Synthetic', value: '...', color: '#8b5cf6' },
    { label: 'Pending', value: '...', color: '#f59e0b' },
  ]);
  const [images, setImages] = useState<DatasetImage[]>([]);
  const [filter, setFilter] = useState<'All' | 'Detection' | 'OK' | 'NG'>('All');

  useEffect(() => {
    let isMounted = true;
    
    // 12082026 - KIET - Đồng bộ danh sách dataset mới nhất từ Edge API.
    const fetchDataset = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/dataset-stats`, { cache: 'no-store' });
        if (res.ok && isMounted) {
          const data = await res.json();
          setDatasetStats(data.stats || []);
          setImages(data.images || []);
        }
      } catch {
        console.error("Failed to fetch dataset stats");
      }
    };
    
    // Fetch immediately on mount
    fetchDataset();
    
    // Auto reload every 5 seconds
    const intervalId = setInterval(fetchDataset, 5000);

    return () => {
      isMounted = false;
      clearInterval(intervalId);
    };
  }, []);

  // 12082026 - KIET - Chọn icon tương ứng với từng thống kê dataset.
  const getIconForLabel = (label: string) => {
    if (label.includes('Total')) return <ImageIcon size={20} />;
    if (label.includes('Labeled')) return <Tag size={20} />;
    if (label.includes('Synthetic')) return <BrainCircuit size={20} />;
    return <Clock size={20} />;
  };

  const filteredImages = images.filter((image) => {
    if (filter === 'All') return true;
    return image.type === filter;
  });

  return (
    <div className="dataset-container">
      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Dataset Management</h1>
          <p className="text-muted">Curate training data, manage labels, and generate synthetic anomalies</p>
        </div>
        <div className="header-actions">
          <button className="btn-secondary"><Layers size={18} /> Manage Classes</button>
          <button className="btn-primary"><Upload size={18} /> Upload Images</button>
        </div>
      </header>

      <div className="dataset-stats-grid">
        {datasetStats.map((stat, idx) => (
          <div key={idx} className="dataset-stat-card">
            <div className="stat-icon-circle" style={{ backgroundColor: `${stat.color}15`, color: stat.color }}>
              {getIconForLabel(stat.label)}
            </div>
            <div className="stat-info">
              <span className="label">{stat.label}</span>
              <span className="value">{stat.value}</span>
            </div>
          </div>
        ))}
      </div>

      <div className="dataset-main-layout">
        <section className="dataset-browser glass-panel">
          <div className="section-header">
            <Database size={20} className="text-primary" />
            <h2>Image Browser</h2>
            <div className="browser-filters">
              <div className="search-mini">
                <Filter size={14} className="text-muted" />
                <input type="text" placeholder="Filter by tag..." />
              </div>
              <div className="tab-group">
                <button className={filter === 'All' ? 'active' : ''} onClick={() => setFilter('All')}>All</button>
                <button className={filter === 'Detection' ? 'active' : ''} onClick={() => setFilter('Detection')}>Detection</button>
                <button className={filter === 'OK' ? 'active' : ''} onClick={() => setFilter('OK')}>OK</button>
                <button className={filter === 'NG' ? 'active' : ''} onClick={() => setFilter('NG')}>NG</button>
              </div>
            </div>
          </div>

          <div className="image-grid">
            {filteredImages.map((img) => (
              <div key={img.id} className="image-card" style={{ minWidth: '220px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                <div style={{ flex: 1 }}>
                  <span className="text-xs text-muted mb-1 block" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {img.product || 'Product'} / {img.date}
                  </span>
                  <div className="image-placeholder" style={{ padding: 0, overflow: 'hidden', height: '160px' }}>
                    <img 
                      src={`${API_BASE_URL}/api/image/${img.name}`} 
                      alt={img.name}
                      style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                      onError={(e) => {
                        e.currentTarget.style.display = 'none';
                        e.currentTarget.parentElement?.classList.add('fallback-icon');
                      }}
                    />
                    <ImageIcon size={32} className="text-muted opacity-20 fallback-svg" style={{ display: 'none' }} />
                    <div className="image-overlay">
                      <button className="overlay-btn"><Eye size={16} /></button>
                      <button className="overlay-btn"><Tag size={16} /></button>
                    </div>
                  </div>
                </div>
                
                <div style={{ flex: 1 }}>
                  <span className="text-xs text-muted mb-1 block">Result</span>
                  <div className="image-placeholder" style={{ padding: 0, overflow: 'hidden', height: '160px' }}>
                    {img.task_type === 'detection' ? (
                      <DatasetDetectionResult recordId={img.id} />
                    ) : (
                      <>
                        <img
                          src={`${API_BASE_URL}/api/result_image/${img.name}`}
                          alt={`Result ${img.name}`}
                          style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                          onError={(e) => {
                            e.currentTarget.style.display = 'none';
                            e.currentTarget.parentElement?.classList.add('fallback-icon');
                          }}
                        />
                        <ImageIcon size={32} className="text-muted opacity-20 fallback-svg" style={{ display: 'none' }} />
                      </>
                    )}
                    <div className="image-overlay">
                      <button className="overlay-btn"><Eye size={16} /></button>
                    </div>
                  </div>
                </div>

                <div className="image-info" style={{ marginTop: 'auto' }}>
                  <span className="img-name text-xs" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{img.name}</span>
                  <div className="img-tags">
                    <span className={`tag status ${img.status}`}>{img.status}</span>
                    <span className={`tag type ${img.type.toLowerCase()}`}>{img.type}</span>
                    {img.task_type === 'detection' && (
                      <span className="tag object-count">{img.total_objects} objects</span>
                    )}
                  </div>
                  {img.task_type === 'detection' && (
                    <div className="dataset-class-counts">
                      {Object.entries(img.counts_by_class || {}).map(([className, count]) => (
                        <span key={className}>{className}: {count}</span>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <button className="add-image-btn" style={{ minWidth: '220px', height: '100%' }}>
              <Plus size={32} />
              <span>Add Images</span>
            </button>
          </div>
        </section>

        <aside className="dataset-sidebar">
          <section className="synthetic-panel glass-panel">
            <div className="section-header">
              <BrainCircuit size={20} className="text-primary" />
              <h2>Synthetic Generator</h2>
            </div>
            <p className="text-xs text-muted mb-4">Generate AI-based defects to augment your training set.</p>
            <div className="gen-controls">
              <div className="control-group">
                <label>Defect Type</label>
                <select>
                  <option>Scratch</option>
                  <option>Crack</option>
                  <option>Dent</option>
                </select>
              </div>
              <div className="control-group">
                <label>Intensity</label>
                <input type="range" min="1" max="100" defaultValue="50" />
              </div>
              <button className="btn-primary w-full mt-2">Generate 50 Samples</button>
            </div>
          </section>

          <section className="labeling-status glass-panel">
            <div className="section-header">
              <CheckCircle size={20} className="text-primary" />
              <h2>Labeling Progress</h2>
            </div>
            <div className="progress-container">
              <div className="progress-info">
                <span>Overall Completion</span>
                <span>87%</span>
              </div>
              <div className="progress-bar-lg">
                <div className="fill" style={{ width: '87%' }}></div>
              </div>
            </div>
            <div className="reviewer-list">
              <div className="reviewer">
                <span className="name">User Admin</span>
                <span className="count">4,200</span>
              </div>
              <div className="reviewer">
                <span className="name">Engineer_1</span>
                <span className="count">3,540</span>
              </div>
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
