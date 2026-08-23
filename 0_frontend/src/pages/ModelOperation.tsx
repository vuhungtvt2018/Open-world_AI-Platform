import React from 'react';
import {
  Database,
  Activity,
  RefreshCcw,
  ArrowUpCircle,
  CheckCircle2,
  Clock,
  HardDrive,
  Loader2,
  XCircle
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie
} from 'recharts';
import './ModelOperation.css';

const modelRegistry = [
  {
    id: 'M-001', name: '[Mockup] YOLO-Detection-V8', type: 'Detection', version: '2.4.1', mAP: '0.942', status: 'Production', update: '2025-05-10',
    format: 'INT8', engine: 'TensorRT',
    details: {
      metrics: [
        { label: 'mAP @.5', value: '0.942' },
        { label: 'mAP @.5:.95', value: '0.814' },
        { label: 'Precision', value: '0.925' },
        { label: 'IoU Score', value: '0.880' }
      ],
      gflops: '8.2', params: '3.2M', memory: '124MB',
      input_size: '640x640', fps: '45.2',
      runtime: { pre: 12, infer: 28, post: 5 }
    }
  },
  {
    id: 'M-002', name: '[Mockup]Bolt-Keypoint-Pro', type: 'Alignment', version: '1.2.0', mAP: '0.915', status: 'Production', update: '2025-05-12',
    format: 'FP16', engine: 'OpenVINO',
    details: {
      metrics: [
        { label: 'mAP (Keypoint)', value: '0.915' },
        { label: 'PCK @0.05', value: '0.962' },
        { label: 'RMSE (Pixels)', value: '2.4px' },
        { label: 'Mean Error', value: '1.8px' }
      ],
      gflops: '2.4', params: '1.1M', memory: '45MB',
      input_size: '320x320', fps: '62.0',
      runtime: { pre: 8, infer: 18, post: 6 }
    }
  },
  {
    id: 'M-003', name: '[Mockup] Anomaly-Unsupervised-S', type: 'Anomaly', version: '3.0.5', mAP: '0.887', status: 'Production', update: '2025-05-14',
    format: 'FP16', engine: 'TensorRT',
    details: {
      metrics: [
        { label: 'Img AUROC', value: '0.982' },
        { label: 'Img AP', value: '0.975' },
        { label: 'Img F1', value: '0.960' },
        { label: 'Pixel AUROC', value: '0.965' },
        { label: 'Pixel AP', value: '0.940' },
        { label: 'Pixel F1', value: '0.925' },
        { label: 'Pixel AUPRO', value: '0.887' }
      ],
      gflops: '15.6', params: '22.4M', memory: '412MB',
      input_size: '1024x1024', fps: '6.8',
      runtime: { pre: 25, infer: 105, post: 15 }
    }
  },
  {
    id: 'M-004', name: '[Mockup] YOLO-Detection-V9-Beta', type: 'Detection', version: '2.5.0-rc', mAP: '0.958', status: 'Staging', update: '2025-05-16',
    format: 'FP32', engine: 'PyTorch',
    details: {
      metrics: [
        { label: 'mAP @.5', value: '0.958' },
        { label: 'mAP @.5:.95', value: '0.842' },
        { label: 'Precision', value: '0.941' },
        { label: 'Recall', value: '0.975' }
      ],
      gflops: '9.8', params: '4.5M', memory: '180MB',
      input_size: '640x640', fps: '22.4',
      runtime: { pre: 15, infer: 35, post: 5 }
    }
  },
  {
    id: 'M-005', name: '[Mockup] Defect-Classifier-ResNet', type: 'Classification', version: '1.0.2', mAP: '0.920', status: 'Inactive', update: '2025-04-20',
    format: 'INT8', engine: 'OpenVINO',
    details: {
      metrics: [
        { label: 'Accuracy', value: '0.920' },
        { label: 'Top-1 Acc', value: '0.920' },
        { label: 'F1-Score', value: '0.920' }
      ],
      confusionMatrix: {
        labels: ['Crack', 'Dent', 'Miss'],
        data: [
          [24, 1, 0],
          [2, 22, 1],
          [0, 0, 25]
        ]
      },
      gflops: '5.2', params: '11.2M', memory: '210MB',
      input_size: '224x224', fps: '120.5',
      runtime: { pre: 5, infer: 12, post: 5 }
    }
  },
];

interface ModelLatencyItem {
  stage: string;
  time: number;
  color: string;
  modelId?: string;
  modelName?: string;
  modelType?: string;
}

// 22082026 - KIET - Không dùng latency mock khi Model Registry API chưa có dữ liệu.
const latencyData: ModelLatencyItem[] = [];

const resourceData = [
  { name: 'GPU Memory', value: 68, color: '#2563eb' },
  { name: 'Free', value: 32, color: '#e2e8f0' },
];

interface RegistryStats {
  activeEngine: string;
  pipelineLatency: number;
  mAP: number;
  models: typeof modelRegistry;
  latencyData: ModelLatencyItem[];
  resourceData: Array<{ name: string; value: number; color: string }>;
  cpuThreads: number;
  npuLoad: number;
}

const API_BASE_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

// 22082026 - KIET - Chuẩn hóa response Model Registry để biểu đồ luôn nhận đủ dữ liệu.
const normalizeRegistryStats = (data: Partial<RegistryStats>): RegistryStats => ({
  activeEngine: data.activeEngine ?? 'v2.4.1 Stable',
  pipelineLatency: data.pipelineLatency ?? 0,
  mAP: data.mAP ?? 0,
  models: data.models ?? modelRegistry,
  latencyData: data.latencyData?.length ? data.latencyData : latencyData,
  resourceData: data.resourceData?.length ? data.resourceData : resourceData,
  cpuThreads: data.cpuThreads ?? 0,
  npuLoad: data.npuLoad ?? 0,
});

export default function ModelOperation() {
  const [selectedModel, setSelectedModel] = React.useState<any>(null);
  // 22082026 - PHUC - Trạng thái popup loading + kết quả khi Sync Model Hub.
  const [isSyncing, setIsSyncing] = React.useState(false);
  const [syncResult, setSyncResult] = React.useState<{ ok: boolean; message: string } | null>(null);
  const [registryStats, setRegistryStats] = React.useState<RegistryStats>({
    activeEngine: "v2.4.1 Stable",
    pipelineLatency: 302,
    mAP: 91.4,
    models: modelRegistry,
    latencyData,
    resourceData,
    cpuThreads: 0,
    npuLoad: 0,
  });

  React.useEffect(() => {
    const fetchModels = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/model-registry`, { cache: 'no-store' });
        if (res.ok) {
          const data = await res.json() as Partial<RegistryStats>;
          setRegistryStats(normalizeRegistryStats(data));
        }
      } catch (err) {
        console.error("Failed to fetch model registry stats");
      }
    };
    fetchModels();
  }, []);

  const handleSync = async () => {
    setIsSyncing(true);
    setSyncResult(null);

    try {
      // 22082026 - KIET - Pull Model Hub trước để metadata latency mới không bị local ghi đè.
      const res = await fetch(`${API_BASE_URL}/api/sync/models-down`, { method: 'POST' });
      if (!res.ok) {
        throw new Error('Failed to pull models from Model Hub');
      }

      // 22082026 - KIET - Push lại registry đã merge và refresh latency theo model.
      const pushRes = await fetch(`${API_BASE_URL}/api/sync/models-up`, { method: 'POST' });
      if (!pushRes.ok) {
        throw new Error('Failed to push merged model registry');
      }

      const modelRes = await fetch(`${API_BASE_URL}/model-registry`, { cache: 'no-store' });
      if (modelRes.ok) {
        const data = await modelRes.json() as Partial<RegistryStats>;
        setRegistryStats(normalizeRegistryStats(data));
      }

      setSyncResult({ ok: true, message: 'Model Registry đã được đồng bộ với Model Hub.' });
    } catch (err) {
      console.error(err);
      setSyncResult({
        ok: false,
        message: err instanceof Error ? err.message : 'Error during sync',
      });
    } finally {
      setIsSyncing(false);
    }
  };

  return (
    <div className="model-op-container">
      {selectedModel && (
        <div className="modal-overlay" onClick={() => setSelectedModel(null)}>
          <div className="modal-content glass-panel" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div>
                <h3 className="text-primary">{selectedModel.name}</h3>
                <span className="text-xs text-muted font-mono">{selectedModel.id} | v{selectedModel.version}</span>
              </div>
              <button className="close-btn" onClick={() => setSelectedModel(null)}>&times;</button>
            </div>

            <div className="modal-body">
              {selectedModel.details && (
                <>
                  <div className="metric-group">
                    <h4>Performance Metrics ({selectedModel.type})</h4>
                    <div className={`metric-grid-mini ${selectedModel.details.metrics.length > 4 ? 'grid-3' : ''}`}>
                      {selectedModel.details.metrics.map((m: any, idx: number) => (
                        <div key={idx} className="m-item">
                          <span>{m.label}</span>
                          <strong>{m.value}</strong>
                        </div>
                      ))}
                    </div>
                  </div>

                  {selectedModel.details.confusionMatrix && (
                    <div className="metric-group">
                      <h4>Confusion Matrix</h4>
                      <div className="confusion-matrix-wrapper">
                        <div className="cm-header">
                          <span></span>
                          {selectedModel.details.confusionMatrix.labels.map((l: string) => <span key={l}>{l}</span>)}
                        </div>
                        {selectedModel.details.confusionMatrix.data.map((row: number[], ri: number) => (
                          <div key={ri} className="cm-row">
                            <span className="cm-label">{selectedModel.details.confusionMatrix.labels[ri]}</span>
                            {row.map((val: number, ci: number) => (
                              <div
                                key={ci}
                                className="cm-cell"
                                style={{
                                  background: ri === ci
                                    ? `rgba(37, 99, 235, ${Math.max(0.1, val / 30)})`
                                    : `rgba(239, 68, 68, ${Math.max(0.05, val / 10)})`
                                }}
                              >
                                {val}
                              </div>
                            ))}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  <div className="metric-group">
                    <h4>Edge Deployment Specs</h4>
                    <div className="metric-grid-mini">
                      <div className="m-item"><span>Quantization</span><strong>{selectedModel.format}</strong></div>
                      <div className="m-item"><span>Target Engine</span><strong>{selectedModel.engine || 'PyTorch'}</strong></div>
                      <div className="m-item"><span>Input Resolution</span><strong>{selectedModel.details.input_size}</strong></div>
                      <div className="m-item"><span>Throughput (FPS)</span><strong>{selectedModel.details.fps}</strong></div>
                    </div>
                  </div>

                  <div className="metric-group">
                    <h4>Computing Resources</h4>
                    <div className="metric-grid-mini">
                      <div className="m-item"><span>GFLOPS</span><strong>{selectedModel.details.gflops}</strong></div>
                      <div className="m-item"><span>Parameters</span><strong>{selectedModel.details.params}</strong></div>
                      <div className="m-item"><span>VRAM Footprint</span><strong>{selectedModel.details.memory}</strong></div>
                    </div>
                  </div>

                  <div className="metric-group">
                    <h4>Runtime Timing (ms)</h4>
                    <div className="timing-stack">
                      <div className="t-bar pre" style={{ flex: selectedModel.details.runtime.pre }} title={`Pre: ${selectedModel.details.runtime.pre}ms`}></div>
                      <div className="t-bar infer" style={{ flex: selectedModel.details.runtime.infer }} title={`Infer: ${selectedModel.details.runtime.infer}ms`}></div>
                      <div className="t-bar post" style={{ flex: selectedModel.details.runtime.post }} title={`Post: ${selectedModel.details.runtime.post}ms`}></div>
                    </div>
                    <div className="timing-labels">
                      <span>Pre: {selectedModel.details.runtime.pre}ms</span>
                      <span>Infer: {selectedModel.details.runtime.infer}ms</span>
                      <span>Post: {selectedModel.details.runtime.post}ms</span>
                    </div>
                  </div>
                </>
              )}
              {!selectedModel.details && (
                <div className="metric-group">
                  <p>No detailed metrics available for this model.</p>
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="btn-secondary" onClick={() => setSelectedModel(null)}>Close</button>
              <button className="btn-primary">Export Report</button>
            </div>
          </div>
        </div>
      )}

      {/* 22082026 - PHUC - Popup loading khi đang sync (chặn tương tác cho tới khi xong). */}
      {isSyncing && (
        <div className="modal-overlay">
          <div className="modal-content glass-panel sync-popup" onClick={(e) => e.stopPropagation()}>
            <Loader2 size={44} className="text-primary sync-spinner" />
            <h3>Syncing with Model Hub...</h3>
            <p className="text-muted">Pulling models và pushing merged registry, vui lòng đợi.</p>
          </div>
        </div>
      )}

      {/* 22082026 - PHUC - Popup kết quả sync thay cho alert(). */}
      {syncResult && !isSyncing && (
        <div className="modal-overlay" onClick={() => setSyncResult(null)}>
          <div
            className={`modal-content glass-panel sync-popup ${syncResult.ok ? 'success' : 'error'}`}
            onClick={(e) => e.stopPropagation()}
          >
            {syncResult.ok ? (
              <CheckCircle2 size={44} className="text-success" />
            ) : (
              <XCircle size={44} className="text-secondary" />
            )}
            <h3>{syncResult.ok ? 'Sync Successful' : 'Sync Failed'}</h3>
            <p className="text-muted">{syncResult.message}</p>
            <button className="btn-primary" onClick={() => setSyncResult(null)}>
              OK
            </button>
          </div>
        </div>
      )}

      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Model Operation</h1>
          <p className="text-muted">Model version control, performance tracking and deployment management</p>
        </div>
        <button className="sync-btn glow-primary" onClick={handleSync} disabled={isSyncing}>
          {isSyncing ? (
            <>
              <RefreshCcw size={18} className="sync-spinner" />
              Syncing...
            </>
          ) : (
            <>
              <RefreshCcw size={18} />
              Sync with Model Hub
            </>
          )}
        </button>
      </header>

      <div className="op-stats-grid">
        <div className="op-stat-card">
          <div className="stat-label">Active Inference Engine</div>
          <div className="stat-value">{registryStats.activeEngine}</div>
          <div className="stat-footer text-success"><CheckCircle2 size={14} /> System Health: Optimized</div>
        </div>
        <div className="op-stat-card">
          <div className="stat-label">Total Pipeline Latency</div>
          <div className="stat-value">{registryStats.pipelineLatency} <span className="text-sm font-normal">ms</span></div>
          <div className="stat-footer text-primary"><Activity size={14} /> Edge Performance: Nominal</div>
        </div>
        <div className="op-stat-card">
          <div className="stat-label">mAP (Mean Average Precision)</div>
          <div className="stat-value text-secondary">{registryStats.mAP}%</div>
          <div className="stat-footer"><ArrowUpCircle size={14} className="text-secondary" /> +1.2% from previous version</div>
        </div>
      </div>

      <div className="op-main-grid">
        {/* Model Registry */}
        <section className="op-section table-panel glass-panel">
          <div className="section-header">
            <Database size={20} className="text-primary" />
            <h2>Model Registry</h2>
          </div>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Model Name</th>
                  <th>Type</th>
                  <th>Format</th>
                  <th>Version</th>
                  <th>mAP / Acc</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {registryStats.models.map((model: any) => (
                  <tr key={model.id} className="clickable-row" onClick={() => setSelectedModel(model)}>
                    <td className="font-mono text-xs">{model.id}</td>
                    <td><span className="font-bold">{model.name}</span></td>
                    <td><span className={`type-badge ${model.type.toLowerCase()}`}>{model.type}</span></td>
                    <td><span className={`format-badge ${model.format.toLowerCase()}`}>{model.format}</span></td>
                    <td>{model.version}</td>
                    <td><span className="font-mono">{model.mAP}</span></td>
                    <td>
                      <span className={`status-tag ${model.status.toLowerCase()}`}>
                        {model.status}
                      </span>
                    </td>
                    <td>
                      <button className="action-link" onClick={(e) => { e.stopPropagation(); setSelectedModel(model); }}>Details</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Latency & Resources */}
        <div className="op-side-col">
          <section className="op-section chart-panel glass-panel">
            <div className="section-header">
              <Clock size={20} className="text-primary" />
              <h2>Model Latency Breakdown (ms)</h2>
            </div>
            <div className="chart-container">
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={registryStats.latencyData} layout="vertical" margin={{ left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#f1f5f9" />
                  <XAxis type="number" hide />
                  <YAxis
                    dataKey="stage"
                    type="category"
                    axisLine={false}
                    tickLine={false}
                    fontSize={12}
                    width={120}
                  />
                  <Tooltip cursor={{ fill: '#f8fafc' }} formatter={(value) => [`${value} ms`, 'Latency']} />
                  <Bar dataKey="time" radius={[0, 4, 4, 0]} barSize={20}>
                    {registryStats.latencyData.map((entry: ModelLatencyItem, index: number) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="op-section resource-panel glass-panel">
            <div className="section-header">
              <HardDrive size={20} className="text-primary" />
              <h2>Edge Resource Utilization</h2>
            </div>
            <div className="resource-content">
              <div className="pie-chart-wrapper">
                <ResponsiveContainer width="100%" height={150}>
                  <PieChart>
                    <Pie
                      data={registryStats.resourceData || resourceData}
                      innerRadius={50}
                      outerRadius={70}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {(registryStats.resourceData || resourceData).map((entry: any, index: number) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                  </PieChart>
                </ResponsiveContainer>
                <div className="pie-label">
                  <span className="label-val">{registryStats.resourceData[0]?.value ?? 0}%</span>
                  <span className="label-sub">VRAM</span>
                </div>
              </div>
              <div className="resource-list">
                <div className="res-item">
                  <div className="res-info">
                    <span>CPU Threads</span>
                    <span className="res-val">{registryStats.cpuThreads}%</span>
                  </div>
                  <div className="res-bar"><div className="res-fill" style={{ width: `${registryStats.cpuThreads}%`, background: '#3b82f6' }}></div></div>
                </div>
                <div className="res-item">
                  <div className="res-info">
                    <span>NPU Load</span>
                    <span className="res-val">{registryStats.npuLoad}%</span>
                  </div>
                  <div className="res-bar"><div className="res-fill" style={{ width: `${registryStats.npuLoad}%`, background: '#f59e0b' }}></div></div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
