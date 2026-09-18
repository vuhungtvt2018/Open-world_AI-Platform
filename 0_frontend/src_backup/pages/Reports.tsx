import React, { useState } from 'react';
import { 
  FileText, 
  Search, 
  FileDown, 
  FileSpreadsheet, 
  History, 
  Calendar as CalendarIcon, 
  Share2,
  ChevronRight,
  Database
} from 'lucide-react';
import './Reports.css';

const EDGE_API_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

const initialReports = [
  { id: 'REP-2026-001', name: 'Daily Production Summary - 2026-05-16', type: 'System', format: 'PDF', date: '2026-05-16 06:00', size: '2.4 MB', downloads: 42, lastDownloaded: '2026-05-16 08:30' },
  { id: 'REP-2026-002', name: 'Weekly Quality Audit - W20', type: 'Manual', format: 'Excel', date: '2026-05-15 14:20', size: '1.2 MB', downloads: 15, lastDownloaded: '2026-05-15 16:45' },
  { id: 'REP-2026-003', name: 'Traceability Log - Line 01 - May', type: 'System', format: 'PDF', date: '2026-05-14 09:15', size: '15.8 MB', downloads: 128, lastDownloaded: '2026-05-16 10:10' },
  { id: 'REP-2026-004', name: 'Maintenance Event Report', type: 'Manual', format: 'PDF', date: '2026-05-12 11:30', size: '0.8 MB', downloads: 8, lastDownloaded: '2026-05-12 15:00' },
];

export default function Reports() {
  const [reports, setReports] = useState(initialReports);
  const [lastActionId, setLastActionId] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearching, setIsSearching] = useState(false);
  const [downloadLog, setDownloadLog] = useState<{id: string, name: string, time: string, format: string}[]>([
    { id: '1', name: 'Traceability Log - Line 01 - May', time: '2026-05-16 10:10:05', format: 'PDF' },
    { id: '2', name: 'Daily Production Summary - 2026-05-16', time: '2026-05-16 08:30:12', format: 'PDF' },
  ]);

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsSearching(true);
    try {
      const res = await fetch(`${EDGE_API_URL}/api/reports/lookup/${searchQuery.trim()}`);
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'success') {
          // pre-pend to reports
          setReports(prev => [...data.reports, ...prev.filter(r => !r.id.includes(searchQuery.trim()))]);
        }
      }
    } catch (err) {
      console.error('Failed to lookup', err);
    } finally {
      setIsSearching(false);
    }
  };

  const handleDownload = (reportId: string, reportName: string, format: string = 'PDF') => {
    const now = new Date();
    const timestamp = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')} ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    const fullTime = `${timestamp}:${String(now.getSeconds()).padStart(2, '0')}`;
    
    // Cập nhật lượt tải và thời gian tải gần nhất
    setReports(prev => prev.map(r => 
      r.id === reportId ? { ...r, downloads: r.downloads + 1, lastDownloaded: timestamp } : r
    ));
    
    // Thêm vào nhật ký hoạt động
    setDownloadLog(prev => [{
      id: Math.random().toString(36).substr(2, 9),
      name: reportName,
      time: fullTime,
      format: format
    }, ...prev].slice(0, 10)); // Giữ 10 mục gần nhất

    setLastActionId(reportId);
    setTimeout(() => setLastActionId(null), 5000);

    if (reportId.startsWith('REP-INSP-') || reportId.startsWith('REP-QC-')) {
        const table_name = reportId.startsWith('REP-INSP-') ? 'inspection_records' : 'qc_product_photo_library';
        const item_code = reportId.replace('REP-INSP-', '').replace('REP-QC-', '');
        window.open(`${EDGE_API_URL}/api/reports/download/${table_name}/${item_code}`, '_blank');
        return;
    }

    // Giả lập tải file
    const element = document.createElement("a");
    const file = new Blob([`AI Vision Report: ${reportName}\nDownloaded: ${fullTime}`], {type: 'text/plain'});
    element.href = URL.createObjectURL(file);
    element.download = `${reportName.replace(/\s+/g, '_')}.txt`;
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
  };

  return (
    <div className="reports-container">
      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Reporting System</h1>
          <p className="text-muted">Automated quality reports and product traceability logs</p>
        </div>
        <button className="btn-primary" onClick={() => handleDownload('CUSTOM', 'Custom_Report')}>
          <FileText size={18} />
          Generate Custom Report
        </button>
      </header>

      <div className="reports-layout">
        <div className="reports-main-content">
          <section className="traceability-section glass-panel">
            <div className="section-header">
              <Search size={20} className="text-primary" />
              <h2>Product Traceability Lookup</h2>
            </div>
            <div className="search-box">
              <div className="search-input-wrapper">
                <Database size={18} className="search-icon" />
                <input 
                  type="text" 
                  placeholder="Enter Product ItemCode (e.g. sp1, sp2)..." 
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                />
              </div>
              <button className="search-btn" onClick={handleSearch} disabled={isSearching}>
                {isSearching ? 'Searching...' : 'Search Database'}
              </button>
            </div>
          </section>

          <section className="reports-section glass-panel">
            <div className="section-header">
              <FileDown size={20} className="text-primary" />
              <h2>Generated Reports</h2>
            </div>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Report ID</th>
                    <th>Report Name</th>
                    <th>Format</th>
                    <th className="text-center">Downloads</th>
                    <th>Created Date</th>
                    <th className="text-right">Last Download</th>
                  </tr>
                </thead>
                <tbody>
                  {reports.map((report) => (
                    <tr 
                      key={report.id} 
                      className={`clickable-row ${lastActionId === report.id ? 'highlight-row' : ''}`}
                      onClick={() => handleDownload(report.id, report.name, report.format)}
                    >
                      <td className="font-mono text-xs">{report.id}</td>
                      <td>
                        <div className="report-name-cell">
                          {report.format === 'PDF' ? <FileText size={16} className="text-danger" /> : <FileSpreadsheet size={16} className="text-success" />}
                          <span className="font-bold">{report.name}</span>
                        </div>
                      </td>
                      <td><span className="format-badge">{report.format}</span></td>
                      <td className="text-center">
                        <div className="download-count-pill">
                          <FileDown size={12} />
                          {report.downloads}
                        </div>
                      </td>
                      <td><span className="text-muted font-mono">{report.date}</span></td>
                      <td className="text-right">
                        <div className="last-download-cell justify-end">
                          <span className="font-mono text-xs">{report.lastDownloaded}</span>
                          {lastActionId === report.id && (
                            <span className="recent-badge">JUST NOW</span>
                          )}
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </div>

        <aside className="reports-sidebar">
          <section className="activity-log-section glass-panel">
            <div className="section-header">
              <div className="header-left">
                <History size={18} className="text-primary" />
                <h2>Activity Log</h2>
              </div>
              <span className="count-tag">{downloadLog.length}</span>
            </div>
            <div className="activity-timeline">
              {downloadLog.map((log) => (
                <div key={log.id} className="activity-item-premium">
                  <div className="activity-status-line">
                    <div className="status-dot-pulse"></div>
                    <div className="line"></div>
                  </div>
                  <div className="activity-content-premium">
                    <div className="activity-header-premium">
                      <span className="activity-type">{log.format} Export</span>
                      <span className="activity-time-premium font-mono">{log.time.split(' ')[1]}</span>
                    </div>
                    <div className="activity-name-with-icon">
                      {log.format === 'PDF' ? 
                        <FileText size={14} className="text-danger" /> : 
                        <FileSpreadsheet size={14} className="text-success" />
                      }
                      <p className="activity-name-premium">{log.name}</p>
                    </div>
                  </div>
                </div>
              ))}
              {downloadLog.length === 0 && (
                <div className="empty-activity">
                  <History size={32} className="opacity-20 mb-2" />
                  <p>No recent activity tracked</p>
                </div>
              )}
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
