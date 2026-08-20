import React, { useState } from 'react';
import { Search, Image as ImageIcon, Upload, Target, Plus, X, CheckCircle } from 'lucide-react';
import './Analytics.css'; // Reuse styles for now
import './DefectSearch.css';

const CLOUD_API_URL = import.meta.env.VITE_CLOUD_API_URL || 'http://localhost:8031';
const EDGE_API_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

interface DefectSearchResult {
  ID: number;
  SM_ID?: string | null;
  ItemCode?: string | null;
  ImageName?: string | null;
  ErrorDetail?: string | null;
  distance: number;
}

export default function DefectSearch() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [results, setResults] = useState<DefectSearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [addFile, setAddFile] = useState<File | null>(null);
  const [addPreview, setAddPreview] = useState('');
  const [itemCode, setItemCode] = useState('sp1');
  const [errorDetail, setErrorDetail] = useState('');
  const [sampleId, setSampleId] = useState('');
  const [isAdding, setIsAdding] = useState(false);
  const [addError, setAddError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResults([]); // Clear previous results
    }
  };

  const handleSearch = async () => {
    if (!selectedFile) return;

    setIsSearching(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('top_k', '10');
    formData.append('metric', 'cosine');

    try {
      const response = await fetch(`${CLOUD_API_URL}/search/by-image`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) throw new Error('Network response was not ok');
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Search failed:', error);
      alert('Không thể kết nối đến Cloud Server (port 8031). Vui lòng kiểm tra lại.');
    } finally {
      setIsSearching(false);
    }
  };

  const closeAddDialog = () => {
    if (isAdding) return;
    setShowAddDialog(false);
    setAddError('');
  };

  const handleAddFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAddFile(file);
    setAddPreview(URL.createObjectURL(file));
    setAddError('');
  };

  const handleAddDefect = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!addFile || !itemCode.trim() || !errorDetail.trim()) {
      setAddError('Vui lòng chọn ảnh, nhập mã sản phẩm và loại lỗi.');
      return;
    }

    setIsAdding(true);
    setAddError('');
    const formData = new FormData();
    formData.append('file', addFile);
    formData.append('ItemCode', itemCode.trim());
    formData.append('ErrorDetail', errorDetail.trim());
    formData.append('ImageType', '1');
    formData.append('Insert_PIC', 'Defect Search');
    if (sampleId.trim()) formData.append('SM_ID', sampleId.trim());

    try {
      const response = await fetch(`${CLOUD_API_URL}/photos`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const errorBody = await response.json().catch(() => null);
        throw new Error(errorBody?.detail || `Không thể thêm ảnh (HTTP ${response.status}).`);
      }

      const created = await response.json();
      setSelectedFile(addFile);
      setPreview(addPreview);
      setResults([]);
      setSuccessMessage(`Đã thêm ảnh lỗi ID ${created.ID} vào Global Database.`);
      setShowAddDialog(false);
      setAddFile(null);
      setAddPreview('');
      setErrorDetail('');
      setSampleId('');
    } catch (error) {
      setAddError(error instanceof Error ? error.message : 'Không thể thêm ảnh lỗi.');
    } finally {
      setIsAdding(false);
    }
  };

  return (
    <div className="analytics-container">
      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Defect Search</h1>
          <p className="text-muted">Global Defect Similarity Search across all factories (via Cloud PGVector)</p>
        </div>
        <div className="header-actions">
          <button className="btn-primary" onClick={() => { setSuccessMessage(''); setShowAddDialog(true); }}>
            <Plus size={18} /> Add Defect Image
          </button>
        </div>
      </header>

      {successMessage && (
        <div className="defect-success" role="status">
          <CheckCircle size={18} />
          <span>{successMessage}</span>
          <button type="button" onClick={() => setSuccessMessage('')} aria-label="Close notification"><X size={16} /></button>
        </div>
      )}

      <div style={{ display: 'flex', gap: '2rem', marginTop: '1rem' }}>
        {/* Upload Panel */}
        <div className="glass-panel" style={{ flex: 1, padding: '2rem', display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center' }}>
          <h3 className="text-muted">Query Image</h3>
          
          <div 
            style={{ 
              width: '100%', height: '250px', border: '2px dashed var(--border)', 
              borderRadius: '8px', display: 'flex', alignItems: 'center', 
              justifyContent: 'center', overflow: 'hidden', cursor: 'pointer',
              position: 'relative'
            }}
            onClick={() => document.getElementById('search-upload')?.click()}
          >
            {preview ? (
              <img src={preview} alt="Query" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', color: 'var(--text-muted)' }}>
                <Upload size={32} />
                <span style={{ marginTop: '10px' }}>Click to Upload Defect Image</span>
              </div>
            )}
            <input 
              id="search-upload" 
              type="file" 
              accept="image/*" 
              style={{ display: 'none' }} 
              onChange={handleFileChange}
            />
          </div>

          <button 
            className="btn btn-primary" 
            style={{ width: '100%', padding: '0.8rem', display: 'flex', justifyContent: 'center', gap: '10px' }}
            onClick={handleSearch}
            disabled={!selectedFile || isSearching}
          >
            {isSearching ? <span className="spinner"></span> : <Search size={20} />}
            {isSearching ? 'Searching...' : 'Search Global Database'}
          </button>
        </div>

        {/* Results Panel */}
        <div className="glass-panel" style={{ flex: 2, padding: '2rem' }}>
          <h3 className="text-muted mb-4" style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <Target size={20} /> Search Results ({results.length})
          </h3>
          
          {results.length === 0 && !isSearching && (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px', color: 'var(--text-muted)' }}>
              No results found. Upload an image to search.
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: '1rem' }}>
            {results.map((r, idx) => (
              <div key={idx} style={{ background: 'var(--bg-secondary)', padding: '10px', borderRadius: '8px', border: '1px solid var(--border)' }}>
                <div style={{ height: '120px', background: '#000', marginBottom: '10px', borderRadius: '4px', overflow: 'hidden', position: 'relative' }}>
                  {r.ImageName ? (
                    <img 
                      src={`${EDGE_API_URL}/sync_images/${r.ItemCode ? r.ItemCode + '/' : ''}${r.ImageName}`} 
                      alt="Defect Result" 
                      style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                      onError={(e) => {
                        e.currentTarget.style.display = 'none';
                        if (e.currentTarget.nextElementSibling) {
                          (e.currentTarget.nextElementSibling as HTMLElement).style.display = 'flex';
                        }
                      }}
                    />
                  ) : null}
                  <div style={{ display: r.ImageName ? 'none' : 'flex', flexDirection: 'column', height: '100%', alignItems: 'center', justifyContent: 'center', color: '#888', textAlign: 'center', padding: '10px' }}>
                    <ImageIcon size={24} style={{ marginBottom: '5px' }} />
                    <span style={{ fontSize: '0.7rem' }}>Ảnh chưa đồng bộ từ Cloud về Edge</span>
                  </div>
                </div>
                <div style={{ fontSize: '0.9rem', color: 'var(--text-primary)', marginBottom: '5px' }}>
                  <strong>ID:</strong> {r.ID}
                </div>
                <div style={{ fontSize: '0.85rem', color: 'var(--text-primary)', marginBottom: '5px' }}>
                  <strong>Sản phẩm:</strong> <span style={{ color: 'var(--primary)' }}>{r.ItemCode || 'N/A'}</span>
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Distance: {(r.distance || 0).toFixed(4)}
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--danger)', marginTop: '5px' }}>
                  {r.ErrorDetail || 'Unknown Defect'}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {showAddDialog && (
        <div className="defect-modal-overlay" onMouseDown={closeAddDialog}>
          <div className="defect-modal" role="dialog" aria-modal="true" aria-labelledby="add-defect-title" onMouseDown={(e) => e.stopPropagation()}>
            <div className="defect-modal-header">
              <div>
                <h2 id="add-defect-title">Add Defect Image</h2>
                <p>Ảnh sẽ được tạo embedding và lưu vào thư viện PostgreSQL.</p>
              </div>
              <button type="button" className="icon-button" onClick={closeAddDialog} disabled={isAdding} aria-label="Close">
                <X size={20} />
              </button>
            </div>

            <form onSubmit={handleAddDefect}>
              <div className="defect-modal-body">
                <label className="defect-upload-field">
                  {addPreview ? <img src={addPreview} alt="Defect preview" /> : <><Upload size={28} /><span>Select defect image</span></>}
                  <input type="file" accept="image/*" onChange={handleAddFileChange} />
                </label>

                <div className="defect-form-grid">
                  <label>
                    <span>Item code *</span>
                    <input value={itemCode} onChange={(e) => setItemCode(e.target.value)} placeholder="sp1" required />
                  </label>
                  <label>
                    <span>Sample ID</span>
                    <input value={sampleId} onChange={(e) => setSampleId(e.target.value)} placeholder="Optional unique ID" />
                  </label>
                  <label className="full-width">
                    <span>Defect type *</span>
                    <input value={errorDetail} onChange={(e) => setErrorDetail(e.target.value)} placeholder="Scratch, crack, dent..." required />
                  </label>
                </div>

                {addError && <div className="defect-form-error" role="alert">{addError}</div>}
              </div>

              <div className="defect-modal-footer">
                <button type="button" className="btn-secondary" onClick={closeAddDialog} disabled={isAdding}>Cancel</button>
                <button type="submit" className="btn-primary" disabled={isAdding || !addFile}>
                  {isAdding ? <span className="spinner" /> : <Plus size={18} />}
                  {isAdding ? 'Adding...' : 'Add to Database'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
