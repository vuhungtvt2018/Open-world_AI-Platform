import React, { useState } from 'react';
import { Search, Image as ImageIcon, AlertTriangle, Upload, Target } from 'lucide-react';
import './Analytics.css'; // Reuse styles for now

const CLOUD_API_URL = import.meta.env.VITE_CLOUD_API_URL || 'http://localhost:8031';
const EDGE_API_URL = import.meta.env.VITE_EDGE_API_URL || 'http://localhost:8000';

export default function DefectSearch() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [results, setResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);

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

  return (
    <div className="analytics-container">
      <header className="page-header">
        <div className="header-info">
          <h1 className="text-primary">Defect Search</h1>
          <p className="text-muted">Global Defect Similarity Search across all factories (via Cloud PGVector)</p>
        </div>
      </header>

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
    </div>
  );
}
