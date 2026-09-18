import React from 'react';
import './GlobalMonitoring.css'; // Reuse basic styling

interface PlaceholderPageProps {
  title: string;
  description: string;
  layer: string;
}

export default function PlaceholderPage({ title, description, layer }: PlaceholderPageProps) {
  return (
    <div className="dashboard-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      <div className="panel glass-panel" style={{ maxWidth: '600px', textAlign: 'center', padding: '48px' }}>
        <h1 className="text-primary" style={{ fontSize: '32px', marginBottom: '16px' }}>{title}</h1>
        <div style={{ display: 'inline-block', padding: '4px 12px', backgroundColor: 'rgba(0, 240, 255, 0.1)', border: '1px solid var(--primary)', borderRadius: '16px', fontSize: '12px', color: 'var(--primary)', marginBottom: '24px' }}>
          {layer}
        </div>
        <p className="text-muted" style={{ fontSize: '16px', lineHeight: '1.6' }}>
          {description}
        </p>
        <div style={{ marginTop: '32px', padding: '16px', border: '1px dashed var(--border)', borderRadius: '8px', color: 'var(--text-muted)' }}>
          This module is currently under construction.
        </div>
      </div>
    </div>
  );
}
