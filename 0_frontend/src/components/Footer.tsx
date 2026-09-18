import React from 'react';
import './Footer.css';

export const Footer: React.FC = () => {
  return (
    <footer className="main-footer">
      <div className="footer-left">
        <span className="footer-copyright">© 2026 Edgify Vision AI</span>
      </div>
      <div className="footer-right">
        <span className="footer-status-indicator">
          <span className="footer-dot"></span> System Ready
        </span>
        <span className="footer-version">v2.4.0</span>
      </div>
    </footer>
  );
};

export default Footer;
