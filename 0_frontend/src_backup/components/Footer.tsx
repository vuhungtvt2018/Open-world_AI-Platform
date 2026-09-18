import type { FC } from 'react';
import './Footer.css';

export const Footer: FC = () => {
  return (
    <footer className="main-footer">
      <div className="footer-left">
        <span className="footer-copyright">© 2026 Smart Inspection & Counting</span>
      </div>
      <div className="footer-right">
        <span className="footer-status-indicator">
          <span className="footer-dot"></span> System Ready
        </span>
      </div>
    </footer>
  );
};

export default Footer;
