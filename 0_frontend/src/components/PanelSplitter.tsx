import React, { useState, useEffect, useCallback } from 'react';
import './PanelSplitter.css';

interface PanelSplitterProps {
  direction: 'vertical' | 'horizontal';
  onDrag: (delta: number) => void;
  className?: string;
}

export const PanelSplitter: React.FC<PanelSplitterProps> = ({
  direction,
  onDrag,
  className = ''
}) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  useEffect(() => {
    if (!isDragging) return;

    let startPos = 0;
    const handleMouseMove = (e: MouseEvent) => {
      const currentPos = direction === 'vertical' ? e.clientX : e.clientY;
      if (startPos !== 0) {
        const delta = currentPos - startPos;
        onDrag(delta);
      }
      startPos = currentPos;
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    // Attach listeners on window for smooth drag outside component boundaries
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // Prevent text selection while dragging
    document.body.style.userSelect = 'none';
    document.body.style.cursor = direction === 'vertical' ? 'col-resize' : 'row-resize';

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [isDragging, direction, onDrag]);

  return (
    <div
      className={`panel-splitter splitter-${direction} ${isDragging ? 'dragging' : ''} ${className}`}
      onMouseDown={handleMouseDown}
      title={direction === 'vertical' ? 'Kéo để chỉnh độ rộng' : 'Kéo để chỉnh chiều cao'}
    >
      <div className="splitter-handle">
        <div className="splitter-line" />
        <div className="splitter-grip">
          <span className="dot" />
          <span className="dot" />
          <span className="dot" />
        </div>
      </div>
    </div>
  );
};
