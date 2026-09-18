import './DetectionResultImage.css';

export interface DetectionObject {
  index: number;
  bbox: number[];
  score: number;
  class_id: number | null;
  class_name: string | null;
}

interface DetectionResultImageProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  objects: DetectionObject[];
}

// 12082026 - KIET - Chọn màu bounding box tương ứng với từng class Detection.
const getDetectionColor = (className: string | null) => {
  const classColors: Record<string, string> = {
    screw: '#2563eb',
    washer: '#10b981',
    wood_screw: '#f59e0b',
  };

  return classColors[className || ''] || '#ef4444';
};

// 12082026 - KIET - Vẽ ảnh Detection cùng bbox, tên class và confidence bằng SVG.
export default function DetectionResultImage({
  imageUrl,
  imageWidth,
  imageHeight,
  objects,
}: DetectionResultImageProps) {
  const fontSize = Math.max(14, imageWidth * 0.012);
  const labelHeight = fontSize * 1.45;
  const strokeWidth = Math.max(2, imageWidth * 0.0015);

  return (
    <svg
      className="detection-result-svg"
      viewBox={`0 0 ${imageWidth} ${imageHeight}`}
      preserveAspectRatio="xMidYMid meet"
      role="img"
      aria-label="Object Detection result"
    >
      <image
        href={imageUrl}
        width={imageWidth}
        height={imageHeight}
        preserveAspectRatio="none"
      />

      {objects.map((object) => {
        if (object.bbox.length < 4) return null;

        const [x1, y1, x2, y2] = object.bbox;
        const boxWidth = Math.max(0, x2 - x1);
        const boxHeight = Math.max(0, y2 - y1);
        const color = getDetectionColor(object.class_name);
        const label = `${object.class_name || 'unknown'} ${(object.score * 100).toFixed(1)}%`;
        const labelWidth = Math.min(
          imageWidth - x1,
          Math.max(fontSize * 5, label.length * fontSize * 0.62 + 10),
        );
        const labelY = Math.max(0, y1 - labelHeight);

        return (
          <g key={`${object.index}-${object.class_id}`}>
            <rect
              x={x1}
              y={y1}
              width={boxWidth}
              height={boxHeight}
              fill="transparent"
              stroke={color}
              strokeWidth={strokeWidth}
            />
            <rect
              x={x1}
              y={labelY}
              width={labelWidth}
              height={labelHeight}
              fill={color}
            />
            <text
              x={x1 + 5}
              y={labelY + fontSize}
              fill="#ffffff"
              fontSize={fontSize}
              fontWeight="700"
            >
              {label}
            </text>
          </g>
        );
      })}
    </svg>
  );
}
