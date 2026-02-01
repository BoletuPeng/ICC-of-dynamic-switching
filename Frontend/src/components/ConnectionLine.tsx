import { useMemo } from 'react';
import type { Position } from '../types';

interface ConnectionLineProps {
  start: Position;
  end: Position;
  color?: string;
  isTemporary?: boolean;
  onRightClick?: (e: React.MouseEvent) => void;
}

export function ConnectionLine({
  start,
  end,
  color = '#0ea5e9',
  isTemporary = false,
  onRightClick,
}: ConnectionLineProps) {
  const path = useMemo(() => {
    const dx = end.x - start.x;
    const controlOffset = Math.min(Math.abs(dx) * 0.5, 150);

    // Create a smooth bezier curve
    const cp1x = start.x + controlOffset;
    const cp1y = start.y;
    const cp2x = end.x - controlOffset;
    const cp2y = end.y;

    return `M ${start.x} ${start.y} C ${cp1x} ${cp1y}, ${cp2x} ${cp2y}, ${end.x} ${end.y}`;
  }, [start, end]);

  return (
    <g className={onRightClick ? 'cursor-pointer' : ''}>
      {/* Wider invisible stroke for easier clicking */}
      {onRightClick && (
        <path
          d={path}
          fill="none"
          stroke="rgba(255,255,255,0.01)"
          strokeWidth={24}
          onContextMenu={onRightClick}
          style={{ pointerEvents: 'auto', cursor: 'pointer' }}
        />
      )}

      {/* Glow effect */}
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={isTemporary ? 2 : 4}
        strokeOpacity={0.3}
        className="blur-sm"
        style={{ pointerEvents: 'none' }}
      />

      {/* Main line */}
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={isTemporary ? 2 : 3}
        strokeLinecap="round"
        className="transition-all duration-200"
        style={isTemporary ? { strokeDasharray: '8 4', pointerEvents: 'none' } : { pointerEvents: 'none' }}
      />

      {/* Animated flow particles (only for established connections) */}
      {!isTemporary && (
        <>
          <circle r={4} fill={color} style={{ pointerEvents: 'none' }}>
            <animateMotion dur="2s" repeatCount="indefinite" path={path} />
          </circle>
          <circle r={4} fill={color} opacity={0.5} style={{ pointerEvents: 'none' }}>
            <animateMotion dur="2s" repeatCount="indefinite" path={path} begin="0.66s" />
          </circle>
          <circle r={4} fill={color} opacity={0.3} style={{ pointerEvents: 'none' }}>
            <animateMotion dur="2s" repeatCount="indefinite" path={path} begin="1.33s" />
          </circle>
        </>
      )}
    </g>
  );
}
