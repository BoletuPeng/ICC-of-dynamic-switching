import { useMemo } from 'react';
import type { Position } from '../types';

interface ConnectionLineProps {
  start: Position;
  end: Position;
  color?: string;
  isTemporary?: boolean;
  onClick?: () => void;
}

export function ConnectionLine({
  start,
  end,
  color = '#0ea5e9',
  isTemporary = false,
  onClick,
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
    <g>
      {/* Wider invisible stroke for easier clicking */}
      {onClick && (
        <path
          d={path}
          fill="none"
          stroke="transparent"
          strokeWidth={20}
          onClick={onClick}
          className="cursor-pointer"
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
      />

      {/* Main line */}
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={isTemporary ? 2 : 3}
        strokeLinecap="round"
        className={`
          transition-all duration-200
          ${isTemporary ? 'stroke-dasharray-4-4 opacity-60' : ''}
          ${onClick ? 'hover:stroke-[4px] cursor-pointer' : ''}
        `}
        style={isTemporary ? { strokeDasharray: '8 4' } : undefined}
        onClick={onClick}
      />

      {/* Animated flow particles (only for established connections) */}
      {!isTemporary && (
        <>
          <circle r={4} fill={color}>
            <animateMotion dur="2s" repeatCount="indefinite" path={path} />
          </circle>
          <circle r={4} fill={color} opacity={0.5}>
            <animateMotion dur="2s" repeatCount="indefinite" path={path} begin="0.66s" />
          </circle>
          <circle r={4} fill={color} opacity={0.3}>
            <animateMotion dur="2s" repeatCount="indefinite" path={path} begin="1.33s" />
          </circle>
        </>
      )}
    </g>
  );
}
