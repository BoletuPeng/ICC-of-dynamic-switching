import { useCallback, useRef } from 'react';
import type { PortType, PortDefinition } from '../types';

interface PortProps {
  port: PortDefinition;
  type: PortType;
  nodeId: string;
  color: string;
  isConnected: boolean;
  onStartConnection: (nodeId: string, portId: string, portType: PortType, position: { x: number; y: number }) => void;
  onEndConnection: (nodeId: string, portId: string, portType: PortType) => void;
}

export function Port({
  port,
  type,
  nodeId,
  color,
  isConnected,
  onStartConnection,
  onEndConnection,
}: PortProps) {
  const portRef = useRef<HTMLDivElement>(null);
  const isInput = type === 'input';

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      e.preventDefault();
      if (portRef.current) {
        const rect = portRef.current.getBoundingClientRect();
        const position = {
          x: rect.left + rect.width / 2,
          y: rect.top + rect.height / 2,
        };
        onStartConnection(nodeId, port.id, type, position);
      }
    },
    [nodeId, port.id, type, onStartConnection]
  );

  const handleMouseUp = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onEndConnection(nodeId, port.id, type);
    },
    [nodeId, port.id, type, onEndConnection]
  );

  return (
    <div
      className={`flex items-center gap-2 h-7 ${
        isInput ? 'flex-row' : 'flex-row-reverse'
      }`}
    >
      <div
        ref={portRef}
        className={`
          w-3 h-3 rounded-full cursor-crosshair
          transition-all duration-200 ease-out
          border-2 hover:scale-150
          ${isConnected
            ? 'border-white bg-current shadow-lg shadow-current/50'
            : 'border-current bg-transparent hover:bg-current/30'
          }
        `}
        style={{ color }}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onPointerDown={(e) => e.stopPropagation()}
        title={`${port.name}: ${port.description}`}
      />
      <span className="text-xs text-surface-400 truncate max-w-[80px]">
        {port.name}
      </span>
    </div>
  );
}
