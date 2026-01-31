import { useCallback, useRef, useEffect } from 'react';
import type { PortType, PortDefinition } from '../types';

interface PortProps {
  port: PortDefinition;
  type: PortType;
  nodeId: string;
  color: string;
  isConnected: boolean;
  onStartConnection: (nodeId: string, portId: string, portType: PortType, position: { x: number; y: number }) => void;
  onEndConnection: (nodeId: string, portId: string, portType: PortType) => void;
  registerPort: (nodeId: string, portId: string, portType: PortType, element: HTMLDivElement | null) => void;
}

export function Port({
  port,
  type,
  nodeId,
  color,
  isConnected,
  onStartConnection,
  onEndConnection,
  registerPort,
}: PortProps) {
  const portRef = useRef<HTMLDivElement>(null);
  const isInput = type === 'input';

  useEffect(() => {
    registerPort(nodeId, port.id, type, portRef.current);
    return () => {
      registerPort(nodeId, port.id, type, null);
    };
  }, [nodeId, port.id, type, registerPort]);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
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
      className={`flex items-center gap-2 py-1 ${
        isInput ? 'flex-row' : 'flex-row-reverse'
      }`}
    >
      <div
        ref={portRef}
        className={`
          w-3 h-3 rounded-full cursor-crosshair
          transition-all duration-200 ease-out
          border-2 hover:scale-125
          ${isConnected
            ? 'border-white bg-current shadow-lg shadow-current/50'
            : 'border-current bg-transparent hover:bg-current/30'
          }
        `}
        style={{ color }}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        title={`${port.name}: ${port.description}`}
      />
      <span className="text-xs text-surface-400 truncate max-w-[100px]">
        {port.name}
      </span>
    </div>
  );
}
