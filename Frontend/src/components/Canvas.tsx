import { useCallback, useRef, useState, useEffect } from 'react';
import { useDroppable } from '@dnd-kit/core';
import { NodeCard } from './NodeCard';
import { ConnectionLine } from './ConnectionLine';
import { usePipelineStore } from '../store/pipelineStore';
import { nodeDefinitionsMap } from '../data';
import type { Position, PortType } from '../types';

interface PortPosition {
  nodeId: string;
  portId: string;
  portType: PortType;
  position: Position;
}

interface ConnectionState {
  sourceNodeId: string;
  sourcePortId: string;
  sourceType: PortType;
  startPosition: Position;
  currentPosition: Position;
}

export function Canvas() {
  const { nodes, connections, selectNode, addConnection, removeConnection } = usePipelineStore();
  const canvasRef = useRef<HTMLDivElement>(null);
  const [portPositions, setPortPositions] = useState<Map<string, PortPosition>>(new Map());
  const [connectionState, setConnectionState] = useState<ConnectionState | null>(null);
  const [viewOffset, setViewOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  const { setNodeRef, isOver } = useDroppable({
    id: 'canvas-dropzone',
  });

  // Register port positions
  const registerPort = useCallback(
    (nodeId: string, portId: string, portType: PortType, element: HTMLDivElement | null) => {
      const key = `${nodeId}-${portId}-${portType}`;
      if (element) {
        const rect = element.getBoundingClientRect();
        const canvasRect = canvasRef.current?.getBoundingClientRect();
        if (canvasRect) {
          setPortPositions((prev) => {
            const next = new Map(prev);
            next.set(key, {
              nodeId,
              portId,
              portType,
              position: {
                x: rect.left + rect.width / 2 - canvasRect.left,
                y: rect.top + rect.height / 2 - canvasRect.top,
              },
            });
            return next;
          });
        }
      } else {
        setPortPositions((prev) => {
          const next = new Map(prev);
          next.delete(key);
          return next;
        });
      }
    },
    []
  );

  // Update port positions on scroll/resize
  useEffect(() => {
    const updatePositions = () => {
      if (!canvasRef.current) return;
      const canvasRect = canvasRef.current.getBoundingClientRect();

      setPortPositions((prev) => {
        const next = new Map<string, PortPosition>();
        prev.forEach((value, key) => {
          const element = document.querySelector(`[data-port-key="${key}"]`) as HTMLDivElement;
          if (element) {
            const rect = element.getBoundingClientRect();
            next.set(key, {
              ...value,
              position: {
                x: rect.left + rect.width / 2 - canvasRect.left,
                y: rect.top + rect.height / 2 - canvasRect.top,
              },
            });
          } else {
            next.set(key, value);
          }
        });
        return next;
      });
    };

    window.addEventListener('resize', updatePositions);
    return () => window.removeEventListener('resize', updatePositions);
  }, []);

  // Handle connection start
  const handleStartConnection = useCallback(
    (nodeId: string, portId: string, portType: PortType, position: Position) => {
      const canvasRect = canvasRef.current?.getBoundingClientRect();
      if (canvasRect) {
        setConnectionState({
          sourceNodeId: nodeId,
          sourcePortId: portId,
          sourceType: portType,
          startPosition: {
            x: position.x - canvasRect.left,
            y: position.y - canvasRect.top,
          },
          currentPosition: {
            x: position.x - canvasRect.left,
            y: position.y - canvasRect.top,
          },
        });
      }
    },
    []
  );

  // Handle connection end
  const handleEndConnection = useCallback(
    (nodeId: string, portId: string, portType: PortType) => {
      if (connectionState && connectionState.sourceNodeId !== nodeId) {
        // Can only connect output to input
        if (connectionState.sourceType === 'output' && portType === 'input') {
          addConnection({
            sourceNodeId: connectionState.sourceNodeId,
            sourcePortId: connectionState.sourcePortId,
            targetNodeId: nodeId,
            targetPortId: portId,
          });
        } else if (connectionState.sourceType === 'input' && portType === 'output') {
          addConnection({
            sourceNodeId: nodeId,
            sourcePortId: portId,
            targetNodeId: connectionState.sourceNodeId,
            targetPortId: connectionState.sourcePortId,
          });
        }
      }
      setConnectionState(null);
    },
    [connectionState, addConnection]
  );

  // Handle mouse move for connection drawing
  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (connectionState && canvasRef.current) {
        const canvasRect = canvasRef.current.getBoundingClientRect();
        setConnectionState((prev) =>
          prev
            ? {
                ...prev,
                currentPosition: {
                  x: e.clientX - canvasRect.left,
                  y: e.clientY - canvasRect.top,
                },
              }
            : null
        );
      }

      if (isPanning) {
        const dx = e.clientX - panStart.x;
        const dy = e.clientY - panStart.y;
        setViewOffset((prev) => ({
          x: prev.x + dx,
          y: prev.y + dy,
        }));
        setPanStart({ x: e.clientX, y: e.clientY });
      }
    };

    const handleMouseUp = () => {
      setConnectionState(null);
      setIsPanning(false);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [connectionState, isPanning, panStart]);

  // Handle canvas pan
  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      // Middle click or Alt+click
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
    }
  };

  // Get connection line positions
  const getConnectionPositions = useCallback(
    (sourceNodeId: string, sourcePortId: string, targetNodeId: string, targetPortId: string) => {
      const sourceKey = `${sourceNodeId}-${sourcePortId}-output`;
      const targetKey = `${targetNodeId}-${targetPortId}-input`;
      const source = portPositions.get(sourceKey);
      const target = portPositions.get(targetKey);

      if (source && target) {
        return { start: source.position, end: target.position };
      }
      return null;
    },
    [portPositions]
  );

  // Get color for connection
  const getConnectionColor = useCallback((sourceNodeId: string) => {
    const node = nodes.find((n) => n.id === sourceNodeId);
    if (node) {
      const definition = nodeDefinitionsMap[node.definitionId];
      return definition?.color || '#0ea5e9';
    }
    return '#0ea5e9';
  }, [nodes]);

  return (
    <div
      ref={(el) => {
        setNodeRef(el);
        (canvasRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
      }}
      className={`
        flex-1 relative overflow-hidden
        bg-gradient-to-br from-surface-950 via-surface-900 to-surface-950
        ${isOver ? 'ring-2 ring-inset ring-primary-500/50' : ''}
        ${isPanning ? 'cursor-grabbing' : 'cursor-default'}
      `}
      onClick={() => selectNode(null)}
      onMouseDown={handleCanvasMouseDown}
    >
      {/* Grid background */}
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: `
            linear-gradient(rgba(148, 163, 184, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148, 163, 184, 0.1) 1px, transparent 1px)
          `,
          backgroundSize: '40px 40px',
          backgroundPosition: `${viewOffset.x % 40}px ${viewOffset.y % 40}px`,
        }}
      />

      {/* Large grid */}
      <div
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage: `
            linear-gradient(rgba(148, 163, 184, 0.2) 2px, transparent 2px),
            linear-gradient(90deg, rgba(148, 163, 184, 0.2) 2px, transparent 2px)
          `,
          backgroundSize: '200px 200px',
          backgroundPosition: `${viewOffset.x % 200}px ${viewOffset.y % 200}px`,
        }}
      />

      {/* Connection SVG layer */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none">
        <defs>
          <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="coloredBlur" />
            <feMerge>
              <feMergeNode in="coloredBlur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        <g filter="url(#glow)">
          {/* Existing connections */}
          {connections.map((conn) => {
            const positions = getConnectionPositions(
              conn.sourceNodeId,
              conn.sourcePortId,
              conn.targetNodeId,
              conn.targetPortId
            );
            if (!positions) return null;

            return (
              <ConnectionLine
                key={conn.id}
                start={positions.start}
                end={positions.end}
                color={getConnectionColor(conn.sourceNodeId)}
                onClick={() => removeConnection(conn.id)}
              />
            );
          })}

          {/* Temporary connection being drawn */}
          {connectionState && (
            <ConnectionLine
              start={connectionState.startPosition}
              end={connectionState.currentPosition}
              color="#0ea5e9"
              isTemporary
            />
          )}
        </g>
      </svg>

      {/* Nodes */}
      {nodes.map((node) => (
        <NodeCard
          key={node.id}
          node={node}
          connections={connections}
          isSelected={usePipelineStore.getState().selectedNodeId === node.id}
          onStartConnection={handleStartConnection}
          onEndConnection={handleEndConnection}
          registerPort={registerPort}
        />
      ))}

      {/* Empty state */}
      {nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div className="w-24 h-24 mx-auto mb-4 rounded-full bg-surface-800/50 flex items-center justify-center">
              <svg
                className="w-12 h-12 text-surface-600"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M12 4v16m8-8H4"
                />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-surface-400">No nodes yet</h3>
            <p className="text-sm text-surface-500 mt-1">
              Drag nodes from the sidebar to start building your pipeline
            </p>
          </div>
        </div>
      )}

      {/* Controls hint */}
      <div className="absolute bottom-4 left-4 text-xs text-surface-500 glass px-3 py-2 rounded-lg">
        <span className="text-surface-400">Alt + Drag</span> to pan &nbsp;·&nbsp;
        <span className="text-surface-400">Click connection</span> to delete
      </div>
    </div>
  );
}
