import { useCallback, useRef, useState, useEffect, useMemo } from 'react';
import { useDroppable } from '@dnd-kit/core';
import { NodeCard } from './NodeCard';
import { ConnectionLine } from './ConnectionLine';
import { usePipelineStore } from '../store/pipelineStore';
import { nodeDefinitionsMap } from '../data';
import type { Position, PortType, PipelineNode } from '../types';

interface ConnectionState {
  sourceNodeId: string;
  sourcePortId: string;
  sourceType: PortType;
  startPosition: Position;
  currentPosition: Position;
}

// Constants for node dimensions
const NODE_WIDTH = 220;
const NODE_HEADER_HEIGHT = 36;
const PORT_HEIGHT = 28;
const PORT_PADDING = 12;
const PORT_RADIUS = 6;

// Calculate port position based on node position and port index
function calculatePortPosition(
  node: PipelineNode,
  portId: string,
  portType: PortType,
  viewOffset: Position
): Position | null {
  const definition = nodeDefinitionsMap[node.definitionId];
  if (!definition) return null;

  const ports = portType === 'input' ? definition.inputs : definition.outputs;
  const portIndex = ports.findIndex((p) => p.id === portId);
  if (portIndex === -1) return null;

  const x = portType === 'input'
    ? node.position.x + PORT_RADIUS + viewOffset.x
    : node.position.x + NODE_WIDTH - PORT_RADIUS + viewOffset.x;

  const y = node.position.y + NODE_HEADER_HEIGHT + PORT_PADDING + (portIndex * PORT_HEIGHT) + PORT_HEIGHT / 2 + viewOffset.y;

  return { x, y };
}

export function Canvas() {
  const { nodes, connections, selectNode, addConnection, removeConnection, setEditingNode } = usePipelineStore();
  const canvasRef = useRef<HTMLDivElement>(null);
  const [connectionState, setConnectionState] = useState<ConnectionState | null>(null);
  const [viewOffset, setViewOffset] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [spacePressed, setSpacePressed] = useState(false);

  const { setNodeRef, isOver } = useDroppable({
    id: 'canvas-dropzone',
  });

  // Handle keyboard events for space bar panning
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat) {
        e.preventDefault();
        setSpacePressed(true);
      }
    };

    const handleKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space') {
        setSpacePressed(false);
        setIsPanning(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
    };
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

  // Handle connection end with validation
  const handleEndConnection = useCallback(
    (nodeId: string, portId: string, portType: PortType) => {
      if (!connectionState) {
        return;
      }

      // Prevent self-connection
      if (connectionState.sourceNodeId === nodeId) {
        setConnectionState(null);
        return;
      }

      // Determine source and target
      let sourceNodeId: string, sourcePortId: string, targetNodeId: string, targetPortId: string;

      if (connectionState.sourceType === 'output' && portType === 'input') {
        sourceNodeId = connectionState.sourceNodeId;
        sourcePortId = connectionState.sourcePortId;
        targetNodeId = nodeId;
        targetPortId = portId;
      } else if (connectionState.sourceType === 'input' && portType === 'output') {
        sourceNodeId = nodeId;
        sourcePortId = portId;
        targetNodeId = connectionState.sourceNodeId;
        targetPortId = connectionState.sourcePortId;
      } else {
        // Invalid connection type (output to output or input to input)
        setConnectionState(null);
        return;
      }

      // Check for duplicate connection
      const isDuplicate = connections.some(
        (c) =>
          c.sourceNodeId === sourceNodeId &&
          c.sourcePortId === sourcePortId &&
          c.targetNodeId === targetNodeId &&
          c.targetPortId === targetPortId
      );

      if (!isDuplicate) {
        addConnection({
          sourceNodeId,
          sourcePortId,
          targetNodeId,
          targetPortId,
        });
      }

      setConnectionState(null);
    },
    [connectionState, connections, addConnection]
  );

  // Handle mouse move for connection drawing and panning
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

  // Handle canvas mouse events
  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    // Middle click, Alt+click, or Space+click for panning
    if (e.button === 1 || (e.button === 0 && e.altKey) || (e.button === 0 && spacePressed)) {
      e.preventDefault();
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
    }
  };

  // Handle right-click on connection to delete
  const handleConnectionRightClick = useCallback(
    (connectionId: string, e: React.MouseEvent) => {
      e.preventDefault();
      e.stopPropagation();
      removeConnection(connectionId);
    },
    [removeConnection]
  );

  // Handle double-click on node to open editor
  const handleNodeDoubleClick = useCallback(
    (nodeId: string) => {
      setEditingNode(nodeId);
    },
    [setEditingNode]
  );

  // Calculate connection positions based on node positions
  const connectionPositions = useMemo(() => {
    return connections.map((conn) => {
      const sourceNode = nodes.find((n) => n.id === conn.sourceNodeId);
      const targetNode = nodes.find((n) => n.id === conn.targetNodeId);

      if (!sourceNode || !targetNode) return null;

      const start = calculatePortPosition(sourceNode, conn.sourcePortId, 'output', viewOffset);
      const end = calculatePortPosition(targetNode, conn.targetPortId, 'input', viewOffset);

      if (!start || !end) return null;

      const sourceDefinition = nodeDefinitionsMap[sourceNode.definitionId];
      const color = sourceDefinition?.color || '#0ea5e9';

      return { conn, start, end, color };
    }).filter(Boolean);
  }, [connections, nodes, viewOffset]);

  // Get color for connection being drawn
  const getDrawingConnectionColor = useMemo(() => {
    if (!connectionState) return '#0ea5e9';
    const node = nodes.find((n) => n.id === connectionState.sourceNodeId);
    if (node) {
      const definition = nodeDefinitionsMap[node.definitionId];
      return definition?.color || '#0ea5e9';
    }
    return '#0ea5e9';
  }, [connectionState, nodes]);

  return (
    <div
      ref={(el) => {
        setNodeRef(el);
        (canvasRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
      }}
      data-canvas
      className={`
        flex-1 relative overflow-hidden h-full
        bg-gradient-to-br from-surface-950 via-surface-900 to-surface-950
        ${isOver ? 'ring-2 ring-inset ring-primary-500/50' : ''}
        ${isPanning || spacePressed ? 'cursor-grab' : 'cursor-default'}
        ${isPanning ? 'cursor-grabbing' : ''}
      `}
      onClick={() => selectNode(null)}
      onMouseDown={handleCanvasMouseDown}
      onContextMenu={(e) => e.preventDefault()}
    >
      {/* Grid background */}
      <div
        className="absolute inset-0 opacity-20 pointer-events-none"
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
        className="absolute inset-0 opacity-10 pointer-events-none"
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
      <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 5 }}>
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
          {connectionPositions.map((item) => {
            if (!item) return null;
            const { conn, start, end, color } = item;

            return (
              <ConnectionLine
                key={conn.id}
                start={start}
                end={end}
                color={color}
                onRightClick={(e) => handleConnectionRightClick(conn.id, e)}
              />
            );
          })}

          {/* Temporary connection being drawn */}
          {connectionState && (
            <ConnectionLine
              start={connectionState.startPosition}
              end={connectionState.currentPosition}
              color={getDrawingConnectionColor}
              isTemporary
            />
          )}
        </g>
      </svg>

      {/* Nodes container with view offset */}
      <div
        className="absolute inset-0"
        style={{
          transform: `translate(${viewOffset.x}px, ${viewOffset.y}px)`,
          zIndex: 10,
        }}
      >
        {nodes.map((node) => (
          <NodeCard
            key={node.id}
            node={node}
            connections={connections}
            isSelected={usePipelineStore.getState().selectedNodeId === node.id}
            onStartConnection={handleStartConnection}
            onEndConnection={handleEndConnection}
            onDoubleClick={handleNodeDoubleClick}
          />
        ))}
      </div>

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
      <div className="absolute bottom-4 left-4 text-xs text-surface-500 glass px-3 py-2 rounded-lg z-20">
        <span className="text-surface-400">Space/Alt + Drag</span> to pan &nbsp;·&nbsp;
        <span className="text-surface-400">Right-click line</span> to delete &nbsp;·&nbsp;
        <span className="text-surface-400">Double-click node</span> to edit
      </div>
    </div>
  );
}
