import { useMemo, useCallback } from 'react';
import { useDraggable } from '@dnd-kit/core';
import { CSS } from '@dnd-kit/utilities';
import { Icon, Settings, Trash2 } from './Icons';
import { Port } from './Port';
import { nodeDefinitionsMap } from '../data';
import { usePipelineStore } from '../store/pipelineStore';
import type { PipelineNode, Connection, PortType } from '../types';

interface NodeCardProps {
  node: PipelineNode;
  connections: Connection[];
  isSelected: boolean;
  onStartConnection: (nodeId: string, portId: string, portType: PortType, position: { x: number; y: number }) => void;
  onEndConnection: (nodeId: string, portId: string, portType: PortType) => void;
  onDoubleClick: (nodeId: string) => void;
}

export function NodeCard({
  node,
  connections,
  isSelected,
  onStartConnection,
  onEndConnection,
  onDoubleClick,
}: NodeCardProps) {
  const definition = nodeDefinitionsMap[node.definitionId];
  const { selectNode, setEditingNode, removeNode } = usePipelineStore();

  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: node.id,
    data: {
      type: 'pipeline-node',
      node,
    },
  });

  const style = useMemo(
    () => ({
      transform: CSS.Translate.toString(transform),
      left: node.position.x,
      top: node.position.y,
    }),
    [transform, node.position.x, node.position.y]
  );

  const connectedInputs = useMemo(
    () => new Set(connections.filter((c) => c.targetNodeId === node.id).map((c) => c.targetPortId)),
    [connections, node.id]
  );

  const connectedOutputs = useMemo(
    () => new Set(connections.filter((c) => c.sourceNodeId === node.id).map((c) => c.sourcePortId)),
    [connections, node.id]
  );

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onDoubleClick(node.id);
    },
    [node.id, onDoubleClick]
  );

  if (!definition) {
    return null;
  }

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`
        absolute select-none
        w-[220px]
        rounded-xl overflow-hidden
        transition-shadow duration-200
        ${isDragging ? 'z-50 cursor-grabbing opacity-80' : 'z-10 cursor-grab'}
        ${isSelected ? 'ring-2 ring-primary-400 ring-offset-2 ring-offset-surface-900' : ''}
        ${isDragging ? 'shadow-2xl shadow-black/50' : 'shadow-lg shadow-black/30'}
      `}
      onClick={(e) => {
        e.stopPropagation();
        selectNode(node.id);
      }}
      onDoubleClick={handleDoubleClick}
      {...attributes}
      {...listeners}
    >
      {/* Header */}
      <div
        className="px-3 py-2 flex items-center gap-2"
        style={{ backgroundColor: definition.color }}
      >
        <Icon name={definition.icon} className="text-white" size={18} />
        <span className="flex-1 text-sm font-semibold text-white truncate">
          {definition.name}
        </span>
        <button
          onClick={(e) => {
            e.stopPropagation();
            setEditingNode(node.id);
          }}
          onPointerDown={(e) => e.stopPropagation()}
          className="p-1 rounded hover:bg-white/20 transition-colors"
          title="Edit parameters"
        >
          <Settings size={14} className="text-white" />
        </button>
        <button
          onClick={(e) => {
            e.stopPropagation();
            removeNode(node.id);
          }}
          onPointerDown={(e) => e.stopPropagation()}
          className="p-1 rounded hover:bg-white/20 transition-colors"
          title="Delete node"
        >
          <Trash2 size={14} className="text-white" />
        </button>
      </div>

      {/* Body */}
      <div className="bg-surface-800 p-3">
        <div className="flex justify-between">
          {/* Inputs */}
          <div className="flex flex-col gap-1">
            {definition.inputs.map((input) => (
              <Port
                key={input.id}
                port={input}
                type="input"
                nodeId={node.id}
                color={definition.color}
                isConnected={connectedInputs.has(input.id)}
                onStartConnection={onStartConnection}
                onEndConnection={onEndConnection}
              />
            ))}
          </div>

          {/* Outputs */}
          <div className="flex flex-col gap-1">
            {definition.outputs.map((output) => (
              <Port
                key={output.id}
                port={output}
                type="output"
                nodeId={node.id}
                color={definition.color}
                isConnected={connectedOutputs.has(output.id)}
                onStartConnection={onStartConnection}
                onEndConnection={onEndConnection}
              />
            ))}
          </div>
        </div>

        {/* Category badge */}
        <div className="mt-2 pt-2 border-t border-surface-700">
          <span
            className="text-[10px] font-medium px-2 py-0.5 rounded-full"
            style={{
              backgroundColor: `${definition.color}20`,
              color: definition.color,
            }}
          >
            {definition.category}
          </span>
        </div>
      </div>
    </div>
  );
}
