import { memo, useCallback } from 'react';
import { Handle, Position } from '@xyflow/react';
import { Settings, Trash2, Copy } from 'lucide-react';
import { Icon } from '../Icons';
import { usePipelineStore } from '../../store/pipelineStore';
import { useNodeShapeInfo } from '../../hooks/useShapeInference';
import type { PipelineNodeProps, ResolvedPortShape } from '../../types';
import { CATEGORY_LABELS } from '../../types';

/**
 * Shape display component for ports
 */
const ShapeDisplay = memo(function ShapeDisplay({
  shape,
  color,
}: {
  shape: ResolvedPortShape | undefined;
  color: string;
}) {
  if (!shape) return null;

  const displayText = shape.isFullyResolved && shape.resolved
    ? shape.resolved
    : shape.symbolic;

  const isResolved = shape.isFullyResolved;

  return (
    <span
      className={`node-port-shape ${isResolved ? 'resolved' : 'symbolic'}`}
      title={`Shape: ${shape.symbolic}${shape.resolved ? ` = ${shape.resolved}` : ''}`}
      style={{
        color: isResolved ? color : undefined,
      }}
    >
      [{displayText}]
    </span>
  );
});

export const PipelineNode = memo(function PipelineNode({
  id,
  data,
  selected,
}: PipelineNodeProps) {
  const { setEditingNode, removeNode, duplicateNode } = usePipelineStore();
  const shapeInfo = useNodeShapeInfo(id);

  const handleEdit = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setEditingNode(id);
    },
    [id, setEditingNode]
  );

  const handleDelete = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      removeNode(id);
    },
    [id, removeNode]
  );

  const handleDuplicate = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      duplicateNode(id);
    },
    [id, duplicateNode]
  );

  const handleDoubleClick = useCallback(() => {
    setEditingNode(id);
  }, [id, setEditingNode]);

  const categoryLabel = CATEGORY_LABELS[data.category] || data.category;

  // Check if this is a control flow node
  const isControlFlow = data.isControlFlow || data.category === 'control';

  return (
    <div
      className={`pipeline-node group ${selected ? 'selected' : ''} ${isControlFlow ? 'control-flow' : ''}`}
      onDoubleClick={handleDoubleClick}
      style={{ '--node-color': data.color } as React.CSSProperties}
    >
      {/* Header */}
      <div
        className="node-header"
        style={{ backgroundColor: data.color }}
      >
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <Icon name={data.icon} className="text-white shrink-0" size={16} />
          <span className="text-sm font-semibold text-white truncate">
            {data.label}
          </span>
        </div>

        {/* Action buttons */}
        <div className="node-actions">
          <button
            onClick={handleDuplicate}
            className="node-action-btn"
            title="Duplicate node"
          >
            <Copy size={12} />
          </button>
          <button
            onClick={handleEdit}
            className="node-action-btn"
            title="Edit parameters"
          >
            <Settings size={12} />
          </button>
          <button
            onClick={handleDelete}
            className="node-action-btn node-action-btn-danger"
            title="Delete node"
          >
            <Trash2 size={12} />
          </button>
        </div>
      </div>

      {/* Body */}
      <div className="node-body">
        {/* Ports container */}
        <div className="node-ports">
          {/* Inputs */}
          <div className="node-port-column">
            {data.inputs.map((input, index) => {
              const inputShape = shapeInfo?.inputShapes[input.id];
              return (
                <div key={input.id} className="node-port node-port-input">
                  <Handle
                    type="target"
                    position={Position.Left}
                    id={input.id}
                    className="node-handle node-handle-input"
                    style={{
                      top: `${((index + 1) / (data.inputs.length + 1)) * 100}%`,
                      backgroundColor: data.color,
                    }}
                  />
                  <div className="node-port-info">
                    <span className="node-port-label" title={input.description}>
                      {input.name}
                    </span>
                    <ShapeDisplay shape={inputShape} color={data.color} />
                  </div>
                </div>
              );
            })}
          </div>

          {/* Outputs */}
          <div className="node-port-column node-port-column-output">
            {data.outputs.map((output, index) => {
              const outputShape = shapeInfo?.outputShapes[output.id];
              return (
                <div key={output.id} className="node-port node-port-output">
                  <div className="node-port-info">
                    <ShapeDisplay shape={outputShape} color={data.color} />
                    <span className="node-port-label" title={output.description}>
                      {output.name}
                    </span>
                  </div>
                  <Handle
                    type="source"
                    position={Position.Right}
                    id={output.id}
                    className="node-handle node-handle-output"
                    style={{
                      top: `${((index + 1) / (data.outputs.length + 1)) * 100}%`,
                      backgroundColor: data.color,
                    }}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="node-footer">
        <span
          className="node-category-badge"
          style={{ backgroundColor: `${data.color}20`, color: data.color }}
        >
          {categoryLabel}
        </span>
      </div>
    </div>
  );
});

export default PipelineNode;
