import { memo, useCallback, type CSSProperties } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from '@xyflow/react';
import { X } from 'lucide-react';
import { usePipelineStore } from '../../store/pipelineStore';
import type { PipelineEdge } from '../../types';

type PipelineEdgeComponentProps = EdgeProps<PipelineEdge>;

export const PipelineEdgeComponent = memo(function PipelineEdgeComponent({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  markerEnd,
  selected,
}: PipelineEdgeComponentProps) {
  const { removeEdge } = usePipelineStore();

  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const handleDelete = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      removeEdge(id);
    },
    [id, removeEdge]
  );

  const edgeStyle: CSSProperties = {
    ...(style || {}),
    strokeWidth: 2,
    stroke: selected ? '#0ea5e9' : '#64748b',
  };

  return (
    <>
      {/* Invisible wider path for easier selection */}
      <path
        d={edgePath}
        fill="none"
        strokeWidth={20}
        stroke="transparent"
        className="react-flow__edge-interaction"
      />

      {/* Glow effect */}
      <path
        d={edgePath}
        fill="none"
        strokeWidth={6}
        stroke="rgba(14, 165, 233, 0.3)"
        strokeLinecap="round"
        filter="blur(4px)"
        className="pipeline-edge-glow"
      />

      {/* Main edge path */}
      <BaseEdge
        path={edgePath}
        markerEnd={markerEnd}
        style={edgeStyle}
        className="pipeline-edge-path"
      />

      {/* Animated flow indicator */}
      <path
        d={edgePath}
        fill="none"
        strokeWidth={2}
        stroke={selected ? '#0ea5e9' : '#94a3b8'}
        strokeLinecap="round"
        strokeDasharray="8 4"
        className="pipeline-edge-flow"
      />

      {/* Delete button on hover/select */}
      <EdgeLabelRenderer>
        <div
          className={`pipeline-edge-label ${selected ? 'visible' : ''}`}
          style={{
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
          }}
        >
          <button
            onClick={handleDelete}
            className="pipeline-edge-delete-btn"
            title="Delete connection"
          >
            <X size={12} />
          </button>
        </div>
      </EdgeLabelRenderer>
    </>
  );
});

export { PipelineEdgeComponent as PipelineEdge };
export default PipelineEdgeComponent;
