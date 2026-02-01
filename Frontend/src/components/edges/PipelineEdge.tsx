import { memo, useCallback, useMemo, type CSSProperties } from 'react';
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  type EdgeProps,
} from '@xyflow/react';
import { X, AlertTriangle, AlertCircle } from 'lucide-react';
import { usePipelineStore } from '../../store/pipelineStore';
import { useConnectionInfo } from '../../hooks/useShapeInference';
import { CONNECTION_COLORS, type ConnectionCompatibility } from '../../engine/connectionValidator';
import type { PipelineEdge } from '../../types';

type PipelineEdgeComponentProps = EdgeProps<PipelineEdge>;

/**
 * Get the color configuration for a connection compatibility level
 */
function getConnectionColors(compatibility: ConnectionCompatibility, selected: boolean) {
  const colors = CONNECTION_COLORS[compatibility];

  if (selected) {
    return {
      stroke: colors.stroke,
      glow: colors.glow,
      dashStroke: colors.stroke,
    };
  }

  // When not selected, use slightly muted colors
  switch (compatibility) {
    case 'exact':
      return {
        stroke: '#22c55e',
        glow: 'rgba(34, 197, 94, 0.2)',
        dashStroke: '#4ade80',
      };
    case 'name_mismatch':
      return {
        stroke: '#eab308',
        glow: 'rgba(234, 179, 8, 0.2)',
        dashStroke: '#facc15',
      };
    case 'dimension_mismatch':
      return {
        stroke: '#ef4444',
        glow: 'rgba(239, 68, 68, 0.2)',
        dashStroke: '#f87171',
      };
    default:
      return {
        stroke: '#64748b',
        glow: 'rgba(100, 116, 139, 0.2)',
        dashStroke: '#94a3b8',
      };
  }
}

/**
 * Get warning icon for connection state
 */
function getWarningIcon(compatibility: ConnectionCompatibility) {
  switch (compatibility) {
    case 'name_mismatch':
      return <AlertTriangle size={12} className="text-yellow-500" />;
    case 'dimension_mismatch':
      return <AlertCircle size={12} className="text-red-500" />;
    default:
      return null;
  }
}

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
  const connectionInfo = useConnectionInfo(id);

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

  // Get the appropriate colors based on validation result
  const compatibility = connectionInfo?.compatibility || 'exact';
  const colors = useMemo(
    () => getConnectionColors(compatibility, selected || false),
    [compatibility, selected]
  );

  const edgeStyle: CSSProperties = {
    ...(style || {}),
    strokeWidth: 2,
    stroke: colors.stroke,
  };

  // Animation class based on compatibility
  const animationClass = useMemo(() => {
    if (compatibility === 'dimension_mismatch') {
      return 'pipeline-edge-error';
    }
    if (compatibility === 'name_mismatch') {
      return 'pipeline-edge-warning';
    }
    return 'pipeline-edge-flow';
  }, [compatibility]);

  // Warning message for tooltip
  const warningMessage = connectionInfo?.validation?.message;
  const warningIcon = getWarningIcon(compatibility);

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
        stroke={colors.glow}
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
        stroke={colors.dashStroke}
        strokeLinecap="round"
        strokeDasharray={compatibility === 'dimension_mismatch' ? '4 4' : '8 4'}
        className={animationClass}
      />

      {/* Edge label with delete button and warning indicator */}
      <EdgeLabelRenderer>
        <div
          className={`pipeline-edge-label ${selected || compatibility !== 'exact' ? 'visible' : ''}`}
          style={{
            transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
          }}
        >
          {/* Warning indicator */}
          {warningIcon && (
            <div
              className="pipeline-edge-warning-indicator"
              title={warningMessage}
            >
              {warningIcon}
            </div>
          )}

          {/* Delete button */}
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
