import { memo, useCallback, useRef, type DragEvent } from 'react';
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Panel,
  useReactFlow,
  type OnSelectionChangeFunc,
  type IsValidConnection,
  ConnectionMode,
  BackgroundVariant,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { nodeTypes } from './nodes';
import { edgeTypes } from './edges';
import { usePipelineStore, selectNodeCount, selectEdgeCount } from '../store/pipelineStore';
import { nodeDefinitionsMap } from '../data';
import type { PipelineNode, PipelineEdge } from '../types';

// =============================================================================
// Flow Canvas Component
// =============================================================================

const FlowCanvas = memo(function FlowCanvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();

  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode,
    selectNode,
    setEditingNode,
  } = usePipelineStore();

  const nodeCount = usePipelineStore(selectNodeCount);
  const edgeCount = usePipelineStore(selectEdgeCount);

  // Handle drag over for dropping new nodes
  const handleDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  // Handle drop to create new nodes
  const handleDrop = useCallback(
    (event: DragEvent) => {
      event.preventDefault();

      const definitionId = event.dataTransfer.getData('application/reactflow');
      if (!definitionId || !nodeDefinitionsMap[definitionId]) {
        return;
      }

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      addNode(definitionId, position);
    },
    [screenToFlowPosition, addNode]
  );

  // Handle selection changes
  const handleSelectionChange: OnSelectionChangeFunc = useCallback(
    ({ nodes: selectedNodes }) => {
      if (selectedNodes.length === 1) {
        selectNode(selectedNodes[0].id);
      } else if (selectedNodes.length === 0) {
        selectNode(null);
      }
    },
    [selectNode]
  );

  // Handle node double-click to edit
  const handleNodeDoubleClick = useCallback(
    (_event: React.MouseEvent, node: PipelineNode) => {
      setEditingNode(node.id);
    },
    [setEditingNode]
  );

  // Validate connections (only allow output -> input)
  const isValidConnection: IsValidConnection = useCallback(
    (connection) => {
      // Prevent self-connections
      if (connection.source === connection.target) {
        return false;
      }

      // Ensure we have all required connection info
      if (
        !connection.source ||
        !connection.target ||
        !connection.sourceHandle ||
        !connection.targetHandle
      ) {
        return false;
      }

      // Allow connections - store handles replacing existing connections to same target
      return true;
    },
    []
  );

  // Custom minimap node color
  const minimapNodeColor = useCallback((node: PipelineNode) => {
    return node.data?.color || '#64748b';
  }, []);

  return (
    <div
      ref={reactFlowWrapper}
      className="flow-canvas"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      <ReactFlow<PipelineNode, PipelineEdge>
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onSelectionChange={handleSelectionChange}
        onNodeDoubleClick={handleNodeDoubleClick}
        isValidConnection={isValidConnection}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        connectionMode={ConnectionMode.Loose}
        defaultEdgeOptions={{
          type: 'pipeline',
          animated: true,
        }}
        fitView
        fitViewOptions={{
          padding: 0.2,
          maxZoom: 1.5,
        }}
        minZoom={0.1}
        maxZoom={2}
        snapToGrid
        snapGrid={[15, 15]}
        deleteKeyCode={['Backspace', 'Delete']}
        multiSelectionKeyCode={['Control', 'Meta']}
        panOnScroll
        selectionOnDrag
        panOnDrag={[1, 2]} // Middle mouse button or right click
        selectNodesOnDrag={false}
        proOptions={{ hideAttribution: true }}
      >
        {/* Background */}
        <Background
          variant={BackgroundVariant.Dots}
          gap={20}
          size={1}
          color="rgba(148, 163, 184, 0.15)"
        />

        {/* Controls */}
        <Controls
          showZoom
          showFitView
          showInteractive={false}
          className="flow-controls"
        />

        {/* MiniMap */}
        <MiniMap
          nodeColor={minimapNodeColor}
          maskColor="rgba(15, 23, 42, 0.8)"
          className="flow-minimap"
          pannable
          zoomable
        />

        {/* Stats Panel */}
        <Panel position="bottom-left" className="flow-stats-panel">
          <div className="flow-stats">
            <span className="flow-stat">
              <strong>{nodeCount}</strong> nodes
            </span>
            <span className="flow-stat-divider">•</span>
            <span className="flow-stat">
              <strong>{edgeCount}</strong> connections
            </span>
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
});

// =============================================================================
// Export
// =============================================================================

export { FlowCanvas };
export default FlowCanvas;
