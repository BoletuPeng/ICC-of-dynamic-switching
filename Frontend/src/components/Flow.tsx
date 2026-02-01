import { memo, useCallback, useRef } from 'react';
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
import { useNodeDrag } from '../hooks';
import { usePipelineStore, selectNodeCount, selectEdgeCount } from '../store/pipelineStore';
import { nodeDefinitionsMap } from '../data';
import type { PipelineNode, PipelineEdge } from '../types';

const FlowCanvas = memo(function FlowCanvas() {
  const reactFlowWrapper = useRef<HTMLDivElement>(null);
  const { screenToFlowPosition } = useReactFlow();
  const { handleDragOver, getDroppedDefinitionId } = useNodeDrag();

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

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      const definitionId = getDroppedDefinitionId(event);
      if (!definitionId || !nodeDefinitionsMap[definitionId]) return;
      const position = screenToFlowPosition({ x: event.clientX, y: event.clientY });
      addNode(definitionId, position);
    },
    [screenToFlowPosition, addNode, getDroppedDefinitionId]
  );

  const handleSelectionChange: OnSelectionChangeFunc = useCallback(
    ({ nodes: selectedNodes }) => {
      selectNode(selectedNodes.length === 1 ? selectedNodes[0].id : null);
    },
    [selectNode]
  );

  const handleNodeDoubleClick = useCallback(
    (_: React.MouseEvent, node: PipelineNode) => setEditingNode(node.id),
    [setEditingNode]
  );

  const isValidConnection: IsValidConnection = useCallback(
    (conn) =>
      conn.source !== conn.target &&
      !!conn.source &&
      !!conn.target &&
      !!conn.sourceHandle &&
      !!conn.targetHandle,
    []
  );

  const minimapNodeColor = useCallback(
    (node: PipelineNode) => node.data?.color || '#64748b',
    []
  );

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
        defaultEdgeOptions={{ type: 'pipeline', animated: true }}
        fitView
        fitViewOptions={{ padding: 0.2, maxZoom: 1.5 }}
        minZoom={0.1}
        maxZoom={2}
        snapToGrid
        snapGrid={[15, 15]}
        deleteKeyCode={['Backspace', 'Delete']}
        multiSelectionKeyCode={['Control', 'Meta']}
        panOnScroll
        selectionOnDrag
        panOnDrag={[1, 2]}
        selectNodesOnDrag={false}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="rgba(148, 163, 184, 0.15)" />
        <Controls showZoom showFitView showInteractive={false} className="flow-controls" />
        <MiniMap nodeColor={minimapNodeColor} maskColor="rgba(15, 23, 42, 0.8)" className="flow-minimap" pannable zoomable />
        <Panel position="bottom-left" className="flow-stats-panel">
          <div className="flow-stats">
            <span><strong>{nodeCount}</strong> nodes</span>
            <span className="flow-stat-divider">•</span>
            <span><strong>{edgeCount}</strong> connections</span>
          </div>
        </Panel>
      </ReactFlow>
    </div>
  );
});

export { FlowCanvas };
export default FlowCanvas;
