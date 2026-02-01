import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import {
  applyNodeChanges,
  applyEdgeChanges,
  type NodeChange,
  type EdgeChange,
  type Connection,
  type XYPosition,
} from '@xyflow/react';
import type {
  PipelineNode,
  PipelineEdge,
  PipelineNodeData,
  PipelineEdgeData,
  NodeCategory,
  ViewportState,
  SerializedPipeline,
  SerializedNode,
  SerializedConnection,
} from '../types';
import { getNodeDefinition, getDefaultParameters } from '../data';
import { demoPipelineNodes, demoPipelineConnections } from '../data/demoPipeline';

// =============================================================================
// Store Types
// =============================================================================

interface PipelineStore {
  // State
  nodes: PipelineNode[];
  edges: PipelineEdge[];
  selectedNodeId: string | null;
  editingNodeId: string | null;
  viewport: ViewportState;

  // React Flow event handlers
  onNodesChange: (changes: NodeChange<PipelineNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<PipelineEdge>[]) => void;
  onConnect: (connection: Connection) => void;

  // Node operations
  addNode: (definitionId: string, position: XYPosition) => string;
  removeNode: (nodeId: string) => void;
  updateNodeParameters: (nodeId: string, parameters: Record<string, unknown>) => void;
  duplicateNode: (nodeId: string) => string | null;

  // Edge operations
  removeEdge: (edgeId: string) => void;

  // Selection
  selectNode: (nodeId: string | null) => void;
  setEditingNode: (nodeId: string | null) => void;

  // Viewport
  setViewport: (viewport: ViewportState) => void;

  // Pipeline operations
  clearPipeline: () => void;
  loadPipeline: (pipeline: SerializedPipeline) => void;
  exportPipeline: () => SerializedPipeline;
  loadDemoPipeline: () => void;
}

// =============================================================================
// Helper Functions
// =============================================================================

let nodeIdCounter = 100;

function generateNodeId(): string {
  return `node-${++nodeIdCounter}`;
}

function generateEdgeId(source: string, sourceHandle: string, target: string, targetHandle: string): string {
  return `edge-${source}-${sourceHandle}-${target}-${targetHandle}`;
}

function createPipelineNode(
  definitionId: string,
  position: XYPosition,
  id?: string,
  parameters?: Record<string, unknown>
): PipelineNode {
  const definition = getNodeDefinition(definitionId);
  const nodeId = id || generateNodeId();

  const data: PipelineNodeData = {
    definitionId,
    label: definition.name,
    category: definition.category as NodeCategory,
    color: definition.color,
    icon: definition.icon,
    description: definition.description,
    inputs: definition.inputs,
    outputs: definition.outputs,
    parameters: parameters || getDefaultParameters(definition),
    parameterDefinitions: definition.parameters,
  };

  return {
    id: nodeId,
    type: 'pipeline',
    position,
    data,
    selected: false,
  };
}

function createPipelineEdge(
  source: string,
  sourceHandle: string,
  target: string,
  targetHandle: string,
  id?: string
): PipelineEdge {
  const edgeData: PipelineEdgeData = {
    sourcePortId: sourceHandle,
    targetPortId: targetHandle,
    animated: true,
  };

  return {
    id: id || generateEdgeId(source, sourceHandle, target, targetHandle),
    source,
    target,
    sourceHandle,
    targetHandle,
    type: 'pipeline',
    data: edgeData,
  };
}

// Convert demo pipeline to React Flow format
function convertDemoPipeline(): { nodes: PipelineNode[]; edges: PipelineEdge[] } {
  const nodes = demoPipelineNodes.map((node: SerializedNode) =>
    createPipelineNode(node.definitionId, node.position, node.id, node.parameters)
  );

  const edges = demoPipelineConnections.map((conn: SerializedConnection) =>
    createPipelineEdge(
      conn.sourceNodeId,
      conn.sourcePortId,
      conn.targetNodeId,
      conn.targetPortId,
      conn.id
    )
  );

  return { nodes, edges };
}

// Convert from serialized format
function deserializePipeline(pipeline: SerializedPipeline): { nodes: PipelineNode[]; edges: PipelineEdge[] } {
  const nodes = pipeline.nodes.map((node: SerializedNode) =>
    createPipelineNode(node.definitionId, node.position, node.id, node.parameters)
  );

  const edges = pipeline.connections.map((conn: SerializedConnection) =>
    createPipelineEdge(
      conn.sourceNodeId,
      conn.sourcePortId,
      conn.targetNodeId,
      conn.targetPortId,
      conn.id
    )
  );

  return { nodes, edges };
}

// =============================================================================
// Initial State
// =============================================================================

const initialData = convertDemoPipeline();

// =============================================================================
// Store Implementation
// =============================================================================

export const usePipelineStore = create<PipelineStore>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    nodes: initialData.nodes,
    edges: initialData.edges,
    selectedNodeId: null,
    editingNodeId: null,
    viewport: { x: 0, y: 0, zoom: 1 },

    // React Flow event handlers
    onNodesChange: (changes) => {
      set({
        nodes: applyNodeChanges(changes, get().nodes) as PipelineNode[],
      });

      // Handle selection changes
      const selectChange = changes.find(
        (c) => c.type === 'select' && c.selected
      );
      if (selectChange && 'id' in selectChange) {
        set({ selectedNodeId: selectChange.id });
      }
    },

    onEdgesChange: (changes) => {
      set({
        edges: applyEdgeChanges(changes, get().edges) as PipelineEdge[],
      });
    },

    onConnect: (connection) => {
      if (!connection.source || !connection.target || !connection.sourceHandle || !connection.targetHandle) {
        return;
      }

      const { edges } = get();

      // Check if connection already exists
      const exists = edges.some(
        (e) =>
          e.source === connection.source &&
          e.sourceHandle === connection.sourceHandle &&
          e.target === connection.target &&
          e.targetHandle === connection.targetHandle
      );

      if (exists) return;

      // Remove existing connections to the same target handle (input can only have one connection)
      const filteredEdges = edges.filter(
        (e) => !(e.target === connection.target && e.targetHandle === connection.targetHandle)
      );

      const newEdge = createPipelineEdge(
        connection.source,
        connection.sourceHandle,
        connection.target,
        connection.targetHandle
      );

      set({ edges: [...filteredEdges, newEdge] });
    },

    // Node operations
    addNode: (definitionId, position) => {
      const node = createPipelineNode(definitionId, position);
      set({ nodes: [...get().nodes, node] });
      return node.id;
    },

    removeNode: (nodeId) => {
      const { nodes, edges, selectedNodeId, editingNodeId } = get();

      set({
        nodes: nodes.filter((n) => n.id !== nodeId),
        edges: edges.filter((e) => e.source !== nodeId && e.target !== nodeId),
        selectedNodeId: selectedNodeId === nodeId ? null : selectedNodeId,
        editingNodeId: editingNodeId === nodeId ? null : editingNodeId,
      });
    },

    updateNodeParameters: (nodeId, parameters) => {
      set({
        nodes: get().nodes.map((node) =>
          node.id === nodeId
            ? {
                ...node,
                data: {
                  ...node.data,
                  parameters: { ...node.data.parameters, ...parameters },
                },
              }
            : node
        ),
      });
    },

    duplicateNode: (nodeId) => {
      const node = get().nodes.find((n) => n.id === nodeId);
      if (!node) return null;

      const newNode = createPipelineNode(
        node.data.definitionId,
        { x: node.position.x + 50, y: node.position.y + 50 },
        undefined,
        { ...node.data.parameters }
      );

      set({ nodes: [...get().nodes, newNode] });
      return newNode.id;
    },

    // Edge operations
    removeEdge: (edgeId) => {
      set({
        edges: get().edges.filter((e) => e.id !== edgeId),
      });
    },

    // Selection
    selectNode: (nodeId) => {
      const { nodes } = get();
      set({
        selectedNodeId: nodeId,
        nodes: nodes.map((n) => ({
          ...n,
          selected: n.id === nodeId,
        })),
      });
    },

    setEditingNode: (nodeId) => {
      set({ editingNodeId: nodeId });
    },

    // Viewport
    setViewport: (viewport) => {
      set({ viewport });
    },

    // Pipeline operations
    clearPipeline: () => {
      set({
        nodes: [],
        edges: [],
        selectedNodeId: null,
        editingNodeId: null,
      });
    },

    loadPipeline: (pipeline) => {
      const { nodes, edges } = deserializePipeline(pipeline);
      set({
        nodes,
        edges,
        selectedNodeId: null,
        editingNodeId: null,
        viewport: pipeline.viewport || { x: 0, y: 0, zoom: 1 },
      });
    },

    exportPipeline: () => {
      const { nodes, edges, viewport } = get();

      const serializedNodes: SerializedNode[] = nodes.map((node) => ({
        id: node.id,
        definitionId: node.data.definitionId,
        position: node.position,
        parameters: node.data.parameters,
      }));

      const serializedConnections: SerializedConnection[] = edges.map((edge) => ({
        id: edge.id,
        sourceNodeId: edge.source,
        sourcePortId: edge.sourceHandle || '',
        targetNodeId: edge.target,
        targetPortId: edge.targetHandle || '',
      }));

      return {
        version: '1.0.0',
        name: 'Pipeline',
        nodes: serializedNodes,
        connections: serializedConnections,
        viewport,
      };
    },

    loadDemoPipeline: () => {
      const { nodes, edges } = convertDemoPipeline();
      set({
        nodes,
        edges,
        selectedNodeId: null,
        editingNodeId: null,
      });
    },
  }))
);

// =============================================================================
// Selectors
// =============================================================================

export const selectNodeById = (nodeId: string) => (state: PipelineStore) =>
  state.nodes.find((n) => n.id === nodeId);

export const selectSelectedNode = (state: PipelineStore) =>
  state.selectedNodeId ? state.nodes.find((n) => n.id === state.selectedNodeId) : null;

export const selectEditingNode = (state: PipelineStore) =>
  state.editingNodeId ? state.nodes.find((n) => n.id === state.editingNodeId) : null;

export const selectNodeCount = (state: PipelineStore) => state.nodes.length;

export const selectEdgeCount = (state: PipelineStore) => state.edges.length;
