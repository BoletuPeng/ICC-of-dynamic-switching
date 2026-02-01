import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';
import type { PipelineNode, Connection, Position } from '../types';
import { demoPipelineNodes, demoPipelineConnections } from '../data/demoPipeline';

interface PipelineStore {
  nodes: PipelineNode[];
  connections: Connection[];
  selectedNodeId: string | null;
  editingNodeId: string | null;

  // Node actions
  addNode: (definitionId: string, position: Position) => string;
  removeNode: (nodeId: string) => void;
  updateNodePosition: (nodeId: string, position: Position) => void;
  updateNodeParameters: (nodeId: string, parameters: Record<string, unknown>) => void;

  // Connection actions
  addConnection: (connection: Omit<Connection, 'id'>) => void;
  removeConnection: (connectionId: string) => void;
  removeConnectionsForNode: (nodeId: string) => void;

  // Selection
  selectNode: (nodeId: string | null) => void;
  setEditingNode: (nodeId: string | null) => void;

  // Utility
  clearPipeline: () => void;
  loadPipeline: (nodes: PipelineNode[], connections: Connection[]) => void;
}

export const usePipelineStore = create<PipelineStore>((set, get) => ({
  nodes: demoPipelineNodes,
  connections: demoPipelineConnections,
  selectedNodeId: null,
  editingNodeId: null,

  addNode: (definitionId, position) => {
    const id = uuidv4();
    const newNode: PipelineNode = {
      id,
      definitionId,
      position,
      parameters: {},
    };
    set((state) => ({
      nodes: [...state.nodes, newNode],
    }));
    return id;
  },

  removeNode: (nodeId) => {
    get().removeConnectionsForNode(nodeId);
    set((state) => ({
      nodes: state.nodes.filter((n) => n.id !== nodeId),
      selectedNodeId: state.selectedNodeId === nodeId ? null : state.selectedNodeId,
      editingNodeId: state.editingNodeId === nodeId ? null : state.editingNodeId,
    }));
  },

  updateNodePosition: (nodeId, position) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, position } : n
      ),
    }));
  },

  updateNodeParameters: (nodeId, parameters) => {
    set((state) => ({
      nodes: state.nodes.map((n) =>
        n.id === nodeId ? { ...n, parameters: { ...n.parameters, ...parameters } } : n
      ),
    }));
  },

  addConnection: (connection) => {
    const id = uuidv4();
    // Check if connection already exists
    const exists = get().connections.some(
      (c) =>
        c.sourceNodeId === connection.sourceNodeId &&
        c.sourcePortId === connection.sourcePortId &&
        c.targetNodeId === connection.targetNodeId &&
        c.targetPortId === connection.targetPortId
    );
    if (!exists) {
      // Remove existing connections to the same target port
      set((state) => ({
        connections: [
          ...state.connections.filter(
            (c) =>
              !(c.targetNodeId === connection.targetNodeId &&
                c.targetPortId === connection.targetPortId)
          ),
          { id, ...connection },
        ],
      }));
    }
  },

  removeConnection: (connectionId) => {
    set((state) => ({
      connections: state.connections.filter((c) => c.id !== connectionId),
    }));
  },

  removeConnectionsForNode: (nodeId) => {
    set((state) => ({
      connections: state.connections.filter(
        (c) => c.sourceNodeId !== nodeId && c.targetNodeId !== nodeId
      ),
    }));
  },

  selectNode: (nodeId) => {
    set({ selectedNodeId: nodeId });
  },

  setEditingNode: (nodeId) => {
    set({ editingNodeId: nodeId });
  },

  clearPipeline: () => {
    set({ nodes: [], connections: [], selectedNodeId: null, editingNodeId: null });
  },

  loadPipeline: (nodes, connections) => {
    set({ nodes, connections, selectedNodeId: null, editingNodeId: null });
  },
}));
