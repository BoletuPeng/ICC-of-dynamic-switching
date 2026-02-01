import type { Node, Edge, NodeProps, BuiltInNode, BuiltInEdge } from '@xyflow/react';

// =============================================================================
// Port & Parameter Definitions
// =============================================================================

export interface PortDefinition {
  id: string;
  name: string;
  type: string;
  dtype: string;
  shape: (string | number)[];
  description: string;
}

export interface SelectOption {
  value: string | number;
  label: string;
}

export type ParameterType = 'int' | 'float' | 'boolean' | 'select' | 'string' | 'array';

export interface ParameterDefinition {
  id: string;
  name: string;
  type: ParameterType;
  default: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: SelectOption[] | string[];
  dtype?: string;
  description: string;
}

// =============================================================================
// Node Definition (Template for creating nodes)
// =============================================================================

export type NodeCategory =
  | 'preprocessing'
  | 'connectivity'
  | 'community'
  | 'metrics'
  | 'analysis'
  | 'clustering'
  | 'output';

export interface NodeDefinition {
  id: string;
  name: string;
  description: string;
  category: NodeCategory;
  color: string;
  icon: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  parameters: ParameterDefinition[];
}

// =============================================================================
// React Flow Node Types
// =============================================================================

export type PipelineNodeData = {
  definitionId: string;
  label: string;
  category: NodeCategory;
  color: string;
  icon: string;
  description: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  parameters: Record<string, unknown>;
  parameterDefinitions: ParameterDefinition[];
  [key: string]: unknown; // Index signature for React Flow compatibility
};

export type PipelineNode = Node<PipelineNodeData, 'pipeline'>;

export type PipelineNodeProps = NodeProps<PipelineNode>;

// =============================================================================
// React Flow Edge Types
// =============================================================================

export type PipelineEdgeData = {
  sourcePortId: string;
  targetPortId: string;
  animated?: boolean;
  [key: string]: unknown; // Index signature for React Flow compatibility
};

export type PipelineEdge = Edge<PipelineEdgeData, 'pipeline'>;

// All node types for React Flow
export type AppNode = PipelineNode | BuiltInNode;
export type AppEdge = PipelineEdge | BuiltInEdge;

// =============================================================================
// Handle (Port) Connection
// =============================================================================

export type HandleType = 'source' | 'target';

export interface HandleInfo {
  nodeId: string;
  handleId: string;
  handleType: HandleType;
}

// =============================================================================
// UI State Types
// =============================================================================

export interface ViewportState {
  x: number;
  y: number;
  zoom: number;
}

export interface DragItem {
  type: 'node-definition';
  definitionId: string;
}

// =============================================================================
// Pipeline State (for save/load)
// =============================================================================

export interface SerializedNode {
  id: string;
  definitionId: string;
  position: { x: number; y: number };
  parameters: Record<string, unknown>;
}

export interface SerializedConnection {
  id: string;
  sourceNodeId: string;
  sourcePortId: string;
  targetNodeId: string;
  targetPortId: string;
}

export interface SerializedPipeline {
  version: string;
  name: string;
  description?: string;
  nodes: SerializedNode[];
  connections: SerializedConnection[];
  viewport?: ViewportState;
}

// =============================================================================
// Execution Types (for future backend integration)
// =============================================================================

export type ExecutionStatus = 'idle' | 'running' | 'completed' | 'error';

export interface NodeExecutionState {
  nodeId: string;
  status: ExecutionStatus;
  progress?: number;
  error?: string;
  startTime?: number;
  endTime?: number;
}

export interface PipelineExecutionState {
  status: ExecutionStatus;
  currentNodeId?: string;
  nodeStates: Record<string, NodeExecutionState>;
  startTime?: number;
  endTime?: number;
}

// =============================================================================
// Constants
// =============================================================================

export const CATEGORY_COLORS: Record<NodeCategory, string> = {
  preprocessing: '#10b981',
  connectivity: '#3b82f6',
  community: '#8b5cf6',
  metrics: '#f59e0b',
  analysis: '#ec4899',
  clustering: '#06b6d4',
  output: '#ef4444',
} as const;

export const CATEGORY_LABELS: Record<NodeCategory, string> = {
  preprocessing: 'Preprocessing',
  connectivity: 'Connectivity',
  community: 'Community Detection',
  metrics: 'Network Metrics',
  analysis: 'Analysis',
  clustering: 'Clustering',
  output: 'Output',
} as const;
