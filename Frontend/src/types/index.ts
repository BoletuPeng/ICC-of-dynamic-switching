import type { Node, Edge, NodeProps } from '@xyflow/react';

// =============================================================================
// Core Definitions
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

export type NodeCategory =
  | 'input' | 'preprocessing' | 'connectivity' | 'community'
  | 'metrics' | 'analysis' | 'clustering' | 'output' | 'control';

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
// React Flow Types
// =============================================================================

/**
 * Resolved shape information for display
 */
export interface ResolvedPortShape {
  /** Original symbolic shape definition */
  symbolic: string;
  /** Resolved numeric values (if known) */
  resolved: string | null;
  /** Whether all dimensions are resolved */
  isFullyResolved: boolean;
}

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

  /** Resolved input shapes (computed by shape engine) */
  resolvedInputShapes?: Record<string, ResolvedPortShape>;
  /** Resolved output shapes (computed by shape engine) */
  resolvedOutputShapes?: Record<string, ResolvedPortShape>;
  /** Known dimension bindings for this node */
  dimensionBindings?: Record<string, number>;

  /** Control flow specific fields */
  isControlFlow?: boolean;
  controlFlowType?: 'for' | 'stack';

  [key: string]: unknown;
};

export type PipelineNode = Node<PipelineNodeData, 'pipeline'>;
export type PipelineNodeProps = NodeProps<PipelineNode>;

export type PipelineEdgeData = {
  sourcePortId: string;
  targetPortId: string;
  animated?: boolean;
  [key: string]: unknown;
};

export type PipelineEdge = Edge<PipelineEdgeData, 'pipeline'>;

// =============================================================================
// Serialization
// =============================================================================

export interface ViewportState {
  x: number;
  y: number;
  zoom: number;
}

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
// Constants
// =============================================================================

export const CATEGORY_COLORS: Record<NodeCategory, string> = {
  input: '#22c55e',
  preprocessing: '#10b981',
  connectivity: '#3b82f6',
  community: '#8b5cf6',
  metrics: '#f59e0b',
  analysis: '#ec4899',
  clustering: '#06b6d4',
  output: '#ef4444',
  control: '#6366f1',
};

export const CATEGORY_LABELS: Record<NodeCategory, string> = {
  input: 'Input',
  preprocessing: 'Preprocessing',
  connectivity: 'Connectivity',
  community: 'Community Detection',
  metrics: 'Network Metrics',
  analysis: 'Analysis',
  clustering: 'Clustering',
  output: 'Output',
  control: 'Control Flow',
};
