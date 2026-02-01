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

export interface ParameterDefinition {
  id: string;
  name: string;
  type: 'int' | 'float' | 'boolean' | 'select' | 'string' | 'array';
  default: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: SelectOption[] | string[];
  dtype?: string;
  description: string;
}

export interface NodeDefinition {
  id: string;
  name: string;
  description: string;
  category: string;
  color: string;
  icon: string;
  inputs: PortDefinition[];
  outputs: PortDefinition[];
  parameters: ParameterDefinition[];
}

export interface Position {
  x: number;
  y: number;
}

export interface PipelineNode {
  id: string;
  definitionId: string;
  position: Position;
  parameters: Record<string, unknown>;
}

export interface Connection {
  id: string;
  sourceNodeId: string;
  sourcePortId: string;
  targetNodeId: string;
  targetPortId: string;
}

export interface PipelineState {
  nodes: PipelineNode[];
  connections: Connection[];
}

export interface DraggedNode {
  definitionId: string;
  isNew: boolean;
  nodeId?: string;
}

export type PortType = 'input' | 'output';

export interface PortInfo {
  nodeId: string;
  portId: string;
  portType: PortType;
  position: Position;
}
