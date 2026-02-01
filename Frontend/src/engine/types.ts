/**
 * Shape Inference Engine - Core Types
 *
 * This module defines the type system for shape inference in the data flow pipeline.
 * It supports symbolic shapes like "n_timepoints" and expressions like "n_timepoints - window - 1".
 */

// =============================================================================
// Shape Expression Types
// =============================================================================

/**
 * A dimension can be:
 * - A concrete number (e.g., 106)
 * - A symbolic variable (e.g., "n_timepoints")
 * - An expression object for complex calculations
 */
export type ShapeDimension = number | string | ShapeExpression;

/**
 * A shape expression for complex dimension calculations
 * Examples:
 *   { op: 'sub', left: 'n_timepoints', right: 'window' }
 *   { op: 'sub', left: { op: 'sub', left: 'n_timepoints', right: 'window' }, right: 1 }
 */
export interface ShapeExpression {
  op: 'add' | 'sub' | 'mul' | 'div' | 'ref';
  left?: ShapeDimension;
  right?: ShapeDimension;
  // For 'ref' operation - reference to a parameter
  param?: string;
}

/**
 * A complete shape definition for a port
 */
export type ShapeDefinition = ShapeDimension[];

// =============================================================================
// Resolved Shape Types
// =============================================================================

/**
 * A resolved dimension that may or may not have a concrete value
 */
export interface ResolvedDimension {
  /** The original symbolic expression */
  symbolic: string;
  /** The resolved numeric value, if known */
  value?: number;
  /** Whether this dimension is fully resolved */
  isResolved: boolean;
}

/**
 * A fully resolved shape with all dimensions evaluated where possible
 */
export type ResolvedShape = ResolvedDimension[];

// =============================================================================
// Shape Context Types
// =============================================================================

/**
 * Context for shape resolution containing known variable values
 */
export interface ShapeContext {
  /** Known dimension variables (e.g., n_timepoints -> 242) */
  dimensions: Record<string, number>;
  /** Parameter values from the node configuration */
  parameters: Record<string, unknown>;
}

/**
 * Binding information for connecting shapes between ports
 */
export interface ShapeBinding {
  /** The source port's resolved shape */
  sourceShape: ResolvedShape;
  /** Variable bindings inferred from the connection */
  bindings: Record<string, number>;
}

// =============================================================================
// Module Definition Types (Extended)
// =============================================================================

/**
 * Extended port definition with rich shape information
 */
export interface ModulePortDefinition {
  id: string;
  name: string;
  type: string;
  dtype: string;
  /** Shape definition using symbolic expressions */
  shape: ShapeDefinition;
  description: string;
}

/**
 * Parameter definition with optional shape influence
 */
export interface ModuleParameterDefinition {
  id: string;
  name: string;
  type: 'int' | 'float' | 'boolean' | 'select' | 'string' | 'array';
  default: unknown;
  min?: number;
  max?: number;
  step?: number;
  options?: Array<{ value: string | number; label: string }> | string[];
  dtype?: string;
  description: string;
  /** Whether this parameter affects output shape */
  affectsShape?: boolean;
}

/**
 * Extended node/module definition
 */
export interface ModuleDefinition {
  id: string;
  name: string;
  description: string;
  category: string;
  color: string;
  icon: string;

  inputs: ModulePortDefinition[];
  outputs: ModulePortDefinition[];
  parameters: ModuleParameterDefinition[];

  /** Optional: Explicit shape transformation rules */
  shapeRules?: ShapeRule[];
}

/**
 * Explicit shape transformation rule
 */
export interface ShapeRule {
  /** Target output port ID */
  outputPort: string;
  /** Expression defining how each dimension is computed */
  dimensions: ShapeDimension[];
}

// =============================================================================
// Control Flow Types (For/Stack)
// =============================================================================

/**
 * For loop configuration - unpacks dimensions
 */
export interface ForLoopConfig {
  /** Dimensions to iterate over (from outer to inner) */
  iterateDimensions: string[];
  /** Variable names for iteration indices */
  indexVariables: string[];
}

/**
 * Stack configuration - repacks dimensions
 */
export interface StackConfig {
  /** Dimensions to stack (from outer to inner) */
  stackDimensions: string[];
  /** Expected values for each dimension */
  expectedSizes: Record<string, number | string>;
}

// =============================================================================
// Graph Types for Shape Propagation
// =============================================================================

/**
 * Node state in the shape propagation graph
 */
export interface NodeShapeState {
  nodeId: string;
  definitionId: string;

  /** Current context with known values */
  context: ShapeContext;

  /** Resolved input shapes */
  inputShapes: Record<string, ResolvedShape>;

  /** Resolved output shapes */
  outputShapes: Record<string, ResolvedShape>;

  /** Whether all shapes are fully resolved */
  isFullyResolved: boolean;
}

/**
 * Connection for shape propagation
 */
export interface ShapeConnection {
  sourceNodeId: string;
  sourcePortId: string;
  targetNodeId: string;
  targetPortId: string;
}

/**
 * Complete shape propagation graph state
 */
export interface ShapePropagationState {
  nodes: Record<string, NodeShapeState>;
  connections: ShapeConnection[];
}
