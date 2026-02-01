/**
 * Shape Propagation Engine
 *
 * Propagates shape information through the pipeline graph:
 * 1. Collects known values from source nodes
 * 2. Infers variable bindings at connection points
 * 3. Computes output shapes based on input shapes and parameters
 * 4. Handles For/Stack control flow nodes
 */

import type {
  ShapeContext,
  ResolvedShape,
  ShapeConnection,
  NodeShapeState,
  ShapePropagationState,
  ModuleDefinition,
} from './types';
import { resolveShape, matchShapes, extractShapeVariables } from './shapeResolver';

// =============================================================================
// Node Shape State Management
// =============================================================================

/**
 * Create initial shape state for a node
 */
export function createNodeShapeState(
  nodeId: string,
  definition: ModuleDefinition,
  parameters: Record<string, unknown>
): NodeShapeState {
  const context: ShapeContext = {
    dimensions: {},
    parameters,
  };

  // Initially resolve shapes with empty context
  const inputShapes: Record<string, ResolvedShape> = {};
  const outputShapes: Record<string, ResolvedShape> = {};

  for (const input of definition.inputs) {
    inputShapes[input.id] = resolveShape(input.shape, context);
  }

  for (const output of definition.outputs) {
    outputShapes[output.id] = resolveShape(output.shape, context);
  }

  const isFullyResolved =
    Object.values(inputShapes).every((s) => s.every((d) => d.isResolved)) &&
    Object.values(outputShapes).every((s) => s.every((d) => d.isResolved));

  return {
    nodeId,
    definitionId: definition.id,
    context,
    inputShapes,
    outputShapes,
    isFullyResolved,
  };
}

/**
 * Update a node's shape state with new dimension bindings
 */
export function updateNodeShapeState(
  state: NodeShapeState,
  definition: ModuleDefinition,
  newBindings: Record<string, number>
): NodeShapeState {
  // Merge new bindings with existing context
  const context: ShapeContext = {
    dimensions: { ...state.context.dimensions, ...newBindings },
    parameters: state.context.parameters,
  };

  // Re-resolve all shapes with updated context
  const inputShapes: Record<string, ResolvedShape> = {};
  const outputShapes: Record<string, ResolvedShape> = {};

  for (const input of definition.inputs) {
    inputShapes[input.id] = resolveShape(input.shape, context);
  }

  for (const output of definition.outputs) {
    outputShapes[output.id] = resolveShape(output.shape, context);
  }

  const isFullyResolved =
    Object.values(inputShapes).every((s) => s.every((d) => d.isResolved)) &&
    Object.values(outputShapes).every((s) => s.every((d) => d.isResolved));

  return {
    ...state,
    context,
    inputShapes,
    outputShapes,
    isFullyResolved,
  };
}

// =============================================================================
// Graph Propagation
// =============================================================================

/**
 * Build a dependency graph from connections
 */
function buildDependencyGraph(
  connections: ShapeConnection[]
): Map<string, Set<string>> {
  const deps = new Map<string, Set<string>>();

  for (const conn of connections) {
    if (!deps.has(conn.targetNodeId)) {
      deps.set(conn.targetNodeId, new Set());
    }
    deps.get(conn.targetNodeId)!.add(conn.sourceNodeId);
  }

  return deps;
}

/**
 * Topological sort of nodes based on connections
 */
function topologicalSort(
  nodeIds: string[],
  connections: ShapeConnection[]
): string[] {
  const deps = buildDependencyGraph(connections);
  const visited = new Set<string>();
  const result: string[] = [];

  function visit(nodeId: string) {
    if (visited.has(nodeId)) return;
    visited.add(nodeId);

    const nodeDeps = deps.get(nodeId) || new Set();
    for (const dep of nodeDeps) {
      visit(dep);
    }

    result.push(nodeId);
  }

  for (const nodeId of nodeIds) {
    visit(nodeId);
  }

  return result;
}

/**
 * Propagate shapes through the entire graph
 */
export function propagateShapes(
  nodes: Record<string, NodeShapeState>,
  connections: ShapeConnection[],
  definitions: Record<string, ModuleDefinition>
): ShapePropagationState {
  // Create a mutable copy
  const updatedNodes: Record<string, NodeShapeState> = {};
  for (const [id, state] of Object.entries(nodes)) {
    updatedNodes[id] = { ...state };
  }

  // Sort nodes topologically
  const sortedIds = topologicalSort(Object.keys(nodes), connections);

  // Track if we made progress (for iterative refinement)
  let madeProgress = true;
  let iterations = 0;
  const maxIterations = 10; // Prevent infinite loops

  while (madeProgress && iterations < maxIterations) {
    madeProgress = false;
    iterations++;

    // Process nodes in topological order
    for (const nodeId of sortedIds) {
      const nodeState = updatedNodes[nodeId];
      if (!nodeState) continue;

      const definition = definitions[nodeState.definitionId];
      if (!definition) continue;

      // Find all incoming connections
      const incomingConns = connections.filter((c) => c.targetNodeId === nodeId);

      // Collect bindings from all incoming connections
      let allBindings: Record<string, number> = { ...nodeState.context.dimensions };

      for (const conn of incomingConns) {
        const sourceState = updatedNodes[conn.sourceNodeId];
        if (!sourceState) continue;

        const sourceShape = sourceState.outputShapes[conn.sourcePortId];
        if (!sourceShape) continue;

        // Find target port definition
        const targetPort = definition.inputs.find((p) => p.id === conn.targetPortId);
        if (!targetPort) continue;

        // Match shapes to infer bindings
        const matchResult = matchShapes(sourceShape, targetPort.shape, allBindings);

        if (matchResult.isCompatible) {
          // Check if we have new bindings
          for (const [key, value] of Object.entries(matchResult.bindings)) {
            if (allBindings[key] === undefined) {
              allBindings[key] = value;
              madeProgress = true;
            }
          }
        }
      }

      // If we have new bindings, update the node
      const hasNewBindings = Object.keys(allBindings).some(
        (k) => nodeState.context.dimensions[k] !== allBindings[k]
      );

      if (hasNewBindings) {
        updatedNodes[nodeId] = updateNodeShapeState(nodeState, definition, allBindings);
        madeProgress = true;
      }
    }
  }

  return {
    nodes: updatedNodes,
    connections,
  };
}

// =============================================================================
// Shape Display Helpers
// =============================================================================

/**
 * Get the effective shape for a port, considering incoming connections
 */
export function getEffectiveInputShape(
  nodeState: NodeShapeState,
  portId: string,
  connections: ShapeConnection[],
  allNodes: Record<string, NodeShapeState>
): ResolvedShape | null {
  // Find incoming connection to this port
  const conn = connections.find(
    (c) => c.targetNodeId === nodeState.nodeId && c.targetPortId === portId
  );

  if (!conn) {
    // No connection, return the port's own shape
    return nodeState.inputShapes[portId] || null;
  }

  // Get source node's output shape
  const sourceState = allNodes[conn.sourceNodeId];
  if (!sourceState) return null;

  return sourceState.outputShapes[conn.sourcePortId] || null;
}

/**
 * Collect all resolved dimension values from a graph state
 */
export function collectAllDimensions(
  state: ShapePropagationState
): Record<string, number> {
  const all: Record<string, number> = {};

  for (const nodeState of Object.values(state.nodes)) {
    Object.assign(all, nodeState.context.dimensions);
  }

  return all;
}

// =============================================================================
// For/Stack Special Handling
// =============================================================================

/**
 * Special handling for For loop nodes
 * For loops "unwrap" outer dimensions, making inner processing work on slices
 */
export function handleForLoopShape(
  inputShape: ResolvedShape,
  iterateDimensions: number // How many outer dimensions to iterate
): ResolvedShape {
  // Remove the first N dimensions
  return inputShape.slice(iterateDimensions);
}

/**
 * Special handling for Stack nodes
 * Stack "wraps" outputs back into the original dimensions
 */
export function handleStackShape(
  innerShape: ResolvedShape,
  stackDimensions: ResolvedShape // The dimensions to prepend
): ResolvedShape {
  return [...stackDimensions, ...innerShape];
}

/**
 * Get required dimension variables for a module
 */
export function getRequiredVariables(definition: ModuleDefinition): string[] {
  const inputVars = definition.inputs.flatMap((p) => extractShapeVariables(p.shape));
  const outputVars = definition.outputs.flatMap((p) => extractShapeVariables(p.shape));
  return [...new Set([...inputVars, ...outputVars])];
}
