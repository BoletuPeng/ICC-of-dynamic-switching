/**
 * Shape Inference Hook
 *
 * Manages shape propagation through the pipeline graph.
 * Automatically recomputes shapes when nodes, connections, or parameters change.
 */

import { useMemo } from 'react';
import { usePipelineStore } from '../store/pipelineStore';
import { nodeDefinitionsMap } from '../data';
import type { PipelineNode, PipelineEdge, ResolvedPortShape, PortDefinition } from '../types';
import type {
  ShapeContext,
  ResolvedShape,
  ModuleDefinition,
  ShapeDefinition,
} from '../engine/types';
import {
  resolveShape,
  formatShapeCompact,
  parseShapeDimension,
} from '../engine';

// =============================================================================
// Types
// =============================================================================

interface NodeShapeInfo {
  inputShapes: Record<string, ResolvedPortShape>;
  outputShapes: Record<string, ResolvedPortShape>;
  dimensionBindings: Record<string, number>;
}

interface ShapeInferenceResult {
  nodeShapes: Record<string, NodeShapeInfo>;
  globalDimensions: Record<string, number>;
}

// =============================================================================
// Shape Computation
// =============================================================================

/**
 * Build dependency graph for topological sorting
 */
function buildDependencyOrder(
  nodes: PipelineNode[],
  edges: PipelineEdge[]
): string[] {
  const nodeIds = new Set(nodes.map((n) => n.id));
  const deps = new Map<string, Set<string>>();

  // Initialize all nodes
  for (const id of nodeIds) {
    deps.set(id, new Set());
  }

  // Add dependencies from edges
  for (const edge of edges) {
    if (nodeIds.has(edge.source) && nodeIds.has(edge.target)) {
      deps.get(edge.target)!.add(edge.source);
    }
  }

  // Topological sort
  const result: string[] = [];
  const visited = new Set<string>();
  const visiting = new Set<string>();

  function visit(nodeId: string) {
    if (visited.has(nodeId)) return;
    if (visiting.has(nodeId)) return; // Cycle detected, skip

    visiting.add(nodeId);

    for (const dep of deps.get(nodeId) || []) {
      visit(dep);
    }

    visiting.delete(nodeId);
    visited.add(nodeId);
    result.push(nodeId);
  }

  for (const id of nodeIds) {
    visit(id);
  }

  return result;
}

/**
 * Parse shape definition from port, handling expression strings like "n_timepoints - @window - 1"
 */
function parsePortShape(port: PortDefinition): ShapeDefinition {
  return port.shape.map((dim) => {
    if (typeof dim === 'number') {
      return dim;
    }
    // Parse string dimensions - this handles both simple variables and expressions
    return parseShapeDimension(dim);
  });
}

/**
 * Convert ResolvedShape to ResolvedPortShape for UI display
 */
function toPortShape(resolved: ResolvedShape): ResolvedPortShape {
  const result = formatShapeCompact(resolved);
  return {
    symbolic: result.symbolic,
    resolved: result.resolved,
    isFullyResolved: result.isFullyResolved,
  };
}

/**
 * Extract dimension bindings from node parameters
 */
function extractParameterBindings(
  node: PipelineNode,
  definition: ModuleDefinition | undefined
): Record<string, number> {
  const bindings: Record<string, number> = {};

  if (!definition) return bindings;

  // Check for affectsShape parameters
  for (const paramDef of definition.parameters) {
    if (paramDef.affectsShape) {
      const value = node.data.parameters[paramDef.id];
      if (typeof value === 'number' && value > 0) {
        // Use the parameter ID as the dimension name
        bindings[paramDef.id] = value;
      }
    }
  }

  return bindings;
}

/**
 * Compute shapes for all nodes in the graph
 */
function computeShapes(
  nodes: PipelineNode[],
  edges: PipelineEdge[]
): ShapeInferenceResult {
  const nodeMap = new Map(nodes.map((n) => [n.id, n]));
  const nodeShapes: Record<string, NodeShapeInfo> = {};
  const globalDimensions: Record<string, number> = {};

  // Sort nodes topologically
  const sortedIds = buildDependencyOrder(nodes, edges);

  // Process each node in order
  for (const nodeId of sortedIds) {
    const node = nodeMap.get(nodeId);
    if (!node) continue;

    const definition = nodeDefinitionsMap[node.data.definitionId] as ModuleDefinition | undefined;
    if (!definition) continue;

    // Start with parameter-based bindings
    const bindings = extractParameterBindings(node, definition);

    // Find incoming connections and gather bindings
    const incomingEdges = edges.filter((e) => e.target === nodeId);

    for (const edge of incomingEdges) {
      const sourceShapes = nodeShapes[edge.source];

      if (!sourceShapes) continue;

      const sourcePortId = edge.sourceHandle || '';
      const targetPortId = edge.targetHandle || '';

      const sourceShape = sourceShapes.outputShapes[sourcePortId];
      const targetPort = definition.inputs.find((p) => p.id === targetPortId);

      if (!sourceShape || !targetPort) continue;

      // If source shape is resolved, try to match and infer bindings
      if (sourceShape.resolved) {
        const sourceValues = sourceShape.resolved.split(' × ').map(Number);
        const targetShapeDef = parsePortShape(targetPort as unknown as PortDefinition);

        if (sourceValues.length === targetShapeDef.length) {
          for (let i = 0; i < targetShapeDef.length; i++) {
            const dim = targetShapeDef[i];
            // Only bind simple string variables, not expressions
            if (typeof dim === 'string' && !isNaN(sourceValues[i])) {
              bindings[dim] = sourceValues[i];
            }
          }
        }
      }
    }

    // Merge with global dimensions
    Object.assign(globalDimensions, bindings);
    Object.assign(bindings, globalDimensions);

    // Create context with both dimensions and parameters
    const context: ShapeContext = {
      dimensions: bindings,
      parameters: node.data.parameters,
    };

    // Resolve input shapes - parse expressions and evaluate
    const inputShapes: Record<string, ResolvedPortShape> = {};
    for (const input of node.data.inputs) {
      const shapeDef = parsePortShape(input);
      const resolved = resolveShape(shapeDef, context);
      inputShapes[input.id] = toPortShape(resolved);
    }

    // Resolve output shapes - parse expressions and evaluate
    const outputShapes: Record<string, ResolvedPortShape> = {};
    for (const output of node.data.outputs) {
      const shapeDef = parsePortShape(output);
      const resolved = resolveShape(shapeDef, context);
      outputShapes[output.id] = toPortShape(resolved);
    }

    nodeShapes[nodeId] = {
      inputShapes,
      outputShapes,
      dimensionBindings: bindings,
    };
  }

  return { nodeShapes, globalDimensions };
}

// =============================================================================
// Hook
// =============================================================================

/**
 * Hook to compute and subscribe to shape inference
 */
export function useShapeInference() {
  const nodes = usePipelineStore((state) => state.nodes);
  const edges = usePipelineStore((state) => state.edges);

  const result = useMemo(() => {
    return computeShapes(nodes, edges);
  }, [nodes, edges]);

  return result;
}

/**
 * Hook to get shape info for a specific node
 */
export function useNodeShapeInfo(nodeId: string): NodeShapeInfo | null {
  const { nodeShapes } = useShapeInference();
  return nodeShapes[nodeId] || null;
}

/**
 * Selector for getting shape-enhanced node data
 */
export function selectNodeWithShapes(nodeId: string) {
  return (state: { nodes: PipelineNode[]; edges: PipelineEdge[] }) => {
    const node = state.nodes.find((n) => n.id === nodeId);
    if (!node) return null;

    const { nodeShapes } = computeShapes(state.nodes, state.edges);
    const shapeInfo = nodeShapes[nodeId];

    if (!shapeInfo) return node;

    return {
      ...node,
      data: {
        ...node.data,
        resolvedInputShapes: shapeInfo.inputShapes,
        resolvedOutputShapes: shapeInfo.outputShapes,
        dimensionBindings: shapeInfo.dimensionBindings,
      },
    };
  };
}

export default useShapeInference;
