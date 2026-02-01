/**
 * Shape Inference Hook
 *
 * Manages shape propagation through the pipeline graph with support for:
 * - Dynamic shape computation for control flow nodes (For Loop, Stack)
 * - Connection validation with compatibility levels
 * - Loop Id Ref metadata propagation
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
  LoopIdRefData,
} from '../engine/types';
import {
  resolveShape,
  formatShapeCompact,
  parseShapeDimension,
} from '../engine';
import {
  validateConnection,
  type ConnectionCompatibility,
  type ConnectionValidationResult,
} from '../engine/connectionValidator';
import {
  computeForLoopOutputs,
  computeStackOutput,
  isForLoopNode,
  isStackNode,
  formatLoopIdRefDisplay,
} from '../engine/controlFlowHandler';

// =============================================================================
// Types
// =============================================================================

export interface NodeShapeInfo {
  inputShapes: Record<string, ResolvedPortShape>;
  outputShapes: Record<string, ResolvedPortShape>;
  dimensionBindings: Record<string, number>;
  /** Loop Id Ref data for For Loop nodes */
  loopIdRefData?: LoopIdRefData;
}

export interface ConnectionInfo {
  edgeId: string;
  sourceNodeId: string;
  sourcePortId: string;
  targetNodeId: string;
  targetPortId: string;
  compatibility: ConnectionCompatibility;
  validation: ConnectionValidationResult;
}

export interface ShapeInferenceResult {
  nodeShapes: Record<string, NodeShapeInfo>;
  globalDimensions: Record<string, number>;
  connectionValidations: Record<string, ConnectionInfo>;
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
 * Parse shape definition from port, handling expression strings
 */
function parsePortShape(port: PortDefinition): ShapeDefinition {
  return port.shape.map((dim) => {
    if (typeof dim === 'number') {
      return dim;
    }
    // Handle dynamic shape marker
    if (dim === '...') {
      return dim;
    }
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

  for (const paramDef of definition.parameters) {
    if (paramDef.affectsShape) {
      const value = node.data.parameters[paramDef.id];
      if (typeof value === 'number' && value > 0) {
        bindings[paramDef.id] = value;
      }
    }
  }

  return bindings;
}

/**
 * Get incoming shape for a specific port
 */
function getIncomingShape(
  nodeId: string,
  portId: string,
  edges: PipelineEdge[],
  nodeShapes: Record<string, NodeShapeInfo>
): ResolvedShape | undefined {
  const edge = edges.find(
    e => e.target === nodeId && e.targetHandle === portId
  );

  if (!edge) return undefined;

  const sourceShapeInfo = nodeShapes[edge.source];
  if (!sourceShapeInfo) return undefined;

  const sourcePortShape = sourceShapeInfo.outputShapes[edge.sourceHandle || ''];
  if (!sourcePortShape || !sourcePortShape.resolved) return undefined;

  // Convert ResolvedPortShape back to ResolvedShape
  const parts = sourcePortShape.resolved.split(' × ');
  const symbolicParts = sourcePortShape.symbolic.split(' × ');

  return parts.map((value, i) => ({
    symbolic: symbolicParts[i] || value,
    value: parseInt(value, 10),
    isResolved: !isNaN(parseInt(value, 10)),
  }));
}

/**
 * Get Loop Id Ref data from incoming connection
 */
function getIncomingLoopIdRef(
  nodeId: string,
  portId: string,
  edges: PipelineEdge[],
  nodeShapes: Record<string, NodeShapeInfo>
): LoopIdRefData | undefined {
  const edge = edges.find(
    e => e.target === nodeId && e.targetHandle === portId
  );

  if (!edge) return undefined;

  const sourceShapeInfo = nodeShapes[edge.source];
  return sourceShapeInfo?.loopIdRefData;
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
  const connectionValidations: Record<string, ConnectionInfo> = {};

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

      // Skip dynamic shape ports for binding extraction
      if (targetPort.shape.includes('...')) continue;

      // If source shape is resolved, try to match and infer bindings
      if (sourceShape.resolved) {
        const sourceValues = sourceShape.resolved.split(' × ').map(Number);
        const targetShapeDef = parsePortShape(targetPort as unknown as PortDefinition);

        if (sourceValues.length === targetShapeDef.length) {
          for (let i = 0; i < targetShapeDef.length; i++) {
            const dim = targetShapeDef[i];
            if (typeof dim === 'string' && dim !== '...' && !isNaN(sourceValues[i])) {
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

    // Handle control flow nodes specially
    let inputShapes: Record<string, ResolvedPortShape> = {};
    let outputShapes: Record<string, ResolvedPortShape> = {};
    let loopIdRefData: LoopIdRefData | undefined;

    if (isForLoopNode(node.data.definitionId)) {
      // For Loop: dynamically compute output from input
      const dataInShape = getIncomingShape(nodeId, 'data_in', edges, nodeShapes);
      const nIterateDims = (node.data.parameters.n_iterate_dims as number) || 1;

      const { loopBody, loopIdRef } = computeForLoopOutputs(
        dataInShape,
        nIterateDims,
        nodeId
      );

      // Set input shape to what's actually coming in
      if (dataInShape) {
        inputShapes['data_in'] = toPortShape(dataInShape);
      } else {
        inputShapes['data_in'] = { symbolic: '...', resolved: null, isFullyResolved: false };
      }

      // Set output shapes
      outputShapes['loop_body'] = toPortShape(loopBody.shape);

      // For Loop Id Ref, show the stripped dimensions info
      const refData = loopIdRef.metadata?.loopIdRefData as LoopIdRefData | undefined;
      if (refData) {
        loopIdRefData = refData;
        outputShapes['loop_id_ref'] = {
          symbolic: formatLoopIdRefDisplay(refData),
          resolved: refData.dimensionValue?.toString() || null,
          isFullyResolved: refData.strippedDimensions.every(d => d.isResolved),
        };
      } else {
        outputShapes['loop_id_ref'] = { symbolic: 'ref', resolved: null, isFullyResolved: false };
      }
    } else if (isStackNode(node.data.definitionId)) {
      // Stack: combine loop output with Loop Id Ref
      const loopOutputShape = getIncomingShape(nodeId, 'loop_output', edges, nodeShapes);
      const incomingLoopIdRef = getIncomingLoopIdRef(nodeId, 'loop_id_ref', edges, nodeShapes);
      const stackOrder = (node.data.parameters.stack_order as 'prepend' | 'append') || 'prepend';

      const stackResult = computeStackOutput(loopOutputShape, incomingLoopIdRef, stackOrder);

      // Set input shapes
      if (loopOutputShape) {
        inputShapes['loop_output'] = toPortShape(loopOutputShape);
      } else {
        inputShapes['loop_output'] = { symbolic: '...', resolved: null, isFullyResolved: false };
      }

      if (incomingLoopIdRef) {
        inputShapes['loop_id_ref'] = {
          symbolic: formatLoopIdRefDisplay(incomingLoopIdRef),
          resolved: incomingLoopIdRef.dimensionValue?.toString() || null,
          isFullyResolved: incomingLoopIdRef.strippedDimensions.every(d => d.isResolved),
        };
      } else {
        inputShapes['loop_id_ref'] = { symbolic: 'ref', resolved: null, isFullyResolved: false };
      }

      // Set output shape
      outputShapes['data_out'] = toPortShape(stackResult.shape);
    } else {
      // Regular node: resolve shapes normally
      for (const input of node.data.inputs) {
        // Check if there's an incoming connection
        const incomingShape = getIncomingShape(nodeId, input.id, edges, nodeShapes);
        if (incomingShape) {
          inputShapes[input.id] = toPortShape(incomingShape);
        } else {
          const shapeDef = parsePortShape(input);
          if (shapeDef.includes('...')) {
            inputShapes[input.id] = { symbolic: '...', resolved: null, isFullyResolved: false };
          } else {
            const resolved = resolveShape(shapeDef as ShapeDefinition, context);
            inputShapes[input.id] = toPortShape(resolved);
          }
        }
      }

      for (const output of node.data.outputs) {
        const shapeDef = parsePortShape(output);
        if (shapeDef.includes('...')) {
          outputShapes[output.id] = { symbolic: '...', resolved: null, isFullyResolved: false };
        } else {
          const resolved = resolveShape(shapeDef as ShapeDefinition, context);
          outputShapes[output.id] = toPortShape(resolved);
        }
      }
    }

    nodeShapes[nodeId] = {
      inputShapes,
      outputShapes,
      dimensionBindings: bindings,
      loopIdRefData,
    };
  }

  // Validate all connections
  for (const edge of edges) {
    const sourceShapes = nodeShapes[edge.source];
    const targetNode = nodeMap.get(edge.target);
    const targetShapes = nodeShapes[edge.target];

    if (!sourceShapes || !targetNode || !targetShapes) continue;

    const sourcePortId = edge.sourceHandle || '';
    const targetPortId = edge.targetHandle || '';

    const sourceShape = sourceShapes.outputShapes[sourcePortId];
    const targetShape = targetShapes.inputShapes[targetPortId];

    // Skip validation for loop_ref type connections
    const targetPort = targetNode.data.inputs.find(p => p.id === targetPortId);
    if (targetPort?.type === 'loop_ref') {
      connectionValidations[edge.id] = {
        edgeId: edge.id,
        sourceNodeId: edge.source,
        sourcePortId,
        targetNodeId: edge.target,
        targetPortId,
        compatibility: 'exact',
        validation: { compatibility: 'exact', isValid: true },
      };
      continue;
    }

    if (!sourceShape || !targetShape) continue;

    // Skip validation for dynamic shape inputs (e.g., For Loop's "..." input)
    // Dynamic shapes accept any input shape
    if (targetShape.symbolic === '...' || targetShape.symbolic.includes('...')) {
      connectionValidations[edge.id] = {
        edgeId: edge.id,
        sourceNodeId: edge.source,
        sourcePortId,
        targetNodeId: edge.target,
        targetPortId,
        compatibility: 'exact',
        validation: { compatibility: 'exact', isValid: true },
      };
      continue;
    }

    // Convert to ResolvedShape for validation
    const sourceResolved = sourceShape.resolved
      ? sourceShape.resolved.split(' × ').map((v, i) => ({
          symbolic: sourceShape.symbolic.split(' × ')[i] || v,
          value: parseInt(v, 10),
          isResolved: !isNaN(parseInt(v, 10)),
        }))
      : sourceShape.symbolic.split(' × ').map(s => ({
          symbolic: s,
          value: undefined,
          isResolved: false,
        }));

    const targetResolved = targetShape.resolved
      ? targetShape.resolved.split(' × ').map((v, i) => ({
          symbolic: targetShape.symbolic.split(' × ')[i] || v,
          value: parseInt(v, 10),
          isResolved: !isNaN(parseInt(v, 10)),
        }))
      : targetShape.symbolic.split(' × ').map(s => ({
          symbolic: s,
          value: undefined,
          isResolved: false,
        }));

    const validation = validateConnection(sourceResolved, targetResolved);

    connectionValidations[edge.id] = {
      edgeId: edge.id,
      sourceNodeId: edge.source,
      sourcePortId,
      targetNodeId: edge.target,
      targetPortId,
      compatibility: validation.compatibility,
      validation,
    };
  }

  return { nodeShapes, globalDimensions, connectionValidations };
}

// =============================================================================
// Hooks
// =============================================================================

/**
 * Hook to compute and subscribe to shape inference
 */
export function useShapeInference(): ShapeInferenceResult {
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
 * Hook to get connection validation info for a specific edge
 */
export function useConnectionInfo(edgeId: string): ConnectionInfo | null {
  const { connectionValidations } = useShapeInference();
  return connectionValidations[edgeId] || null;
}

/**
 * Hook to get all connection validations
 */
export function useConnectionValidations(): Record<string, ConnectionInfo> {
  const { connectionValidations } = useShapeInference();
  return connectionValidations;
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
