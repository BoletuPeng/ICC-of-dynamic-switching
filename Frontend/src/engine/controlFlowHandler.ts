/**
 * Control Flow Handler - Manages dynamic shape computation for For Loop and Stack nodes
 *
 * For Loop:
 *   - Strips the first N dimensions from input
 *   - Outputs the remaining dimensions as loop_body
 *   - Creates LoopIdRefData with stripped dimension info
 *
 * Stack:
 *   - Receives LoopIdRefData from loop_id_ref input
 *   - Prepends (or appends) the stripped dimensions to loop_output
 *   - Outputs the reconstructed shape
 */

import type {
  ResolvedShape,
  LoopIdRefData,
  DynamicShapeResult,
} from './types';

// =============================================================================
// For Loop Handler
// =============================================================================

/**
 * Compute the output shapes for a For Loop node
 *
 * @param inputShape - The resolved input shape from the data_in port
 * @param nIterateDims - Number of dimensions to strip (from parameters)
 * @param nodeId - The For Loop node's ID
 * @returns Dynamic shape results for loop_body and loop_id_ref outputs
 */
export function computeForLoopOutputs(
  inputShape: ResolvedShape | undefined,
  nIterateDims: number,
  nodeId: string
): { loopBody: DynamicShapeResult; loopIdRef: DynamicShapeResult } {
  const warnings: string[] = [];

  // Handle undefined or empty input
  if (!inputShape || inputShape.length === 0) {
    return {
      loopBody: {
        shape: [],
        warnings: ['No input connected to For Loop'],
      },
      loopIdRef: {
        shape: [],
        metadata: { loopIdRefData: undefined },
        warnings: ['Cannot determine loop reference without input'],
      },
    };
  }

  // Validate that we have enough dimensions to strip
  if (nIterateDims > inputShape.length) {
    warnings.push(
      `Cannot iterate over ${nIterateDims} dimensions when input only has ${inputShape.length}`
    );
    return {
      loopBody: {
        shape: inputShape,
        warnings,
      },
      loopIdRef: {
        shape: [],
        metadata: { loopIdRefData: undefined },
        warnings,
      },
    };
  }

  // Strip the first N dimensions
  const strippedDimensions = inputShape.slice(0, nIterateDims);
  const remainingShape = inputShape.slice(nIterateDims);

  // Create Loop Id Ref metadata
  const loopIdRefData: LoopIdRefData = {
    dimensionName: strippedDimensions.map(d => d.symbolic).join(' × '),
    dimensionValue: strippedDimensions.every(d => d.isResolved)
      ? strippedDimensions.reduce((acc, d) => acc * (d.value || 1), 1)
      : undefined,
    sourceNodeId: nodeId,
    strippedDimensions,
  };

  return {
    loopBody: {
      shape: remainingShape,
    },
    loopIdRef: {
      shape: [], // Loop Id Ref is a reference, not data with dimensions
      metadata: { loopIdRefData },
    },
  };
}

/**
 * Format the Loop Id Ref display text
 */
export function formatLoopIdRefDisplay(data: LoopIdRefData | undefined): string {
  if (!data) {
    return 'No Loop';
  }

  const dims = data.strippedDimensions;
  if (dims.length === 0) {
    return 'Empty Loop';
  }

  // Format as "dim_name" or "dim_name = value"
  return dims.map(d => {
    if (d.isResolved && d.value !== undefined) {
      return `${d.symbolic} = ${d.value}`;
    }
    return d.symbolic;
  }).join(' × ');
}

// =============================================================================
// Stack Handler
// =============================================================================

/**
 * Compute the output shape for a Stack node
 *
 * @param loopOutputShape - The shape from the loop_output input
 * @param loopIdRefData - The LoopIdRefData from the loop_id_ref input
 * @param stackOrder - 'prepend' or 'append' from parameters
 * @returns Dynamic shape result for data_out
 */
export function computeStackOutput(
  loopOutputShape: ResolvedShape | undefined,
  loopIdRefData: LoopIdRefData | undefined,
  stackOrder: 'prepend' | 'append' = 'prepend'
): DynamicShapeResult {
  const warnings: string[] = [];

  // Handle missing inputs
  if (!loopOutputShape) {
    warnings.push('No loop output connected to Stack');
  }
  if (!loopIdRefData) {
    warnings.push('No Loop Id Ref connected to Stack');
  }

  if (!loopOutputShape || !loopIdRefData) {
    return {
      shape: loopOutputShape || [],
      warnings,
    };
  }

  // Combine the stripped dimensions with the loop output
  const strippedDims = loopIdRefData.strippedDimensions;

  if (stackOrder === 'append') {
    return {
      shape: [...loopOutputShape, ...strippedDims],
    };
  }

  // Default: prepend
  return {
    shape: [...strippedDims, ...loopOutputShape],
  };
}

// =============================================================================
// Control Flow Detection
// =============================================================================

/**
 * Check if a node is a For Loop
 */
export function isForLoopNode(definitionId: string): boolean {
  return definitionId === 'for_loop';
}

/**
 * Check if a node is a Stack
 */
export function isStackNode(definitionId: string): boolean {
  return definitionId === 'stack';
}

/**
 * Check if a port is a Loop Id Ref output
 */
export function isLoopIdRefPort(portId: string): boolean {
  return portId === 'loop_id_ref';
}

/**
 * Check if a port expects Loop Id Ref input
 */
export function expectsLoopIdRefInput(portId: string): boolean {
  return portId === 'loop_id_ref';
}

// =============================================================================
// Shape Display Helpers
// =============================================================================

/**
 * Create a special display format for dynamic shapes
 * Shows "..." when shape is dynamically computed
 */
export function createDynamicShapeDisplay(
  shape: ResolvedShape,
  label?: string
): { symbolic: string; resolved: string | null; isFullyResolved: boolean } {
  if (shape.length === 0) {
    return {
      symbolic: label || 'dynamic',
      resolved: null,
      isFullyResolved: false,
    };
  }

  const symbolic = shape.map(d => d.symbolic).join(' × ');
  const isFullyResolved = shape.every(d => d.isResolved);
  const resolved = isFullyResolved
    ? shape.map(d => String(d.value)).join(' × ')
    : null;

  return { symbolic, resolved, isFullyResolved };
}

/**
 * Validate that a loop output shape is compatible with the original input shape
 * (minus the stripped dimensions)
 *
 * @param originalInputShape - The shape that went into the For Loop
 * @param loopOutputShape - The shape coming out of the loop body processing
 * @param nIterateDims - Number of dimensions that were stripped
 * @returns Whether the shapes are compatible
 */
export function validateLoopBodyOutput(
  originalInputShape: ResolvedShape,
  loopOutputShape: ResolvedShape,
  nIterateDims: number
): { isCompatible: boolean; expectedShape: ResolvedShape; message?: string } {
  const expectedShape = originalInputShape.slice(nIterateDims);

  // Allow any shape for loop output - the user might transform it
  // But we can warn if dimension count doesn't make sense
  if (loopOutputShape.length === 0 && expectedShape.length > 0) {
    return {
      isCompatible: true,
      expectedShape,
      message: 'Loop output has no dimensions but input suggests otherwise',
    };
  }

  return {
    isCompatible: true,
    expectedShape,
  };
}
