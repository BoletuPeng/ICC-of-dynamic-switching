/**
 * Connection Validator - Validates shape compatibility between connected ports
 *
 * Provides three levels of compatibility:
 * - Exact: All dimensions match exactly (green connection)
 * - NameMismatch: Dimension counts match but names differ (yellow connection)
 * - DimensionMismatch: Dimension counts don't match (red connection)
 */

import type { ResolvedShape, ResolvedDimension } from './types';

// =============================================================================
// Connection Compatibility Types
// =============================================================================

/**
 * Compatibility level for a connection
 */
export type ConnectionCompatibility = 'exact' | 'name_mismatch' | 'dimension_mismatch';

/**
 * Detailed result of connection validation
 */
export interface ConnectionValidationResult {
  /** Overall compatibility level */
  compatibility: ConnectionCompatibility;
  /** Whether the connection is valid (exact or name_mismatch) */
  isValid: boolean;
  /** Human-readable warning/error message */
  message?: string;
  /** Detailed dimension-level comparison */
  dimensionComparison?: DimensionComparison[];
}

/**
 * Comparison result for a single dimension
 */
export interface DimensionComparison {
  index: number;
  sourceSymbolic: string;
  targetSymbolic: string;
  sourceValue?: number;
  targetValue?: number;
  matches: boolean;
  nameMatches: boolean;
}

// =============================================================================
// Connection Colors
// =============================================================================

/**
 * CSS color values for connection states
 */
export const CONNECTION_COLORS: Record<ConnectionCompatibility, { stroke: string; glow: string }> = {
  exact: {
    stroke: '#22c55e',  // Green
    glow: 'rgba(34, 197, 94, 0.3)',
  },
  name_mismatch: {
    stroke: '#eab308',  // Yellow
    glow: 'rgba(234, 179, 8, 0.3)',
  },
  dimension_mismatch: {
    stroke: '#ef4444',  // Red
    glow: 'rgba(239, 68, 68, 0.3)',
  },
};

// =============================================================================
// Validation Functions
// =============================================================================

/**
 * Normalize dimension symbolic name for comparison
 * Handles cases like "n_vertices" vs "n_rois"
 */
function normalizeSymbolic(symbolic: string): string {
  return symbolic.trim().toLowerCase();
}

/**
 * Check if two symbolic dimension names are equivalent
 */
function areSymbolicNamesEquivalent(source: string, target: string): boolean {
  const normalizedSource = normalizeSymbolic(source);
  const normalizedTarget = normalizeSymbolic(target);

  // Exact match
  if (normalizedSource === normalizedTarget) {
    return true;
  }

  // Check if both are numeric
  const sourceIsNumeric = /^\d+$/.test(source);
  const targetIsNumeric = /^\d+$/.test(target);

  if (sourceIsNumeric && targetIsNumeric) {
    return source === target;
  }

  return false;
}

/**
 * Compare two dimensions for compatibility
 */
function compareDimension(
  source: ResolvedDimension,
  target: ResolvedDimension,
  index: number
): DimensionComparison {
  const nameMatches = areSymbolicNamesEquivalent(source.symbolic, target.symbolic);

  // Check value match if both are resolved
  let valueMatches = true;
  if (source.isResolved && target.isResolved &&
      source.value !== undefined && target.value !== undefined) {
    valueMatches = source.value === target.value;
  }

  return {
    index,
    sourceSymbolic: source.symbolic,
    targetSymbolic: target.symbolic,
    sourceValue: source.value,
    targetValue: target.value,
    matches: nameMatches && valueMatches,
    nameMatches,
  };
}

/**
 * Validate connection between a source output and target input
 *
 * @param sourceShape - The resolved output shape from the source node
 * @param targetShape - The resolved input shape expected by the target node
 * @returns Validation result with compatibility level and details
 */
export function validateConnection(
  sourceShape: ResolvedShape,
  targetShape: ResolvedShape
): ConnectionValidationResult {
  // Check dimension count first
  if (sourceShape.length !== targetShape.length) {
    return {
      compatibility: 'dimension_mismatch',
      isValid: false,
      message: `Dimension count mismatch: source has ${sourceShape.length} dimensions, target expects ${targetShape.length}`,
      dimensionComparison: [],
    };
  }

  // If both are empty (scalar), they match exactly
  if (sourceShape.length === 0) {
    return {
      compatibility: 'exact',
      isValid: true,
    };
  }

  // Compare each dimension
  const comparisons: DimensionComparison[] = [];
  let allNamesMatch = true;
  let allMatch = true;

  for (let i = 0; i < sourceShape.length; i++) {
    const comparison = compareDimension(sourceShape[i], targetShape[i], i);
    comparisons.push(comparison);

    if (!comparison.nameMatches) {
      allNamesMatch = false;
    }
    if (!comparison.matches) {
      allMatch = false;
    }
  }

  // Determine compatibility level
  if (allMatch && allNamesMatch) {
    return {
      compatibility: 'exact',
      isValid: true,
      dimensionComparison: comparisons,
    };
  }

  if (!allNamesMatch && sourceShape.length === targetShape.length) {
    // Find which dimensions don't match names
    const mismatchedDims = comparisons
      .filter(c => !c.nameMatches)
      .map(c => `dim ${c.index}: ${c.sourceSymbolic} ≠ ${c.targetSymbolic}`);

    return {
      compatibility: 'name_mismatch',
      isValid: true,  // Still valid, just a warning
      message: `Dimension name mismatch: ${mismatchedDims.join(', ')}`,
      dimensionComparison: comparisons,
    };
  }

  // This shouldn't be reached, but handle it as exact for safety
  return {
    compatibility: 'exact',
    isValid: true,
    dimensionComparison: comparisons,
  };
}

/**
 * Create a default unconnected validation result
 */
export function createUnconnectedResult(): ConnectionValidationResult {
  return {
    compatibility: 'exact',
    isValid: true,
    message: 'Port is not connected',
  };
}

/**
 * Format a shape for display in error messages
 */
export function formatShapeForMessage(shape: ResolvedShape): string {
  if (shape.length === 0) {
    return 'scalar';
  }
  return shape.map(d => d.symbolic).join(' × ');
}

// =============================================================================
// Loop Reference Validation
// =============================================================================

/**
 * Metadata carried by a Loop Id Ref
 */
export interface LoopIdRefMetadata {
  /** The dimension name that was stripped (e.g., "n_sessions") */
  dimensionName: string;
  /** The resolved value if known (e.g., 6) */
  dimensionValue?: number;
  /** Reference to the source For Loop node */
  sourceLoopNodeId: string;
}

/**
 * Validate that a Loop Id Ref connection is valid for a Stack node
 */
export function validateLoopIdRef(
  loopMetadata: LoopIdRefMetadata | undefined,
  expectedDimensionName?: string
): ConnectionValidationResult {
  if (!loopMetadata) {
    return {
      compatibility: 'dimension_mismatch',
      isValid: false,
      message: 'Missing Loop Id Ref connection',
    };
  }

  if (expectedDimensionName &&
      normalizeSymbolic(loopMetadata.dimensionName) !== normalizeSymbolic(expectedDimensionName)) {
    return {
      compatibility: 'name_mismatch',
      isValid: true,
      message: `Loop dimension name mismatch: ${loopMetadata.dimensionName} ≠ ${expectedDimensionName}`,
    };
  }

  return {
    compatibility: 'exact',
    isValid: true,
  };
}
