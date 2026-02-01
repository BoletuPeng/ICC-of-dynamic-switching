/**
 * Shape Resolver - Evaluates shape expressions and resolves dimensions
 *
 * This module handles:
 * - Evaluating shape expressions with known variable values
 * - Inferring variable bindings from shape matching
 * - Propagating shape information through the graph
 */

import type {
  ShapeDimension,
  ShapeExpression,
  ShapeContext,
  ResolvedDimension,
  ResolvedShape,
  ShapeDefinition,
} from './types';
import { dimensionToString } from './shapeParser';

// =============================================================================
// Dimension Evaluation
// =============================================================================

/**
 * Evaluate a shape dimension to a numeric value if possible
 */
export function evaluateDimension(
  dim: ShapeDimension,
  context: ShapeContext
): number | undefined {
  // Concrete number
  if (typeof dim === 'number') {
    return dim;
  }

  // Variable reference
  if (typeof dim === 'string') {
    return context.dimensions[dim];
  }

  // Expression
  const expr = dim as ShapeExpression;

  // Parameter reference
  if (expr.op === 'ref') {
    const paramValue = context.parameters[expr.param!];
    if (typeof paramValue === 'number') {
      return paramValue;
    }
    return undefined;
  }

  // Binary operation
  const left = evaluateDimension(expr.left!, context);
  const right = evaluateDimension(expr.right!, context);

  if (left === undefined || right === undefined) {
    return undefined;
  }

  switch (expr.op) {
    case 'add':
      return left + right;
    case 'sub':
      return left - right;
    case 'mul':
      return left * right;
    case 'div':
      return Math.floor(left / right);
    default:
      return undefined;
  }
}

/**
 * Resolve a single dimension to a ResolvedDimension
 */
export function resolveDimension(
  dim: ShapeDimension,
  context: ShapeContext
): ResolvedDimension {
  const symbolic = dimensionToString(dim);
  const value = evaluateDimension(dim, context);

  return {
    symbolic,
    value,
    isResolved: value !== undefined,
  };
}

/**
 * Resolve a complete shape definition
 */
export function resolveShape(
  shape: ShapeDefinition,
  context: ShapeContext
): ResolvedShape {
  return shape.map((dim) => resolveDimension(dim, context));
}

// =============================================================================
// Variable Extraction
// =============================================================================

/**
 * Extract all variable names from a dimension
 */
export function extractVariables(dim: ShapeDimension): string[] {
  if (typeof dim === 'number') {
    return [];
  }

  if (typeof dim === 'string') {
    return [dim];
  }

  const expr = dim as ShapeExpression;

  if (expr.op === 'ref') {
    return []; // Parameter references are not dimension variables
  }

  const leftVars = expr.left ? extractVariables(expr.left) : [];
  const rightVars = expr.right ? extractVariables(expr.right) : [];

  return [...new Set([...leftVars, ...rightVars])];
}

/**
 * Extract all variables from a shape definition
 */
export function extractShapeVariables(shape: ShapeDefinition): string[] {
  const allVars = shape.flatMap(extractVariables);
  return [...new Set(allVars)];
}

// =============================================================================
// Shape Matching and Binding Inference
// =============================================================================

/**
 * Result of matching two shapes
 */
export interface ShapeMatchResult {
  /** Whether the shapes can be matched */
  isCompatible: boolean;
  /** Inferred variable bindings */
  bindings: Record<string, number>;
  /** Conflict information if not compatible */
  conflicts?: string[];
}

/**
 * Try to match a source shape against a target shape definition
 * This infers variable bindings when the source has known values
 */
export function matchShapes(
  sourceResolved: ResolvedShape,
  targetDefinition: ShapeDefinition,
  existingBindings: Record<string, number> = {}
): ShapeMatchResult {
  // Dimension count must match
  if (sourceResolved.length !== targetDefinition.length) {
    return {
      isCompatible: false,
      bindings: {},
      conflicts: [
        `Dimension count mismatch: source has ${sourceResolved.length}, target expects ${targetDefinition.length}`,
      ],
    };
  }

  const bindings: Record<string, number> = { ...existingBindings };
  const conflicts: string[] = [];

  for (let i = 0; i < sourceResolved.length; i++) {
    const sourceDim = sourceResolved[i];
    const targetDim = targetDefinition[i];

    // If source is not resolved, we can't infer anything
    if (!sourceDim.isResolved || sourceDim.value === undefined) {
      continue;
    }

    const sourceValue = sourceDim.value;

    // Target is a concrete number - must match exactly
    if (typeof targetDim === 'number') {
      if (sourceValue !== targetDim) {
        conflicts.push(
          `Dimension ${i}: source has ${sourceValue}, target expects exactly ${targetDim}`
        );
      }
      continue;
    }

    // Target is a variable - bind it
    if (typeof targetDim === 'string') {
      if (bindings[targetDim] !== undefined && bindings[targetDim] !== sourceValue) {
        conflicts.push(
          `Variable ${targetDim} has conflicting values: ${bindings[targetDim]} vs ${sourceValue}`
        );
      } else {
        bindings[targetDim] = sourceValue;
      }
      continue;
    }

    // Target is an expression - we can't directly infer bindings from complex expressions
    // But we can verify if the result matches when we have all variables
    // This will be handled during full propagation
  }

  return {
    isCompatible: conflicts.length === 0,
    bindings,
    conflicts: conflicts.length > 0 ? conflicts : undefined,
  };
}

// =============================================================================
// Shape Formatting
// =============================================================================

/**
 * Format a resolved shape for display
 * Shows both symbolic names and resolved values when available
 */
export function formatResolvedShape(shape: ResolvedShape): string {
  return shape
    .map((dim) => {
      if (dim.isResolved && dim.value !== undefined) {
        // If it's just a number, show the number
        if (/^\d+$/.test(dim.symbolic)) {
          return dim.value.toString();
        }
        // Otherwise show both symbolic and value
        return `${dim.value}`;
      }
      return dim.symbolic;
    })
    .join(' × ');
}

/**
 * Format a resolved shape with symbolic annotations
 */
export function formatResolvedShapeWithSymbols(shape: ResolvedShape): string {
  return shape
    .map((dim) => {
      if (dim.isResolved && dim.value !== undefined) {
        if (/^\d+$/.test(dim.symbolic)) {
          return dim.value.toString();
        }
        return `${dim.symbolic}=${dim.value}`;
      }
      return dim.symbolic;
    })
    .join(' × ');
}

/**
 * Create a compact display string for shapes
 */
export function formatShapeCompact(shape: ResolvedShape): {
  symbolic: string;
  resolved: string | null;
  isFullyResolved: boolean;
} {
  const symbolic = shape.map((d) => d.symbolic).join(' × ');
  const isFullyResolved = shape.every((d) => d.isResolved);

  if (isFullyResolved) {
    const resolved = shape.map((d) => d.value!.toString()).join(' × ');
    return { symbolic, resolved, isFullyResolved };
  }

  return { symbolic, resolved: null, isFullyResolved };
}
