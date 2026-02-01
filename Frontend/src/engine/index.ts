/**
 * Shape Inference Engine - Main Export
 *
 * This module provides the complete shape inference system for the pipeline UI.
 */

// Core types
export type {
  ShapeDimension,
  ShapeExpression,
  ShapeDefinition,
  ResolvedDimension,
  ResolvedShape,
  ShapeContext,
  ShapeBinding,
  ModulePortDefinition,
  ModuleParameterDefinition,
  ModuleDefinition,
  ShapeRule,
  ForLoopConfig,
  StackConfig,
  NodeShapeState,
  ShapeConnection,
  ShapePropagationState,
} from './types';

// Parser
export {
  parseShapeDimension,
  parseShapeDefinition,
  dimensionToString,
  shapeToString,
} from './shapeParser';

// Resolver
export {
  evaluateDimension,
  resolveDimension,
  resolveShape,
  extractVariables,
  extractShapeVariables,
  matchShapes,
  formatResolvedShape,
  formatResolvedShapeWithSymbols,
  formatShapeCompact,
} from './shapeResolver';
export type { ShapeMatchResult } from './shapeResolver';

// Propagation
export {
  createNodeShapeState,
  updateNodeShapeState,
  propagateShapes,
  getEffectiveInputShape,
  collectAllDimensions,
  handleForLoopShape,
  handleStackShape,
  getRequiredVariables,
} from './shapePropagation';

// Module loader
export {
  moduleRegistry,
  validateModuleDefinition,
  loadModules,
  loadModulesSync,
  getModule,
  getAllModules,
  getModulesByCategory,
  toLegacyNodeDefinition,
  getAllModulesLegacy,
} from './moduleLoader';

// Control flow handler
export {
  computeForLoopOutputs,
  computeStackOutput,
  isForLoopNode,
  isStackNode,
  isLoopIdRefPort,
  expectsLoopIdRefInput,
  formatLoopIdRefDisplay,
  createDynamicShapeDisplay,
  validateLoopBodyOutput,
} from './controlFlowHandler';

// Connection validator
export {
  validateConnection,
  createUnconnectedResult,
  formatShapeForMessage,
  validateLoopIdRef,
  CONNECTION_COLORS,
} from './connectionValidator';
export type {
  ConnectionCompatibility,
  ConnectionValidationResult,
  DimensionComparison,
  LoopIdRefMetadata,
} from './connectionValidator';

// Additional types from types.ts
export type { LoopIdRefData, DynamicShapeResult } from './types';
