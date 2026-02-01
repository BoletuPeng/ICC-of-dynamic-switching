// =============================================================================
// Module Definitions - Static Import
// =============================================================================

// Data modules (from data/)
import extractRoiTimeseries from './extract_roi_timeseries.json';
import extractDmnEcn from './extract_dmn_ecn.json';
import couplingOptimized from './coupling_optimized.json';
import modularityLouvain from './modularity_louvain_und_sign_optimized.json';
import agreementWeighted from './agreement_weighted_optimized.json';
import consensusUnd from './consensus_und_optimized.json';
import participationCoef from './participation_coef_sign_optimized.json';
import moduleDegreeZscore from './module_degree_zscore_optimized.json';
import cartographicProfile from './cartographic_profile_optimized.json';
import kmeansOptimized from './kmeans_optimized.json';
import switchFrequency from './switch_frequency_optimized.json';
import tsnrOptimized from './tsnr_optimized.json';

// New modules (from modules/)
import forLoop from '../modules/_for_loop.json';
import stack from '../modules/_stack.json';
import readFmri from '../modules/read_fmri.json';

import type { NodeDefinition, NodeCategory } from '../types';

// =============================================================================
// Module Registration
// =============================================================================

// All available node definitions
export const nodeDefinitions: NodeDefinition[] = [
  // Input nodes
  readFmri as unknown as NodeDefinition,

  // Preprocessing
  extractRoiTimeseries as NodeDefinition,
  extractDmnEcn as NodeDefinition,
  tsnrOptimized as NodeDefinition,

  // Connectivity
  couplingOptimized as NodeDefinition,

  // Community Detection
  modularityLouvain as NodeDefinition,
  agreementWeighted as NodeDefinition,
  consensusUnd as NodeDefinition,

  // Network Metrics
  participationCoef as NodeDefinition,
  moduleDegreeZscore as NodeDefinition,
  cartographicProfile as NodeDefinition,

  // Clustering
  kmeansOptimized as NodeDefinition,

  // Output
  switchFrequency as NodeDefinition,

  // Control Flow
  forLoop as unknown as NodeDefinition,
  stack as unknown as NodeDefinition,
];

// Map for quick lookup by ID
export const nodeDefinitionsMap: Record<string, NodeDefinition> = Object.fromEntries(
  nodeDefinitions.map((def) => [def.id, def])
);

// Group definitions by category
export const nodesByCategory: Record<NodeCategory, NodeDefinition[]> = nodeDefinitions.reduce(
  (acc, def) => {
    const category = def.category as NodeCategory;
    if (!acc[category]) {
      acc[category] = [];
    }
    acc[category].push(def);
    return acc;
  },
  {} as Record<NodeCategory, NodeDefinition[]>
);

// Get sorted categories for display
export const sortedCategories: NodeCategory[] = [
  'input',
  'control',
  'preprocessing',
  'connectivity',
  'community',
  'metrics',
  'analysis',
  'clustering',
  'output',
];

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Get a node definition by ID, throws if not found
 */
export function getNodeDefinition(definitionId: string): NodeDefinition {
  const def = nodeDefinitionsMap[definitionId];
  if (!def) {
    throw new Error(`Unknown node definition: ${definitionId}`);
  }
  return def;
}

/**
 * Get default parameter values for a node definition
 */
export function getDefaultParameters(definition: NodeDefinition): Record<string, unknown> {
  return definition.parameters.reduce(
    (acc, param) => {
      acc[param.id] = param.default;
      return acc;
    },
    {} as Record<string, unknown>
  );
}

/**
 * Check if a node definition is a control flow node
 */
export function isControlFlowNode(definition: NodeDefinition): boolean {
  return definition.category === 'control' ||
    (definition as unknown as { isControlFlow?: boolean }).isControlFlow === true;
}

/**
 * Check if a node definition is an input/source node
 */
export function isSourceNode(definition: NodeDefinition): boolean {
  return definition.category === 'input' ||
    (definition as unknown as { isSource?: boolean }).isSource === true;
}
