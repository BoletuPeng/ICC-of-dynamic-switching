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

import type { NodeDefinition, NodeCategory } from '../types';

// All available node definitions
export const nodeDefinitions: NodeDefinition[] = [
  extractRoiTimeseries as NodeDefinition,
  extractDmnEcn as NodeDefinition,
  tsnrOptimized as NodeDefinition,
  couplingOptimized as NodeDefinition,
  modularityLouvain as NodeDefinition,
  agreementWeighted as NodeDefinition,
  consensusUnd as NodeDefinition,
  participationCoef as NodeDefinition,
  moduleDegreeZscore as NodeDefinition,
  cartographicProfile as NodeDefinition,
  kmeansOptimized as NodeDefinition,
  switchFrequency as NodeDefinition,
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
  'preprocessing',
  'connectivity',
  'community',
  'metrics',
  'analysis',
  'clustering',
  'output',
];

// Helper to get definition or throw
export function getNodeDefinition(definitionId: string): NodeDefinition {
  const def = nodeDefinitionsMap[definitionId];
  if (!def) {
    throw new Error(`Unknown node definition: ${definitionId}`);
  }
  return def;
}

// Helper to get default parameter values for a node definition
export function getDefaultParameters(definition: NodeDefinition): Record<string, unknown> {
  return definition.parameters.reduce(
    (acc, param) => {
      acc[param.id] = param.default;
      return acc;
    },
    {} as Record<string, unknown>
  );
}
