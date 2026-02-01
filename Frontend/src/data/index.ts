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

import type { NodeDefinition } from '../types';

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

export const nodeDefinitionsMap: Record<string, NodeDefinition> = Object.fromEntries(
  nodeDefinitions.map((def) => [def.id, def])
);

export const categoryColors: Record<string, string> = {
  preprocessing: '#10b981',
  connectivity: '#3b82f6',
  community: '#8b5cf6',
  metrics: '#f59e0b',
  analysis: '#ec4899',
  clustering: '#06b6d4',
  output: '#ef4444',
};

export const categoryLabels: Record<string, string> = {
  preprocessing: 'Preprocessing',
  connectivity: 'Connectivity',
  community: 'Community Detection',
  metrics: 'Network Metrics',
  analysis: 'Analysis',
  clustering: 'Clustering',
  output: 'Output',
};
