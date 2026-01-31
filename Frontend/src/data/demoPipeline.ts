import type { PipelineNode, Connection } from '../types';

export const demoPipelineNodes: PipelineNode[] = [
  {
    id: 'node-1',
    definitionId: 'extract_roi_timeseries',
    position: { x: 50, y: 80 },
    parameters: {
      space: 'fsaverage6',
      atlas_name: 'Schaefer300',
      output_level: 'roi',
    },
  },
  {
    id: 'node-2',
    definitionId: 'extract_dmn_ecn',
    position: { x: 320, y: 80 },
    parameters: {},
  },
  {
    id: 'node-3',
    definitionId: 'coupling_optimized',
    position: { x: 590, y: 80 },
    parameters: {
      window: 14,
      direction: 0,
      trim: 0,
      parallel: true,
    },
  },
  {
    id: 'node-4',
    definitionId: 'modularity_louvain_und_sign_optimized',
    position: { x: 50, y: 280 },
    parameters: {
      qtype: 'sta',
      seed: null,
    },
  },
  {
    id: 'node-5',
    definitionId: 'agreement_weighted_optimized',
    position: { x: 320, y: 280 },
    parameters: {
      parallel: false,
    },
  },
  {
    id: 'node-6',
    definitionId: 'consensus_und_optimized',
    position: { x: 590, y: 280 },
    parameters: {
      tau: 0.1,
      reps: 500,
      seed: null,
      verbose: false,
    },
  },
  {
    id: 'node-7',
    definitionId: 'participation_coef_sign_optimized',
    position: { x: 50, y: 480 },
    parameters: {
      parallel: false,
    },
  },
  {
    id: 'node-8',
    definitionId: 'module_degree_zscore_optimized',
    position: { x: 320, y: 480 },
    parameters: {
      flag: 0,
    },
  },
  {
    id: 'node-9',
    definitionId: 'cartographic_profile_optimized',
    position: { x: 590, y: 480 },
    parameters: {
      parallel: true,
    },
  },
  {
    id: 'node-10',
    definitionId: 'kmeans_optimized',
    position: { x: 185, y: 680 },
    parameters: {
      n_clusters: 2,
      n_init: 100,
      max_iter: 300,
      random_state: null,
    },
  },
  {
    id: 'node-11',
    definitionId: 'switch_frequency_optimized',
    position: { x: 455, y: 680 },
    parameters: {},
  },
];

export const demoPipelineConnections: Connection[] = [
  {
    id: 'conn-1',
    sourceNodeId: 'node-1',
    sourcePortId: 'roi_timeseries',
    targetNodeId: 'node-2',
    targetPortId: 'roi_timeseries',
  },
  {
    id: 'conn-2',
    sourceNodeId: 'node-2',
    sourcePortId: 'dmn_ecn_timeseries',
    targetNodeId: 'node-3',
    targetPortId: 'data',
  },
  {
    id: 'conn-3',
    sourceNodeId: 'node-3',
    sourcePortId: 'mtd',
    targetNodeId: 'node-4',
    targetPortId: 'W',
  },
  {
    id: 'conn-4',
    sourceNodeId: 'node-4',
    sourcePortId: 'Ci',
    targetNodeId: 'node-5',
    targetPortId: 'CI',
  },
  {
    id: 'conn-5',
    sourceNodeId: 'node-5',
    sourcePortId: 'D',
    targetNodeId: 'node-6',
    targetPortId: 'd',
  },
  {
    id: 'conn-6',
    sourceNodeId: 'node-3',
    sourcePortId: 'mtd',
    targetNodeId: 'node-7',
    targetPortId: 'W',
  },
  {
    id: 'conn-7',
    sourceNodeId: 'node-6',
    sourcePortId: 'ciu',
    targetNodeId: 'node-7',
    targetPortId: 'Ci',
  },
  {
    id: 'conn-8',
    sourceNodeId: 'node-3',
    sourcePortId: 'mtd',
    targetNodeId: 'node-8',
    targetPortId: 'W',
  },
  {
    id: 'conn-9',
    sourceNodeId: 'node-6',
    sourcePortId: 'ciu',
    targetNodeId: 'node-8',
    targetPortId: 'Ci',
  },
  {
    id: 'conn-10',
    sourceNodeId: 'node-7',
    sourcePortId: 'Ppos',
    targetNodeId: 'node-9',
    targetPortId: 'BT',
  },
  {
    id: 'conn-11',
    sourceNodeId: 'node-8',
    sourcePortId: 'Z',
    targetNodeId: 'node-9',
    targetPortId: 'WT',
  },
  {
    id: 'conn-12',
    sourceNodeId: 'node-9',
    sourcePortId: 'CP',
    targetNodeId: 'node-10',
    targetPortId: 'X',
  },
  {
    id: 'conn-13',
    sourceNodeId: 'node-10',
    sourcePortId: 'labels',
    targetNodeId: 'node-11',
    targetPortId: 'idx',
  },
];
