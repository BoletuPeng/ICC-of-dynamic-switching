import { Play, Trash2, Zap } from './Icons';
import { usePipelineStore } from '../store/pipelineStore';

export function Header() {
  const { nodes, connections, clearPipeline } = usePipelineStore();

  return (
    <header className="h-14 glass border-b border-surface-700/50 flex items-center px-4 gap-4">
      {/* Logo & Title */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
          <Zap className="text-white" size={18} />
        </div>
        <div>
          <h1 className="text-sm font-semibold text-surface-100">
            NeuroFlow Pipeline Builder
          </h1>
          <p className="text-xs text-surface-400">
            Dynamic Brain State Analysis
          </p>
        </div>
      </div>

      {/* Stats */}
      <div className="flex-1 flex items-center justify-center gap-6">
        <div className="text-center">
          <div className="text-lg font-semibold text-surface-100">{nodes.length}</div>
          <div className="text-xs text-surface-400">Nodes</div>
        </div>
        <div className="w-px h-8 bg-surface-700" />
        <div className="text-center">
          <div className="text-lg font-semibold text-surface-100">{connections.length}</div>
          <div className="text-xs text-surface-400">Connections</div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <button
          onClick={clearPipeline}
          className="px-3 py-1.5 rounded-lg text-sm text-surface-300 hover:text-surface-100 hover:bg-surface-800 transition-colors flex items-center gap-2"
          disabled={nodes.length === 0}
        >
          <Trash2 size={16} />
          Clear
        </button>
        <button
          className="px-4 py-1.5 rounded-lg text-sm font-medium text-white bg-gradient-to-r from-primary-500 to-primary-600 hover:from-primary-400 hover:to-primary-500 transition-all flex items-center gap-2 shadow-lg shadow-primary-500/25"
          disabled={nodes.length === 0}
        >
          <Play size={16} />
          Run Pipeline
        </button>
      </div>
    </header>
  );
}
