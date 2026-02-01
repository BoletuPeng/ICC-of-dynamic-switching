import { memo, useCallback } from 'react';
import { Play, Trash2, Zap, Download, Upload, RotateCcw, Maximize2 } from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { Button } from './ui';
import { useFileOperations } from '../hooks';
import { usePipelineStore, selectNodeCount, selectEdgeCount } from '../store/pipelineStore';

export const Header = memo(function Header() {
  const { clearPipeline, loadDemoPipeline, exportPipeline, loadPipeline } = usePipelineStore();
  const nodeCount = usePipelineStore(selectNodeCount);
  const edgeCount = usePipelineStore(selectEdgeCount);
  const { fitView } = useReactFlow();

  const doFitView = useCallback(() => fitView({ padding: 0.2 }), [fitView]);

  const { FileInput, handleExport, handleImport } = useFileOperations({
    onExport: exportPipeline,
    onImport: loadPipeline,
    afterImport: () => setTimeout(doFitView, 100),
  });

  const handleReset = useCallback(() => {
    loadDemoPipeline();
    setTimeout(doFitView, 100);
  }, [loadDemoPipeline, doFitView]);

  const handleFitView = useCallback(() => {
    fitView({ padding: 0.2, duration: 300 });
  }, [fitView]);

  const handleClear = useCallback(() => {
    if (nodeCount > 0 && !confirm('Are you sure you want to clear the pipeline?')) return;
    clearPipeline();
  }, [clearPipeline, nodeCount]);

  const handleRun = useCallback(() => {
    alert('Pipeline execution coming soon! This will connect to the Python backend.');
  }, []);

  return (
    <header className="header">
      {FileInput}

      {/* Brand */}
      <div className="header-brand">
        <div className="header-logo">
          <Zap className="text-white" size={18} />
        </div>
        <div className="header-title-group">
          <h1 className="header-title">NeuroFlow Pipeline Builder</h1>
          <p className="header-subtitle">Dynamic Brain State Analysis</p>
        </div>
      </div>

      {/* Stats */}
      <div className="header-stats">
        <div className="header-stat">
          <div className="header-stat-value">{nodeCount}</div>
          <div className="header-stat-label">Nodes</div>
        </div>
        <div className="header-stat-divider" />
        <div className="header-stat">
          <div className="header-stat-value">{edgeCount}</div>
          <div className="header-stat-label">Connections</div>
        </div>
      </div>

      {/* Actions */}
      <div className="header-actions">
        <div className="header-action-group">
          <Button onClick={handleFitView} title="Fit view" icon={<Maximize2 size={16} />} />
        </div>

        <div className="header-divider" />

        <div className="header-action-group">
          <Button onClick={handleImport} title="Import" icon={<Upload size={16} />}>
            Import
          </Button>
          <Button
            onClick={handleExport}
            title="Export"
            icon={<Download size={16} />}
            disabled={nodeCount === 0}
          >
            Export
          </Button>
        </div>

        <div className="header-divider" />

        <div className="header-action-group">
          <Button onClick={handleReset} title="Reset" icon={<RotateCcw size={16} />}>
            Reset
          </Button>
          <Button
            variant="danger"
            onClick={handleClear}
            title="Clear"
            icon={<Trash2 size={16} />}
            disabled={nodeCount === 0}
          >
            Clear
          </Button>
        </div>

        <div className="header-divider" />

        <Button
          variant="primary"
          onClick={handleRun}
          icon={<Play size={16} />}
          disabled={nodeCount === 0}
        >
          Run Pipeline
        </Button>
      </div>
    </header>
  );
});

export default Header;
