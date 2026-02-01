import { memo, useCallback, useRef } from 'react';
import {
  Play,
  Trash2,
  Zap,
  Download,
  Upload,
  RotateCcw,
  Maximize2,
} from 'lucide-react';
import { useReactFlow } from '@xyflow/react';
import { usePipelineStore, selectNodeCount, selectEdgeCount } from '../store/pipelineStore';

export const Header = memo(function Header() {
  const { clearPipeline, loadDemoPipeline, exportPipeline, loadPipeline } =
    usePipelineStore();
  const nodeCount = usePipelineStore(selectNodeCount);
  const edgeCount = usePipelineStore(selectEdgeCount);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const { fitView } = useReactFlow();

  // Export pipeline to JSON file
  const handleExport = useCallback(() => {
    const pipeline = exportPipeline();
    const blob = new Blob([JSON.stringify(pipeline, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pipeline-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [exportPipeline]);

  // Import pipeline from JSON file
  const handleImport = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const pipeline = JSON.parse(event.target?.result as string);
          loadPipeline(pipeline);
          setTimeout(() => fitView({ padding: 0.2 }), 100);
        } catch (error) {
          console.error('Failed to parse pipeline file:', error);
        }
      };
      reader.readAsText(file);

      // Reset input
      e.target.value = '';
    },
    [loadPipeline, fitView]
  );

  // Reset to demo pipeline
  const handleReset = useCallback(() => {
    loadDemoPipeline();
    setTimeout(() => fitView({ padding: 0.2 }), 100);
  }, [loadDemoPipeline, fitView]);

  // Fit view
  const handleFitView = useCallback(() => {
    fitView({ padding: 0.2, duration: 300 });
  }, [fitView]);

  // Clear pipeline
  const handleClear = useCallback(() => {
    if (nodeCount > 0 && !confirm('Are you sure you want to clear the pipeline?')) {
      return;
    }
    clearPipeline();
  }, [clearPipeline, nodeCount]);

  // Run pipeline (placeholder)
  const handleRun = useCallback(() => {
    alert('Pipeline execution coming soon! This will connect to the Python backend.');
  }, []);

  return (
    <header className="header">
      {/* Hidden file input for import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".json"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Logo & Title */}
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
        {/* View actions */}
        <div className="header-action-group">
          <button
            onClick={handleFitView}
            className="header-btn header-btn-ghost"
            title="Fit view"
          >
            <Maximize2 size={16} />
          </button>
        </div>

        <div className="header-action-divider" />

        {/* File actions */}
        <div className="header-action-group">
          <button
            onClick={handleImport}
            className="header-btn header-btn-ghost"
            title="Import pipeline"
          >
            <Upload size={16} />
            <span>Import</span>
          </button>
          <button
            onClick={handleExport}
            className="header-btn header-btn-ghost"
            title="Export pipeline"
            disabled={nodeCount === 0}
          >
            <Download size={16} />
            <span>Export</span>
          </button>
        </div>

        <div className="header-action-divider" />

        {/* Pipeline actions */}
        <div className="header-action-group">
          <button
            onClick={handleReset}
            className="header-btn header-btn-ghost"
            title="Reset to demo pipeline"
          >
            <RotateCcw size={16} />
            <span>Reset</span>
          </button>
          <button
            onClick={handleClear}
            className="header-btn header-btn-ghost header-btn-danger"
            title="Clear pipeline"
            disabled={nodeCount === 0}
          >
            <Trash2 size={16} />
            <span>Clear</span>
          </button>
        </div>

        <div className="header-action-divider" />

        {/* Run */}
        <button
          onClick={handleRun}
          className="header-btn header-btn-primary"
          disabled={nodeCount === 0}
        >
          <Play size={16} />
          <span>Run Pipeline</span>
        </button>
      </div>
    </header>
  );
});

export default Header;
