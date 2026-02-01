import { useCallback, useRef, type ChangeEvent } from 'react';
import type { SerializedPipeline } from '../types';

interface UseFileOperationsOptions {
  onImport: (pipeline: SerializedPipeline) => void;
  onExport: () => SerializedPipeline;
  afterImport?: () => void;
}

/**
 * Hook for handling pipeline file import/export operations.
 */
export function useFileOperations({
  onImport,
  onExport,
  afterImport,
}: UseFileOperationsOptions) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleExport = useCallback(() => {
    const pipeline = onExport();
    const blob = new Blob([JSON.stringify(pipeline, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `pipeline-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [onExport]);

  const handleImportClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const reader = new FileReader();
      reader.onload = (event) => {
        try {
          const pipeline = JSON.parse(event.target?.result as string);
          onImport(pipeline);
          afterImport?.();
        } catch (error) {
          console.error('Failed to parse pipeline file:', error);
        }
      };
      reader.readAsText(file);
      e.target.value = '';
    },
    [onImport, afterImport]
  );

  // Hidden file input element to be rendered
  const FileInput = (
    <input
      ref={fileInputRef}
      type="file"
      accept=".json"
      onChange={handleFileChange}
      className="hidden"
    />
  );

  return {
    FileInput,
    handleExport,
    handleImport: handleImportClick,
  };
}
