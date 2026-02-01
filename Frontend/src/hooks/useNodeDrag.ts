import { useCallback, type DragEvent } from 'react';

const MIME_TYPE = 'application/reactflow';

/**
 * Hook for handling node drag-and-drop from sidebar to canvas.
 */
export function useNodeDrag() {
  const startDrag = useCallback((event: DragEvent, definitionId: string) => {
    event.dataTransfer.setData(MIME_TYPE, definitionId);
    event.dataTransfer.effectAllowed = 'move';
  }, []);

  const handleDragOver = useCallback((event: DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const getDroppedDefinitionId = useCallback((event: DragEvent): string | null => {
    event.preventDefault();
    const definitionId = event.dataTransfer.getData(MIME_TYPE);
    return definitionId || null;
  }, []);

  return {
    startDrag,
    handleDragOver,
    getDroppedDefinitionId,
  };
}
