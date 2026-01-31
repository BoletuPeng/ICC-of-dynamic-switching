import { useCallback, useState } from 'react';
import {
  DndContext,
  DragOverlay,
  MouseSensor,
  TouchSensor,
  useSensor,
  useSensors,
  type DragStartEvent,
  type DragEndEvent,
} from '@dnd-kit/core';
import { Header } from './components/Header';
import { Canvas } from './components/Canvas';
import { Sidebar } from './components/Sidebar';
import { NodeEditor } from './components/NodeEditor';
import { Icon } from './components/Icons';
import { usePipelineStore } from './store/pipelineStore';
import { nodeDefinitionsMap } from './data';

interface DragState {
  type: 'sidebar-node' | 'pipeline-node';
  definitionId?: string;
  nodeId?: string;
}

function DragOverlayContent({ definitionId }: { definitionId: string }) {
  const definition = nodeDefinitionsMap[definitionId];
  if (!definition) return null;

  return (
    <div
      className="min-w-[200px] rounded-xl overflow-hidden shadow-2xl shadow-black/50 opacity-90 pointer-events-none"
      style={{ transform: 'rotate(-2deg)' }}
    >
      <div
        className="px-3 py-2 flex items-center gap-2"
        style={{ backgroundColor: definition.color }}
      >
        <Icon name={definition.icon} className="text-white" size={18} />
        <span className="text-sm font-semibold text-white">{definition.name}</span>
      </div>
      <div className="bg-surface-800 p-3">
        <div className="flex justify-between text-xs text-surface-400">
          <span>{definition.inputs.length} inputs</span>
          <span>{definition.outputs.length} outputs</span>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const { addNode, removeNode, updateNodePosition, nodes } = usePipelineStore();
  const [dragState, setDragState] = useState<DragState | null>(null);

  const sensors = useSensors(
    useSensor(MouseSensor, {
      activationConstraint: {
        distance: 5,
      },
    }),
    useSensor(TouchSensor, {
      activationConstraint: {
        delay: 100,
        tolerance: 5,
      },
    })
  );

  const handleDragStart = useCallback((event: DragStartEvent) => {
    const { active } = event;
    const data = active.data.current;

    if (data?.type === 'sidebar-node') {
      setDragState({
        type: 'sidebar-node',
        definitionId: data.definitionId,
      });
    } else if (data?.type === 'pipeline-node') {
      setDragState({
        type: 'pipeline-node',
        nodeId: data.node.id,
        definitionId: data.node.definitionId,
      });
    }
  }, []);

  const handleDragEnd = useCallback(
    (event: DragEndEvent) => {
      const { active, over, delta } = event;
      const data = active.data.current;

      if (data?.type === 'sidebar-node' && over?.id === 'canvas-dropzone') {
        // Add new node to canvas
        const canvasElement = document.querySelector('[data-canvas]');
        const rect = canvasElement?.getBoundingClientRect();

        // Calculate position (center of where it was dropped)
        const x = event.activatorEvent && 'clientX' in event.activatorEvent
          ? (event.activatorEvent as MouseEvent).clientX + delta.x - (rect?.left || 0) - 110
          : 100;
        const y = event.activatorEvent && 'clientY' in event.activatorEvent
          ? (event.activatorEvent as MouseEvent).clientY + delta.y - (rect?.top || 0) - 40
          : 100;

        addNode(data.definitionId, { x: Math.max(0, x), y: Math.max(0, y) });
      } else if (data?.type === 'pipeline-node') {
        if (over?.id === 'sidebar-dropzone') {
          // Delete node by dragging to sidebar
          removeNode(data.node.id);
        } else {
          // Update node position
          const node = nodes.find((n) => n.id === data.node.id);
          if (node) {
            updateNodePosition(data.node.id, {
              x: node.position.x + delta.x,
              y: node.position.y + delta.y,
            });
          }
        }
      }

      setDragState(null);
    },
    [addNode, removeNode, updateNodePosition, nodes]
  );

  const handleDragCancel = useCallback(() => {
    setDragState(null);
  }, []);

  return (
    <DndContext
      sensors={sensors}
      onDragStart={handleDragStart}
      onDragEnd={handleDragEnd}
      onDragCancel={handleDragCancel}
    >
      <div className="w-full h-full flex flex-col">
        <Header />
        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1" data-canvas>
            <Canvas />
          </div>
          <Sidebar />
        </div>
        <NodeEditor />
      </div>

      <DragOverlay dropAnimation={null}>
        {dragState?.definitionId && (
          <DragOverlayContent definitionId={dragState.definitionId} />
        )}
      </DragOverlay>
    </DndContext>
  );
}
