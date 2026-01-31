import { useMemo, useState } from 'react';
import { useDraggable, useDroppable } from '@dnd-kit/core';
import { Icon, ChevronDown, ChevronUp, Trash2 } from './Icons';
import { nodeDefinitions, categoryLabels, categoryColors } from '../data';
import type { NodeDefinition } from '../types';

interface SidebarNodeProps {
  definition: NodeDefinition;
}

function SidebarNode({ definition }: SidebarNodeProps) {
  const { attributes, listeners, setNodeRef, transform, isDragging } = useDraggable({
    id: `sidebar-${definition.id}`,
    data: {
      type: 'sidebar-node',
      definitionId: definition.id,
    },
  });

  const style = transform
    ? {
        transform: `translate3d(${transform.x}px, ${transform.y}px, 0)`,
      }
    : undefined;

  return (
    <div
      ref={setNodeRef}
      style={style}
      className={`
        glass rounded-lg p-3 cursor-grab
        transition-all duration-200
        hover:bg-surface-700/50 hover:border-surface-500/30
        ${isDragging ? 'opacity-50 cursor-grabbing shadow-2xl scale-105 z-50' : ''}
      `}
      {...listeners}
      {...attributes}
    >
      <div className="flex items-center gap-2">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: `${definition.color}30` }}
        >
          <Icon name={definition.icon} size={18} style={{ color: definition.color }} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-surface-100 truncate">
            {definition.name}
          </div>
          <div className="text-xs text-surface-400 truncate">
            {definition.inputs.length} in · {definition.outputs.length} out
          </div>
        </div>
      </div>
    </div>
  );
}

interface CategorySectionProps {
  category: string;
  nodes: NodeDefinition[];
  isExpanded: boolean;
  onToggle: () => void;
}

function CategorySection({ category, nodes, isExpanded, onToggle }: CategorySectionProps) {
  const color = categoryColors[category] || '#64748b';

  return (
    <div className="mb-2">
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-2 px-3 py-2 rounded-lg hover:bg-surface-800/50 transition-colors"
      >
        <div
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: color }}
        />
        <span className="flex-1 text-left text-sm font-medium text-surface-200">
          {categoryLabels[category] || category}
        </span>
        <span className="text-xs text-surface-500 mr-2">{nodes.length}</span>
        {isExpanded ? (
          <ChevronUp size={16} className="text-surface-400" />
        ) : (
          <ChevronDown size={16} className="text-surface-400" />
        )}
      </button>

      {isExpanded && (
        <div className="mt-2 space-y-2 px-1 animate-slide-in">
          {nodes.map((def) => (
            <SidebarNode key={def.id} definition={def} />
          ))}
        </div>
      )}
    </div>
  );
}

export function Sidebar() {
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(
    new Set(['preprocessing', 'connectivity', 'community'])
  );

  const { setNodeRef, isOver } = useDroppable({
    id: 'sidebar-dropzone',
  });

  const groupedNodes = useMemo(() => {
    const groups: Record<string, NodeDefinition[]> = {};
    nodeDefinitions.forEach((def) => {
      if (!groups[def.category]) {
        groups[def.category] = [];
      }
      groups[def.category].push(def);
    });
    return groups;
  }, []);

  const toggleCategory = (category: string) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  };

  const categoryOrder = [
    'preprocessing',
    'connectivity',
    'community',
    'metrics',
    'analysis',
    'clustering',
    'output',
  ];

  return (
    <div
      ref={setNodeRef}
      className={`
        w-72 h-full glass border-l border-surface-700/50
        flex flex-col overflow-hidden
        transition-all duration-200
        ${isOver ? 'bg-red-900/20 border-red-500/50' : ''}
      `}
    >
      {/* Header */}
      <div className="p-4 border-b border-surface-700/50">
        <h2 className="text-lg font-semibold text-surface-100">Node Library</h2>
        <p className="text-xs text-surface-400 mt-1">
          Drag nodes to the canvas to build your pipeline
        </p>
      </div>

      {/* Trash zone indicator */}
      {isOver && (
        <div className="mx-4 mt-4 p-4 border-2 border-dashed border-red-500/50 rounded-lg bg-red-900/20 flex items-center justify-center gap-2 animate-fade-in">
          <Trash2 className="text-red-400" size={20} />
          <span className="text-red-400 text-sm font-medium">Drop to delete</span>
        </div>
      )}

      {/* Scrollable content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-1">
        {categoryOrder
          .filter((cat) => groupedNodes[cat])
          .map((category) => (
            <CategorySection
              key={category}
              category={category}
              nodes={groupedNodes[category]}
              isExpanded={expandedCategories.has(category)}
              onToggle={() => toggleCategory(category)}
            />
          ))}
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-surface-700/50">
        <div className="text-xs text-surface-500 text-center">
          {nodeDefinitions.length} nodes available
        </div>
      </div>
    </div>
  );
}
