import { memo, useState, useCallback, type DragEvent } from 'react';
import { ChevronDown, ChevronUp, Search, GripVertical } from 'lucide-react';
import { Icon } from './Icons';
import { nodesByCategory, sortedCategories, nodeDefinitions } from '../data';
import { CATEGORY_COLORS, CATEGORY_LABELS, type NodeDefinition, type NodeCategory } from '../types';

interface NodeItemProps {
  definition: NodeDefinition;
  onDragStart: (event: DragEvent, definition: NodeDefinition) => void;
}

const NodeItem = memo(function NodeItem({ definition, onDragStart }: NodeItemProps) {
  const handleDragStart = useCallback(
    (event: DragEvent) => {
      onDragStart(event, definition);
    },
    [definition, onDragStart]
  );

  return (
    <div
      className="sidebar-node-item"
      draggable
      onDragStart={handleDragStart}
      style={{ '--node-color': definition.color } as React.CSSProperties}
    >
      <div className="sidebar-node-grip">
        <GripVertical size={14} className="text-surface-500" />
      </div>
      <div
        className="sidebar-node-icon"
        style={{ backgroundColor: definition.color }}
      >
        <Icon name={definition.icon} size={14} className="text-white" />
      </div>
      <div className="sidebar-node-info">
        <span className="sidebar-node-name">{definition.name}</span>
        <span className="sidebar-node-ports">
          {definition.inputs.length} in / {definition.outputs.length} out
        </span>
      </div>
    </div>
  );
});

interface CategorySectionProps {
  category: NodeCategory;
  definitions: NodeDefinition[];
  isExpanded: boolean;
  onToggle: () => void;
  onDragStart: (event: DragEvent, definition: NodeDefinition) => void;
}

const CategorySection = memo(function CategorySection({
  category,
  definitions,
  isExpanded,
  onToggle,
  onDragStart,
}: CategorySectionProps) {
  const color = CATEGORY_COLORS[category] || '#64748b';
  const label = CATEGORY_LABELS[category] || category;

  return (
    <div className="sidebar-category">
      <button
        className="sidebar-category-header"
        onClick={onToggle}
        style={{ '--category-color': color } as React.CSSProperties}
      >
        <div className="flex items-center gap-2">
          <div
            className="sidebar-category-indicator"
            style={{ backgroundColor: color }}
          />
          <span className="sidebar-category-label">{label}</span>
          <span className="sidebar-category-count">{definitions.length}</span>
        </div>
        {isExpanded ? (
          <ChevronUp size={16} className="text-surface-400" />
        ) : (
          <ChevronDown size={16} className="text-surface-400" />
        )}
      </button>

      {isExpanded && (
        <div className="sidebar-category-content">
          {definitions.map((def) => (
            <NodeItem
              key={def.id}
              definition={def}
              onDragStart={onDragStart}
            />
          ))}
        </div>
      )}
    </div>
  );
});

interface SidebarProps {
  isCollapsed?: boolean;
  onToggleCollapse?: () => void;
}

export const Sidebar = memo(function Sidebar({
  isCollapsed = false,
  onToggleCollapse,
}: SidebarProps) {
  const [expandedCategories, setExpandedCategories] = useState<Set<NodeCategory>>(
    new Set(sortedCategories)
  );
  const [searchQuery, setSearchQuery] = useState('');

  const handleToggleCategory = useCallback((category: NodeCategory) => {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  }, []);

  const handleDragStart = useCallback(
    (event: DragEvent, definition: NodeDefinition) => {
      event.dataTransfer.setData('application/reactflow', definition.id);
      event.dataTransfer.effectAllowed = 'move';
    },
    []
  );

  const handleSearchChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSearchQuery(e.target.value);
    },
    []
  );

  // Filter nodes by search query
  const filteredCategories = sortedCategories
    .map((category) => {
      const definitions = nodesByCategory[category] || [];
      if (!searchQuery) return { category, definitions };

      const query = searchQuery.toLowerCase();
      const filtered = definitions.filter(
        (def) =>
          def.name.toLowerCase().includes(query) ||
          def.description.toLowerCase().includes(query)
      );
      return { category, definitions: filtered };
    })
    .filter(({ definitions }) => definitions.length > 0);

  if (isCollapsed) {
    return (
      <aside className="sidebar sidebar-collapsed">
        <button
          className="sidebar-expand-btn"
          onClick={onToggleCollapse}
          title="Expand sidebar"
        >
          <ChevronDown size={20} className="rotate-90" />
        </button>
      </aside>
    );
  }

  return (
    <aside className="sidebar">
      {/* Header */}
      <div className="sidebar-header">
        <h2 className="sidebar-title">Node Library</h2>
        <p className="sidebar-subtitle">Drag nodes to canvas</p>
      </div>

      {/* Search */}
      <div className="sidebar-search">
        <Search size={16} className="sidebar-search-icon" />
        <input
          type="text"
          placeholder="Search nodes..."
          value={searchQuery}
          onChange={handleSearchChange}
          className="sidebar-search-input"
        />
      </div>

      {/* Node categories */}
      <div className="sidebar-content">
        {filteredCategories.map(({ category, definitions }) => (
          <CategorySection
            key={category}
            category={category}
            definitions={definitions}
            isExpanded={expandedCategories.has(category)}
            onToggle={() => handleToggleCategory(category)}
            onDragStart={handleDragStart}
          />
        ))}

        {filteredCategories.length === 0 && (
          <div className="sidebar-empty">
            <p>No nodes found</p>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="sidebar-footer">
        <span className="text-xs text-surface-500">
          {nodeDefinitions.length} nodes available
        </span>
      </div>
    </aside>
  );
});

export default Sidebar;
