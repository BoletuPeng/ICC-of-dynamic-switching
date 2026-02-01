import { memo, useState, useCallback, type DragEvent } from 'react';
import { ChevronDown, ChevronUp, Search, GripVertical } from 'lucide-react';
import { Icon } from './Icons';
import { useNodeDrag } from '../hooks';
import { nodesByCategory, sortedCategories, nodeDefinitions } from '../data';
import { CATEGORY_COLORS, CATEGORY_LABELS, type NodeDefinition, type NodeCategory } from '../types';

// Single node item
const NodeItem = memo(function NodeItem({
  definition,
  onDragStart,
}: {
  definition: NodeDefinition;
  onDragStart: (e: DragEvent, def: NodeDefinition) => void;
}) {
  return (
    <div
      className="sidebar-node"
      draggable
      onDragStart={(e) => onDragStart(e, definition)}
      style={{ '--node-color': definition.color } as React.CSSProperties}
    >
      <GripVertical size={14} className="sidebar-node-grip" />
      <div className="sidebar-node-icon" style={{ backgroundColor: definition.color }}>
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

// Category section
const CategorySection = memo(function CategorySection({
  category,
  definitions,
  isExpanded,
  onToggle,
  onDragStart,
}: {
  category: NodeCategory;
  definitions: NodeDefinition[];
  isExpanded: boolean;
  onToggle: () => void;
  onDragStart: (e: DragEvent, def: NodeDefinition) => void;
}) {
  const color = CATEGORY_COLORS[category] || '#64748b';
  const label = CATEGORY_LABELS[category] || category;

  return (
    <div className="sidebar-category">
      <button className="sidebar-category-header" onClick={onToggle}>
        <div className="flex items-center gap-2">
          <div className="sidebar-category-dot" style={{ backgroundColor: color }} />
          <span className="sidebar-category-label">{label}</span>
          <span className="sidebar-category-count">{definitions.length}</span>
        </div>
        {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>
      {isExpanded && (
        <div className="sidebar-category-content">
          {definitions.map((def) => (
            <NodeItem key={def.id} definition={def} onDragStart={onDragStart} />
          ))}
        </div>
      )}
    </div>
  );
});

// Main sidebar component
export const Sidebar = memo(function Sidebar() {
  const [expandedCategories, setExpandedCategories] = useState<Set<NodeCategory>>(
    () => new Set(sortedCategories)
  );
  const [searchQuery, setSearchQuery] = useState('');
  const { startDrag } = useNodeDrag();

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
    (e: DragEvent, definition: NodeDefinition) => {
      startDrag(e, definition.id);
    },
    [startDrag]
  );

  // Filter nodes by search
  const filteredCategories = sortedCategories
    .map((category) => {
      const definitions = nodesByCategory[category] || [];
      if (!searchQuery) return { category, definitions };
      const q = searchQuery.toLowerCase();
      return {
        category,
        definitions: definitions.filter(
          (d) => d.name.toLowerCase().includes(q) || d.description.toLowerCase().includes(q)
        ),
      };
    })
    .filter(({ definitions }) => definitions.length > 0);

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2 className="sidebar-title">Node Library</h2>
        <p className="sidebar-subtitle">Drag nodes to canvas</p>
      </div>

      <div className="sidebar-search">
        <Search size={16} className="sidebar-search-icon" />
        <input
          type="text"
          placeholder="Search nodes..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="sidebar-search-input"
        />
      </div>

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
          <div className="sidebar-empty">No nodes found</div>
        )}
      </div>

      <div className="sidebar-footer">
        <span className="text-xs text-surface-500">
          {nodeDefinitions.length} nodes available
        </span>
      </div>
    </aside>
  );
});

export default Sidebar;
