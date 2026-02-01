import { memo, useState, useCallback } from 'react';
import { Icon } from './Icons';
import { Modal, Button, ParameterInput } from './ui';
import { usePipelineStore, selectEditingNode } from '../store/pipelineStore';
import { useNodeShapeInfo } from '../hooks/useShapeInference';
import type { PipelineNode, ResolvedPortShape } from '../types';

/**
 * Shape display with both symbolic and resolved values
 */
const ShapeInfo = memo(function ShapeInfo({
  shape,
  color,
}: {
  shape: ResolvedPortShape | undefined;
  color: string;
}) {
  if (!shape) {
    return <span className="port-shape port-shape-unknown">unknown</span>;
  }

  if (shape.isFullyResolved && shape.resolved) {
    return (
      <div className="port-shape-container">
        <span className="port-shape port-shape-symbolic">{shape.symbolic}</span>
        <span className="port-shape-equals">=</span>
        <span className="port-shape port-shape-resolved" style={{ color }}>
          {shape.resolved}
        </span>
      </div>
    );
  }

  return <span className="port-shape port-shape-symbolic">{shape.symbolic}</span>;
});

/**
 * Port info display component with shape inference
 */
const PortList = memo(function PortList({
  title,
  ports,
  shapes,
  color,
}: {
  title: string;
  ports: { id: string; name: string; shape: (string | number)[]; description: string }[];
  shapes?: Record<string, ResolvedPortShape>;
  color: string;
}) {
  return (
    <div className="port-section">
      <h4 className="port-title">{title}</h4>
      <div className="port-list">
        {ports.length > 0 ? (
          ports.map((port) => (
            <div key={port.id} className="port-item-detailed">
              <div className="port-item-header">
                <span className="port-name">{port.name}</span>
              </div>
              <ShapeInfo shape={shapes?.[port.id]} color={color} />
              {port.description && (
                <span className="port-description">{port.description}</span>
              )}
            </div>
          ))
        ) : (
          <span className="port-empty">No {title.toLowerCase()}</span>
        )}
      </div>
    </div>
  );
});

/**
 * Dimension bindings display
 */
const DimensionBindings = memo(function DimensionBindings({
  bindings,
  color,
}: {
  bindings: Record<string, number> | undefined;
  color: string;
}) {
  if (!bindings || Object.keys(bindings).length === 0) {
    return null;
  }

  return (
    <div className="dimension-bindings">
      <h4 className="port-title">Known Dimensions</h4>
      <div className="bindings-list">
        {Object.entries(bindings).map(([key, value]) => (
          <div key={key} className="binding-item">
            <span className="binding-name">{key}</span>
            <span className="binding-equals">=</span>
            <span className="binding-value" style={{ color }}>{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
});

/**
 * Editor content
 */
const NodeEditorContent = memo(function NodeEditorContent({
  node,
}: {
  node: PipelineNode;
}) {
  const { setEditingNode, updateNodeParameters } = usePipelineStore();
  const { data } = node;
  const shapeInfo = useNodeShapeInfo(node.id);

  const [localParams, setLocalParams] = useState<Record<string, unknown>>(
    () => ({ ...data.parameters })
  );

  const handleParamChange = useCallback((paramId: string, value: unknown) => {
    setLocalParams((prev) => ({ ...prev, [paramId]: value }));
  }, []);

  const handleSave = useCallback(() => {
    updateNodeParameters(node.id, localParams);
    setEditingNode(null);
  }, [node.id, localParams, updateNodeParameters, setEditingNode]);

  const handleClose = useCallback(() => setEditingNode(null), [setEditingNode]);

  const footer = (
    <>
      <Button variant="ghost" onClick={handleClose}>
        Cancel
      </Button>
      <Button
        variant="primary"
        onClick={handleSave}
        style={{ backgroundColor: data.color }}
      >
        Save Changes
      </Button>
    </>
  );

  return (
    <Modal
      isOpen
      onClose={handleClose}
      title={data.label}
      subtitle={data.description}
      icon={<Icon name={data.icon} className="text-white" size={24} />}
      accentColor={data.color}
      footer={footer}
    >
      {/* Shape Overview */}
      <div className="shape-overview">
        <div className="ports-grid">
          <PortList
            title="Inputs"
            ports={data.inputs}
            shapes={shapeInfo?.inputShapes}
            color={data.color}
          />
          <PortList
            title="Outputs"
            ports={data.outputs}
            shapes={shapeInfo?.outputShapes}
            color={data.color}
          />
        </div>

        <DimensionBindings
          bindings={shapeInfo?.dimensionBindings}
          color={data.color}
        />
      </div>

      {/* Parameters */}
      {data.parameterDefinitions.length > 0 ? (
        <div className="parameters-section">
          <h3 className="section-title">Parameters</h3>
          <div className="parameters-list">
            {data.parameterDefinitions.map((param) => (
              <ParameterInput
                key={param.id}
                param={param}
                value={localParams[param.id]}
                onChange={(value) => handleParamChange(param.id, value)}
                accentColor={data.color}
              />
            ))}
          </div>
        </div>
      ) : (
        <div className="no-params">This node has no configurable parameters</div>
      )}
    </Modal>
  );
});

/**
 * Main component
 */
export const NodeEditor = memo(function NodeEditor() {
  const editingNode = usePipelineStore(selectEditingNode);
  if (!editingNode) return null;
  return <NodeEditorContent key={editingNode.id} node={editingNode} />;
});

export default NodeEditor;
