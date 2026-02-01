import { memo, useState, useCallback } from 'react';
import { Icon } from './Icons';
import { Modal, Button, ParameterInput } from './ui';
import { usePipelineStore, selectEditingNode } from '../store/pipelineStore';
import type { PipelineNode } from '../types';

// Port info display component
const PortList = memo(function PortList({
  title,
  ports,
}: {
  title: string;
  ports: { id: string; name: string; shape: (string | number)[] }[];
}) {
  return (
    <div className="port-section">
      <h4 className="port-title">{title}</h4>
      <div className="port-list">
        {ports.length > 0 ? (
          ports.map((port) => (
            <div key={port.id} className="port-item">
              <span className="port-name">{port.name}</span>
              <span className="port-shape">{port.shape.join(' × ')}</span>
            </div>
          ))
        ) : (
          <span className="port-empty">No {title.toLowerCase()}</span>
        )}
      </div>
    </div>
  );
});

// Editor content
const NodeEditorContent = memo(function NodeEditorContent({
  node,
}: {
  node: PipelineNode;
}) {
  const { setEditingNode, updateNodeParameters } = usePipelineStore();
  const { data } = node;

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
      {/* Ports Info */}
      <div className="ports-grid">
        <PortList title="Inputs" ports={data.inputs} />
        <PortList title="Outputs" ports={data.outputs} />
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

// Main component
export const NodeEditor = memo(function NodeEditor() {
  const editingNode = usePipelineStore(selectEditingNode);
  if (!editingNode) return null;
  return <NodeEditorContent key={editingNode.id} node={editingNode} />;
});

export default NodeEditor;
