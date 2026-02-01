import { memo, useState, useCallback } from 'react';
import { X } from 'lucide-react';
import { Icon } from './Icons';
import { usePipelineStore, selectEditingNode } from '../store/pipelineStore';
import type { ParameterDefinition, SelectOption, PipelineNode } from '../types';

// =============================================================================
// Parameter Input Components
// =============================================================================

interface ParameterInputProps {
  param: ParameterDefinition;
  value: unknown;
  onChange: (value: unknown) => void;
  accentColor: string;
}

const NumberInput = memo(function NumberInput({
  param,
  value,
  onChange,
  accentColor,
}: ParameterInputProps) {
  const currentValue = value ?? param.default;

  return (
    <div className="parameter-field">
      <label className="parameter-label">{param.name}</label>
      <input
        type="number"
        value={currentValue === null ? '' : String(currentValue)}
        onChange={(e) => {
          const val = e.target.value;
          if (val === '') {
            onChange(null);
          } else {
            onChange(param.type === 'int' ? parseInt(val, 10) : parseFloat(val));
          }
        }}
        min={param.min}
        max={param.max}
        step={param.step || (param.type === 'float' ? 0.01 : 1)}
        className="parameter-input"
        placeholder={param.default === null ? 'null (random)' : undefined}
        style={{ '--accent-color': accentColor } as React.CSSProperties}
      />
      <p className="parameter-description">{param.description}</p>
    </div>
  );
});

const BooleanInput = memo(function BooleanInput({
  param,
  value,
  onChange,
  accentColor,
}: ParameterInputProps) {
  const currentValue = value ?? param.default;

  return (
    <div className="parameter-field parameter-field-boolean">
      <div className="parameter-field-boolean-info">
        <label className="parameter-label">{param.name}</label>
        <p className="parameter-description">{param.description}</p>
      </div>
      <button
        onClick={() => onChange(!currentValue)}
        className="parameter-toggle"
        style={{
          backgroundColor: currentValue ? accentColor : undefined,
        }}
        data-checked={currentValue}
      >
        <div className="parameter-toggle-knob" />
      </button>
    </div>
  );
});

const SelectInput = memo(function SelectInput({
  param,
  value,
  onChange,
  accentColor,
}: ParameterInputProps) {
  const currentValue = value ?? param.default;

  return (
    <div className="parameter-field">
      <label className="parameter-label">{param.name}</label>
      <select
        value={String(currentValue)}
        onChange={(e) => {
          const option = param.options?.find((opt) =>
            typeof opt === 'object'
              ? String(opt.value) === e.target.value
              : String(opt) === e.target.value
          );
          if (typeof option === 'object') {
            onChange((option as SelectOption).value);
          } else {
            onChange(option);
          }
        }}
        className="parameter-select"
        style={{ '--accent-color': accentColor } as React.CSSProperties}
      >
        {param.options?.map((option) => {
          const opt =
            typeof option === 'object'
              ? (option as SelectOption)
              : { value: option, label: option };
          return (
            <option key={String(opt.value)} value={String(opt.value)}>
              {String(opt.label)}
            </option>
          );
        })}
      </select>
      <p className="parameter-description">{param.description}</p>
    </div>
  );
});

const TextInput = memo(function TextInput({
  param,
  value,
  onChange,
  accentColor,
}: ParameterInputProps) {
  const currentValue = value ?? param.default;

  return (
    <div className="parameter-field">
      <label className="parameter-label">{param.name}</label>
      <input
        type="text"
        value={String(currentValue ?? '')}
        onChange={(e) => onChange(e.target.value)}
        className="parameter-input"
        style={{ '--accent-color': accentColor } as React.CSSProperties}
      />
      <p className="parameter-description">{param.description}</p>
    </div>
  );
});

function ParameterInput(props: ParameterInputProps) {
  const { param } = props;

  switch (param.type) {
    case 'int':
    case 'float':
      return <NumberInput {...props} />;
    case 'boolean':
      return <BooleanInput {...props} />;
    case 'select':
      return <SelectInput {...props} />;
    case 'string':
    case 'array':
      return <TextInput {...props} />;
    default:
      return null;
  }
}

// =============================================================================
// Node Editor Content
// =============================================================================

interface NodeEditorContentProps {
  node: PipelineNode;
}

const NodeEditorContent = memo(function NodeEditorContent({
  node,
}: NodeEditorContentProps) {
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

  const handleClose = useCallback(() => {
    setEditingNode(null);
  }, [setEditingNode]);

  const handleBackdropClick = useCallback(
    (e: React.MouseEvent) => {
      if (e.target === e.currentTarget) {
        handleClose();
      }
    },
    [handleClose]
  );

  return (
    <div className="node-editor-overlay" onClick={handleBackdropClick}>
      <div className="node-editor-modal" onClick={(e) => e.stopPropagation()}>
        {/* Header */}
        <div
          className="node-editor-header"
          style={{ backgroundColor: data.color }}
        >
          <Icon name={data.icon} className="text-white" size={24} />
          <div className="node-editor-header-info">
            <h2 className="node-editor-title">{data.label}</h2>
            <p className="node-editor-description">{data.description}</p>
          </div>
          <button onClick={handleClose} className="node-editor-close-btn">
            <X size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="node-editor-content">
          {/* Inputs & Outputs Info */}
          <div className="node-editor-ports">
            <div className="node-editor-port-section">
              <h4 className="node-editor-port-title">Inputs</h4>
              <div className="node-editor-port-list">
                {data.inputs.map((input) => (
                  <div key={input.id} className="node-editor-port-item">
                    <span className="node-editor-port-name">{input.name}</span>
                    <span className="node-editor-port-shape">
                      {input.shape.join(' × ')}
                    </span>
                  </div>
                ))}
                {data.inputs.length === 0 && (
                  <span className="node-editor-port-empty">No inputs</span>
                )}
              </div>
            </div>
            <div className="node-editor-port-section">
              <h4 className="node-editor-port-title">Outputs</h4>
              <div className="node-editor-port-list">
                {data.outputs.map((output) => (
                  <div key={output.id} className="node-editor-port-item">
                    <span className="node-editor-port-name">{output.name}</span>
                    <span className="node-editor-port-shape">
                      {output.shape.join(' × ')}
                    </span>
                  </div>
                ))}
                {data.outputs.length === 0 && (
                  <span className="node-editor-port-empty">No outputs</span>
                )}
              </div>
            </div>
          </div>

          {/* Parameters */}
          {data.parameterDefinitions.length > 0 && (
            <div className="node-editor-parameters">
              <h3 className="node-editor-section-title">Parameters</h3>
              <div className="node-editor-parameter-list">
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
          )}

          {data.parameterDefinitions.length === 0 && (
            <div className="node-editor-no-params">
              This node has no configurable parameters
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="node-editor-footer">
          <button onClick={handleClose} className="node-editor-cancel-btn">
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="node-editor-save-btn"
            style={{ backgroundColor: data.color }}
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
});

// =============================================================================
// Node Editor (Main Component)
// =============================================================================

export const NodeEditor = memo(function NodeEditor() {
  const editingNode = usePipelineStore(selectEditingNode);

  if (!editingNode) {
    return null;
  }

  // Use key to force remount when editing different nodes
  return <NodeEditorContent key={editingNode.id} node={editingNode} />;
});

export default NodeEditor;
