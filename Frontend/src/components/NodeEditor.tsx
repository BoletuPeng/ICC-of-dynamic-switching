import { useState, useEffect, useCallback } from 'react';
import { Icon, X } from './Icons';
import { nodeDefinitionsMap } from '../data';
import { usePipelineStore } from '../store/pipelineStore';
import type { ParameterDefinition, SelectOption } from '../types';

interface ParameterInputProps {
  param: ParameterDefinition;
  value: unknown;
  onChange: (value: unknown) => void;
}

function ParameterInput({ param, value, onChange }: ParameterInputProps) {
  const currentValue = value ?? param.default;

  switch (param.type) {
    case 'int':
    case 'float':
      return (
        <div>
          <label className="block text-sm font-medium text-surface-200 mb-1">
            {param.name}
          </label>
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
            className="w-full px-3 py-2 bg-surface-800 border border-surface-600 rounded-lg
                       text-surface-100 focus:outline-none focus:ring-2 focus:ring-primary-500
                       focus:border-transparent transition-all"
            placeholder={param.default === null ? 'null (random)' : undefined}
          />
          <p className="mt-1 text-xs text-surface-400">{param.description}</p>
        </div>
      );

    case 'boolean':
      return (
        <div className="flex items-center justify-between">
          <div>
            <label className="block text-sm font-medium text-surface-200">
              {param.name}
            </label>
            <p className="text-xs text-surface-400">{param.description}</p>
          </div>
          <button
            onClick={() => onChange(!currentValue)}
            className={`
              relative w-12 h-6 rounded-full transition-colors
              ${currentValue ? 'bg-primary-500' : 'bg-surface-600'}
            `}
          >
            <div
              className={`
                absolute top-1 w-4 h-4 bg-white rounded-full transition-transform
                ${currentValue ? 'translate-x-7' : 'translate-x-1'}
              `}
            />
          </button>
        </div>
      );

    case 'select':
      return (
        <div>
          <label className="block text-sm font-medium text-surface-200 mb-1">
            {param.name}
          </label>
          <select
            value={String(currentValue)}
            onChange={(e) => {
              const option = param.options?.find((opt) =>
                typeof opt === 'object'
                  ? String(opt.value) === e.target.value
                  : String(opt) === e.target.value
              );
              if (typeof option === 'object') {
                onChange(option.value);
              } else {
                onChange(option);
              }
            }}
            className="w-full px-3 py-2 bg-surface-800 border border-surface-600 rounded-lg
                       text-surface-100 focus:outline-none focus:ring-2 focus:ring-primary-500
                       focus:border-transparent transition-all appearance-none cursor-pointer"
          >
            {param.options?.map((option) => {
              const opt = typeof option === 'object' ? option as SelectOption : { value: option, label: option };
              return (
                <option key={String(opt.value)} value={String(opt.value)}>
                  {String(opt.label)}
                </option>
              );
            })}
          </select>
          <p className="mt-1 text-xs text-surface-400">{param.description}</p>
        </div>
      );

    case 'string':
    case 'array':
      return (
        <div>
          <label className="block text-sm font-medium text-surface-200 mb-1">
            {param.name}
          </label>
          <input
            type="text"
            value={String(currentValue ?? '')}
            onChange={(e) => onChange(e.target.value)}
            className="w-full px-3 py-2 bg-surface-800 border border-surface-600 rounded-lg
                       text-surface-100 focus:outline-none focus:ring-2 focus:ring-primary-500
                       focus:border-transparent transition-all"
          />
          <p className="mt-1 text-xs text-surface-400">{param.description}</p>
        </div>
      );

    default:
      return null;
  }
}

export function NodeEditor() {
  const { editingNodeId, setEditingNode, nodes, updateNodeParameters } = usePipelineStore();

  const node = nodes.find((n) => n.id === editingNodeId);
  const definition = node ? nodeDefinitionsMap[node.definitionId] : null;

  const [localParams, setLocalParams] = useState<Record<string, unknown>>({});

  useEffect(() => {
    if (node) {
      setLocalParams({ ...node.parameters });
    }
  }, [node]);

  const handleParamChange = useCallback((paramId: string, value: unknown) => {
    setLocalParams((prev) => ({ ...prev, [paramId]: value }));
  }, []);

  const handleSave = useCallback(() => {
    if (editingNodeId) {
      updateNodeParameters(editingNodeId, localParams);
      setEditingNode(null);
    }
  }, [editingNodeId, localParams, updateNodeParameters, setEditingNode]);

  const handleClose = useCallback(() => {
    setEditingNode(null);
  }, [setEditingNode]);

  if (!editingNodeId || !node || !definition) {
    return null;
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in"
      onClick={handleClose}
    >
      <div
        className="w-full max-w-lg max-h-[80vh] glass rounded-2xl overflow-hidden shadow-2xl animate-slide-in"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div
          className="px-6 py-4 flex items-center gap-3"
          style={{ backgroundColor: definition.color }}
        >
          <Icon name={definition.icon} className="text-white" size={24} />
          <div className="flex-1">
            <h2 className="text-lg font-semibold text-white">{definition.name}</h2>
            <p className="text-sm text-white/70">{definition.description}</p>
          </div>
          <button
            onClick={handleClose}
            className="p-2 rounded-lg hover:bg-white/20 transition-colors"
          >
            <X className="text-white" size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[50vh]">
          {/* Inputs & Outputs Info */}
          <div className="mb-6 grid grid-cols-2 gap-4">
            <div className="p-3 bg-surface-800/50 rounded-lg">
              <h4 className="text-xs font-medium text-surface-400 uppercase tracking-wider mb-2">
                Inputs
              </h4>
              <div className="space-y-1">
                {definition.inputs.map((input) => (
                  <div key={input.id} className="text-sm">
                    <span className="text-surface-200">{input.name}</span>
                    <span className="text-surface-500 text-xs ml-2">
                      {input.shape.join(' × ')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
            <div className="p-3 bg-surface-800/50 rounded-lg">
              <h4 className="text-xs font-medium text-surface-400 uppercase tracking-wider mb-2">
                Outputs
              </h4>
              <div className="space-y-1">
                {definition.outputs.map((output) => (
                  <div key={output.id} className="text-sm">
                    <span className="text-surface-200">{output.name}</span>
                    <span className="text-surface-500 text-xs ml-2">
                      {output.shape.join(' × ')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Parameters */}
          {definition.parameters.length > 0 && (
            <div>
              <h3 className="text-sm font-medium text-surface-300 uppercase tracking-wider mb-4">
                Parameters
              </h3>
              <div className="space-y-4">
                {definition.parameters.map((param) => (
                  <ParameterInput
                    key={param.id}
                    param={param}
                    value={localParams[param.id]}
                    onChange={(value) => handleParamChange(param.id, value)}
                  />
                ))}
              </div>
            </div>
          )}

          {definition.parameters.length === 0 && (
            <div className="text-center py-8 text-surface-400">
              This node has no configurable parameters
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-surface-700/50 flex justify-end gap-3">
          <button
            onClick={handleClose}
            className="px-4 py-2 text-surface-300 hover:text-surface-100 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className="px-6 py-2 rounded-lg font-medium text-white transition-all"
            style={{ backgroundColor: definition.color }}
          >
            Save Changes
          </button>
        </div>
      </div>
    </div>
  );
}
