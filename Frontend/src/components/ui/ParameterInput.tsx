import { memo, useMemo } from 'react';
import { Field } from './Field';
import { Toggle } from './Toggle';
import type { ParameterDefinition, SelectOption } from '../../types';

interface ParameterInputProps {
  param: ParameterDefinition;
  value: unknown;
  onChange: (value: unknown) => void;
  accentColor?: string;
}

// Number input renderer (int/float)
const NumberRenderer = memo(function NumberRenderer({
  param,
  value,
  onChange,
  accentColor,
}: ParameterInputProps) {
  const currentValue = value ?? param.default;
  const isFloat = param.type === 'float';

  return (
    <input
      type="number"
      value={currentValue === null ? '' : String(currentValue)}
      onChange={(e) => {
        const val = e.target.value;
        onChange(val === '' ? null : isFloat ? parseFloat(val) : parseInt(val, 10));
      }}
      min={param.min}
      max={param.max}
      step={param.step || (isFloat ? 0.01 : 1)}
      className="input"
      placeholder={param.default === null ? 'null (random)' : undefined}
      style={{ '--accent': accentColor } as React.CSSProperties}
    />
  );
});

// Select input renderer
const SelectRenderer = memo(function SelectRenderer({
  param,
  value,
  onChange,
  accentColor,
}: ParameterInputProps) {
  const currentValue = value ?? param.default;

  const options = useMemo(
    () =>
      param.options?.map((opt) =>
        typeof opt === 'object' ? (opt as SelectOption) : { value: opt, label: opt }
      ) ?? [],
    [param.options]
  );

  return (
    <select
      value={String(currentValue)}
      onChange={(e) => {
        const option = options.find((opt) => String(opt.value) === e.target.value);
        onChange(option?.value);
      }}
      className="select"
      style={{ '--accent': accentColor } as React.CSSProperties}
    >
      {options.map((opt) => (
        <option key={String(opt.value)} value={String(opt.value)}>
          {String(opt.label)}
        </option>
      ))}
    </select>
  );
});

// Text input renderer (string/array)
const TextRenderer = memo(function TextRenderer({
  param,
  value,
  onChange,
  accentColor,
}: ParameterInputProps) {
  const currentValue = value ?? param.default;

  return (
    <input
      type="text"
      value={String(currentValue ?? '')}
      onChange={(e) => onChange(e.target.value)}
      className="input"
      style={{ '--accent': accentColor } as React.CSSProperties}
    />
  );
});

// Boolean toggle renderer
const BooleanRenderer = memo(function BooleanRenderer({
  param,
  value,
  onChange,
  accentColor,
}: ParameterInputProps) {
  const currentValue = (value ?? param.default) as boolean;
  return <Toggle checked={currentValue} onChange={onChange} accentColor={accentColor} />;
});

// Renderer mapping by parameter type
const RENDERERS: Record<string, React.ComponentType<ParameterInputProps>> = {
  int: NumberRenderer,
  float: NumberRenderer,
  boolean: BooleanRenderer,
  select: SelectRenderer,
  string: TextRenderer,
  array: TextRenderer,
};

/**
 * Unified parameter input component.
 * Automatically selects the appropriate renderer based on parameter type.
 */
export const ParameterInput = memo(function ParameterInput(props: ParameterInputProps) {
  const { param } = props;
  const Renderer = RENDERERS[param.type] || TextRenderer;
  const isHorizontal = param.type === 'boolean';

  return (
    <Field label={param.name} description={param.description} horizontal={isHorizontal}>
      <Renderer {...props} />
    </Field>
  );
});

export type { ParameterInputProps };
