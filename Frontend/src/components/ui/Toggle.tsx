import { memo } from 'react';

export interface ToggleProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  accentColor?: string;
}

export const Toggle = memo(function Toggle({
  checked,
  onChange,
  accentColor,
}: ToggleProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="toggle"
      style={{ backgroundColor: checked ? accentColor : undefined }}
      data-checked={checked}
    >
      <span className="toggle-knob" />
    </button>
  );
});
