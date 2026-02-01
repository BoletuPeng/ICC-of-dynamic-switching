import { memo, type ReactNode } from 'react';

export interface FieldProps {
  label: string;
  description?: string;
  horizontal?: boolean;
  children: ReactNode;
}

export const Field = memo(function Field({
  label,
  description,
  horizontal = false,
  children,
}: FieldProps) {
  return (
    <div className={`field ${horizontal ? 'field-horizontal' : ''}`}>
      <div className="field-info">
        <label className="field-label">{label}</label>
        {description && <p className="field-description">{description}</p>}
      </div>
      {children}
    </div>
  );
});
