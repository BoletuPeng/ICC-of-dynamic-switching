import { memo, type ButtonHTMLAttributes, type ReactNode } from 'react';

type ButtonVariant = 'ghost' | 'primary' | 'danger';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  icon?: ReactNode;
}

export const Button = memo(function Button({
  variant = 'ghost',
  icon,
  children,
  className = '',
  ...props
}: ButtonProps) {
  const variantClass = {
    ghost: 'btn-ghost',
    primary: 'btn-primary',
    danger: 'btn-danger',
  }[variant];

  return (
    <button className={`btn ${variantClass} ${className}`} {...props}>
      {icon}
      {children}
    </button>
  );
});
