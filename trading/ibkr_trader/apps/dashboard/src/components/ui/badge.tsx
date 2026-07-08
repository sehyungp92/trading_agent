import * as React from 'react';
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const badgeVariants = cva(
  'inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-semibold font-mono transition-colors',
  {
    variants: {
      variant: {
        default: 'border-transparent bg-gray-700 text-gray-200',
        success: 'border-transparent bg-green-900 text-green-200',
        danger: 'border-transparent bg-red-900 text-red-200',
        warning: 'border-transparent bg-amber-900 text-amber-200',
        info: 'border-transparent bg-blue-900 text-blue-200',
        outline: 'border-gray-700 text-gray-300',
      },
    },
    defaultVariants: {
      variant: 'default',
    },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
