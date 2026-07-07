import { cn } from "@/lib/utils";

const variants: Record<string, string> = {
  green: "bg-accent-green/15 text-accent-green border-accent-green/30",
  red: "bg-accent-red/15 text-accent-red border-accent-red/30",
  amber: "bg-accent-amber/15 text-accent-amber border-accent-amber/30",
  blue: "bg-accent-blue/15 text-accent-blue border-accent-blue/30",
  neutral: "bg-zinc-800 text-zinc-400 border-zinc-700",
};

export function Badge({
  variant = "neutral",
  className,
  children,
}: {
  variant?: keyof typeof variants;
  className?: string;
  children: React.ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-md border px-2 py-0.5 text-xs font-medium",
        variants[variant] ?? variants.neutral,
        className
      )}
    >
      {children}
    </span>
  );
}
