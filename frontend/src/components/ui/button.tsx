import { cva, type VariantProps } from "class-variance-authority";
import type { ButtonHTMLAttributes } from "react";
import { cn } from "../../lib/utils";

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 rounded-lg text-sm font-semibold tracking-wide transition-all duration-200 disabled:pointer-events-none disabled:opacity-40",
  {
    variants: {
      variant: {
        primary:
          "bg-cyan-400 text-slate-950 hover:bg-cyan-300 shadow-[0_0_24px_rgba(34,211,238,0.18)]",
        secondary:
          "border border-white/12 bg-white/[0.04] text-slate-100 hover:border-cyan-400/40 hover:bg-white/[0.07]",
        ghost: "text-slate-300 hover:text-white hover:bg-white/[0.05]",
        brutal:
          "border border-cyan-300/80 bg-cyan-400 text-slate-950 shadow-[4px_4px_0_rgba(34,211,238,0.35)] hover:translate-x-px hover:translate-y-px hover:shadow-[3px_3px_0_rgba(34,211,238,0.35)]",
      },
      size: {
        sm: "h-8 px-3 text-xs",
        md: "h-10 px-4",
        lg: "h-12 px-6 text-[13px]",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "md",
    },
  },
);

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> &
  VariantProps<typeof buttonVariants>;

export function Button({ className, variant, size, ...props }: ButtonProps) {
  return <button className={cn(buttonVariants({ variant, size }), className)} {...props} />;
}
