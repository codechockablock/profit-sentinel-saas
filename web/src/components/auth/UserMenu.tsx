"use client";

import React, { useState, useEffect } from "react";
import { LogOut, User } from "lucide-react";
import { getSupabase } from "@/lib/supabase";
import { robustSignOut } from "@/lib/auth-helpers";
import { useRouter } from "next/navigation";

interface UserMenuProps {
  /** Compact mode for sidebar footer */
  compact?: boolean;
  /** Whether sidebar is collapsed (icon only) */
  collapsed?: boolean;
}

export function UserMenu({ compact = false, collapsed = false }: UserMenuProps) {
  const [email, setEmail] = useState<string | null>(null);
  const [signingOut, setSigningOut] = useState(false);
  const router = useRouter();

  useEffect(() => {
    const supabase = getSupabase();
    if (!supabase) return;

    supabase.auth.getSession().then(({ data: { session } }: { data: { session: { user?: { email?: string } } | null } }) => {
      setEmail(session?.user?.email ?? null);
    });

    const { data: { subscription } } = supabase.auth.onAuthStateChange(
      (_event: string, session: { user?: { email?: string } } | null) => {
        setEmail(session?.user?.email ?? null);
      }
    );

    return () => subscription.unsubscribe();
  }, []);

  const handleSignOut = async () => {
    setSigningOut(true);
    await robustSignOut();
    router.push("/");
  };

  if (!email) return null;

  if (compact) {
    return (
      <div className="space-y-2">
        {!collapsed && (
          <div className="px-3 text-xs text-slate-500 truncate" title={email}>
            {email}
          </div>
        )}
        <button
          onClick={handleSignOut}
          disabled={signingOut}
          className="flex items-center gap-2 w-full px-3 py-2 rounded-lg text-xs text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-colors disabled:opacity-50"
          title={collapsed ? "Sign Out" : undefined}
        >
          <LogOut size={14} className="shrink-0" />
          {!collapsed && (signingOut ? "Signing out..." : "Sign Out")}
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2 text-sm text-slate-300">
        <User size={16} className="text-slate-500" />
        <span className="truncate max-w-[160px]" title={email}>
          {email}
        </span>
      </div>
      <button
        onClick={handleSignOut}
        disabled={signingOut}
        className="text-sm text-slate-400 hover:text-red-400 transition-colors disabled:opacity-50"
      >
        {signingOut ? "..." : "Sign Out"}
      </button>
    </div>
  );
}
