"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ClipboardList,
  Phone,
  TrendingUp,
  Brain,
  Stethoscope,
  History,
  BarChart3,
  AlertTriangle,
  Key,
  Plug,
  ChevronLeft,
  ChevronRight,
  Lock,
  Search,
  Truck,
} from "lucide-react";
import { getSupabase } from "@/lib/supabase";
import { AuthModal } from "@/components/auth/AuthModal";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Morning Digest", icon: LayoutDashboard, description: "Daily issue overview" },
  { href: "/dashboard/tasks", label: "Tasks", icon: ClipboardList, description: "Delegated action items" },
  { href: "/dashboard/vendor", label: "Vendor Prep", icon: Phone, description: "Call talking points" },
  { href: "/dashboard/coop", label: "Co-op Intel", icon: TrendingUp, description: "Rebate & patronage" },
  { href: "/dashboard/explain", label: "Explain", icon: Brain, description: "Symbolic reasoning" },
  { href: "/dashboard/diagnostic", label: "Diagnostic", icon: Stethoscope, description: "Shrinkage wizard" },
  { href: "/dashboard/history", label: "History", icon: History, description: "Track & compare" },
  { href: "/dashboard/vendor-scores", label: "Vendor Scores", icon: BarChart3, description: "Performance scoring" },
  { href: "/dashboard/findings", label: "Findings", icon: Search, description: "Ranked by impact" },
  { href: "/dashboard/predictions", label: "Predictions", icon: AlertTriangle, description: "Inventory forecasts" },
  { href: "/dashboard/transfers", label: "Transfers", icon: Truck, description: "Cross-store moves" },
  { href: "/dashboard/api-keys", label: "API Keys", icon: Key, description: "Enterprise access" },
  { href: "/dashboard/pos", label: "POS Connect", icon: Plug, description: "System integrations" },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [collapsed, setCollapsed] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState<boolean | null>(null);
  const [showAuthModal, setShowAuthModal] = useState(false);

  useEffect(() => {
    const supabase = getSupabase();
    if (!supabase) {
      // No Supabase config — allow access (dev mode)
      setIsAuthenticated(true);
      return;
    }

    supabase.auth.getSession().then(({ data: { session } }: { data: { session: unknown } }) => {
      setIsAuthenticated(!!session);
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event: string, session: unknown) => {
      setIsAuthenticated(!!session);
    });

    return () => subscription.unsubscribe();
  }, []);

  // Loading state
  if (isAuthenticated === null) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  // Auth gate
  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center px-4">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 bg-emerald-500/10 rounded-2xl flex items-center justify-center mx-auto mb-6">
            <Lock size={32} className="text-emerald-400" />
          </div>
          <h1 className="text-2xl font-bold text-white mb-3">Executive Dashboard</h1>
          <p className="text-slate-400 mb-8">
            Sign in to access your morning digest, task delegation, co-op intelligence, and more.
          </p>
          <button
            onClick={() => setShowAuthModal(true)}
            className="px-6 py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg transition-colors"
          >
            Sign In
          </button>
          <AuthModal
            isOpen={showAuthModal}
            onClose={() => setShowAuthModal(false)}
            onSuccess={() => setIsAuthenticated(true)}
            defaultMode="login"
          />
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 flex">
      {/* Sidebar */}
      <aside
        className={`sticky top-0 h-screen bg-slate-800/50 border-r border-slate-700/50 flex flex-col transition-all duration-200 ${
          collapsed ? "w-16" : "w-56"
        }`}
      >
        {/* Collapse toggle */}
        <div className="p-3 flex justify-end">
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1.5 text-slate-500 hover:text-slate-300 rounded-lg hover:bg-slate-700/50 transition-colors"
            aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          >
            {collapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
          </button>
        </div>

        {/* Nav items */}
        <nav className="flex-1 px-2 space-y-1">
          {NAV_ITEMS.map((item) => {
            const isActive =
              item.href === "/dashboard"
                ? pathname === "/dashboard"
                : pathname.startsWith(item.href);

            return (
              <Link
                key={item.href}
                href={item.href}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                    : "text-slate-400 hover:text-white hover:bg-slate-700/50 border border-transparent"
                }`}
                title={collapsed ? item.label : undefined}
              >
                <item.icon size={18} className="shrink-0" />
                {!collapsed && (
                  <div className="min-w-0">
                    <div className="truncate">{item.label}</div>
                    <div className="text-[10px] text-slate-500 truncate">{item.description}</div>
                  </div>
                )}
              </Link>
            );
          })}
        </nav>

        {/* Back to analyze */}
        <div className="p-3 border-t border-slate-700/50">
          <Link
            href="/analyze"
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs text-slate-500 hover:text-slate-300 hover:bg-slate-700/50 transition-colors"
          >
            <ChevronLeft size={14} />
            {!collapsed && "Back to Analyze"}
          </Link>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
}
