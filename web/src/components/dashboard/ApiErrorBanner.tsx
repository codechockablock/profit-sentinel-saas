"use client";

import React, { useState } from "react";
import { AlertTriangle, LogIn, RefreshCw, ShieldX, Clock } from "lucide-react";
import { AuthModal } from "@/components/auth/AuthModal";

interface ApiErrorBannerProps {
  error: string | null;
  /** Override the auto-detected status code */
  statusCode?: number;
  /** Callback to retry the failed request */
  onRetry?: () => void;
}

function parseStatusCode(error: string | null): number | null {
  if (!error) return null;
  const match = error.match(/API (\d{3})/);
  return match ? parseInt(match[1], 10) : null;
}

/**
 * Standardized error banner for all dashboard pages.
 *
 * Interprets API error messages and shows user-friendly text:
 * - 401 → "Session expired" with Sign In button
 * - 403 → "Permission denied"
 * - 429 → "Rate limited" with retry
 * - 5xx → "Server error" with retry
 * - Other → show actual message
 */
export function ApiErrorBanner({ error, statusCode, onRetry }: ApiErrorBannerProps) {
  const [showAuthModal, setShowAuthModal] = useState(false);

  if (!error) return null;

  const code = statusCode ?? parseStatusCode(error);

  let icon = <AlertTriangle className="text-red-400 shrink-0" size={18} />;
  let title = "Something went wrong";
  let message = error;
  let action: React.ReactNode = null;

  if (code === 401) {
    icon = <LogIn className="text-amber-400 shrink-0" size={18} />;
    title = "Your session has expired";
    message = "Please sign in again to access this page.";
    action = (
      <button
        onClick={() => setShowAuthModal(true)}
        className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition-colors"
      >
        Sign In
      </button>
    );
  } else if (code === 403) {
    icon = <ShieldX className="text-red-400 shrink-0" size={18} />;
    title = "Access denied";
    message = "You don't have permission to access this feature.";
  } else if (code === 429) {
    icon = <Clock className="text-amber-400 shrink-0" size={18} />;
    title = "Too many requests";
    message = "Please wait a moment and try again.";
    if (onRetry) {
      action = (
        <button
          onClick={onRetry}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
        >
          <RefreshCw size={14} /> Retry
        </button>
      );
    }
  } else if (code && code >= 500) {
    title = "Server error";
    message = "Something went wrong on our end. Please try again later.";
    if (onRetry) {
      action = (
        <button
          onClick={onRetry}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
        >
          <RefreshCw size={14} /> Retry
        </button>
      );
    }
  }

  const borderColor = code === 401 || code === 429 ? "border-amber-500/30" : "border-red-500/30";
  const bgColor = code === 401 || code === 429 ? "bg-amber-500/10" : "bg-red-500/10";

  return (
    <>
      <div className={`${bgColor} ${borderColor} border rounded-xl p-4 mb-6`}>
        <div className="flex items-start gap-3">
          {icon}
          <div className="flex-1 min-w-0">
            <p className="font-medium text-white text-sm">{title}</p>
            <p className="text-sm text-slate-400 mt-0.5">{message}</p>
          </div>
          {action}
        </div>
      </div>

      <AuthModal
        isOpen={showAuthModal}
        onClose={() => setShowAuthModal(false)}
        onSuccess={() => {
          setShowAuthModal(false);
          onRetry?.();
        }}
        defaultMode="login"
      />
    </>
  );
}
