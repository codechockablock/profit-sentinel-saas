"use client";

import React, { useState, useEffect } from "react";
import { X, Mail } from "lucide-react";
import { usePathname, useRouter } from "next/navigation";
import { getSupabase } from "@/lib/supabase";

interface AuthModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
  defaultMode?: "login" | "signup";
  /** Skip automatic redirect after login (caller handles it) */
  skipRedirect?: boolean;
}

export function AuthModal({
  isOpen,
  onClose,
  onSuccess,
  defaultMode = "login",
  skipRedirect = false,
}: AuthModalProps) {
  const [mode, setMode] = useState<"login" | "signup">(defaultMode);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [confirmationSent, setConfirmationSent] = useState(false);
  const pathname = usePathname();
  const router = useRouter();

  // Reset mode to defaultMode every time the modal opens
  useEffect(() => {
    if (isOpen) {
      setMode(defaultMode);
      setError(null);
      setConfirmationSent(false);
    }
  }, [isOpen, defaultMode]);

  const handlePostLoginRedirect = () => {
    if (skipRedirect) return;

    // If on a dashboard page (lock screen sign-in), just reload to show content
    if (pathname?.startsWith("/dashboard")) {
      router.refresh();
      return;
    }

    // If on /analyze and explicitly signed in, go to dashboard
    if (pathname === "/analyze" || pathname === "/diagnostic") {
      router.push("/dashboard");
      return;
    }

    // Default: go to dashboard
    router.push("/dashboard");
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    const supabase = getSupabase();
    if (!supabase) {
      setError("Authentication service not configured");
      setLoading(false);
      return;
    }

    try {
      if (mode === "login") {
        const { error } = await supabase.auth.signInWithPassword({
          email,
          password,
        });
        if (error) {
          if (error.message === "Invalid login credentials") {
            throw new Error(
              "Invalid login credentials. If you just signed up, check your email to confirm your account first."
            );
          }
          if (error.message === "Email not confirmed") {
            throw new Error(
              "Please confirm your email before signing in. Check your inbox for a confirmation link."
            );
          }
          throw error;
        }
        onSuccess();
        onClose();
        handlePostLoginRedirect();
      } else {
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
        });
        if (error) throw error;

        if (data?.user && !data?.session) {
          // Email confirmation required
          setConfirmationSent(true);
        } else if (data?.session) {
          // No confirmation needed — signed in immediately
          onSuccess();
          onClose();
          handlePostLoginRedirect();
        }
      }
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Authentication failed";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const handleResendConfirmation = async () => {
    const supabase = getSupabase();
    if (!supabase || !email) return;

    setLoading(true);
    setError(null);

    try {
      const { error } = await supabase.auth.resend({
        type: "signup",
        email,
      });
      if (error) throw error;
      setError(null);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : "Failed to resend email";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  // Confirmation sent screen
  if (confirmationSent) {
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
        <div className="relative bg-slate-900 rounded-xl p-8 w-full max-w-md border border-slate-700 shadow-2xl">
          <button
            onClick={onClose}
            className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
            aria-label="Close"
          >
            <X size={20} />
          </button>

          <div className="text-center">
            <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-emerald-500/20 flex items-center justify-center">
              <Mail size={32} className="text-emerald-400" />
            </div>

            <h2 className="text-2xl font-bold text-white mb-3">
              Check your email
            </h2>

            <p className="text-slate-400 mb-2">
              We sent a confirmation link to
            </p>
            <p className="text-white font-medium mb-6">{email}</p>

            <p className="text-slate-500 text-sm mb-6">
              Click the link in the email to activate your account, then come
              back here to sign in.
            </p>

            {error && (
              <div className="text-red-400 text-sm bg-red-950/30 px-3 py-2 rounded-lg border border-red-900/50 mb-4">
                {error}
              </div>
            )}

            <div className="space-y-3">
              <button
                onClick={handleResendConfirmation}
                disabled={loading}
                className="w-full py-3 bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium rounded-lg transition-colors border border-slate-600 disabled:opacity-50"
              >
                {loading ? "Sending..." : "Resend confirmation email"}
              </button>

              <button
                onClick={() => {
                  setConfirmationSent(false);
                  setMode("login");
                  setError(null);
                }}
                className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg transition-colors"
              >
                I confirmed — Sign in
              </button>
            </div>

            <p className="text-slate-600 text-xs mt-4">
              Check your spam folder if you don&apos;t see the email.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="relative bg-slate-900 rounded-xl p-8 w-full max-w-md border border-slate-700 shadow-2xl">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-slate-400 hover:text-white transition-colors"
          aria-label="Close"
        >
          <X size={20} />
        </button>

        <h2 className="text-2xl font-bold text-white mb-6">
          {mode === "login" ? "Welcome back" : "Create your account"}
        </h2>

        {mode === "signup" && (
          <p className="text-slate-400 text-sm mb-6">
            Get 50MB uploads, 100 analyses/hour, and full diagnostic tools.
          </p>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-slate-400 mb-1">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-emerald-500 transition-colors"
              placeholder="you@company.com"
              required
            />
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-4 py-3 bg-slate-800 border border-slate-600 rounded-lg text-white focus:outline-none focus:border-emerald-500 transition-colors"
              placeholder={
                mode === "signup" ? "At least 8 characters" : "Your password"
              }
              required
              minLength={mode === "signup" ? 8 : undefined}
              autoComplete={mode === "signup" ? "new-password" : "current-password"}
            />
          </div>

          {error && (
            <div className="text-red-400 text-sm bg-red-950/30 px-3 py-2 rounded-lg border border-red-900/50">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading
              ? "Please wait..."
              : mode === "login"
                ? "Sign In"
                : "Create Account"}
          </button>
        </form>

        <div className="mt-6 text-center text-slate-400 text-sm">
          {mode === "login" ? (
            <>
              Don&apos;t have an account?{" "}
              <button
                onClick={() => {
                  setMode("signup");
                  setError(null);
                }}
                className="text-emerald-400 hover:underline"
              >
                Sign up free
              </button>
            </>
          ) : (
            <>
              Already have an account?{" "}
              <button
                onClick={() => {
                  setMode("login");
                  setError(null);
                }}
                className="text-emerald-400 hover:underline"
              >
                Sign in
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
