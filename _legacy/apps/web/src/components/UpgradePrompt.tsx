"use client";

import React from "react";
import { Sparkles } from "lucide-react";

interface UpgradePromptProps {
  message: string;
  cta: string;
  onSignUp: () => void;
}

export function UpgradePrompt({ message, cta, onSignUp }: UpgradePromptProps) {
  return (
    <div className="bg-gradient-to-r from-emerald-900/50 to-slate-900 border border-emerald-700/50 rounded-lg p-4 mt-6">
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-emerald-600/20 rounded-full flex items-center justify-center shrink-0">
            <Sparkles size={20} className="text-emerald-400" />
          </div>
          <p className="text-slate-300 text-sm">{message}</p>
        </div>
        <button
          onClick={onSignUp}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition-colors whitespace-nowrap text-sm"
        >
          {cta}
        </button>
      </div>
    </div>
  );
}
