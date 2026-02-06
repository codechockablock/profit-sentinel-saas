"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Key,
  Plus,
  Trash2,
  Copy,
  Check,
  Shield,
  Activity,
  Clock,
  Loader2,
  Eye,
  EyeOff,
} from "lucide-react";
import {
  createApiKey,
  listApiKeys,
  revokeApiKey,
  fetchApiKeyUsage,
  type ApiKeyRecord,
  type ApiKeyUsageStats,
  type ApiTier,
} from "@/lib/sentinel-api";

const TIER_COLORS: Record<string, string> = {
  free: "text-slate-400 bg-slate-500/10 border-slate-500/30",
  pro: "text-blue-400 bg-blue-500/10 border-blue-500/30",
  enterprise: "text-purple-400 bg-purple-500/10 border-purple-500/30",
};

function KeyRow({
  record,
  onRevoke,
}: {
  record: ApiKeyRecord;
  onRevoke: (keyId: string) => void;
}) {
  const [usage, setUsage] = useState<ApiKeyUsageStats | null>(null);
  const [showUsage, setShowUsage] = useState(false);

  const loadUsage = useCallback(async () => {
    if (!showUsage) {
      try {
        const stats = await fetchApiKeyUsage(record.key_id);
        setUsage(stats);
      } catch { /* ignore */ }
    }
    setShowUsage(!showUsage);
  }, [showUsage, record.key_id]);

  return (
    <div className="bg-white/5 rounded-xl border border-slate-700 p-5">
      <div className="flex items-center gap-4">
        <div className={`w-10 h-10 rounded-lg border flex items-center justify-center ${TIER_COLORS[record.tier]}`}>
          <Key size={18} />
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-white font-semibold">{record.name}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full border ${TIER_COLORS[record.tier]}`}>
              {record.tier}
            </span>
            {record.is_test && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/30">
                test
              </span>
            )}
            {!record.is_active && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-red-500/10 text-red-400 border border-red-500/30">
                revoked
              </span>
            )}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            {record.key_id} &middot; {record.usage_count} requests
            {record.last_used_at && ` \u00b7 Last used ${new Date(record.last_used_at).toLocaleDateString()}`}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={loadUsage}
            className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-700/50 transition"
            title="View usage"
          >
            <Activity size={16} />
          </button>
          {record.is_active && (
            <button
              onClick={() => onRevoke(record.key_id)}
              className="p-2 text-slate-400 hover:text-red-400 rounded-lg hover:bg-red-500/10 transition"
              title="Revoke key"
            >
              <Trash2 size={16} />
            </button>
          )}
        </div>
      </div>

      {showUsage && usage && (
        <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3 pt-4 border-t border-slate-700/50">
          <div className="bg-slate-800/50 rounded-lg p-2">
            <div className="text-[10px] text-slate-500">Total Requests</div>
            <div className="text-sm font-bold text-white">{usage.usage_count_total}</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-2">
            <div className="text-[10px] text-slate-500">Last Hour</div>
            <div className="text-sm font-bold text-white">{usage.usage_last_hour}</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-2">
            <div className="text-[10px] text-slate-500">Remaining/hr</div>
            <div className="text-sm font-bold text-emerald-400">{usage.remaining_hourly}</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-2">
            <div className="text-[10px] text-slate-500">Limit/hr</div>
            <div className="text-sm font-bold text-slate-300">{usage.limits.requests_per_hour}</div>
          </div>
        </div>
      )}
    </div>
  );
}

export default function ApiKeysPage() {
  const [keys, setKeys] = useState<ApiKeyRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);
  const [newKey, setNewKey] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [newKeyName, setNewKeyName] = useState("Default");
  const [newKeyTier, setNewKeyTier] = useState<ApiTier>("free");

  const loadKeys = useCallback(async () => {
    try {
      const resp = await listApiKeys();
      setKeys(resp.keys);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadKeys();
  }, [loadKeys]);

  const handleCreate = async () => {
    setCreating(true);
    try {
      const resp = await createApiKey({ name: newKeyName, tier: newKeyTier });
      setNewKey(resp.key);
      setShowKey(true);
      await loadKeys();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create key");
    } finally {
      setCreating(false);
    }
  };

  const handleRevoke = async (keyId: string) => {
    try {
      await revokeApiKey(keyId);
      await loadKeys();
    } catch { /* ignore */ }
  };

  const handleCopy = () => {
    if (newKey) {
      navigator.clipboard.writeText(newKey);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-emerald-400 animate-spin" />
      </div>
    );
  }

  return (
    <div className="p-6 md:p-8 max-w-5xl">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <Key className="w-7 h-7 text-purple-400" />
          <h1 className="text-2xl font-bold text-white">API Keys</h1>
        </div>
        <p className="text-sm text-slate-400">
          Manage API keys for enterprise integrations. Keys use tiered rate limiting.
        </p>
      </div>

      {/* Create key section */}
      <div className="bg-white/5 rounded-xl border border-slate-700 p-5 mb-8">
        <h2 className="text-sm font-semibold text-white mb-4">Create New Key</h2>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Name</label>
            <input
              type="text"
              value={newKeyName}
              onChange={(e) => setNewKeyName(e.target.value)}
              className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
              placeholder="My Integration"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Tier</label>
            <select
              value={newKeyTier}
              onChange={(e) => setNewKeyTier(e.target.value as ApiTier)}
              className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="free">Free (10/hr)</option>
              <option value="pro">Pro (100/hr)</option>
              <option value="enterprise">Enterprise (1,000/hr)</option>
            </select>
          </div>
          <button
            onClick={handleCreate}
            disabled={creating}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition disabled:opacity-50"
          >
            {creating ? <Loader2 size={16} className="animate-spin" /> : <Plus size={16} />}
            Create Key
          </button>
        </div>

        {/* New key display */}
        {newKey && (
          <div className="mt-4 bg-amber-500/10 border border-amber-500/30 rounded-lg p-4">
            <div className="flex items-center gap-2 mb-2">
              <Shield size={16} className="text-amber-400" />
              <span className="text-sm font-medium text-amber-400">Save this key — it won't be shown again</span>
            </div>
            <div className="flex items-center gap-2">
              <code className="flex-1 text-sm text-white bg-slate-800 px-3 py-2 rounded font-mono overflow-x-auto">
                {showKey ? newKey : newKey.slice(0, 12) + "..."}
              </code>
              <button
                onClick={() => setShowKey(!showKey)}
                className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-slate-700/50 transition"
              >
                {showKey ? <EyeOff size={16} /> : <Eye size={16} />}
              </button>
              <button
                onClick={handleCopy}
                className="p-2 text-slate-400 hover:text-emerald-400 rounded-lg hover:bg-emerald-500/10 transition"
              >
                {copied ? <Check size={16} className="text-emerald-400" /> : <Copy size={16} />}
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-6 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Key list */}
      <div className="space-y-3">
        {keys.length === 0 ? (
          <div className="text-center py-12">
            <Key className="w-12 h-12 text-slate-600 mx-auto mb-4" />
            <p className="text-slate-400">No API keys yet. Create one to get started.</p>
          </div>
        ) : (
          keys.map((record) => (
            <KeyRow key={record.key_id} record={record} onRevoke={handleRevoke} />
          ))
        )}
      </div>

      {/* Rate limit info */}
      <div className="mt-8 bg-white/5 rounded-xl border border-slate-700 p-5">
        <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
          <Clock size={14} className="text-slate-400" />
          Rate Limits by Tier
        </h3>
        <div className="grid grid-cols-3 gap-3 text-center">
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-slate-400 mb-1">Free</div>
            <div className="text-sm font-bold text-white">10/hr</div>
            <div className="text-[10px] text-slate-500">100/day</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-blue-400 mb-1">Pro</div>
            <div className="text-sm font-bold text-white">100/hr</div>
            <div className="text-[10px] text-slate-500">2,000/day</div>
          </div>
          <div className="bg-slate-800/50 rounded-lg p-3">
            <div className="text-xs text-purple-400 mb-1">Enterprise</div>
            <div className="text-sm font-bold text-white">1,000/hr</div>
            <div className="text-[10px] text-slate-500">50,000/day</div>
          </div>
        </div>
      </div>
    </div>
  );
}
