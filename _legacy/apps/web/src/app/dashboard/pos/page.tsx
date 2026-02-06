"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Plug,
  Plus,
  RefreshCw,
  Unplug,
  Trash2,
  CheckCircle,
  XCircle,
  Loader2,
  Store,
  ArrowRight,
} from "lucide-react";
import {
  fetchSupportedPosSystems,
  createPosConnection,
  listPosConnections,
  triggerPosSync,
  disconnectPos,
  deletePosConnection,
  type PosSystemInfo,
  type PosConnection,
  type PosSystemType,
} from "@/lib/sentinel-api";

const STATUS_COLORS: Record<string, string> = {
  connected: "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  disconnected: "text-slate-400 bg-slate-500/10 border-slate-500/30",
  syncing: "text-blue-400 bg-blue-500/10 border-blue-500/30",
  error: "text-red-400 bg-red-500/10 border-red-500/30",
};

function ConnectionCard({
  conn,
  onSync,
  onDisconnect,
  onDelete,
}: {
  conn: PosConnection;
  onSync: (id: string) => void;
  onDisconnect: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div className="bg-white/5 rounded-xl border border-slate-700 p-5 hover:bg-white/[0.07] transition">
      <div className="flex items-center gap-4">
        <div className={`w-10 h-10 rounded-lg border flex items-center justify-center ${STATUS_COLORS[conn.status] || STATUS_COLORS.disconnected}`}>
          {conn.status === "connected" ? <CheckCircle size={18} /> : <XCircle size={18} />}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="text-white font-semibold">{conn.store_name}</span>
            <span className={`text-xs px-2 py-0.5 rounded-full border ${STATUS_COLORS[conn.status]}`}>
              {conn.status}
            </span>
          </div>
          <div className="text-xs text-slate-400 mt-0.5">
            {conn.pos_system_display} &middot; Sync: {conn.sync_frequency}
            {conn.items_synced > 0 && ` \u00b7 ${conn.items_synced} items`}
            {conn.last_sync_at && ` \u00b7 Last: ${new Date(conn.last_sync_at).toLocaleString()}`}
          </div>
        </div>
        <div className="flex items-center gap-1">
          {conn.status === "connected" && (
            <>
              <button
                onClick={() => onSync(conn.connection_id)}
                className="p-2 text-slate-400 hover:text-emerald-400 rounded-lg hover:bg-emerald-500/10 transition"
                title="Trigger sync"
              >
                <RefreshCw size={16} />
              </button>
              <button
                onClick={() => onDisconnect(conn.connection_id)}
                className="p-2 text-slate-400 hover:text-amber-400 rounded-lg hover:bg-amber-500/10 transition"
                title="Disconnect"
              >
                <Unplug size={16} />
              </button>
            </>
          )}
          <button
            onClick={() => onDelete(conn.connection_id)}
            className="p-2 text-slate-400 hover:text-red-400 rounded-lg hover:bg-red-500/10 transition"
            title="Delete"
          >
            <Trash2 size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}

export default function PosPage() {
  const [systems, setSystems] = useState<PosSystemInfo[]>([]);
  const [connections, setConnections] = useState<PosConnection[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [selectedSystem, setSelectedSystem] = useState<PosSystemType>("square");
  const [storeName, setStoreName] = useState("");
  const [creating, setCreating] = useState(false);

  const loadData = useCallback(async () => {
    try {
      const [sysResp, connResp] = await Promise.all([
        fetchSupportedPosSystems(),
        listPosConnections(),
      ]);
      setSystems(sysResp.systems);
      setConnections(connResp.connections);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleCreate = async () => {
    if (!storeName.trim()) return;
    setCreating(true);
    try {
      await createPosConnection({
        pos_system: selectedSystem,
        store_name: storeName,
        sync_frequency: "daily",
      });
      setStoreName("");
      setShowCreate(false);
      await loadData();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to create connection");
    } finally {
      setCreating(false);
    }
  };

  const handleSync = async (id: string) => {
    try {
      await triggerPosSync(id);
      await loadData();
    } catch { /* ignore */ }
  };

  const handleDisconnect = async (id: string) => {
    try {
      await disconnectPos(id);
      await loadData();
    } catch { /* ignore */ }
  };

  const handleDelete = async (id: string) => {
    try {
      await deletePosConnection(id);
      await loadData();
    } catch { /* ignore */ }
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
          <Plug className="w-7 h-7 text-blue-400" />
          <h1 className="text-2xl font-bold text-white">POS Integrations</h1>
        </div>
        <p className="text-sm text-slate-400">
          Connect your point-of-sale system for automatic inventory data sync.
        </p>
      </div>

      {/* Supported systems */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        {systems.map((sys) => (
          <div key={sys.system} className="bg-white/5 rounded-xl border border-slate-700 p-4 text-center">
            <Store size={24} className="text-slate-400 mx-auto mb-2" />
            <div className="text-sm font-semibold text-white">{sys.display_name}</div>
            <div className="text-[10px] text-slate-500 mt-1">{sys.auth_type}</div>
          </div>
        ))}
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 mb-6 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Create connection */}
      <div className="mb-8">
        {!showCreate ? (
          <button
            onClick={() => setShowCreate(true)}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition"
          >
            <Plus size={16} />
            Add Connection
          </button>
        ) : (
          <div className="bg-white/5 rounded-xl border border-slate-700 p-5">
            <h3 className="text-sm font-semibold text-white mb-4">New Connection</h3>
            <div className="flex flex-wrap items-end gap-3">
              <div>
                <label className="block text-xs text-slate-400 mb-1">POS System</label>
                <select
                  value={selectedSystem}
                  onChange={(e) => setSelectedSystem(e.target.value as PosSystemType)}
                  className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                >
                  {systems.map((sys) => (
                    <option key={sys.system} value={sys.system}>
                      {sys.display_name}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-xs text-slate-400 mb-1">Store Name</label>
                <input
                  type="text"
                  value={storeName}
                  onChange={(e) => setStoreName(e.target.value)}
                  className="px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-sm text-white focus:outline-none focus:border-emerald-500"
                  placeholder="My Hardware Store"
                />
              </div>
              <button
                onClick={handleCreate}
                disabled={creating || !storeName.trim()}
                className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-lg transition disabled:opacity-50"
              >
                {creating ? <Loader2 size={16} className="animate-spin" /> : <ArrowRight size={16} />}
                Connect
              </button>
              <button
                onClick={() => setShowCreate(false)}
                className="px-4 py-2 text-slate-400 hover:text-white transition"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Connection list */}
      <div className="space-y-3">
        {connections.length === 0 ? (
          <div className="text-center py-12">
            <Plug className="w-12 h-12 text-slate-600 mx-auto mb-4" />
            <p className="text-slate-400">No POS connections yet. Add one to start syncing inventory data.</p>
          </div>
        ) : (
          connections.map((conn) => (
            <ConnectionCard
              key={conn.connection_id}
              conn={conn}
              onSync={handleSync}
              onDisconnect={handleDisconnect}
              onDelete={handleDelete}
            />
          ))
        )}
      </div>
    </div>
  );
}
