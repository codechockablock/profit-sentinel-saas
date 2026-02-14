"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  Plus,
  Loader2,
  RefreshCw,
  Store,
  Upload,
  BarChart3,
  Settings,
  Trash2,
  X,
  MapPin,
  Package,
  DollarSign,
  Clock,
} from "lucide-react";
import Link from "next/link";
import {
  fetchStores,
  createStore,
  updateStore,
  deleteStore,
  type Store as StoreType,
} from "@/lib/sentinel-api";
import { ApiErrorBanner } from "@/components/dashboard/ApiErrorBanner";

// ─── Helpers ─────────────────────────────────────────────────

function formatDollar(amount: number): string {
  if (amount >= 1_000_000) return `$${(amount / 1_000_000).toFixed(1)}M`;
  if (amount >= 1_000) return `$${(amount / 1_000).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

function formatDate(iso: string | null): string {
  if (!iso) return "Never";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
}

function formatNumber(n: number): string {
  return n.toLocaleString("en-US");
}

// ─── Add/Edit Store Modal ────────────────────────────────────

function StoreModal({
  isOpen,
  onClose,
  onSave,
  initial,
  mode,
}: {
  isOpen: boolean;
  onClose: () => void;
  onSave: (name: string, address: string) => Promise<void>;
  initial?: { name: string; address: string };
  mode: "create" | "edit";
}) {
  const [name, setName] = useState(initial?.name || "");
  const [address, setAddress] = useState(initial?.address || "");
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      setName(initial?.name || "");
      setAddress(initial?.address || "");
      setErr(null);
    }
  }, [isOpen, initial]);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim()) {
      setErr("Store name is required");
      return;
    }
    setSaving(true);
    setErr(null);
    try {
      await onSave(name.trim(), address.trim());
      onClose();
    } catch (error) {
      setErr((error as Error).message);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4">
      <div className="bg-slate-800 border border-slate-700 rounded-xl w-full max-w-md p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-bold text-white">
            {mode === "create" ? "Add Store" : "Edit Store"}
          </h2>
          <button onClick={onClose} className="text-slate-400 hover:text-white">
            <X size={20} />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Store Name *
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Pike Creek, Wilmington"
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
              autoFocus
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Address (optional)
            </label>
            <input
              type="text"
              value={address}
              onChange={(e) => setAddress(e.target.value)}
              placeholder="e.g. 123 Main St, Pike Creek, PA"
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
          </div>

          {err && (
            <p className="text-sm text-red-400">{err}</p>
          )}

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={saving}
              className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
            >
              {saving && <Loader2 size={14} className="animate-spin" />}
              {mode === "create" ? "Create Store" : "Save Changes"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ─── Delete Confirmation Modal ───────────────────────────────

function DeleteModal({
  isOpen,
  storeName,
  onClose,
  onConfirm,
}: {
  isOpen: boolean;
  storeName: string;
  onClose: () => void;
  onConfirm: () => Promise<void>;
}) {
  const [deleting, setDeleting] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleDelete = async () => {
    setDeleting(true);
    setErr(null);
    try {
      await onConfirm();
      onClose();
    } catch (error) {
      setErr((error as Error).message);
    } finally {
      setDeleting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4">
      <div className="bg-slate-800 border border-slate-700 rounded-xl w-full max-w-sm p-6">
        <h2 className="text-lg font-bold text-white mb-3">Delete Store</h2>
        <p className="text-sm text-slate-400 mb-4">
          Are you sure you want to delete <span className="text-white font-medium">{storeName}</span>?
          This action cannot be undone.
        </p>
        {err && <p className="text-sm text-red-400 mb-4">{err}</p>}
        <div className="flex justify-end gap-3">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm text-slate-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleDelete}
            disabled={deleting}
            className="px-4 py-2 bg-red-600 hover:bg-red-500 disabled:opacity-50 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
          >
            {deleting && <Loader2 size={14} className="animate-spin" />}
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── Store Card ──────────────────────────────────────────────

function StoreCard({
  store,
  onEdit,
  onDelete,
}: {
  store: StoreType;
  onEdit: () => void;
  onDelete: () => void;
}) {
  const hasData = store.item_count > 0;

  return (
    <div className="bg-white/5 rounded-xl border border-slate-700 p-5 hover:bg-white/[0.07] transition">
      <div className="flex items-start justify-between mb-3">
        <div className="min-w-0">
          <h3 className="text-lg font-semibold text-white truncate">{store.name}</h3>
          {store.address && (
            <div className="flex items-center gap-1.5 mt-1 text-sm text-slate-400">
              <MapPin size={12} className="shrink-0" />
              <span className="truncate">{store.address}</span>
            </div>
          )}
        </div>
        <button
          onClick={onEdit}
          className="p-2 text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors"
          title="Store settings"
        >
          <Settings size={16} />
        </button>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        <div className="bg-slate-800/50 rounded-lg p-2.5">
          <div className="flex items-center gap-1 text-[10px] text-slate-500 mb-0.5">
            <Clock size={10} />
            Last Upload
          </div>
          <div className="text-sm font-medium text-white">
            {formatDate(store.last_upload_at)}
          </div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-2.5">
          <div className="flex items-center gap-1 text-[10px] text-slate-500 mb-0.5">
            <Package size={10} />
            Items
          </div>
          <div className="text-sm font-medium text-white">
            {hasData ? formatNumber(store.item_count) : "No data"}
          </div>
        </div>
        <div className="bg-slate-800/50 rounded-lg p-2.5">
          <div className="flex items-center gap-1 text-[10px] text-slate-500 mb-0.5">
            <DollarSign size={10} />
            Impact
          </div>
          <div className="text-sm font-medium text-emerald-400">
            {hasData ? formatDollar(store.total_impact) : "--"}
          </div>
        </div>
      </div>

      {/* Actions */}
      <div className="flex items-center gap-2">
        <Link
          href={`/dashboard/operations?store_id=${store.id}`}
          className="flex items-center gap-2 px-3 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition-colors"
        >
          <Upload size={14} />
          Upload Data
        </Link>
        {hasData && (
          <Link
            href={`/dashboard?store_id=${store.id}`}
            className="flex items-center gap-2 px-3 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <BarChart3 size={14} />
            Dashboard
          </Link>
        )}
        <button
          onClick={onDelete}
          className="ml-auto p-2 text-slate-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
          title="Delete store"
        >
          <Trash2 size={14} />
        </button>
      </div>
    </div>
  );
}

// ─── Welcome State (first-time user) ─────────────────────────

function WelcomeState({
  store,
  onRename,
}: {
  store: StoreType;
  onRename: () => void;
}) {
  return (
    <div className="text-center py-12 max-w-md mx-auto">
      <div className="w-16 h-16 bg-emerald-500/10 rounded-2xl flex items-center justify-center mx-auto mb-6">
        <Store size={32} className="text-emerald-400" />
      </div>
      <h2 className="text-xl font-bold text-white mb-3">
        Welcome! We&apos;ve created your first store.
      </h2>
      <p className="text-sm text-slate-400 mb-6">
        Give it a name and upload your inventory data to get started.
      </p>
      <div className="flex items-center justify-center gap-3">
        <button
          onClick={onRename}
          className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white text-sm font-medium rounded-lg transition-colors"
        >
          Name Your Store
        </button>
        <Link
          href={`/dashboard/operations?store_id=${store.id}`}
          className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition-colors flex items-center gap-2"
        >
          <Upload size={14} />
          Upload Data
        </Link>
      </div>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────

export default function StoresPage() {
  const [stores, setStores] = useState<StoreType[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Modal state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [editingStore, setEditingStore] = useState<StoreType | null>(null);
  const [deletingStore, setDeletingStore] = useState<StoreType | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchStores();
      setStores(res.stores);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const handleCreate = async (name: string, address: string) => {
    await createStore(name, address);
    await load();
  };

  const handleUpdate = async (name: string, address: string) => {
    if (!editingStore) return;
    await updateStore(editingStore.id, { name, address });
    await load();
  };

  const handleDelete = async () => {
    if (!deletingStore) return;
    await deleteStore(deletingStore.id);
    await load();
  };

  if (loading && stores.length === 0) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Loader2 className="w-8 h-8 text-emerald-400 animate-spin" />
      </div>
    );
  }

  // Check if this is a first-time user with only the auto-created "Main Store" with no data
  const isFirstTime =
    stores.length === 1 &&
    stores[0].name === "Main Store" &&
    stores[0].item_count === 0;

  return (
    <div className="p-6 md:p-8 max-w-4xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Store className="w-7 h-7 text-emerald-400" />
            <h1 className="text-2xl font-bold text-white">My Stores</h1>
          </div>
          <p className="text-sm text-slate-400">
            Manage your store locations and upload inventory data per store.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={load}
            disabled={loading}
            className="flex items-center gap-2 px-3 py-2 text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors text-sm"
          >
            <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-medium rounded-lg transition-colors"
          >
            <Plus size={14} />
            Add Store
          </button>
        </div>
      </div>

      <ApiErrorBanner error={error} onRetry={load} />

      {/* Content */}
      {!error && (
        <>
          {isFirstTime ? (
            <WelcomeState
              store={stores[0]}
              onRename={() => setEditingStore(stores[0])}
            />
          ) : (
            <div className="space-y-4">
              {stores.map((store) => (
                <StoreCard
                  key={store.id}
                  store={store}
                  onEdit={() => setEditingStore(store)}
                  onDelete={() => setDeletingStore(store)}
                />
              ))}
            </div>
          )}
        </>
      )}

      {/* Modals */}
      <StoreModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onSave={handleCreate}
        mode="create"
      />
      <StoreModal
        isOpen={!!editingStore}
        onClose={() => setEditingStore(null)}
        onSave={handleUpdate}
        initial={editingStore ? { name: editingStore.name, address: editingStore.address } : undefined}
        mode="edit"
      />
      <DeleteModal
        isOpen={!!deletingStore}
        storeName={deletingStore?.name || ""}
        onClose={() => setDeletingStore(null)}
        onConfirm={handleDelete}
      />
    </div>
  );
}
