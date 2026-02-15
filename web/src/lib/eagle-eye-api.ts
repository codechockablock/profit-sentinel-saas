/**
 * Eagle's Eye API Client — Typed wrapper for executive dashboard endpoints.
 */

import { apiFetch } from './sentinel-api';

// ─── Types ──────────────────────────────────────────────────

export interface Organization {
  id: string;
  name: string;
  total_stores: number;
  total_exposure: number;
  exposure_trend: number;
  total_pending_actions: number;
  total_completed_actions_30d: number;
}

export interface StoreSummary {
  id: string;
  name: string;
  region_id: string | null;
  status: 'healthy' | 'attention' | 'critical';
  total_impact: number;
  exposure_trend: number;
  item_count: number;
  flagged_count: number;
  last_upload_at: string | null;
  pending_actions: number;
  top_issue: string | null;
}

export interface RegionSummary {
  id: string;
  name: string;
  store_count: number;
  total_exposure: number;
  exposure_trend: number;
  pending_actions: number;
  stores: StoreSummary[];
}

export interface NetworkAlert {
  type: string;
  description: string;
  affected_stores: number;
  total_impact: number;
  recommended_action: string;
}

export interface EagleEyeResponse {
  org: Organization | null;
  regions: RegionSummary[];
  unassigned_stores: StoreSummary[];
  network_alerts: NetworkAlert[];
}

export interface ActionItem {
  id: string;
  org_id: string;
  store_id: string | null;
  user_id: string;
  action_type: string;
  description: string;
  reasoning: string | null;
  financial_impact: number;
  confidence: number;
  status: string;
  source: string;
  created_at: string;
  decided_at: string | null;
  completed_at: string | null;
  deferred_until: string | null;
  outcome_notes: string | null;
}

export interface AgentBriefing {
  briefing: string;
  action_items: Array<{
    type: string;
    store_id: string | null;
    description: string;
    reasoning: string;
    financial_impact: number;
    confidence: number;
  }>;
  generated_at: string | null;
  expires_at: string | null;
}

export interface OrgStoreDetail {
  id: string;
  org_id: string;
  region_id: string | null;
  name: string;
  address: string;
  store_type: string;
  created_at: string;
  updated_at: string;
  last_upload_at: string | null;
  item_count: number;
  total_impact: number;
  exposure_trend: number;
  status: string;
}

export interface RegionDetail {
  id: string;
  org_id: string;
  name: string;
  created_at: string;
}

// ─── API Functions ──────────────────────────────────────────

// Eagle Eye
export async function fetchEagleEye(): Promise<EagleEyeResponse> {
  return apiFetch('/eagle-eye');
}

// Briefing
export async function fetchBriefing(): Promise<AgentBriefing> {
  return apiFetch('/briefing');
}

export async function refreshBriefing(): Promise<AgentBriefing> {
  return apiFetch('/briefing/refresh', { method: 'POST' });
}

// Actions
export async function fetchActions(params?: {
  status?: string;
  store_id?: string;
  limit?: number;
}): Promise<{ actions: ActionItem[]; total: number }> {
  const query = new URLSearchParams();
  if (params?.status) query.set('status', params.status);
  if (params?.store_id) query.set('store_id', params.store_id);
  if (params?.limit) query.set('limit', String(params.limit));
  const qs = query.toString();
  return apiFetch(`/actions${qs ? `?${qs}` : ''}`);
}

export async function approveAction(actionId: string): Promise<ActionItem> {
  return apiFetch(`/actions/${actionId}/approve`, { method: 'POST' });
}

export async function deferAction(actionId: string, until?: string): Promise<ActionItem> {
  return apiFetch(`/actions/${actionId}/defer`, {
    method: 'POST',
    body: JSON.stringify({ deferred_until: until || null }),
  });
}

export async function rejectAction(actionId: string, reason?: string): Promise<ActionItem> {
  return apiFetch(`/actions/${actionId}/reject`, {
    method: 'POST',
    body: JSON.stringify({ reason: reason || '' }),
  });
}

// Organization
export async function fetchOrg(): Promise<{ id: string; name: string; owner_user_id: string; created_at: string; updated_at: string }> {
  return apiFetch('/org');
}

export async function createOrg(name: string): Promise<{ id: string; name: string }> {
  return apiFetch('/org', {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
}

// Regions
export async function fetchRegions(): Promise<{ regions: RegionDetail[]; total: number }> {
  return apiFetch('/regions');
}

export async function createRegion(name: string): Promise<RegionDetail> {
  return apiFetch('/regions', {
    method: 'POST',
    body: JSON.stringify({ name }),
  });
}

export async function updateRegion(regionId: string, name: string): Promise<RegionDetail> {
  return apiFetch(`/regions/${regionId}`, {
    method: 'PUT',
    body: JSON.stringify({ name }),
  });
}

export async function deleteRegion(regionId: string): Promise<void> {
  return apiFetch(`/regions/${regionId}`, { method: 'DELETE' });
}

// Stores
export async function fetchOrgStore(storeId: string): Promise<OrgStoreDetail> {
  return apiFetch(`/stores/${storeId}`);
}

export async function fetchOrgStores(): Promise<{ stores: OrgStoreDetail[]; total: number }> {
  return apiFetch('/stores');
}

// Complete action
export async function completeAction(actionId: string, notes?: string): Promise<ActionItem> {
  return apiFetch(`/actions/${actionId}/complete`, {
    method: 'POST',
    body: JSON.stringify({ outcome_notes: notes || '' }),
  });
}
