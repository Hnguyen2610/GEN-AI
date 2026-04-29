import { createSupabaseBrowserClient } from "../supabase/browser";

export type AssetKind = "dataset" | "knowledge";
export type AssetStatus = "pending" | "processing" | "ready" | "failed";

type DatasetVersionSummary = {
  id: string;
  version_number: number;
  file_size_bytes: number;
  created_at: string;
};

export type JobSummary = {
  id: string;
  status: AssetStatus;
  created_at: string;
  updated_at: string;
};

export type AssetJobSummary = {
  id: string;
  status: AssetStatus;
  error_message: string | null;
};

export type DatasetListItem = {
  id: string;
  workspace_id: string;
  title: string;
  original_filename: string;
  mime_type: string;
  status: AssetStatus;
  created_at: string;
  updated_at: string;
  latest_version: DatasetVersionSummary | null;
  latest_job: JobSummary | null;
};

export type UploadAssetResponse = {
  asset_id: string;
  kind: AssetKind;
  job_id: string;
  status: AssetStatus;
};

export type AssetListItem = {
  id: string;
  kind: AssetKind;
  title: string;
  original_filename: string;
  status: AssetStatus;
  created_at: string;
  updated_at: string;
  latest_job: AssetJobSummary | null;
};

export type AssetVersionSummary = {
  id: string;
  version_number: number;
  storage_path: string;
  file_size_bytes: number;
  created_at: string;
};

export type AssetDetail = AssetListItem & {
  mime_type: string;
  latest_version: AssetVersionSummary | null;
};

export type AssetPreviewResponse = {
  asset_id: string;
  kind: AssetKind;
  preview_data: Record<string, unknown>[];
};

export type AssetProfileResponse = {
  asset_id: string;
  kind: AssetKind;
  profile_data: Record<string, unknown>;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

async function getAccessToken() {
  const supabase = createSupabaseBrowserClient();
  const {
    data: { session },
  } = await supabase.auth.getSession();

  const accessToken = session?.access_token;

  if (!accessToken) {
    throw new Error("You are not authenticated with Supabase.");
  }

  return accessToken;
}

async function apiFetch(path: string, init?: RequestInit) {
  const accessToken = await getAccessToken();
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${accessToken}`,
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const body = await response
      .json()
      .catch(() => ({ detail: "Unexpected API error." }));
    throw new Error(body.detail ?? "Unexpected API error.");
  }

  return response;
}

export async function listAssets() {
  const response = await apiFetch("/v1/assets");
  const payload = (await response.json()) as { items: AssetListItem[] };
  return payload.items;
}

export async function getAsset(assetId: string) {
  const response = await apiFetch(`/v1/assets/${assetId}`);
  return (await response.json()) as AssetDetail;
}

export async function getAssetPreview(assetId: string) {
  const response = await apiFetch(`/v1/assets/${assetId}/preview`);
  return (await response.json()) as AssetPreviewResponse;
}

export async function getAssetProfile(assetId: string) {
  const response = await apiFetch(`/v1/assets/${assetId}/profile`);
  return (await response.json()) as AssetProfileResponse;
}

export async function listDatasets() {
  // Use facade and filter/map to maintain legacy type if needed,
  // but better to just use the facade data.
  const items = await listAssets();
  return items.filter(
    (i) => i.kind === "dataset",
  ) as unknown as DatasetListItem[];
}

export async function uploadAsset(file: File) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiFetch("/v1/assets/upload", {
    method: "POST",
    body: formData,
  });

  return (await response.json()) as UploadAssetResponse;
}

export async function uploadDataset(file: File) {
  // Redirect legacy call to facade
  const res = await uploadAsset(file);
  // Map to legacy response structure if needed by components
  return {
    dataset: { id: res.asset_id, status: res.status },
    job: { id: res.job_id, status: res.status },
  };
}

export async function deleteAsset(assetId: string) {
  await apiFetch(`/v1/assets/${assetId}`, {
    method: "DELETE",
  });
}

export async function deleteDataset(datasetId: string) {
  await deleteAsset(datasetId);
}

export async function getJob(jobId: string) {
  const response = await apiFetch(`/v1/jobs/${jobId}`);
  return (await response.json()) as JobSummary;
}

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  status: "streaming" | "completed" | "failed";
  model_name?: string | null;
  retrieval_used?: boolean;
  metadata_json: any;
  created_at: string;
};

export type ChatSession = {
  id: string;
  title: string;
  created_at: string;
  messages?: ChatMessage[];
};

export async function listChatSessions() {
  const response = await apiFetch("/v1/chat/sessions");
  return (await response.json()) as ChatSession[];
}

export async function createChatSession(title: string) {
  const response = await apiFetch("/v1/chat/sessions", {
    method: "POST",
    body: JSON.stringify({ title }),
    headers: { "Content-Type": "application/json" },
  });
  return (await response.json()) as ChatSession;
}

export async function getChatSession(sessionId: string) {
  const response = await apiFetch(`/v1/chat/sessions/${sessionId}`);
  return (await response.json()) as ChatSession;
}

export async function deleteChatSession(sessionId: string) {
  await apiFetch(`/v1/chat/sessions/${sessionId}`, {
    method: "DELETE",
  });
}

export type KnowledgeListItem = {
  id: string;
  workspace_id: string;
  title: string;
  original_filename: string;
  mime_type: string;
  status: AssetStatus;
  created_at: string;
  updated_at: string;
  latest_version: KnowledgeVersionSummary | null;
};

type KnowledgeVersionSummary = {
  id: string;
  version_number: number;
  file_size_bytes: number;
  created_at: string;
};

export async function listKnowledge() {
  const items = await listAssets();
  return items.filter(
    (i) => i.kind === "knowledge",
  ) as unknown as KnowledgeListItem[];
}

export async function uploadKnowledge(file: File) {
  const res = await uploadAsset(file);
  return { id: res.asset_id, status: res.status };
}

export async function deleteKnowledge(assetId: string) {
  await deleteAsset(assetId);
}

export async function getAuthToken() {
  return await getAccessToken();
}
