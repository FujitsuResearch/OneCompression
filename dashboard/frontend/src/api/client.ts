import type { ChatTaskStatus, InferenceStatus, JobStatus } from "../types/status";
export { ChatTaskStatus, InferenceStatus, JobStatus } from "../types/status";

const BASE = "/api";

export interface QuantParams {
  bits: number;
  group_size: number;
  use_qep: boolean;
  dataset: string;
  num_samples: number;
  auto_bits?: boolean;
  total_vram_gb?: number | null;
}

export interface JobCreate {
  model_name: string;
  quant_method: string;
  quant_params: QuantParams;
}

export interface EstimateWbitsRequest {
  model_name: string;
  total_vram_gb: number;
  group_size?: number;
  vram_ratio?: number;
}

export interface EstimateWbitsResponse {
  target_bitwidth: number;
  total_vram_gb: number;
  budget_gb: number;
  non_quant_weight_gb: number;
  available_for_quant_gb: number;
  total_params: number;
  quantizable_params: number;
  meta_bits_per_param: number;
}

export interface Job {
  id: string;
  status: JobStatus;
  progress: number;
  model_name: string;
  quant_method: string;
  quant_params: Record<string, unknown>;
  result_path: string | null;
  error_message: string | null;
  inference_status: InferenceStatus;
  created_at: string;
  updated_at: string;
}

export interface JobListResponse {
  jobs: Job[];
  total: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatRequest {
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
}

export interface ChatResponse {
  message: ChatMessage | null;
  task_id: string | null;
}

export interface ChatTaskResult {
  status: ChatTaskStatus;
  message?: ChatMessage;
  error?: string;
}

export const api = {
  createJob: (data: JobCreate) =>
    request<Job>("/jobs", { method: "POST", body: JSON.stringify(data) }),

  getJob: (id: string) => request<Job>(`/jobs/${id}`),

  listJobs: (limit = 20, offset = 0) =>
    request<JobListResponse>(`/jobs?limit=${limit}&offset=${offset}`),

  deploy: (jobId: string) =>
    request<Job>(`/jobs/${jobId}/deploy`, { method: "POST" }),

  stopInference: (jobId: string) =>
    request<Job>(`/jobs/${jobId}/stop`, { method: "POST" }),

  chat: (jobId: string, data: ChatRequest) =>
    request<ChatResponse>(`/jobs/${jobId}/chat`, {
      method: "POST",
      body: JSON.stringify(data),
    }),

  getChatResult: (taskId: string) =>
    request<ChatTaskResult>(`/jobs/chat-result/${taskId}`),

  estimateWbits: (data: EstimateWbitsRequest) =>
    request<EstimateWbitsResponse>(`/jobs/estimate-wbits`, {
      method: "POST",
      body: JSON.stringify(data),
    }),
};
