import { useState } from "react";
import { useParams, Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";
import { InferenceStatus, JobStatus } from "../types/status";
import ChatTab from "../components/ChatTab";

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-yellow-100 text-yellow-800",
  running: "bg-blue-100 text-blue-800",
  completed: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
};

const TABS = ["overview", "chat"] as const;
type Tab = (typeof TABS)[number];

const TAB_LABELS: Record<Tab, string> = {
  overview: "Overview",
  chat: "Chat",
};

export default function JobDetail() {
  const { id } = useParams<{ id: string }>();
  const [tab, setTab] = useState<Tab>("overview");

  const { data: job, isLoading, error } = useQuery({
    queryKey: ["job", id],
    queryFn: () => api.getJob(id!),
    refetchInterval: (query) => {
      const data = query.state.data;
      if (!data) return 3000;
      const jobDone = data.status === JobStatus.COMPLETED || data.status === JobStatus.FAILED;
      const inferDone =
        data.inference_status === InferenceStatus.READY
        || data.inference_status === InferenceStatus.NONE
        || data.inference_status === InferenceStatus.FAILED;
      if (jobDone && inferDone) return false;
      return 3000;
    },
    enabled: !!id,
  });

  if (isLoading) {
    return (
      <div className="flex justify-center py-20">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="max-w-2xl mx-auto">
        <div className="rounded-lg bg-red-50 border border-red-200 text-red-700 px-4 py-3">
          {error?.message || "Job not found"}
        </div>
        <Link to="/" className="text-indigo-600 hover:underline mt-4 inline-block">
          Back to home
        </Link>
      </div>
    );
  }

  const isCompleted = job.status === JobStatus.COMPLETED;

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h1 className="text-2xl font-bold">{job.model_name}</h1>
          <div className="flex items-center gap-3 mt-1">
            <code className="text-xs text-gray-400 font-mono">{job.id}</code>
            <span
              className={`text-xs font-medium px-2.5 py-0.5 rounded-full ${STATUS_STYLES[job.status] || ""}`}
            >
              {job.status}
            </span>
          </div>
        </div>
        <Link to="/" className="text-indigo-600 hover:underline text-sm">
          Back
        </Link>
      </div>

      {/* Tab navigation */}
      <div className="border-b border-gray-200 mb-4">
        <nav className="flex gap-6">
          {TABS.map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              disabled={t !== "overview" && !isCompleted}
              className={`pb-3 text-sm font-medium border-b-2 transition cursor-pointer ${
                tab === t
                  ? "border-indigo-600 text-indigo-600"
                  : "border-transparent text-gray-500 hover:text-gray-700"
              } ${t !== "overview" && !isCompleted ? "opacity-40 cursor-not-allowed" : ""}`}
            >
              {TAB_LABELS[t]}
            </button>
          ))}
        </nav>
      </div>

      {/* Tab content */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        {tab === "overview" && <OverviewContent job={job} />}
        {tab === "chat" && isCompleted && <ChatTab job={job} />}
      </div>
    </div>
  );
}

function OverviewContent({ job }: { job: ReturnType<typeof api.getJob> extends Promise<infer T> ? T : never }) {
  const isActive = job.status === JobStatus.PENDING || job.status === JobStatus.RUNNING;

  return (
    <>
      {isActive && (
        <div className="px-6 py-4 border-b border-gray-100">
          <div className="flex justify-between text-sm mb-1">
            <span className="text-gray-600">Progress</span>
            <span className="font-medium">{job.progress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2.5">
            <div
              className="bg-indigo-600 h-2.5 rounded-full transition-all duration-500"
              style={{ width: `${job.progress}%` }}
            />
          </div>
        </div>
      )}

      <dl className="divide-y divide-gray-100">
        <Row label="Model" value={job.model_name} />
        <Row label="Method" value={job.quant_method} />
        <Row
          label={
            job.quant_method === "auto_run"
              ? "Bit Width (auto-estimated)"
              : "Bit Width"
          }
          value={formatAutoRunBits(job)}
        />
        <Row label="Group Size" value={String(job.quant_params.group_size ?? "-")} />
        <Row label="Created" value={new Date(job.created_at).toLocaleString()} />
        <Row label="Updated" value={new Date(job.updated_at).toLocaleString()} />
      </dl>

      {job.status === JobStatus.FAILED && job.error_message && (
        <div className="px-6 py-4">
          <div className="rounded-lg bg-red-50 border border-red-200 text-red-700 px-4 py-3 text-sm">
            {job.error_message}
          </div>
        </div>
      )}
    </>
  );
}

function formatBits(value: unknown): string {
  if (value === undefined || value === null) return "-";
  const n = Number(value);
  if (!Number.isFinite(n)) return String(value);
  return Number.isInteger(n) ? `${n}-bit` : `${n.toFixed(2)}-bit`;
}

function formatAutoRunBits(
  job: ReturnType<typeof api.getJob> extends Promise<infer T> ? T : never,
): string {
  if (job.quant_method !== "auto_run") {
    return formatBits(job.quant_params.bits);
  }
  if (
    job.status === JobStatus.PENDING
    || (job.status === JobStatus.RUNNING && job.progress < 10)
  ) {
    return "Estimating from VRAM…";
  }
  return formatBits(job.quant_params.bits);
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="px-6 py-3 flex justify-between">
      <dt className="text-sm text-gray-500">{label}</dt>
      <dd className="text-sm font-medium text-gray-900">{value}</dd>
    </div>
  );
}
