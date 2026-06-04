import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { JobStatus } from "../types/status";

const STATUS_STYLES: Record<string, string> = {
  pending: "bg-yellow-100 text-yellow-800",
  running: "bg-blue-100 text-blue-800",
  completed: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
};

function formatBitsCompact(value: unknown): string {
  if (value === undefined || value === null) return "?bit";
  const n = Number(value);
  if (!Number.isFinite(n)) return `${value}bit`;
  return Number.isInteger(n) ? `${n}bit` : `${n.toFixed(2)}bit`;
}

export default function JobList() {
  const { data, isLoading } = useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.listJobs(),
    refetchInterval: 5000,
  });

  return (
    <div className="max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold">Jobs</h1>
        <Link
          to="/new"
          className="rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white font-medium px-4 py-2 text-sm transition"
        >
          + New Job
        </Link>
      </div>

      {isLoading ? (
        <div className="flex justify-center py-20">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-600" />
        </div>
      ) : !data?.jobs.length ? (
        <div className="text-center py-20 text-gray-500">
          <p className="mb-4">No jobs yet.</p>
          <Link to="/new" className="text-indigo-600 hover:underline">
            Create your first quantization job
          </Link>
        </div>
      ) : (
        <div className="space-y-3">
          {data.jobs.map((job) => (
            <Link
              key={job.id}
              to={`/jobs/${job.id}`}
              className="block bg-white rounded-xl border border-gray-200 shadow-sm hover:shadow-md transition px-6 py-4"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-gray-900 truncate mr-4">
                  {job.model_name}
                </span>
                <span
                  className={`text-xs font-medium px-2.5 py-1 rounded-full shrink-0 ${STATUS_STYLES[job.status] || ""}`}
                >
                  {job.status}
                </span>
              </div>
              <div className="flex items-center justify-between text-sm text-gray-500">
                <span>
                  {job.quant_method} / {formatBitsCompact(job.quant_params.bits)}
                </span>
                <span>{new Date(job.created_at).toLocaleString()}</span>
              </div>
              {(job.status === JobStatus.RUNNING || job.status === JobStatus.PENDING) && (
                <div className="w-full bg-gray-200 rounded-full h-1.5 mt-3">
                  <div
                    className="bg-indigo-600 h-1.5 rounded-full transition-all duration-500"
                    style={{ width: `${job.progress}%` }}
                  />
                </div>
              )}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
