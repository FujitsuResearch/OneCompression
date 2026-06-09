import { useMemo, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api, type EstimateWbitsResponse, type JobCreate } from "../api/client";

const QUANT_METHODS = [
  { value: "gptq", label: "GPTQ" },
  { value: "autobit", label: "AutoBit" },
  { value: "jointq", label: "JointQ" },
  { value: "auto_run", label: "auto_run" },
] as const;

const INT_BIT_OPTIONS = [2, 3, 4, 8] as const;

function isFractionalMethod(method: string): boolean {
  return method === "autobit" || method === "auto_run";
}

function jointqQepError(method: string, useQep: boolean): string | null {
  if (method === "jointq" && useQep) {
    return "JointQ does not support QEP. Disable QEP or choose a different quantization method.";
  }
  return null;
}

/** Match onecomp.Runner.auto_run / worker floor convention. */
function floorWbits(value: number): number {
  return Math.floor(value * 100) / 100;
}

export default function NewJob() {
  const navigate = useNavigate();
  const [form, setForm] = useState<JobCreate>({
    model_name: "",
    quant_method: "gptq",
    quant_params: {
      bits: 4,
      group_size: 128,
      use_qep: true,
      dataset: "c4",
      num_samples: 128,
      auto_bits: false,
      total_vram_gb: null,
    },
  });

  // VRAM estimation UI state
  const [vramInputGb, setVramInputGb] = useState<string>("");
  const [estimation, setEstimation] = useState<EstimateWbitsResponse | null>(null);

  const mutation = useMutation({
    mutationFn: api.createJob,
    onSuccess: (job) => {
      const saved: string[] = JSON.parse(localStorage.getItem("job_ids") || "[]");
      saved.unshift(job.id);
      localStorage.setItem("job_ids", JSON.stringify(saved.slice(0, 100)));
      navigate(`/jobs/${job.id}`);
    },
  });

  const estimateMutation = useMutation({
    mutationFn: api.estimateWbits,
    onSuccess: (data) => {
      setEstimation(data);
    },
  });

  const set = (field: string, value: unknown) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const setParam = (field: string, value: unknown) => {
    setForm((prev) => ({
      ...prev,
      quant_params: { ...prev.quant_params, [field]: value },
    }));
  };

  const allowsFractionalBits = isFractionalMethod(form.quant_method);
  const isAutoRun = form.quant_method === "auto_run";
  const isJointq = form.quant_method === "jointq";
  const qepValidationError = jointqQepError(form.quant_method, form.quant_params.use_qep);

  // Snap bits to the nearest valid integer when switching to an integer-only method.
  const handleMethodChange = (value: string) => {
    setForm((prev) => {
      const next: JobCreate = { ...prev, quant_method: value };
      if (!isFractionalMethod(value)) {
        const rounded = Math.round(prev.quant_params.bits);
        const snapped = INT_BIT_OPTIONS.includes(rounded as 2 | 3 | 4 | 8) ? rounded : 4;
        next.quant_params = {
          ...prev.quant_params,
          bits: snapped,
          auto_bits: false,
          ...(value === "jointq" ? { use_qep: false } : {}),
        };
      } else if (value === "auto_run") {
        next.quant_params = {
          ...prev.quant_params,
          auto_bits: true,
          group_size: 128,
        };
      } else {
        next.quant_params = { ...prev.quant_params, auto_bits: false };
      }
      return next;
    });
  };

  const handleEstimate = () => {
    if (!form.model_name.trim()) {
      estimateMutation.reset();
      return;
    }
    const v = Number(vramInputGb);
    if (!Number.isFinite(v) || v <= 0) {
      estimateMutation.reset();
      return;
    }
    estimateMutation.mutate({
      model_name: form.model_name.trim(),
      total_vram_gb: v,
      group_size: form.quant_params.group_size,
    });
  };

  const applyEstimation = () => {
    if (!estimation) return;
    // Choose a fractional method if the user is on an int-only one
    setForm((prev) => {
      const method = isFractionalMethod(prev.quant_method) ? prev.quant_method : "autobit";
      return {
        ...prev,
        quant_method: method,
        quant_params: {
          ...prev.quant_params,
          bits: floorWbits(estimation.target_bitwidth),
          total_vram_gb: Number(vramInputGb) || estimation.total_vram_gb,
        },
      };
    });
  };

  const integerBitsValue = useMemo(() => {
    const rounded = Math.round(form.quant_params.bits);
    return INT_BIT_OPTIONS.includes(rounded as 2 | 3 | 4 | 8) ? rounded : 4;
  }, [form.quant_params.bits]);

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">New Quantization Job</h1>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (qepValidationError) return;
          mutation.mutate(form);
        }}
        className="space-y-6"
      >
        <div>
          <label className="block text-sm font-medium mb-1">
            HuggingFace Model
          </label>
          <input
            type="text"
            required
            value={form.model_name}
            onChange={(e) => set("model_name", e.target.value)}
            placeholder="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none transition"
          />
          <p className="text-xs text-gray-500 mt-1">
            The server verifies that the model exists on huggingface.co
            before queuing the job.
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">
            Quantization Method
          </label>
          <select
            value={form.quant_method}
            onChange={(e) => handleMethodChange(e.target.value)}
            className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none transition"
          >
            {QUANT_METHODS.map((m) => (
              <option key={m.value} value={m.value}>
                {m.label}
              </option>
            ))}
          </select>
          {isAutoRun && (
            <p className="text-xs text-gray-500 mt-1">
              auto_run uses AutoBit + QEP. Bit width and group size are
              chosen automatically from GPU VRAM when the job starts; the
              resolved values appear on the job detail page.
            </p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              Bit Width
              {allowsFractionalBits && (
                <span className="ml-2 text-xs text-gray-400">
                  (fractional allowed)
                </span>
              )}
            </label>
            {allowsFractionalBits ? (
              <input
                type="number"
                min={2}
                max={8}
                step={0.1}
                disabled={isAutoRun}
                value={Number.isFinite(form.quant_params.bits) ? form.quant_params.bits : ""}
                onChange={(e) => setParam("bits", Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none transition disabled:bg-gray-100 disabled:text-gray-400"
              />
            ) : (
              <select
                value={integerBitsValue}
                disabled={isAutoRun}
                onChange={(e) => setParam("bits", Number(e.target.value))}
                className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none transition disabled:bg-gray-100 disabled:text-gray-400"
              >
                {INT_BIT_OPTIONS.map((b) => (
                  <option key={b} value={b}>
                    {b}-bit
                  </option>
                ))}
              </select>
            )}
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Group Size
            </label>
            <select
              value={form.quant_params.group_size}
              disabled={isAutoRun}
              onChange={(e) => setParam("group_size", Number(e.target.value))}
              className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none transition disabled:bg-gray-100 disabled:text-gray-400"
            >
              <option value={-1}>Channel-wise (-1)</option>
              {[32, 64, 128, 256].map((g) => (
                <option key={g} value={g}>
                  {g}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* VRAM-based bit width recommender (AutoBit use case) */}
        {!isAutoRun && (
        <details className="rounded-lg border border-gray-200 bg-gray-50/50 px-4 py-3 group">
          <summary className="cursor-pointer text-sm font-medium text-gray-700 select-none">
            Recommend a bit width from VRAM budget
          </summary>
          <div className="mt-3 space-y-3">
            <div className="flex flex-wrap items-end gap-2">
              <div className="flex-1 min-w-[160px]">
                <label className="block text-xs text-gray-500 mb-1">
                  Available VRAM (GB)
                </label>
                <input
                  type="number"
                  min={0.5}
                  step={0.1}
                  value={vramInputGb}
                  onChange={(e) => setVramInputGb(e.target.value)}
                  placeholder="24"
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none transition"
                />
              </div>
              <button
                type="button"
                onClick={handleEstimate}
                disabled={
                  estimateMutation.isPending
                  || !form.model_name.trim()
                  || !vramInputGb
                }
                className="rounded-lg bg-gray-700 hover:bg-gray-800 disabled:bg-gray-300 text-white text-sm font-medium px-4 py-2 transition cursor-pointer"
              >
                {estimateMutation.isPending ? "Estimating..." : "Estimate"}
              </button>
            </div>

            {estimateMutation.isError && (
              <div className="rounded-lg bg-red-50 border border-red-200 text-red-700 px-3 py-2 text-xs">
                {estimateMutation.error.message}
              </div>
            )}

            {estimation && (
              <div className="rounded-lg bg-white border border-gray-200 px-3 py-2 text-xs space-y-1">
                <div className="flex justify-between">
                  <span className="text-gray-500">Recommended bit width</span>
                  <span className="font-medium text-indigo-600">
                    {floorWbits(estimation.target_bitwidth).toFixed(2)} bpw
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">VRAM budget</span>
                  <span>
                    {estimation.budget_gb.toFixed(2)} / {estimation.total_vram_gb.toFixed(2)} GB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-500">Quantizable params</span>
                  <span>
                    {Math.round(estimation.quantizable_params / 1e6)}M /{" "}
                    {Math.round(estimation.total_params / 1e6)}M
                  </span>
                </div>
                <button
                  type="button"
                  onClick={applyEstimation}
                  className="mt-2 w-full rounded-md bg-indigo-50 hover:bg-indigo-100 text-indigo-700 text-xs font-medium py-1.5 transition cursor-pointer"
                >
                  Apply to Bit Width
                </button>
              </div>
            )}
          </div>
        </details>
        )}

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-1">
              Calibration Dataset
            </label>
            <input
              type="text"
              value={form.quant_params.dataset}
              onChange={(e) => setParam("dataset", e.target.value)}
              className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none transition"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">
              Calibration Samples
            </label>
            <input
              type="number"
              min={1}
              max={1024}
              value={form.quant_params.num_samples}
              onChange={(e) => setParam("num_samples", Number(e.target.value))}
              className="w-full rounded-lg border border-gray-300 px-4 py-2 focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none transition"
            />
          </div>
        </div>

        <label className={`flex items-center gap-2 ${isJointq ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}>
          <input
            type="checkbox"
            checked={form.quant_params.use_qep}
            disabled={isJointq}
            onChange={(e) => setParam("use_qep", e.target.checked)}
            className="rounded border-gray-300 text-indigo-600 focus:ring-indigo-200 disabled:cursor-not-allowed"
          />
          <span className="text-sm">
            Enable QEP (Quantization Error Propagation)
          </span>
        </label>
        {isJointq && (
          <p className="text-xs text-gray-500 -mt-4">
            QEP is not available with JointQ.
          </p>
        )}

        {qepValidationError && (
          <div className="rounded-lg bg-red-50 border border-red-200 text-red-700 px-4 py-3 text-sm">
            {qepValidationError}
          </div>
        )}

        {mutation.isError && (
          <div className="rounded-lg bg-red-50 border border-red-200 text-red-700 px-4 py-3 text-sm">
            {mutation.error.message}
          </div>
        )}

        <button
          type="submit"
          disabled={mutation.isPending || !!qepValidationError}
          className="w-full rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white font-medium py-3 transition cursor-pointer"
        >
          {mutation.isPending ? "Submitting..." : "Start Quantization"}
        </button>
      </form>
    </div>
  );
}
