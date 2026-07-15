import { useState, useRef, useEffect, useCallback } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, type ChatMessage, type Job } from "../api/client";
import { ChatTaskStatus, InferenceStatus } from "../types/status";

const INFERENCE_STATUS_STYLES: Record<string, string> = {
  none: "bg-gray-100 text-gray-600",
  deploying: "bg-yellow-100 text-yellow-800",
  ready: "bg-green-100 text-green-800",
  failed: "bg-red-100 text-red-800",
};

const POLL_INTERVAL = 3000;

export default function ChatTab({ job }: { job: Job }) {
  const queryClient = useQueryClient();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [isWaiting, setIsWaiting] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef(false);

  const deployMutation = useMutation({
    mutationFn: () => api.deploy(job.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["job", job.id] });
    },
  });

  const stopMutation = useMutation({
    mutationFn: () => api.stopInference(job.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["job", job.id] });
    },
  });

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isWaiting]);

  useEffect(() => {
    if (!isWaiting) {
      setElapsed(0);
      return;
    }
    const timer = setInterval(() => setElapsed((e) => e + 1), 1000);
    return () => clearInterval(timer);
  }, [isWaiting]);

  const pollForResult = useCallback(async (taskId: string) => {
    while (!abortRef.current) {
      await new Promise((r) => setTimeout(r, POLL_INTERVAL));
      if (abortRef.current) break;

      try {
        const result = await api.getChatResult(taskId);
        if (result.status === ChatTaskStatus.COMPLETED && result.message) {
          setMessages((prev) => [...prev, result.message!]);
          setIsWaiting(false);
          return;
        }
        if (result.status === ChatTaskStatus.FAILED) {
          setError(result.error || "Inference failed");
          setIsWaiting(false);
          return;
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Polling failed");
        setIsWaiting(false);
        return;
      }
    }
  }, []);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || isWaiting) return;

    const userMsg: ChatMessage = { role: "user", content: text };
    const updated = [...messages, userMsg];
    setMessages(updated);
    setInput("");
    setError(null);
    setIsWaiting(true);
    abortRef.current = false;

    try {
      const res = await api.chat(job.id, { messages: updated });

      if (res.message) {
        setMessages((prev) => [...prev, res.message!]);
        setIsWaiting(false);
        return;
      }

      if (res.task_id) {
        pollForResult(res.task_id);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Chat request failed");
      setIsWaiting(false);
    }
  }, [input, isWaiting, messages, job.id, pollForResult]);

  useEffect(() => {
    return () => {
      abortRef.current = true;
    };
  }, []);

  if (job.inference_status === InferenceStatus.NONE || job.inference_status === InferenceStatus.FAILED) {
    return (
      <div className="flex flex-col items-center justify-center py-16 px-6">
        <div className="text-center mb-6">
          <h3 className="text-lg font-semibold mb-2">Deploy Model for Chat</h3>
          <p className="text-sm text-gray-500 max-w-md">
            Deploy the quantized model for inference.
            Once deployed, you can chat with the model in real time.
          </p>
        </div>

        {job.inference_status === InferenceStatus.FAILED && (
          <div className="rounded-lg bg-red-50 border border-red-200 text-red-700 px-4 py-3 text-sm mb-4 max-w-md">
            Deployment failed. Please try again.
          </div>
        )}

        <button
          onClick={() => deployMutation.mutate()}
          disabled={deployMutation.isPending}
          className="rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white font-medium px-8 py-3 transition cursor-pointer"
        >
          {deployMutation.isPending ? "Requesting..." : "Deploy & Start Chat"}
        </button>

        {deployMutation.isError && (
          <p className="text-red-600 text-sm mt-3">{deployMutation.error.message}</p>
        )}
      </div>
    );
  }

  if (job.inference_status === InferenceStatus.DEPLOYING) {
    return (
      <div className="flex flex-col items-center justify-center py-16">
        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600 mb-4" />
        <p className="text-gray-600 font-medium">Loading model...</p>
        <p className="text-sm text-gray-400 mt-1">This may take a moment</p>
      </div>
    );
  }

  const formatElapsed = (s: number) => {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
  };

  return (
    <div className="flex flex-col h-[60vh]">
      {/* Header bar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-100 bg-gray-50/50">
        <div className="flex items-center gap-2">
          <span
            className={`text-xs font-medium px-2 py-0.5 rounded-full ${INFERENCE_STATUS_STYLES[job.inference_status]}`}
          >
            Inference: {job.inference_status}
          </span>
          <span className="text-xs text-gray-400">{job.model_name}</span>
        </div>
        <button
          onClick={() => stopMutation.mutate()}
          disabled={stopMutation.isPending}
          className="text-xs text-red-500 hover:text-red-700 cursor-pointer"
        >
          Stop server
        </button>
      </div>

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto space-y-4 p-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 py-8">
            <p className="text-sm">Model is ready. Send a message to start chatting.</p>
          </div>
        )}

        {messages.map((msg, i) => (
          <div
            key={i}
            className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
          >
            <div
              className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap ${
                msg.role === "user"
                  ? "bg-indigo-600 text-white"
                  : "bg-gray-100 text-gray-900"
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}

        {isWaiting && (
          <div className="flex justify-start">
            <div className="bg-gray-100 rounded-2xl px-4 py-2.5 text-sm text-gray-400">
              <span className="inline-flex items-center gap-2">
                <span className="inline-flex gap-1">
                  <span className="animate-bounce">.</span>
                  <span className="animate-bounce" style={{ animationDelay: "0.1s" }}>.</span>
                  <span className="animate-bounce" style={{ animationDelay: "0.2s" }}>.</span>
                </span>
                {elapsed >= 5 && (
                  <span className="text-xs text-gray-400">
                    {formatElapsed(elapsed)}
                  </span>
                )}
              </span>
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-lg bg-red-50 border border-red-200 text-red-700 px-4 py-3 text-sm">
            {error}
          </div>
        )}

        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 p-4">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
          className="flex gap-2"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            className="flex-1 rounded-lg border border-gray-300 px-4 py-2.5 text-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 outline-none transition"
            disabled={isWaiting}
          />
          <button
            type="submit"
            disabled={isWaiting || !input.trim()}
            className="rounded-lg bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-300 text-white px-5 py-2.5 text-sm font-medium transition cursor-pointer"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
