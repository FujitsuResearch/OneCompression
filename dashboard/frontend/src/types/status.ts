export const JobStatus = {
  PENDING: "pending",
  RUNNING: "running",
  COMPLETED: "completed",
  FAILED: "failed",
} as const;

export type JobStatus = (typeof JobStatus)[keyof typeof JobStatus];

export const InferenceStatus = {
  NONE: "none",
  DEPLOYING: "deploying",
  READY: "ready",
  FAILED: "failed",
} as const;

export type InferenceStatus = (typeof InferenceStatus)[keyof typeof InferenceStatus];

export const ChatTaskStatus = {
  PENDING: "pending",
  COMPLETED: "completed",
  FAILED: "failed",
} as const;

export type ChatTaskStatus = (typeof ChatTaskStatus)[keyof typeof ChatTaskStatus];
