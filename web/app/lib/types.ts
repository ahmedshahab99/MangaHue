export type JobStatus = "queued" | "running" | "complete" | "cancelled" | "failed";
export type JobKind = "unknown" | "page" | "batch" | "link" | "chapter";

export interface JobProgress {
  stage: string;
  completed: number;
  total: number;
  percent: number;
}

export interface MangaHueJob {
  id: string;
  kind: JobKind;
  status: JobStatus;
  progress: JobProgress;
  created_at: string;
  updated_at: string;
  result_url?: string | null;
  report_url?: string | null;
  output_path?: string | null;
  error?: string | null;
  result?: JobResult | null;
}

export interface JobPage {
  index?: number;
  source_url?: string;
  input_path: string;
  output_path: string;
  status: string;
  verdict?: string | null;
}

export interface JobResult {
  output_path?: string;
  output_dir?: string;
  archive_path?: string | null;
  export_path?: string | null;
  processed?: number;
  failed?: number;
  total_pages?: number;
  pages?: JobPage[];
  report?: { director_verdict?: string; input_path?: string };
}

export interface StyleSummary {
  id: string;
  name: string;
  path: string;
  notes?: string | null;
}

export interface UploadResult {
  id: string;
  path: string;
  size: number;
  kind: "page" | "chapter";
}

export interface QuotaSnapshot {
  policy: {
    daily_pages: number;
    daily_jobs: number;
    max_pages_per_job: number;
    max_concurrent_jobs: number;
    max_output_bytes: number;
  };
  pages_used: number;
  jobs_used: number;
  active_jobs: number;
  output_bytes: number;
}

export interface JobEvent {
  status: string;
  stage?: string;
  progress?: number;
  percent?: number;
  completed?: number;
  total?: number;
  error?: string;
}
