import type { JobEvent, MangaHueJob, StyleSummary, UploadResult } from "./types";

const configuredBase = process.env.NEXT_PUBLIC_MANGAHUE_API_URL?.replace(/\/$/, "");

export function apiBase(): string {
  if (configuredBase) return configuredBase;
  return "";
}

function apiUrl(path: string): string {
  return `${apiBase()}${path}`;
}

function requestHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  headers.set("x-mangahue-local", "1");
  const token = typeof window !== "undefined" ? window.localStorage.getItem("mangahue-token") : null;
  if (token) headers.set("authorization", `Bearer ${token}`);
  return headers;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { error?: string };
    return body.error || `${response.status} ${response.statusText}`;
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = requestHeaders(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set("content-type", "application/json");
  const response = await fetch(apiUrl(path), { ...init, headers, cache: "no-store" });
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as T;
}

export async function healthcheck(): Promise<boolean> {
  try {
    const response = await fetch(apiUrl("/health"), { cache: "no-store" });
    return response.ok;
  } catch {
    return false;
  }
}

export function getStyles(): Promise<StyleSummary[]> {
  return apiRequest<StyleSummary[]>("/api/styles");
}

export function getJobs(): Promise<MangaHueJob[]> {
  return apiRequest<MangaHueJob[]>("/api/jobs");
}

export function getJob(id: string): Promise<MangaHueJob> {
  return apiRequest<MangaHueJob>(`/api/jobs/${encodeURIComponent(id)}`);
}

export function cancelJob(id: string): Promise<{ status: string; id: string }> {
  return apiRequest(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
}

export async function uploadSource(file: File): Promise<UploadResult> {
  const body = new FormData();
  body.append("file", file);
  return apiRequest<UploadResult>("/api/upload", { method: "POST", body });
}

export async function fetchAsset(path: string): Promise<string> {
  const response = await fetch(apiUrl(`/api/file?path=${encodeURIComponent(path)}`), {
    headers: requestHeaders(),
    cache: "no-store",
  });
  if (!response.ok) throw new Error(await readError(response));
  return URL.createObjectURL(await response.blob());
}

export async function downloadAsset(path: string, filename: string): Promise<void> {
  const objectUrl = await fetchAsset(path);
  const link = document.createElement("a");
  link.href = objectUrl;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(objectUrl), 2_000);
}

export async function streamJob(
  id: string,
  onEvent: (event: JobEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(apiUrl(`/api/jobs/${encodeURIComponent(id)}/events`), {
    headers: requestHeaders({ accept: "text/event-stream" }),
    signal,
    cache: "no-store",
  });
  if (!response.ok || !response.body) throw new Error(await readError(response));

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (!signal.aborted) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const data = frame
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trim())
        .join("\n");
      if (!data) continue;
      try {
        onEvent(JSON.parse(data) as JobEvent);
      } catch {
        // Ignore malformed keep-alive frames without interrupting the job.
      }
    }
  }
}
