"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Image from "next/image";
import {
  Check,
  ChevronDown,
  ChevronUp,
  CircleStop,
  CloudOff,
  Download,
  FileArchive,
  FileImage,
  FolderOpen,
  Gauge,
  History,
  ImageIcon,
  KeyRound,
  Link2,
  LoaderCircle,
  Menu,
  Play,
  RefreshCw,
  Sparkles,
  Trash2,
  Upload,
  WandSparkles,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { Toaster } from "sonner";

import {
  apiRequest,
  cancelJob,
  downloadAsset,
  fetchAsset,
  getAccessToken,
  getJob,
  getJobs,
  getQuota,
  getStyles,
  healthcheck,
  setAccessToken,
  streamJob,
  uploadSource,
} from "@/lib/api";
import type { JobEvent, JobPage, MangaHueJob, StyleSummary } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { Switch } from "@/components/ui/switch";
import { cn } from "@/lib/utils";

type View = "create" | "queue" | "library";
type SourceMode = "link" | "upload";
type Quality = "fast" | "studio" | "masterpiece";

const terminalStatuses = new Set(["complete", "failed", "cancelled"]);

const STAGE_LABELS: Record<string, string> = {
  queued: "Waiting for the engine",
  running: "Working",
  downloading: "Downloading chapter",
  preparing_chapter: "Preparing chapter",
  inference_started: "Coloring pages",
  colorize: "Coloring pages",
  palette: "Picking palette",
  consistency: "Matching character colors",
  export: "Building CBZ",
  complete: "Complete",
  cancelled: "Cancelled",
  failed: "Failed",
};

const QUALITY_COPY: Record<Quality, { title: string; hint: string }> = {
  fast: { title: "Fast", hint: "Quick preview · lower detail" },
  studio: { title: "Studio", hint: "Balanced quality · recommended" },
  masterpiece: { title: "Masterpiece", hint: "Ensemble + pyramid · slowest" },
};

function friendlyError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (/401|token|unauthor/i.test(message)) {
    return "This token is invalid or expired — update it in Access.";
  }
  if (/quota|limit/i.test(message)) {
    return "You have used today's page budget. It resets at midnight UTC.";
  }
  if (/413|too large|exceeds 50/i.test(message)) {
    return "That file is too large — try a single page or a smaller chapter.";
  }
  if (/not found|404/i.test(message)) {
    return "We could not reach that chapter — it may be behind a login or geo-block.";
  }
  return message;
}

function statusLabel(value: string): string {
  return value.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

function shortId(id: string): string {
  return id.slice(0, 8).toUpperCase();
}

function timeLabel(value?: string): string {
  if (!value) return "Now";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Now";
  return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(date);
}

function jobPages(job: MangaHueJob | null): JobPage[] {
  if (!job?.result) return [];
  if (job.result.pages?.length) return job.result.pages;
  const output = job.result.output_path || (job.kind === "page" ? job.output_path : null);
  return output
    ? [{ input_path: job.result.report?.input_path || "", output_path: output, status: job.status }]
    : [];
}

function archivePath(job: MangaHueJob | null): string | null {
  return job?.result?.archive_path || job?.result?.export_path || null;
}

export function MangaHueStudio() {
  const [view, setView] = useState<View>("create");
  const [sourceMode, setSourceMode] = useState<SourceMode>("upload");
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [styles, setStyles] = useState<StyleSummary[]>([]);
  const [stylePath, setStylePath] = useState("");
  const [quality, setQuality] = useState<Quality>("studio");
  const [maxPages, setMaxPages] = useState(80);
  const [pageCeiling, setPageCeiling] = useState(500);
  const [autoPalette, setAutoPalette] = useState(true);
  const [consistency, setConsistency] = useState(true);
  const [jobs, setJobs] = useState<MangaHueJob[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<MangaHueJob | null>(null);
  const [selectedPage, setSelectedPage] = useState(0);
  const [beforeImage, setBeforeImage] = useState<string | null>(null);
  const [afterImage, setAfterImage] = useState<string | null>(null);
  const [compare, setCompare] = useState(52);
  const [connected, setConnected] = useState(false);
  const [starting, setStarting] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [accessOpen, setAccessOpen] = useState(false);
  const [tokenDraft, setTokenDraft] = useState("");
  const [savingToken, setSavingToken] = useState(false);
  const [quota, setQuota] = useState<{ used: number; limit: number } | null>(null);
  const [demoLoading, setDemoLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const streamAbortRef = useRef<AbortController | null>(null);

  const activeJob = useMemo(
    () => jobs.find((job) => job.id === activeId) || (selectedJob?.id === activeId ? selectedJob : null),
    [activeId, jobs, selectedJob],
  );
  const pages = useMemo(() => jobPages(selectedJob), [selectedJob]);

  const refreshJobs = useCallback(async () => {
    try {
      const nextJobs = await getJobs();
      setJobs(nextJobs);
      setConnected(true);
      if (selectedJob) {
        const fresh = nextJobs.find((job) => job.id === selectedJob.id);
        if (fresh) setSelectedJob(fresh);
      }
      const running = nextJobs.find((job) => job.status === "running" || job.status === "queued");
      if (running) setActiveId((current) => current || running.id);
    } catch {
      setConnected(false);
    }
  }, [selectedJob]);

  const refreshQuota = useCallback(async () => {
    try {
      const snapshot = await getQuota();
      setQuota({
        used: Number(snapshot.pages_used) || 0,
        limit: Number(snapshot.policy.daily_pages) || 0,
      });
    } catch {
      setQuota(null);
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    void (async () => {
      const health = await healthcheck();
      try {
        const [nextStyles, nextJobs] = await Promise.all([getStyles(), getJobs()]);
        if (!mounted) return;
        setConnected(health);
        setStyles(nextStyles);
        setJobs(nextJobs);
        void refreshQuota();
        const preferred = nextStyles.find((style) => style.id === "target_style") || nextStyles[0];
        if (preferred) setStylePath(preferred.path);
        const current = nextJobs.find((job) => job.status === "running" || job.status === "queued");
        if (current) setActiveId(current.id);
      } catch {
        if (!mounted) return;
        setConnected(false);
        if (health) {
          setTokenDraft(getAccessToken());
          setAccessOpen(true);
        }
      }
    })();
    const poller = window.setInterval(() => {
      void refreshJobs();
      void refreshQuota();
    }, 10000);
    return () => {
      mounted = false;
      window.clearInterval(poller);
    };
  }, [refreshJobs, refreshQuota]);

  useEffect(() => {
    if (!activeId) return;
    streamAbortRef.current?.abort();
    const controller = new AbortController();
    streamAbortRef.current = controller;

    const updateFromEvent = (event: JobEvent) => {
      setJobs((current) =>
        current.map((job) =>
          job.id === activeId
            ? {
                ...job,
                status: (event.status === "progress" ? "running" : event.status) as MangaHueJob["status"],
                progress: {
                  stage: event.stage || job.progress.stage,
                  completed: event.completed ?? event.tiles_completed ?? job.progress.completed,
                  total: event.total ?? event.tiles_total ?? job.progress.total,
                  percent: event.progress ?? event.percent ?? job.progress.percent,
                },
                error: event.error || job.error,
              }
            : job,
        ),
      );
      if (terminalStatuses.has(event.status)) {
        if (event.status === "failed") {
          setError(friendlyError(event.error || "Colorization failed"));
        } else if (event.status === "complete") {
          toast.success("Colorization complete");
          void refreshQuota();
        }
        void getJob(activeId).then((job) => {
          setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
          setSelectedJob(job);
          setSelectedPage(0);
          setView("create");
        });
      }
    };

    void streamJob(activeId, updateFromEvent, controller.signal).catch((streamError: unknown) => {
      if (!controller.signal.aborted) {
        setError(friendlyError(streamError));
      }
    });
    return () => controller.abort();
  }, [activeId, refreshQuota]);

  useEffect(() => {
    let live = true;
    let beforeObject: string | null = null;
    let afterObject: string | null = null;
    const page = pages[selectedPage];
    queueMicrotask(() => {
      if (!live) return;
      setBeforeImage(sourceFile && selectedJob?.kind === "page" ? sourcePreview : null);
    });
    if (!page?.output_path) return;

    Promise.all([
      page.input_path ? fetchAsset(page.input_path, selectedJob?.id).catch(() => null) : Promise.resolve(null),
      fetchAsset(page.output_path, selectedJob?.id),
    ]).then(([before, after]) => {
      beforeObject = before;
      afterObject = after;
      if (!live) {
        if (before) URL.revokeObjectURL(before);
        URL.revokeObjectURL(after);
        return;
      }
      if (before) setBeforeImage(before);
      setAfterImage(after);
    }).catch((assetError: unknown) => {
      if (live) setError(friendlyError(assetError));
    });
    return () => {
      live = false;
      if (beforeObject) URL.revokeObjectURL(beforeObject);
      if (afterObject) URL.revokeObjectURL(afterObject);
    };
  }, [pages, selectedPage, selectedJob?.id, selectedJob?.kind, sourceFile, sourcePreview]);

  useEffect(() => {
    return () => {
      if (sourcePreview) URL.revokeObjectURL(sourcePreview);
    };
  }, [sourcePreview]);

  const chooseFile = useCallback((file: File | null) => {
    if (!file) return;
    setError(null);
    setSourceFile(file);
    if (sourcePreview) URL.revokeObjectURL(sourcePreview);
    setSourcePreview(file.type.startsWith("image/") ? URL.createObjectURL(file) : null);
  }, [sourcePreview]);

  async function loadDemoPage() {
    setDemoLoading(true);
    setError(null);
    try {
      const response = await fetch("/samples/demo-page.jpg");
      if (!response.ok) throw new Error("Demo page unavailable");
      const blob = await response.blob();
      const file = new File([blob], "demo-page.jpg", { type: "image/jpeg" });
      setSourceMode("upload");
      chooseFile(file);
      toast.success("Demo page loaded — press Start to colorize");
    } catch (demoError: unknown) {
      setError(friendlyError(demoError));
    } finally {
      setDemoLoading(false);
    }
  }

  async function startJob() {
    if (sourceMode === "link" && !sourceUrl.trim()) {
      setError("Enter a manga or chapter link.");
      return;
    }
    if (sourceMode === "upload" && !sourceFile) {
      setError("Choose a manga page or chapter archive.");
      return;
    }
    setStarting(true);
    setError(null);
    try {
      const root = `.mangahue/jobs/web/${crypto.randomUUID()}`;
      const template = {
        input_path: "",
        output_path: `${root}/result.png`,
        model_path: null,
        reference_image_path: null,
        config_path: "configs/engine.generative-studio.yaml",
        preset_path: null,
        style_path: stylePath || null,
        project_path: null,
        region_locks_path: null,
        character_ids: [],
        user_hints_path: null,
        debug_dir: null,
        use_mock_backend: false,
        ensemble_model_paths: [],
        enable_ensemble: quality === "masterpiece",
        pyramid_enabled: quality !== "fast",
        hint_propagation_enabled: consistency,
        palette_path: null,
        auto_palette: autoPalette,
        draft_path: null,
      };

      let job: MangaHueJob;
      if (sourceMode === "link") {
        job = await apiRequest<MangaHueJob>("/api/link/colorize", {
          method: "POST",
          body: JSON.stringify({
            source_url: sourceUrl.trim(),
            output_dir: `${root}/out`,
            downloads_dir: `${root}/downloads`,
            template,
            max_pages: maxPages,
            max_bytes_per_page: 64 * 1024 * 1024,
            continue_on_error: true,
            user_agent: null,
          }),
        });
      } else {
        const upload = await uploadSource(sourceFile as File);
        template.input_path = upload.path;
        if (upload.kind === "chapter") {
          job = await apiRequest<MangaHueJob>("/api/chapter/colorize", {
            method: "POST",
            body: JSON.stringify({
              input_path: upload.path,
              work_dir: `${root}/work`,
              output_dir: `${root}/out`,
              export_path: `${root}/mangahue-colored.cbz`,
              gallery_path: `${root}/gallery.html`,
              manifest_path: `${root}/manifest.json`,
              brain_report_path: `${root}/brain.json`,
              locked_style_path: `${root}/locked-style.json`,
              palette_memory_path: `${root}/palette-memory.json`,
              palette_memory_style_path: `${root}/palette-style.json`,
              safety_report_path: `${root}/safety.json`,
              template,
              max_pages: maxPages,
              max_bytes_per_page: 64 * 1024 * 1024,
              resume: false,
              safety_gate: true,
              enforce_safety_gate: false,
              consistency_retry: consistency,
              retry_drift_threshold: 0.18,
              retry_tone_threshold: 0.34,
              retry_max_pages: maxPages,
            }),
          });
        } else {
          job = await apiRequest<MangaHueJob>("/api/colorize", {
            method: "POST",
            body: JSON.stringify({ request: template, ads_token: null, premium_account_id: null }),
          });
        }
      }
      setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
      setActiveId(job.id);
      setSelectedJob(job);
      toast.info("Job started");
    } catch (startError: unknown) {
      setError(friendlyError(startError));
    } finally {
      setStarting(false);
    }
  }

  async function stopActiveJob() {
    if (!activeId) return;
    try {
      await cancelJob(activeId);
      await refreshJobs();
      toast.info("Job cancelled — finished pages stay in your Library");
    } catch (cancelError: unknown) {
      setError(friendlyError(cancelError));
    }
  }

  async function openJob(job: MangaHueJob) {
    const fresh = await getJob(job.id).catch(() => job);
    setSelectedJob(fresh);
    setSelectedPage(0);
    setView("create");
    if (!terminalStatuses.has(fresh.status)) setActiveId(fresh.id);
  }

  async function saveAccessToken() {
    const token = tokenDraft.trim();
    if (!token) {
      toast.error("Enter an access token");
      return;
    }
    setSavingToken(true);
    setAccessToken(token);
    try {
      await getQuota();
      setAccessOpen(false);
      toast.success("Connected to the engine");
      setConnected(true);
      void refreshJobs();
      void refreshQuota();
    } catch {
      setAccessToken("");
      toast.error("That token was rejected by the engine — check it and try again");
    } finally {
      setSavingToken(false);
    }
  }

  function clearAccessToken() {
    setAccessToken("");
    setTokenDraft("");
    setAccessOpen(false);
    setConnected(false);
    setQuota(null);
    toast.info("Access token removed");
  }

  const progress = activeJob?.progress.percent || (activeJob?.status === "complete" ? 100 : 0);
  const canStart = sourceMode === "link" ? Boolean(sourceUrl.trim()) : Boolean(sourceFile);
  const stageLabel = STAGE_LABELS[activeJob?.progress.stage || ""] || statusLabel(activeJob?.progress.stage || activeJob?.status || "");
  // The engine reports unlimited quotas as a u64 max sentinel; show a plain
  // count instead of an absurd number.
  const quotaUnlimited = quota !== null && quota.limit >= 1e15;
  const quotaText =
    quota !== null && !quotaUnlimited && quota.limit > 0
      ? `${quota.used}/${quota.limit} pages today`
      : quota !== null
        ? `${quota.used} pages today`
        : null;

  const navItem = (target: View, label: string, icon: React.ReactNode) => (
    <button
      key={label}
      onClick={() => {
        setView(target);
        setSidebarOpen(false);
      }}
      className={cn(
        "flex w-full items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors",
        view === target
          ? "bg-accent text-accent-foreground"
          : "text-muted-foreground hover:bg-accent/60 hover:text-foreground",
      )}
    >
      {icon}
      <span>{label}</span>
    </button>
  );

  return (
    <div className="flex min-h-screen bg-background text-foreground">
      <Toaster theme="dark" position="bottom-right" richColors />

      {/* Sidebar */}
      <aside
        className={cn(
          "fixed inset-y-0 left-0 z-40 flex w-56 flex-col border-r border-border bg-sidebar p-3 transition-transform",
          sidebarOpen ? "translate-x-0" : "-translate-x-full lg:translate-x-0",
        )}
      >
        <div className="flex items-center gap-2 px-2 py-3">
          <div className="flex size-9 items-center justify-center rounded-md bg-primary font-bold text-primary-foreground">
            MH
          </div>
          <div>
            <div className="text-sm font-semibold leading-none">MangaHue</div>
            <div className="text-xs text-muted-foreground">Color studio</div>
          </div>
        </div>
        <Separator className="my-3" />
        <nav className="flex flex-col gap-1" aria-label="Workspace">
          {navItem("create", "Create", <WandSparkles size={18} />)}
          {navItem("queue", "Queue", <History size={18} />)}
          {navItem("library", "Library", <FolderOpen size={18} />)}
        </nav>
        <div className="mt-auto flex items-center gap-2 rounded-md px-2 py-2 text-xs text-muted-foreground">
          <span className={cn("size-2 rounded-full", connected ? "bg-emerald-500" : "bg-destructive")} />
          {connected ? "Engine ready" : "Engine offline"}
        </div>
        <Button variant="ghost" size="sm" className="justify-start text-muted-foreground" onClick={() => { setTokenDraft(getAccessToken()); setAccessOpen(true); setSidebarOpen(false); }}>
          <KeyRound size={16} /> API access
        </Button>
      </aside>
      {sidebarOpen && (
        <button className="fixed inset-0 z-30 bg-black/50 lg:hidden" onClick={() => setSidebarOpen(false)} aria-label="Close navigation" />
      )}

      {/* Main */}
      <main className="flex min-h-screen flex-1 flex-col lg:pl-56">
        <header className="sticky top-0 z-20 flex items-center gap-3 border-b border-border bg-background/90 px-4 py-3 backdrop-blur">
          <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setSidebarOpen(true)} title="Open navigation">
            <Menu size={20} />
          </Button>
          <div className="min-w-0">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">MangaHue Studio</div>
            <h1 className="truncate text-lg font-semibold leading-tight">
              {view === "create" ? "Color workspace" : view === "queue" ? "Processing queue" : "Color library"}
            </h1>
          </div>
          <div className="ml-auto flex items-center gap-2">
            {quotaText && (
              <Badge variant="outline" className="hidden sm:inline-flex items-center gap-1.5">
                <Gauge size={13} /> {quotaText}
              </Badge>
            )}
            <Badge variant={connected ? "default" : "destructive"} className="gap-1.5">
              {connected ? <Gauge size={13} /> : <CloudOff size={13} />}
              {connected ? "Engine ready" : "Disconnected"}
            </Badge>
            <Button variant="ghost" size="icon" onClick={() => { setTokenDraft(getAccessToken()); setAccessOpen(true); }} title="API access">
              <KeyRound size={18} />
            </Button>
            <Button variant="ghost" size="icon" onClick={() => { void refreshJobs(); void refreshQuota(); }} title="Refresh">
              <RefreshCw size={18} />
            </Button>
          </div>
        </header>

        {error && (
          <div role="alert" className="mx-4 mt-4 flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            <span className="flex-1">{error}</span>
            <button onClick={() => setError(null)} aria-label="Dismiss error">
              <X size={16} />
            </button>
          </div>
        )}

        {view === "create" && (
          <div className="grid flex-1 gap-4 p-4 xl:grid-cols-[400px_minmax(0,1fr)]">
            {/* Control panel */}
            <Card className="h-fit">
              <CardHeader>
                <CardTitle className="text-base">Source</CardTitle>
                <CardDescription>Upload a page or chapter, or paste a manga link.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex rounded-md border border-border p-1">
                  {(["upload", "link"] as SourceMode[]).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setSourceMode(mode)}
                      className={cn(
                        "flex flex-1 items-center justify-center gap-1.5 rounded px-3 py-1.5 text-sm transition-colors",
                        sourceMode === mode ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:text-foreground",
                      )}
                    >
                      {mode === "upload" ? <Upload size={15} /> : <Link2 size={15} />}
                      {mode === "upload" ? "Upload" : "Link"}
                    </button>
                  ))}
                </div>

                {sourceMode === "link" ? (
                  <div className="space-y-2">
                    <Label htmlFor="source-url">Manga or chapter URL</Label>
                    <Input
                      id="source-url"
                      placeholder="https://..."
                      value={sourceUrl}
                      onChange={(event) => setSourceUrl(event.target.value)}
                    />
                    <p className="text-xs text-muted-foreground">
                      nhentai, MangaDex, e-hentai, MangaPlus, Webtoons, Batoto and direct image links.
                    </p>
                  </div>
                ) : (
                  <div
                    className={cn(
                      "flex flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed border-border px-4 py-8 text-center transition-colors",
                      dragging && "border-primary bg-accent/50",
                    )}
                    onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(event) => {
                      event.preventDefault();
                      setDragging(false);
                      chooseFile(event.dataTransfer.files[0] || null);
                    }}
                    onClick={() => fileInputRef.current?.click()}
                    role="button"
                    tabIndex={0}
                    onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") fileInputRef.current?.click(); }}
                  >
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept="image/png,image/jpeg,image/webp,image/gif,application/zip,application/x-cbz,application/vnd.comicbook+zip,.cbz,.zip"
                      className="hidden"
                      onChange={(event) => chooseFile(event.target.files?.[0] || null)}
                    />
                    {sourcePreview ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={sourcePreview} alt="Source preview" className="max-h-40 rounded-md object-contain" />
                    ) : (
                      <>
                        <ImageIcon size={28} className="text-muted-foreground" />
                        <p className="text-sm font-medium">Drop a page or chapter here</p>
                        <p className="text-xs text-muted-foreground">PNG · JPG · WEBP · CBZ · ZIP (up to 50 MB)</p>
                      </>
                    )}
                    {sourceFile && (
                      <p className="max-w-full truncate text-xs text-muted-foreground">
                        {sourceFile.name} · {(sourceFile.size / 1048576).toFixed(1)} MB
                      </p>
                    )}
                  </div>
                )}

                <Separator />

                <div className="space-y-2">
                  <Label>Color direction</Label>
                  <div className="grid grid-cols-1 gap-1">
                    {styles.length === 0 ? (
                      <Skeleton className="h-8 w-full" />
                    ) : (
                      styles.map((style) => (
                        <button
                          key={style.id}
                          onClick={() => setStylePath(style.path)}
                          className={cn(
                            "flex items-center justify-between rounded-md border px-3 py-2 text-left text-sm transition-colors",
                            stylePath === style.path
                              ? "border-primary bg-accent/60 text-foreground"
                              : "border-border text-muted-foreground hover:text-foreground",
                          )}
                        >
                          <span>{style.name}</span>
                          {stylePath === style.path && <Check size={15} className="text-primary" />}
                        </button>
                      ))
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label>Render quality</Label>
                  <div className="grid grid-cols-3 gap-1">
                    {(Object.keys(QUALITY_COPY) as Quality[]).map((level) => (
                      <button
                        key={level}
                        onClick={() => setQuality(level)}
                        title={QUALITY_COPY[level].hint}
                        className={cn(
                          "rounded-md border px-2 py-2 text-xs font-medium transition-colors",
                          quality === level
                            ? "border-primary bg-accent/60 text-foreground"
                            : "border-border text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {QUALITY_COPY[level].title}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="flex items-center justify-between gap-3">
                  <div>
                    <Label htmlFor="palette-toggle">Palette intelligence</Label>
                    <p className="text-xs text-muted-foreground">Automatic scene palette</p>
                  </div>
                  <Switch id="palette-toggle" checked={autoPalette} onCheckedChange={setAutoPalette} />
                </div>
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <Label htmlFor="consistency-toggle">Chapter consistency</Label>
                    <p className="text-xs text-muted-foreground">Carry identity and color memory</p>
                  </div>
                  <Switch id="consistency-toggle" checked={consistency} onCheckedChange={setConsistency} />
                </div>

                <div className="flex items-center justify-between gap-3">
                  <Label>Maximum pages</Label>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="icon" className="size-7" onClick={() => setMaxPages((value) => Math.max(1, value - 10))} disabled={maxPages <= 1}>
                      <ChevronDown size={14} />
                    </Button>
                    <span className="w-12 text-center text-sm font-medium tabular-nums">{maxPages}</span>
                    <Button variant="outline" size="icon" className="size-7" onClick={() => setMaxPages((value) => Math.min(pageCeiling, value + 10))} disabled={maxPages >= pageCeiling}>
                      <ChevronUp size={14} />
                    </Button>
                  </div>
                </div>

                <Button className="w-full" size="lg" disabled={!canStart || starting} onClick={() => void startJob()}>
                  {starting ? <LoaderCircle size={18} className="animate-spin" /> : <Play size={18} />}
                  {starting ? "Starting…" : "Start colorization"}
                </Button>
              </CardContent>
            </Card>

            {/* Viewer */}
            <Card className="flex min-h-[420px] flex-col">
              <CardHeader className="flex-row items-center justify-between space-y-0">
                <div>
                  <CardTitle className="text-base">Color preview</CardTitle>
                  <CardDescription>
                    {selectedJob ? `${statusLabel(selectedJob.status)} · ${shortId(selectedJob.id)}` : "Choose a source to begin"}
                  </CardDescription>
                </div>
                <div className="flex items-center gap-1.5">
                  {afterImage && (
                    <>
                      <Button variant="outline" size="sm" onClick={() => void downloadAsset(pages[selectedPage]?.output_path || "", `mangahue-${shortId(selectedJob?.id || "")}-page-${selectedPage + 1}.png`, selectedJob?.id)} title="Download page as PNG">
                        <FileImage size={15} /> PNG
                      </Button>
                      <Button variant="outline" size="sm" onClick={() => void downloadAsset(archivePath(selectedJob) || "", `mangahue-${shortId(selectedJob?.id || "")}-chapter.cbz`, selectedJob?.id)} title="Download chapter as CBZ">
                        <FileArchive size={15} /> CBZ
                      </Button>
                    </>
                  )}
                </div>
              </CardHeader>
              <CardContent className="flex flex-1 flex-col">
                {activeJob && !terminalStatuses.has(activeJob.status) && (
                  <div className="mb-3 space-y-2 rounded-md border border-border bg-accent/30 p-3" aria-live="polite">
                    <div className="flex items-center justify-between text-sm">
                      <span className="flex items-center gap-2 font-medium">
                        <LoaderCircle size={15} className="animate-spin" />
                        {stageLabel}
                      </span>
                      <span className="text-xs text-muted-foreground tabular-nums">{Math.round(progress)}%</span>
                    </div>
                    <Progress value={progress} />
                    {activeJob.progress.total > 0 && (
                      <p className="text-xs text-muted-foreground">
                        {activeJob.progress.completed} of {activeJob.progress.total} tiles
                      </p>
                    )}
                    <Button variant="destructive" size="sm" onClick={() => void stopActiveJob()}>
                      <CircleStop size={14} /> Cancel
                    </Button>
                  </div>
                )}

                <div className="relative flex-1 overflow-hidden rounded-md border border-border bg-black/30">
                  {beforeImage ? (
                    <>
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={beforeImage} alt="Original manga page" className="absolute inset-0 h-full w-full object-contain" />
                      {afterImage && (
                        <div className="absolute inset-0" style={{ clipPath: `inset(0 0 0 ${compare}%)` }}>
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={afterImage} alt="Colorized manga page" className="h-full w-full object-contain" />
                        </div>
                      )}
                      <div className="pointer-events-none absolute inset-0" style={{ left: `${compare}%` }}>
                        <div className="h-full w-px bg-primary" />
                      </div>
                      <input
                        aria-label="Before and after comparison"
                        type="range"
                        min={0}
                        max={100}
                        value={compare}
                        onChange={(event) => setCompare(Number(event.target.value))}
                        className="absolute inset-x-0 bottom-0 z-10 h-6 w-full cursor-ew-resize opacity-0"
                      />
                      <div className="pointer-events-none absolute bottom-1 left-2 rounded bg-black/60 px-2 py-0.5 text-[11px] text-white">
                        BEFORE
                      </div>
                      <div className="pointer-events-none absolute bottom-1 right-2 rounded bg-black/60 px-2 py-0.5 text-[11px] text-white">
                        AFTER
                      </div>
                    </>
                  ) : (
                    <div className="flex h-full min-h-[300px] flex-col items-center justify-center gap-3 p-6 text-center">
                      <Sparkles size={32} className="text-primary" />
                      <div>
                        <p className="font-semibold">Color your first page in seconds</p>
                        <p className="mt-1 text-sm text-muted-foreground">
                          Upload a page or paste a chapter link — the engine handles ink, screentones and character colors.
                        </p>
                      </div>
                      <Button variant="secondary" size="sm" disabled={demoLoading} onClick={() => void loadDemoPage()}>
                        {demoLoading ? <LoaderCircle size={15} className="animate-spin" /> : <WandSparkles size={15} />}
                        Try a demo page
                      </Button>
                    </div>
                  )}
                </div>

                {pages.length > 1 && (
                  <div className="mt-3 flex items-center gap-2 overflow-x-auto pb-1" aria-label="Result pages">
                    <span className="shrink-0 text-xs text-muted-foreground">{pages.length} PAGES</span>
                    {pages.map((page, index) => (
                      <button
                        key={`${page.output_path}-${index}`}
                        onClick={() => setSelectedPage(index)}
                        aria-label={`Page ${index + 1} of ${pages.length}`}
                        className={cn(
                          "size-8 shrink-0 rounded-md border text-xs font-medium tabular-nums transition-colors",
                          selectedPage === index
                            ? "border-primary bg-accent text-foreground"
                            : "border-border text-muted-foreground hover:text-foreground",
                        )}
                      >
                        {String(index + 1).padStart(2, "0")}
                      </button>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {view === "queue" && (
          <div className="p-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Processing queue</CardTitle>
                <CardDescription>Live jobs and their progress.</CardDescription>
              </CardHeader>
              <CardContent>
                {jobs.length === 0 ? (
                  <p className="py-8 text-center text-sm text-muted-foreground">No jobs yet — start colorizing from the Create tab.</p>
                ) : (
                  <div className="space-y-2">
                    {jobs.map((job) => (
                      <div key={job.id} className="flex items-center gap-3 rounded-md border border-border px-3 py-2.5">
                        <Badge variant={job.status === "complete" ? "default" : job.status === "failed" || job.status === "cancelled" ? "destructive" : "outline"}>
                          {statusLabel(job.status)}
                        </Badge>
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm font-medium">
                            {shortId(job.id)} · {job.kind}
                          </div>
                          {(job.status === "running" || job.status === "queued") && (
                            <Progress value={job.progress.percent} className="mt-1.5 h-1.5" />
                          )}
                        </div>
                        <span className="text-xs text-muted-foreground">{timeLabel(job.updated_at || job.created_at)}</span>
                        {job.status === "running" || job.status === "queued" ? (
                          <Button variant="ghost" size="icon" className="size-7" onClick={() => void (async () => { try { await cancelJob(job.id); void refreshJobs(); } catch { /* already finished */ } })()} title="Cancel job">
                            <CircleStop size={15} />
                          </Button>
                        ) : (
                          <Button variant="ghost" size="icon" className="size-7" onClick={() => void openJob(job)} title="Open result">
                            <Download size={15} />
                          </Button>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}

        {view === "library" && (
          <div className="p-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">Color library</CardTitle>
                <CardDescription>Everything you have colorized on this engine.</CardDescription>
              </CardHeader>
              <CardContent>
                {jobs.length === 0 ? (
                  <p className="py-8 text-center text-sm text-muted-foreground">Nothing here yet.</p>
                ) : (
                  <div className="space-y-2">
                    {jobs.map((job) => (
                      <button
                        key={job.id}
                        onClick={() => void openJob(job)}
                        className="flex w-full items-center gap-3 rounded-md border border-border px-3 py-2.5 text-left transition-colors hover:bg-accent/50"
                      >
                        <Badge variant={job.status === "complete" ? "default" : job.status === "failed" || job.status === "cancelled" ? "destructive" : "outline"}>
                          {statusLabel(job.status)}
                        </Badge>
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm font-medium">
                            {shortId(job.id)} · {job.kind}
                            {job.result?.pages?.length ? ` · ${job.result.pages.length} pages` : ""}
                          </div>
                          <div className="truncate text-xs text-muted-foreground">
                            {job.result?.report?.director_verdict || job.progress.stage || "—"}
                          </div>
                        </div>
                        <span className="shrink-0 text-xs text-muted-foreground">{timeLabel(job.created_at)}</span>
                        {archivePath(job) && (
                          <Button variant="ghost" size="icon" className="size-7" onClick={(event) => { event.stopPropagation(); void downloadAsset(archivePath(job) || "", `mangahue-${shortId(job.id)}-chapter.cbz`, job.id); }} title="Download CBZ">
                            <FileArchive size={15} />
                          </Button>
                        )}
                      </button>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        )}
      </main>

      {/* Access dialog */}
      <Dialog open={accessOpen} onOpenChange={setAccessOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <KeyRound size={18} /> API access
            </DialogTitle>
            <DialogDescription>
              Paste the token from your engine to connect. The token is stored only in this browser and sent only to your MangaHue engine.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-2">
            <Label htmlFor="access-token">Access token</Label>
            <Input
              id="access-token"
              type="password"
              value={tokenDraft}
              onChange={(event) => setTokenDraft(event.target.value)}
              placeholder="mh_..."
              autoComplete="off"
              spellCheck={false}
            />
            <p className="text-xs text-muted-foreground">
              Generate one on your engine: <code className="rounded bg-accent px-1.5 py-0.5">mangahue production token --user &lt;user_id&gt;</code>
            </p>
          </div>
          <DialogFooter className="gap-2">
            <Button variant="secondary" onClick={clearAccessToken}>
              <Trash2 size={15} /> Clear
            </Button>
            <Button onClick={() => void saveAccessToken()} disabled={savingToken}>
              {savingToken ? <LoaderCircle size={15} className="animate-spin" /> : <Check size={15} />}
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
