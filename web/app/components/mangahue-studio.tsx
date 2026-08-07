"use client";

import {
  Archive,
  ArrowRight,
  Check,
  ChevronDown,
  CircleStop,
  Clock3,
  CloudOff,
  Download,
  FileArchive,
  FileImage,
  FolderOpen,
  Gauge,
  History,
  Image as ImageIcon,
  KeyRound,
  Layers3,
  Link2,
  LoaderCircle,
  Menu,
  PanelLeftClose,
  Play,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Save,
  Trash2,
  Upload,
  WandSparkles,
  X,
} from "lucide-react";
import Image from "next/image";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  streamJob,
  setAccessToken,
  uploadSource,
} from "@/lib/api";
import type { JobEvent, JobPage, MangaHueJob, StyleSummary } from "@/lib/types";

type View = "create" | "queue" | "library";
type SourceMode = "link" | "upload";
type Quality = "fast" | "studio" | "masterpiece";

const terminalStatuses = new Set(["complete", "failed", "cancelled"]);

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

function StatusMark({ status }: { status: MangaHueJob["status"] }) {
  if (status === "running") return <LoaderCircle size={14} className="spin" aria-hidden="true" />;
  if (status === "complete") return <Check size={14} aria-hidden="true" />;
  if (status === "failed" || status === "cancelled") return <X size={14} aria-hidden="true" />;
  return <Clock3 size={14} aria-hidden="true" />;
}

export function MangaHueStudio() {
  const [view, setView] = useState<View>("create");
  const [sourceMode, setSourceMode] = useState<SourceMode>("link");
  const [sourceUrl, setSourceUrl] = useState("");
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [styles, setStyles] = useState<StyleSummary[]>([]);
  const [stylePath, setStylePath] = useState("");
  const [quality, setQuality] = useState<Quality>("masterpiece");
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

  useEffect(() => {
    let mounted = true;
    void (async () => {
      const health = await healthcheck();
      try {
        const [nextStyles, nextJobs, quota] = await Promise.all([
          getStyles(),
          getJobs(),
          getAccessToken() ? getQuota().catch(() => null) : Promise.resolve(null),
        ]);
        if (!mounted) return;
        setConnected(health);
        setStyles(nextStyles);
        setJobs(nextJobs);
        if (quota) {
          const ceiling = Math.max(1, Math.min(500, Number(quota.policy.max_pages_per_job)));
          setPageCeiling(ceiling);
          setMaxPages((current) => Math.min(current, ceiling));
        }
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
    const poller = window.setInterval(refreshJobs, 5000);
    return () => {
      mounted = false;
      window.clearInterval(poller);
    };
  }, [refreshJobs]);

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
                  completed: event.completed ?? job.progress.completed,
                  total: event.total ?? job.progress.total,
                  percent: event.progress ?? event.percent ?? job.progress.percent,
                },
                error: event.error || job.error,
              }
            : job,
        ),
      );
      if (terminalStatuses.has(event.status)) {
        if (event.status === "failed") setError(event.error || "Colorization failed");
        void getJob(activeId).then((job) => {
          setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
          setSelectedJob(job);
          setSelectedPage(0);
          setView("create");
        });
      }
    };

    void streamJob(activeId, updateFromEvent, controller.signal).catch((streamError: unknown) => {
      if (!controller.signal.aborted) setError(streamError instanceof Error ? streamError.message : "Progress stream closed");
    });
    return () => controller.abort();
  }, [activeId]);

  useEffect(() => {
    let live = true;
    let beforeObject: string | null = null;
    let afterObject: string | null = null;
    const page = pages[selectedPage];
    queueMicrotask(() => {
      if (!live) return;
      setBeforeImage(sourceFile && selectedJob?.kind === "page" ? sourcePreview : null);
      setAfterImage(null);
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
      if (live) setError(assetError instanceof Error ? assetError.message : "Could not load result preview");
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
        config_path: "configs/engine.studio.yaml",
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
    } catch (startError: unknown) {
      setError(startError instanceof Error ? startError.message : "Could not start colorization");
    } finally {
      setStarting(false);
    }
  }

  async function stopActiveJob() {
    if (!activeId) return;
    try {
      await cancelJob(activeId);
      await refreshJobs();
    } catch (cancelError: unknown) {
      setError(cancelError instanceof Error ? cancelError.message : "Could not cancel job");
    }
  }

  async function openJob(job: MangaHueJob) {
    const fresh = await getJob(job.id).catch(() => job);
    setSelectedJob(fresh);
    setSelectedPage(0);
    setView("create");
    if (!terminalStatuses.has(fresh.status)) setActiveId(fresh.id);
  }

  function saveAccessToken() {
    setAccessToken(tokenDraft);
    setAccessOpen(false);
    window.location.reload();
  }

  function clearAccessToken() {
    setAccessToken("");
    setTokenDraft("");
    setAccessOpen(false);
    window.location.reload();
  }

  const progress = activeJob?.progress.percent || (activeJob?.status === "complete" ? 100 : 0);
  const canStart = sourceMode === "link" ? Boolean(sourceUrl.trim()) : Boolean(sourceFile);

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "sidebar-open" : ""}`}>
        <div className="brand" aria-label="MangaHue">
          <div className="brand-mark"><span>M</span><span>H</span></div>
          <div className="brand-name">MangaHue</div>
        </div>
        <nav className="primary-nav" aria-label="Workspace">
          <button className={view === "create" ? "active" : ""} onClick={() => setView("create")} title="Create">
            <WandSparkles size={20} /><span>Create</span>
          </button>
          <button className={view === "queue" ? "active" : ""} onClick={() => setView("queue")} title="Queue">
            <History size={20} /><span>Queue</span>
            {jobs.some((job) => !terminalStatuses.has(job.status)) && <i />}
          </button>
          <button className={view === "library" ? "active" : ""} onClick={() => setView("library")} title="Library">
            <FolderOpen size={20} /><span>Library</span>
          </button>
          <button onClick={() => { setTokenDraft(getAccessToken()); setAccessOpen(true); setSidebarOpen(false); }} title="API access">
            <KeyRound size={20} /><span>Access</span>
          </button>
        </nav>
        <div className="sidebar-foot">
          <span className={`connection-dot ${connected ? "online" : ""}`} />
          <span>{connected ? "Engine ready" : "Engine offline"}</span>
        </div>
      </aside>

      <main className="main-shell">
        <header className="topbar">
          <button className="icon-button mobile-menu" onClick={() => setSidebarOpen((value) => !value)} title="Open navigation">
            <Menu size={20} />
          </button>
          <div>
            <span className="eyebrow">MANGAHUE STUDIO</span>
            <h1>{view === "create" ? "Color workspace" : view === "queue" ? "Processing queue" : "Color library"}</h1>
          </div>
          <div className="topbar-actions">
            <span className={`engine-state ${connected ? "connected" : ""}`}>
              {connected ? <Gauge size={15} /> : <CloudOff size={15} />}
              {connected ? "Engine ready" : "Disconnected"}
            </span>
            <button className="icon-button" onClick={() => { setTokenDraft(getAccessToken()); setAccessOpen(true); }} title="API access"><KeyRound size={18} /></button>
            <button className="icon-button" onClick={() => void refreshJobs()} title="Refresh jobs"><RefreshCw size={18} /></button>
          </div>
        </header>

        {view === "create" && (
          <div className="create-layout">
            <section className="control-panel" aria-label="Colorization controls">
              <div className="panel-heading">
                <div><span className="step-number">01</span><h2>Source</h2></div>
                <PanelLeftClose size={18} aria-hidden="true" />
              </div>

              <div className="segmented" role="tablist" aria-label="Source type">
                <button className={sourceMode === "link" ? "selected" : ""} onClick={() => setSourceMode("link")}>
                  <Link2 size={16} /> Link
                </button>
                <button className={sourceMode === "upload" ? "selected" : ""} onClick={() => setSourceMode("upload")}>
                  <Upload size={16} /> Upload
                </button>
              </div>

              {sourceMode === "link" ? (
                <label className="field-block">
                  <span>Manga or chapter URL</span>
                  <div className="url-field"><Link2 size={17} /><input value={sourceUrl} onChange={(event) => setSourceUrl(event.target.value)} placeholder="https://..." inputMode="url" /></div>
                </label>
              ) : (
                <div
                  className={`drop-zone ${dragging ? "dragging" : ""} ${sourceFile ? "has-file" : ""}`}
                  onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
                  onDragLeave={() => setDragging(false)}
                  onDrop={(event) => { event.preventDefault(); setDragging(false); chooseFile(event.dataTransfer.files[0] || null); }}
                >
                  <input ref={fileInputRef} type="file" accept="image/png,image/jpeg,image/webp,.zip,.cbz" onChange={(event) => chooseFile(event.target.files?.[0] || null)} />
                  {sourceFile ? (
                    <><FileArchive size={24} /><strong>{sourceFile.name}</strong><span>{(sourceFile.size / 1024 / 1024).toFixed(1)} MB</span></>
                  ) : (
                    <><FileImage size={24} /><strong>Drop page or chapter</strong><span>PNG, JPEG, WEBP, ZIP, CBZ</span></>
                  )}
                  <button onClick={() => fileInputRef.current?.click()}><FolderOpen size={15} /> Browse</button>
                </div>
              )}

              <div className="section-rule" />
              <div className="panel-heading compact"><div><span className="step-number">02</span><h2>Direction</h2></div><SlidersHorizontal size={18} /></div>

              <label className="field-block">
                <span>Color direction</span>
                <div className="select-wrap">
                  <select value={stylePath} onChange={(event) => setStylePath(event.target.value)}>
                    {styles.map((style) => <option key={style.id} value={style.path}>{style.name}</option>)}
                  </select>
                  <ChevronDown size={16} />
                </div>
              </label>

              <div className="field-block">
                <span>Render quality</span>
                <div className="quality-grid">
                  {(["fast", "studio", "masterpiece"] as Quality[]).map((item) => (
                    <button key={item} className={quality === item ? "selected" : ""} onClick={() => setQuality(item)}>
                      {item === "fast" ? <Gauge size={17} /> : item === "studio" ? <Layers3 size={17} /> : <Sparkles size={17} />}
                      <span>{statusLabel(item)}</span>
                    </button>
                  ))}
                </div>
              </div>

              <div className="setting-row">
                <div><strong>Palette intelligence</strong><span>Automatic scene palette</span></div>
                <button className={`switch ${autoPalette ? "on" : ""}`} onClick={() => setAutoPalette((value) => !value)} role="switch" aria-checked={autoPalette}><i /></button>
              </div>
              <div className="setting-row">
                <div><strong>Chapter consistency</strong><span>Carry identity and color memory</span></div>
                <button className={`switch ${consistency ? "on" : ""}`} onClick={() => setConsistency((value) => !value)} role="switch" aria-checked={consistency}><i /></button>
              </div>

              <label className="page-limit">
                <span>Maximum pages</span>
                <div><button onClick={() => setMaxPages((value) => Math.max(1, value - 10))}>-</button><input type="number" min="1" max={pageCeiling} value={maxPages} onChange={(event) => setMaxPages(Math.min(pageCeiling, Math.max(1, Number(event.target.value))))} /><button onClick={() => setMaxPages((value) => Math.min(pageCeiling, value + 10))}>+</button></div>
              </label>

              {error && <div className="error-banner" role="alert"><X size={16} /><span>{error}</span><button onClick={() => setError(null)} title="Dismiss"><X size={14} /></button></div>}

              <button className="start-button" disabled={!canStart || starting || !connected} onClick={() => void startJob()}>
                {starting ? <LoaderCircle size={18} className="spin" /> : <Play size={18} fill="currentColor" />}
                {starting ? "Preparing" : "Start colorization"}
                <ArrowRight size={18} />
              </button>
            </section>

            <section className="viewer-panel" aria-label="Result preview">
              <div className="viewer-toolbar">
                <div className="viewer-title"><ImageIcon size={18} /><span>{selectedJob ? `${statusLabel(selectedJob.kind)} ${shortId(selectedJob.id)}` : "Preview"}</span></div>
                <div className="viewer-actions">
                  {afterImage && <span className="comparison-label">BEFORE <b>{compare}</b> AFTER</span>}
                  {archivePath(selectedJob) && <button className="icon-button light" onClick={() => void downloadAsset(archivePath(selectedJob) as string, "mangahue-colored.cbz", selectedJob?.id)} title="Download chapter"><Download size={18} /></button>}
                  {pages[selectedPage]?.output_path && <button className="icon-button light" onClick={() => void downloadAsset(pages[selectedPage].output_path, `mangahue-page-${selectedPage + 1}.png`, selectedJob?.id)} title="Download page"><Download size={18} /></button>}
                </div>
              </div>

              <div className="canvas-stage">
                {afterImage ? (
                  <div className="comparison-canvas">
                    {beforeImage ? <Image src={beforeImage} alt="Original manga page" className="before-layer" fill sizes="(max-width: 780px) 100vw, 70vw" unoptimized /> : <div className="color-backdrop" />}
                    <div className="after-clip" style={{ clipPath: `inset(0 0 0 ${compare}%)` }}><Image src={afterImage} alt="Colorized manga page" fill sizes="(max-width: 780px) 100vw, 70vw" unoptimized /></div>
                    {beforeImage && <><div className="compare-line" style={{ left: `${compare}%` }}><i /></div><input aria-label="Before and after comparison" type="range" min="0" max="100" value={compare} onChange={(event) => setCompare(Number(event.target.value))} /></>}
                  </div>
                ) : sourcePreview ? (
                  <div className="source-canvas"><Image src={sourcePreview} alt="Selected manga page" fill sizes="(max-width: 780px) 100vw, 70vw" unoptimized /></div>
                ) : (
                  <div className="empty-canvas"><div className="frame-corners"><ImageIcon size={34} /></div><strong>Color preview</strong><span>Choose a source to begin</span></div>
                )}
              </div>

              {pages.length > 1 && (
                <div className="page-strip" aria-label="Result pages">
                  <span>{pages.length} PAGES</span>
                  <div>{pages.map((page, index) => <button key={`${page.output_path}-${index}`} className={selectedPage === index ? "selected" : ""} onClick={() => setSelectedPage(index)}>{String(index + 1).padStart(2, "0")}</button>)}</div>
                </div>
              )}

              {activeJob && !terminalStatuses.has(activeJob.status) && (
                <div className="active-job">
                  <div className="active-job-head"><div><LoaderCircle size={17} className="spin" /><strong>{statusLabel(activeJob.progress.stage || activeJob.status)}</strong></div><span>{Math.round(progress)}%</span></div>
                  <div className="progress-track"><i style={{ width: `${Math.max(1, progress)}%` }} /></div>
                  <div className="active-job-foot"><span>{activeJob.progress.total ? `${activeJob.progress.completed} of ${activeJob.progress.total}` : `JOB ${shortId(activeJob.id)}`}</span><button onClick={() => void stopActiveJob()}><CircleStop size={16} /> Cancel</button></div>
                </div>
              )}
            </section>
          </div>
        )}

        {view === "queue" && <JobTable jobs={jobs.filter((job) => job.status !== "complete")} empty="No jobs in the queue" onOpen={openJob} onRefresh={refreshJobs} />}
        {view === "library" && <JobTable jobs={jobs.filter((job) => job.status === "complete")} empty="No finished colorizations yet" onOpen={openJob} onRefresh={refreshJobs} />}
      </main>
      {sidebarOpen && <button className="sidebar-scrim" onClick={() => setSidebarOpen(false)} aria-label="Close navigation" />}
      {accessOpen && (
        <div className="dialog-layer">
          <button className="dialog-scrim" onClick={() => setAccessOpen(false)} aria-label="Close API access" />
          <section className="access-dialog" role="dialog" aria-modal="true" aria-labelledby="access-title">
            <div className="dialog-head"><div><KeyRound size={18} /><h2 id="access-title">API access</h2></div><button className="icon-button" onClick={() => setAccessOpen(false)} title="Close"><X size={17} /></button></div>
            <label className="field-block"><span>Access token</span><input className="token-input" type="password" value={tokenDraft} onChange={(event) => setTokenDraft(event.target.value)} autoComplete="off" spellCheck={false} /></label>
            <div className="dialog-actions"><button className="secondary-command" onClick={clearAccessToken}><Trash2 size={16} /> Clear</button><button className="primary-command" onClick={saveAccessToken}><Save size={16} /> Save</button></div>
          </section>
        </div>
      )}
    </div>
  );
}

function JobTable({ jobs, empty, onOpen, onRefresh }: { jobs: MangaHueJob[]; empty: string; onOpen: (job: MangaHueJob) => void; onRefresh: () => Promise<void> }) {
  return (
    <section className="jobs-view">
      <div className="jobs-command"><div><h2>{jobs.length} {jobs.length === 1 ? "job" : "jobs"}</h2><span>Persistent Rust engine queue</span></div><button onClick={() => void onRefresh()}><RefreshCw size={16} /> Refresh</button></div>
      {jobs.length === 0 ? (
        <div className="jobs-empty"><Archive size={30} /><strong>{empty}</strong><span>New work will appear here automatically.</span></div>
      ) : (
        <div className="job-table">
          <div className="job-row job-header"><span>Job</span><span>Type</span><span>Progress</span><span>Started</span><span>Status</span><span /></div>
          {jobs.map((job) => (
            <button className="job-row" key={job.id} onClick={() => void onOpen(job)}>
              <span className="job-identity"><i><ImageIcon size={16} /></i><strong>{shortId(job.id)}</strong></span>
              <span>{statusLabel(job.kind)}</span>
              <span className="table-progress"><i><b style={{ width: `${job.progress.percent || (job.status === "complete" ? 100 : 0)}%` }} /></i><em>{Math.round(job.progress.percent || 0)}%</em></span>
              <span>{timeLabel(job.created_at)}</span>
              <span className={`status-badge ${job.status}`}><StatusMark status={job.status} />{statusLabel(job.status)}</span>
              <span><ArrowRight size={17} /></span>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}
