use anyhow::{Context, Result};
use chrono::Utc;
use mangahue_brain::{
    analyze_chapter_brain, analyze_safety_gate, build_palette_memory, write_chapter_brain_report,
    write_locked_style_profile, write_palette_memory, write_palette_memory_style,
    write_safety_gate_report, BrainPageInput, ChapterBrainReport, ChapterBrainRequest,
    ChapterSafetyReport, PaletteMemoryRequest, SafetyGatePageInput, SafetyGateRequest,
    SafetyGateThresholds,
};
use mangahue_config::load_style_profile;
#[cfg(test)]
use mangahue_core::ProcessingReport;
use mangahue_core::StyleProfile;
use mangahue_image::{
    export_chapter_archive, import_chapter_to_dir, write_chapter_manifest, ChapterImportOptions,
    ChapterManifest,
};
use mangahue_orchestrator::{delivery_quality_approved, ColorizeRequest, MangaHueEngine};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterJobRequest {
    pub input_path: PathBuf,
    pub work_dir: PathBuf,
    pub output_dir: PathBuf,
    pub export_path: Option<PathBuf>,
    pub gallery_path: Option<PathBuf>,
    pub manifest_path: Option<PathBuf>,
    pub brain_report_path: Option<PathBuf>,
    pub locked_style_path: Option<PathBuf>,
    pub palette_memory_path: Option<PathBuf>,
    pub palette_memory_style_path: Option<PathBuf>,
    pub safety_report_path: Option<PathBuf>,
    pub colorize_template: ColorizeRequest,
    pub import_options: ChapterImportOptions,
    #[serde(default)]
    pub safety_gate: ChapterSafetyGateConfig,
    #[serde(default)]
    pub consistency_retry: ChapterConsistencyRetry,
    #[serde(default)]
    pub resume: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterConsistencyRetry {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_retry_drift_threshold")]
    pub palette_drift_threshold: f32,
    #[serde(default = "default_retry_tone_threshold")]
    pub tone_drift_threshold: f32,
    #[serde(default = "default_retry_max_pages")]
    pub max_pages: usize,
    /// How many retries a single page may receive across runs. Without a
    /// cap, resume runs re-retried the same drifting pages forever.
    #[serde(default = "default_retry_max_rounds")]
    pub max_rounds: usize,
}

pub fn default_retry_max_rounds() -> usize {
    2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterSafetyGateConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub enforce: bool,
    #[serde(default)]
    pub thresholds: SafetyGateThresholds,
}

impl Default for ChapterSafetyGateConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enforce: false,
            thresholds: SafetyGateThresholds::default(),
        }
    }
}

impl Default for ChapterConsistencyRetry {
    fn default() -> Self {
        Self {
            enabled: false,
            palette_drift_threshold: default_retry_drift_threshold(),
            tone_drift_threshold: default_retry_tone_threshold(),
            max_pages: default_retry_max_pages(),
            max_rounds: default_retry_max_rounds(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterJobManifest {
    pub id: String,
    pub status: ChapterJobStatus,
    pub input_path: PathBuf,
    pub import_manifest_path: PathBuf,
    pub import_dir: PathBuf,
    pub output_dir: PathBuf,
    pub export_path: Option<PathBuf>,
    pub gallery_path: Option<PathBuf>,
    #[serde(default)]
    pub brain_report_path: Option<PathBuf>,
    #[serde(default)]
    pub locked_style_path: Option<PathBuf>,
    #[serde(default)]
    pub palette_memory_path: Option<PathBuf>,
    #[serde(default)]
    pub palette_memory_style_path: Option<PathBuf>,
    #[serde(default)]
    pub safety_report_path: Option<PathBuf>,
    pub processed: usize,
    pub failed: usize,
    pub skipped: usize,
    #[serde(default)]
    pub consistency_retried: usize,
    #[serde(default)]
    pub brain_consistency_score: Option<f32>,
    #[serde(default)]
    pub brain_verdict: Option<String>,
    #[serde(default)]
    pub safety_gate_pass: Option<bool>,
    #[serde(default)]
    pub safety_verdict: Option<String>,
    pub total_pages: usize,
    pub created_at: String,
    pub updated_at: String,
    pub pages: Vec<ChapterJobPage>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChapterJobStatus {
    Pending,
    Running,
    Complete,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChapterPageStatus {
    Pending,
    Running,
    Complete,
    Failed,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterJobPage {
    pub index: usize,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub status: ChapterPageStatus,
    pub total_ms: u128,
    pub tiles: usize,
    pub verdict: Option<String>,
    pub report_path: Option<PathBuf>,
    pub quality_report_path: Option<PathBuf>,
    #[serde(default)]
    pub consistency_retry_count: usize,
    #[serde(default)]
    pub palette_drift_score: Option<f32>,
    #[serde(default)]
    pub tone_drift_score: Option<f32>,
    #[serde(default)]
    pub brain_consistency_score: Option<f32>,
    #[serde(default)]
    pub safety_gate_pass: Option<bool>,
    #[serde(default)]
    pub safety_severity: Option<f32>,
    #[serde(default)]
    pub safety_violations: Vec<String>,
    pub error: Option<String>,
}

pub fn run_chapter_job(request: ChapterJobRequest) -> Result<ChapterJobManifest> {
    fs::create_dir_all(&request.work_dir)
        .with_context(|| format!("failed to create {}", request.work_dir.display()))?;
    fs::create_dir_all(&request.output_dir)
        .with_context(|| format!("failed to create {}", request.output_dir.display()))?;

    let manifest_path = request
        .manifest_path
        .clone()
        .unwrap_or_else(|| request.output_dir.join("mangahue_chapter_job.json"));

    if request.resume && manifest_path.exists() {
        resume_chapter_job(request, &manifest_path)
    } else {
        start_chapter_job(request, &manifest_path)
    }
}

pub fn write_gallery(manifest: &ChapterJobManifest, output_path: impl AsRef<Path>) -> Result<()> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut html = String::new();
    html.push_str("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">");
    html.push_str("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">");
    html.push_str("<title>MangaHue Chapter Gallery</title>");
    html.push_str("<style>");
    html.push_str(
        "body{font-family:Inter,system-ui,sans-serif;margin:0;background:#f7f7f4;color:#171717}",
    );
    html.push_str("header{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:16px 22px;z-index:2}");
    html.push_str("main{padding:18px;display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(280px,1fr))}");
    html.push_str(".page{border:1px solid #ddd;background:#fff;border-radius:8px;padding:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}");
    html.push_str("img{width:100%;background:#eef0f3;border:1px solid #ddd}.meta{font-size:12px;color:#667085;line-height:1.45;margin-top:8px}");
    html.push_str(".status{font-weight:700}.complete{color:#25756b}.failed{color:#b44939}.skipped{color:#8a6a12}");
    html.push_str("</style></head><body>");
    html.push_str(&format!(
        "<header><h1>MangaHue Chapter Gallery</h1><p>{} pages | {} complete | {} failed | {}</p></header><main>",
        manifest.total_pages, manifest.processed, manifest.failed, escape_html(&manifest.id)
    ));
    let base_dir = output_path.parent().unwrap_or_else(|| Path::new("."));
    for page in &manifest.pages {
        let status_class = match page.status {
            ChapterPageStatus::Complete => "complete",
            ChapterPageStatus::Failed => "failed",
            ChapterPageStatus::Skipped => "skipped",
            ChapterPageStatus::Pending | ChapterPageStatus::Running => "status",
        };
        html.push_str("<section class=\"page\">");
        html.push_str(&format!("<h2>Page {:03}</h2>", page.index));
        html.push_str("<div class=\"pair\">");
        html.push_str(&image_tag(base_dir, &page.input_path, "Input"));
        html.push_str(&image_tag(base_dir, &page.output_path, "Output"));
        html.push_str("</div>");
        html.push_str(&format!(
            "<div class=\"meta\"><span class=\"status {status_class}\">{:?}</span><br>time: {}ms<br>tiles: {}<br>verdict: {}</div>",
            page.status,
            page.total_ms,
            page.tiles,
            escape_html(page.verdict.as_deref().unwrap_or("none"))
        ));
        if let Some(error) = &page.error {
            html.push_str(&format!("<pre>{}</pre>", escape_html(error)));
        }
        html.push_str("</section>");
    }
    html.push_str("</main></body></html>");
    fs::write(output_path, html)
        .with_context(|| format!("failed to write {}", output_path.display()))
}

fn start_chapter_job(
    request: ChapterJobRequest,
    manifest_path: &Path,
) -> Result<ChapterJobManifest> {
    let job_id = Uuid::new_v4().to_string();
    let import_dir = request.work_dir.join(&job_id).join("imported");
    let import_manifest = import_chapter_to_dir(
        &request.input_path,
        &import_dir,
        request.import_options.clone(),
    )?;
    let import_manifest_path = request
        .output_dir
        .join(format!("mangahue_import_manifest_{job_id}.json"));
    write_chapter_manifest(&import_manifest, &import_manifest_path)?;
    let now = Utc::now().to_rfc3339();
    let pages = make_pages(&import_manifest, &request.output_dir);
    let mut manifest = ChapterJobManifest {
        id: job_id,
        status: ChapterJobStatus::Pending,
        input_path: request.input_path.clone(),
        import_manifest_path,
        import_dir,
        output_dir: request.output_dir.clone(),
        export_path: request.export_path.clone(),
        gallery_path: request.gallery_path.clone(),
        brain_report_path: request
            .brain_report_path
            .clone()
            .or_else(|| Some(request.output_dir.join("mangahue_chapter_brain.json"))),
        locked_style_path: request
            .locked_style_path
            .clone()
            .or_else(|| Some(request.output_dir.join("mangahue_locked_style.json"))),
        palette_memory_path: request
            .palette_memory_path
            .clone()
            .or_else(|| Some(request.output_dir.join("mangahue_palette_memory.json"))),
        palette_memory_style_path: request.palette_memory_style_path.clone().or_else(|| {
            Some(
                request
                    .output_dir
                    .join("mangahue_palette_memory_style.json"),
            )
        }),
        safety_report_path: request
            .safety_report_path
            .clone()
            .or_else(|| Some(request.output_dir.join("mangahue_safety_gate.json"))),
        processed: 0,
        failed: 0,
        skipped: 0,
        consistency_retried: 0,
        brain_consistency_score: None,
        brain_verdict: None,
        safety_gate_pass: None,
        safety_verdict: None,
        total_pages: pages.len(),
        created_at: now.clone(),
        updated_at: now,
        pages,
        warnings: import_manifest.warnings,
    };
    save_job_manifest(&mut manifest, manifest_path)?;
    process_manifest_pages(&request, manifest, manifest_path)
}

fn resume_chapter_job(
    request: ChapterJobRequest,
    manifest_path: &Path,
) -> Result<ChapterJobManifest> {
    let text = fs::read_to_string(manifest_path)
        .with_context(|| format!("failed to read {}", manifest_path.display()))?;
    let mut manifest: ChapterJobManifest = serde_json::from_str(&text)
        .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
    manifest.export_path = request.export_path.clone().or(manifest.export_path);
    manifest.gallery_path = request.gallery_path.clone().or(manifest.gallery_path);
    manifest.brain_report_path = request
        .brain_report_path
        .clone()
        .or(manifest.brain_report_path)
        .or_else(|| Some(manifest.output_dir.join("mangahue_chapter_brain.json")));
    manifest.locked_style_path = request
        .locked_style_path
        .clone()
        .or(manifest.locked_style_path)
        .or_else(|| Some(manifest.output_dir.join("mangahue_locked_style.json")));
    manifest.palette_memory_path = request
        .palette_memory_path
        .clone()
        .or(manifest.palette_memory_path)
        .or_else(|| Some(manifest.output_dir.join("mangahue_palette_memory.json")));
    manifest.palette_memory_style_path = request
        .palette_memory_style_path
        .clone()
        .or(manifest.palette_memory_style_path)
        .or_else(|| {
            Some(
                manifest
                    .output_dir
                    .join("mangahue_palette_memory_style.json"),
            )
        });
    manifest.safety_report_path = request
        .safety_report_path
        .clone()
        .or(manifest.safety_report_path)
        .or_else(|| Some(manifest.output_dir.join("mangahue_safety_gate.json")));
    process_manifest_pages(&request, manifest, manifest_path)
}

fn process_manifest_pages(
    request: &ChapterJobRequest,
    mut manifest: ChapterJobManifest,
    manifest_path: &Path,
) -> Result<ChapterJobManifest> {
    manifest.status = ChapterJobStatus::Running;
    save_job_manifest(&mut manifest, manifest_path)?;
    let engine = MangaHueEngine::new();

    for index in 0..manifest.pages.len() {
        let can_skip = request.resume
            && manifest.pages[index].status == ChapterPageStatus::Complete
            && manifest.pages[index].output_path.exists();
        if can_skip {
            manifest.pages[index].status = ChapterPageStatus::Skipped;
            manifest.skipped += 1;
            save_job_manifest(&mut manifest, manifest_path)?;
            continue;
        }

        manifest.pages[index].status = ChapterPageStatus::Running;
        manifest.pages[index].error = None;
        save_job_manifest(&mut manifest, manifest_path)?;

        let start = Instant::now();
        let mut colorize = request.colorize_template.clone();
        colorize.input_path = manifest.pages[index].input_path.clone();
        colorize.output_path = manifest.pages[index].output_path.clone();
        colorize.region_locks_path = resolve_region_locks_path(
            request.colorize_template.region_locks_path.as_deref(),
            &manifest.pages[index].input_path,
        );
        configure_color_bible_page(&mut colorize, &manifest, index);
        let result = engine.colorize(colorize);
        match result {
            Ok(result) => {
                let verdict = result.report.director_verdict.clone();
                let approved = delivery_quality_approved(&result.report);
                manifest.pages[index].status = if approved {
                    ChapterPageStatus::Complete
                } else {
                    ChapterPageStatus::Failed
                };
                manifest.pages[index].total_ms = start.elapsed().as_millis();
                manifest.pages[index].tiles = result.report.tiles_count;
                manifest.pages[index].verdict = verdict.clone();
                manifest.pages[index].quality_report_path = result.quality_report_path;
                manifest.pages[index].report_path = Some(mangahue_image::sibling_with_suffix(
                    &manifest.pages[index].output_path,
                    ".report",
                    "json",
                ));
                if !approved {
                    manifest.pages[index].error = Some(format!(
                        "delivery blocked by quality gate: {}",
                        verdict.as_deref().unwrap_or("missing verdict")
                    ));
                    let _ = fs::remove_file(&manifest.pages[index].output_path);
                }
            }
            Err(error) => {
                manifest.pages[index].status = ChapterPageStatus::Failed;
                manifest.pages[index].total_ms = start.elapsed().as_millis();
                manifest.pages[index].error = Some(error.to_string());
            }
        }
        recompute_counts(&mut manifest);
        save_job_manifest(&mut manifest, manifest_path)?;
    }

    recompute_counts(&mut manifest);
    record_partial_delivery_warning(&mut manifest);
    manifest.status = chapter_job_status(&manifest);
    if let Some(report) = write_chapter_brain_outputs(request, &manifest)? {
        apply_brain_report_to_manifest(&mut manifest, &report);
        save_job_manifest(&mut manifest, manifest_path)?;
        if manifest.status == ChapterJobStatus::Complete && request.consistency_retry.enabled {
            let retried =
                run_consistency_retry(request, &mut manifest, &report, manifest_path, &engine)?;
            if retried > 0 {
                if let Some(report) = write_chapter_brain_outputs(request, &manifest)? {
                    apply_brain_report_to_manifest(&mut manifest, &report);
                    save_job_manifest(&mut manifest, manifest_path)?;
                }
            }
        }
    }
    recompute_counts(&mut manifest);
    record_partial_delivery_warning(&mut manifest);
    manifest.status = chapter_job_status(&manifest);
    if let Some(report) = write_chapter_safety_outputs(request, &manifest)? {
        apply_safety_report_to_manifest(&mut manifest, &report);
        if request.safety_gate.enforce && !report.pass {
            manifest.status = ChapterJobStatus::Failed;
        }
        save_job_manifest(&mut manifest, manifest_path)?;
    }
    if manifest.status == ChapterJobStatus::Complete {
        if let Some(export_path) = manifest.export_path.clone() {
            export_chapter_archive(&manifest.output_dir, export_path)?;
        }
    } else if manifest.export_path.is_some() {
        manifest
            .warnings
            .push("archive export skipped because the chapter is incomplete".to_string());
    }
    if let Some(gallery_path) = manifest.gallery_path.clone() {
        write_gallery(&manifest, gallery_path)?;
    }
    save_job_manifest(&mut manifest, manifest_path)?;
    Ok(manifest)
}

fn write_chapter_brain_outputs(
    request: &ChapterJobRequest,
    manifest: &ChapterJobManifest,
) -> Result<Option<ChapterBrainReport>> {
    if manifest.brain_report_path.is_none()
        && manifest.locked_style_path.is_none()
        && manifest.palette_memory_path.is_none()
        && manifest.palette_memory_style_path.is_none()
    {
        return Ok(None);
    }

    let pages = manifest
        .pages
        .iter()
        .filter(|page| {
            matches!(
                page.status,
                ChapterPageStatus::Complete | ChapterPageStatus::Skipped
            ) && page.output_path.exists()
        })
        .map(|page| BrainPageInput {
            index: page.index,
            input_path: page.input_path.clone(),
            output_path: page.output_path.clone(),
        })
        .collect::<Vec<_>>();

    if pages.is_empty() {
        return Ok(None);
    }

    let report = analyze_chapter_brain(ChapterBrainRequest {
        pages,
        ..ChapterBrainRequest::default()
    })?;
    if let Some(report_path) = &manifest.brain_report_path {
        write_chapter_brain_report(&report, report_path)?;
    }
    if let Some(style_path) = &manifest.locked_style_path {
        let base_style = load_base_style(request.colorize_template.style_path.as_deref())?;
        write_locked_style_profile(&report, &base_style, style_path)?;
    }
    if manifest.palette_memory_path.is_some() || manifest.palette_memory_style_path.is_some() {
        let memory = build_palette_memory(
            &report,
            PaletteMemoryRequest {
                id: Some(format!("chapter_{}", manifest.id.replace('-', "_"))),
                name: Some("MangaHue Chapter Palette Memory".to_string()),
                source_report_path: manifest.brain_report_path.clone(),
                ..PaletteMemoryRequest::default()
            },
        );
        if let Some(memory_path) = &manifest.palette_memory_path {
            write_palette_memory(&memory, memory_path)?;
        }
        if let Some(style_path) = &manifest.palette_memory_style_path {
            let base_style = load_base_style(request.colorize_template.style_path.as_deref())?;
            write_palette_memory_style(&memory, &base_style, style_path)?;
        }
    }
    Ok(Some(report))
}

fn run_consistency_retry(
    request: &ChapterJobRequest,
    manifest: &mut ChapterJobManifest,
    report: &ChapterBrainReport,
    manifest_path: &Path,
    engine: &MangaHueEngine,
) -> Result<usize> {
    let Some(locked_style_path) = manifest
        .locked_style_path
        .as_ref()
        .filter(|path| path.exists())
        .cloned()
    else {
        manifest
            .warnings
            .push("consistency retry skipped because locked style was not written".to_string());
        return Ok(0);
    };

    let mut candidates = report
        .pages
        .iter()
        .filter(|page| {
            page.palette_drift_score >= request.consistency_retry.palette_drift_threshold
                || page.tone_drift_score >= request.consistency_retry.tone_drift_threshold
        })
        .map(|page| {
            let severity = page.palette_drift_score * 0.72 + page.tone_drift_score * 0.28;
            (page.index, severity)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
    if request.consistency_retry.max_pages > 0 {
        candidates.truncate(request.consistency_retry.max_pages);
    } else {
        candidates.clear();
    }
    // A page is never retried more than max_rounds times across all runs:
    // a retry that did not fix the drift must not be re-executed forever.
    let max_rounds = request.consistency_retry.max_rounds.max(1);
    candidates.retain(|(page_index, _)| {
        manifest
            .pages
            .iter()
            .find(|page| page.index == *page_index)
            .map(|page| page.consistency_retry_count < max_rounds)
            .unwrap_or(false)
    });

    let mut retried = 0usize;
    for (page_index, _) in candidates {
        let Some(index) = manifest
            .pages
            .iter()
            .position(|page| page.index == page_index && page.output_path.exists())
        else {
            continue;
        };

        manifest.pages[index].status = ChapterPageStatus::Running;
        manifest.pages[index].error = None;
        save_job_manifest(manifest, manifest_path)?;

        let start = Instant::now();
        let mut colorize = request.colorize_template.clone();
        colorize.input_path = manifest.pages[index].input_path.clone();
        colorize.output_path = manifest.pages[index].output_path.clone();
        colorize.style_path = Some(locked_style_path.clone());
        colorize.region_locks_path = resolve_region_locks_path(
            request.colorize_template.region_locks_path.as_deref(),
            &manifest.pages[index].input_path,
        );
        configure_color_bible_page(&mut colorize, manifest, index);

        match engine.colorize(colorize) {
            Ok(result) => {
                let verdict = result.report.director_verdict.clone();
                let approved = delivery_quality_approved(&result.report);
                manifest.pages[index].status = if approved {
                    ChapterPageStatus::Complete
                } else {
                    ChapterPageStatus::Failed
                };
                manifest.pages[index].total_ms = start.elapsed().as_millis();
                manifest.pages[index].tiles = result.report.tiles_count;
                manifest.pages[index].verdict = verdict.clone();
                manifest.pages[index].quality_report_path = result.quality_report_path;
                manifest.pages[index].report_path = Some(mangahue_image::sibling_with_suffix(
                    &manifest.pages[index].output_path,
                    ".report",
                    "json",
                ));
                if approved {
                    manifest.pages[index].consistency_retry_count += 1;
                    manifest.consistency_retried += 1;
                    retried += 1;
                } else {
                    manifest.pages[index].error = Some(format!(
                        "delivery blocked by quality gate: {}",
                        verdict.as_deref().unwrap_or("missing verdict")
                    ));
                    let _ = fs::remove_file(&manifest.pages[index].output_path);
                }
            }
            Err(error) => {
                manifest.pages[index].status = ChapterPageStatus::Failed;
                manifest.pages[index].total_ms = start.elapsed().as_millis();
                manifest.pages[index].error = Some(error.to_string());
            }
        }
        recompute_counts(manifest);
        save_job_manifest(manifest, manifest_path)?;
    }
    Ok(retried)
}

fn configure_color_bible_page(
    colorize: &mut ColorizeRequest,
    manifest: &ChapterJobManifest,
    page_index: usize,
) {
    if colorize.color_bible_path.is_none() && colorize.structure_graph_output_path.is_none() {
        return;
    }
    let page = &manifest.pages[page_index];
    let stem = page
        .input_path
        .file_stem()
        .and_then(|value| value.to_str())
        .map(str::to_string)
        .unwrap_or_else(|| format!("page_{:04}", page.index));
    colorize.page_id = Some(format!("{}:{stem}", manifest.id));
    colorize
        .chapter_id
        .get_or_insert_with(|| manifest.id.clone());
    colorize.structure_graph_output_path = Some(
        manifest
            .output_dir
            .join("structure-graphs")
            .join(format!("{stem}.json")),
    );
}

fn resolve_region_locks_path(root: Option<&Path>, page_path: &Path) -> Option<PathBuf> {
    let root = root?;
    if root.is_file() {
        return Some(root.to_path_buf());
    }
    if !root.is_dir() {
        return Some(root.to_path_buf());
    }
    let stem = page_path.file_stem()?.to_str()?;
    [
        format!("{stem}.region-locks.json"),
        format!("{stem}.regions.json"),
        format!("{stem}.json"),
    ]
    .into_iter()
    .map(|name| root.join(name))
    .find(|candidate| candidate.is_file())
}

fn apply_brain_report_to_manifest(manifest: &mut ChapterJobManifest, report: &ChapterBrainReport) {
    manifest.brain_consistency_score = Some(report.consistency_score);
    manifest.brain_verdict = Some(report.director_verdict.clone());
    for brain_page in &report.pages {
        if let Some(page) = manifest
            .pages
            .iter_mut()
            .find(|page| page.index == brain_page.index)
        {
            page.palette_drift_score = Some(brain_page.palette_drift_score);
            page.tone_drift_score = Some(brain_page.tone_drift_score);
            page.brain_consistency_score = Some(brain_page.consistency_score);
        }
    }
}

fn write_chapter_safety_outputs(
    request: &ChapterJobRequest,
    manifest: &ChapterJobManifest,
) -> Result<Option<ChapterSafetyReport>> {
    if !request.safety_gate.enabled || manifest.safety_report_path.is_none() {
        return Ok(None);
    }

    let pages = manifest
        .pages
        .iter()
        .filter(|page| {
            matches!(
                page.status,
                ChapterPageStatus::Complete | ChapterPageStatus::Skipped
            )
        })
        .map(|page| SafetyGatePageInput {
            index: page.index,
            input_path: page.input_path.clone(),
            output_path: page.output_path.clone(),
            quality_report_path: page.quality_report_path.clone(),
        })
        .collect::<Vec<_>>();

    if pages.is_empty() {
        return Ok(None);
    }

    let report = analyze_safety_gate(SafetyGateRequest {
        pages,
        thresholds: request.safety_gate.thresholds.clone(),
    })?;
    if let Some(path) = &manifest.safety_report_path {
        write_safety_gate_report(&report, path)?;
    }
    Ok(Some(report))
}

fn apply_safety_report_to_manifest(
    manifest: &mut ChapterJobManifest,
    report: &ChapterSafetyReport,
) {
    manifest.safety_gate_pass = Some(report.pass);
    manifest.safety_verdict = Some(report.verdict.clone());
    for safety_page in &report.pages {
        if let Some(page) = manifest
            .pages
            .iter_mut()
            .find(|page| page.index == safety_page.index)
        {
            page.safety_gate_pass = Some(safety_page.pass);
            page.safety_severity = Some(safety_page.severity);
            page.safety_violations = safety_page.violations.clone();
        }
    }
}

fn load_base_style(path: Option<&Path>) -> Result<StyleProfile> {
    match path {
        Some(path) => load_style_profile(path),
        None => Ok(StyleProfile::default()),
    }
}

fn default_true() -> bool {
    true
}

fn default_retry_drift_threshold() -> f32 {
    0.18
}

fn default_retry_tone_threshold() -> f32 {
    0.34
}

fn default_retry_max_pages() -> usize {
    24
}

fn make_pages(import_manifest: &ChapterManifest, output_dir: &Path) -> Vec<ChapterJobPage> {
    import_manifest
        .pages
        .iter()
        .map(|page| ChapterJobPage {
            index: page.index,
            input_path: page.path.clone(),
            output_path: output_dir.join(format!("page_{:03}.png", page.index)),
            status: ChapterPageStatus::Pending,
            total_ms: 0,
            tiles: 0,
            verdict: None,
            report_path: None,
            quality_report_path: None,
            consistency_retry_count: 0,
            palette_drift_score: None,
            tone_drift_score: None,
            brain_consistency_score: None,
            safety_gate_pass: None,
            safety_severity: None,
            safety_violations: Vec::new(),
            error: None,
        })
        .collect()
}

fn save_job_manifest(manifest: &mut ChapterJobManifest, path: &Path) -> Result<()> {
    manifest.updated_at = Utc::now().to_rfc3339();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    // Atomic write: a crash mid-write must never truncate the live manifest
    // (a truncated manifest previously forced the whole chapter to restart).
    let bytes = serde_json::to_vec_pretty(manifest)?;
    let tmp = path.with_extension("json.tmp");
    fs::write(&tmp, bytes).with_context(|| format!("failed to write {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("failed to replace {}", path.display()))
}

fn recompute_counts(manifest: &mut ChapterJobManifest) {
    manifest.processed = manifest
        .pages
        .iter()
        .filter(|page| page.status == ChapterPageStatus::Complete)
        .count();
    manifest.failed = manifest
        .pages
        .iter()
        .filter(|page| page.status == ChapterPageStatus::Failed)
        .count();
    manifest.skipped = manifest
        .pages
        .iter()
        .filter(|page| page.status == ChapterPageStatus::Skipped)
        .count();
}

/// A chapter is complete only when every admitted page has an approved output.
/// Individual successful pages remain available for diagnosis and resume, but
/// a partial set must never be promoted or exported as a completed chapter.
fn chapter_job_status(manifest: &ChapterJobManifest) -> ChapterJobStatus {
    if manifest.failed == 0
        && manifest.processed.saturating_add(manifest.skipped) == manifest.total_pages
    {
        ChapterJobStatus::Complete
    } else {
        ChapterJobStatus::Failed
    }
}

fn record_partial_delivery_warning(manifest: &mut ChapterJobManifest) {
    if manifest.failed == 0 || manifest.processed == 0 {
        return;
    }
    let pages = manifest
        .pages
        .iter()
        .filter(|page| page.status == ChapterPageStatus::Failed)
        .map(|page| page.index)
        .collect::<Vec<_>>();
    let message = format!(
        "{} page(s) withheld ({pages:?}); {}/{} pages delivered",
        manifest.failed, manifest.processed, manifest.total_pages
    );
    if !manifest.warnings.iter().any(|warning| warning == &message) {
        manifest.warnings.push(message);
    }
}

fn image_tag(base_dir: &Path, path: &Path, label: &str) -> String {
    let src = relative_or_original(base_dir, path)
        .to_string_lossy()
        .replace('\\', "/");
    format!(
        "<figure><img src=\"{}\" alt=\"{}\"><figcaption>{}</figcaption></figure>",
        escape_html(&src),
        escape_html(label),
        escape_html(label)
    )
}

fn relative_or_original(base_dir: &Path, path: &Path) -> PathBuf {
    let base = base_dir
        .canonicalize()
        .unwrap_or_else(|_| base_dir.to_path_buf());
    let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    resolved
        .strip_prefix(&base)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| path.to_path_buf())
}

fn escape_html(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn delivery_requires_ready_verdict_and_completed_inference() {
        let mut report = ProcessingReport {
            director_verdict: Some("needs_protected_region_cleanup".to_string()),
            ..ProcessingReport::default()
        };
        assert!(!delivery_quality_approved(&report));
        report.add_stage_report(
            mangahue_core::StageReport::new("inference").metric("tiles_completed", 1.0),
        );
        assert!(!delivery_quality_approved(&report));
        report.add_stage_report(
            mangahue_core::StageReport::new("colorization_coverage").metric("passed", 1.0),
        );
        assert!(!delivery_quality_approved(&report));
        report.add_stage_report(
            mangahue_core::StageReport::new("colorization_transform").metric("passed", 1.0),
        );
        assert!(!delivery_quality_approved(&report));
        report.director_verdict = Some("production_ready_preview".to_string());
        assert!(delivery_quality_approved(&report));

        // A page without real color (coverage failed) must not ship as a
        // grayscale pass-through.
        let mut gray = ProcessingReport {
            director_verdict: Some("needs_insufficient_colorization".to_string()),
            ..ProcessingReport::default()
        };
        gray.add_stage_report(
            mangahue_core::StageReport::new("inference").metric("tiles_completed", 1.0),
        );
        gray.add_stage_report(
            mangahue_core::StageReport::new("colorization_coverage").metric("passed", 0.0),
        );
        gray.add_stage_report(
            mangahue_core::StageReport::new("colorization_transform").metric("passed", 1.0),
        );
        assert!(!delivery_quality_approved(&gray));

        // An identity pass-through (no visible transformation) is not a
        // colored page and stays undeliverable.
        let mut identity = ProcessingReport {
            director_verdict: Some("production_ready_preview".to_string()),
            ..ProcessingReport::default()
        };
        identity.add_stage_report(
            mangahue_core::StageReport::new("inference").metric("tiles_completed", 1.0),
        );
        identity.add_stage_report(
            mangahue_core::StageReport::new("colorization_coverage").metric("passed", 1.0),
        );
        identity.add_stage_report(
            mangahue_core::StageReport::new("colorization_transform").metric("passed", 0.0),
        );
        assert!(!delivery_quality_approved(&identity));
    }

    #[test]
    fn resolves_page_specific_region_locks_from_directory() {
        let dir = tempfile::tempdir().unwrap();
        let lock_path = dir.path().join("page_001.region-locks.json");
        fs::write(&lock_path, "{}").unwrap();
        let resolved =
            resolve_region_locks_path(Some(dir.path()), Path::new("chapter/page_001.png"));
        assert_eq!(resolved.as_deref(), Some(lock_path.as_path()));
        assert!(
            resolve_region_locks_path(Some(dir.path()), Path::new("chapter/page_999.png"))
                .is_none()
        );
    }

    #[test]
    fn chapter_job_runs_and_resumes_completed_pages() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input");
        let output = dir.path().join("output");
        let work = dir.path().join("work");
        let manifest_path = output.join("job.json");
        fs::create_dir_all(&input).unwrap();
        // Midtone pages exercise the full mock colorization path. Uniform
        // near-black fixtures are correctly classified as ink-only and are
        // therefore not valid evidence for a colorized delivery.
        write_test_png(&input.join("page2.png"), [180, 180, 180]);
        write_test_png(&input.join("page1.png"), [160, 160, 160]);

        let request = ChapterJobRequest {
            input_path: input,
            work_dir: work,
            output_dir: output.clone(),
            export_path: Some(output.join("chapter.cbz")),
            gallery_path: Some(output.join("gallery.html")),
            manifest_path: Some(manifest_path.clone()),
            brain_report_path: Some(output.join("brain.json")),
            locked_style_path: Some(output.join("locked_style.json")),
            palette_memory_path: Some(output.join("palette_memory.json")),
            palette_memory_style_path: Some(output.join("palette_memory_style.json")),
            safety_report_path: Some(output.join("safety.json")),
            colorize_template: ColorizeRequest {
                use_mock_backend: true,
                ..ColorizeRequest::default()
            },
            import_options: ChapterImportOptions::default(),
            safety_gate: ChapterSafetyGateConfig::default(),
            consistency_retry: ChapterConsistencyRetry::default(),
            resume: false,
        };
        let first = run_chapter_job(request.clone()).unwrap();
        assert_eq!(first.status, ChapterJobStatus::Complete);
        assert_eq!(first.processed, 2);
        assert!(output.join("chapter.cbz").exists());
        assert!(output.join("gallery.html").exists());
        assert!(output.join("brain.json").exists());
        assert!(output.join("locked_style.json").exists());
        assert!(output.join("palette_memory.json").exists());
        assert!(output.join("palette_memory_style.json").exists());
        assert!(output.join("safety.json").exists());

        let resumed = run_chapter_job(ChapterJobRequest {
            resume: true,
            ..request
        })
        .unwrap();
        assert_eq!(resumed.status, ChapterJobStatus::Complete);
        assert_eq!(resumed.skipped, 2);
        assert_eq!(resumed.pages[0].status, ChapterPageStatus::Skipped);
        assert!(manifest_path.exists());
    }

    #[test]
    fn incomplete_chapter_is_not_promoted_or_exported() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input");
        let output = dir.path().join("output");
        let work = dir.path().join("work");
        let manifest_path = output.join("job.json");
        fs::create_dir_all(&input).unwrap();
        // Two midtone pages deliver; the uniform black fixture is classified
        // as ink-only and is correctly withheld by the delivery gates.
        write_test_png(&input.join("page1.png"), [180, 180, 180]);
        write_test_png(&input.join("page2.png"), [0, 0, 0]);
        write_test_png(&input.join("page3.png"), [160, 160, 160]);

        let request = ChapterJobRequest {
            input_path: input,
            work_dir: work,
            output_dir: output.clone(),
            export_path: Some(output.join("chapter.cbz")),
            gallery_path: Some(output.join("gallery.html")),
            manifest_path: Some(manifest_path.clone()),
            brain_report_path: Some(output.join("brain.json")),
            locked_style_path: Some(output.join("locked_style.json")),
            palette_memory_path: Some(output.join("palette_memory.json")),
            palette_memory_style_path: Some(output.join("palette_memory_style.json")),
            safety_report_path: Some(output.join("safety.json")),
            colorize_template: ColorizeRequest {
                use_mock_backend: true,
                ..ColorizeRequest::default()
            },
            import_options: ChapterImportOptions::default(),
            safety_gate: ChapterSafetyGateConfig::default(),
            consistency_retry: ChapterConsistencyRetry::default(),
            resume: false,
        };
        let manifest = run_chapter_job(request).unwrap();

        // Successful pages remain resumable, but the chapter cannot be marked
        // complete or exported while one admitted page is withheld.
        assert_eq!(manifest.status, ChapterJobStatus::Failed);
        assert_eq!(manifest.processed, 2);
        assert_eq!(manifest.failed, 1);
        assert_eq!(manifest.pages[1].status, ChapterPageStatus::Failed);
        assert!(manifest
            .warnings
            .iter()
            .any(|warning| warning.contains("withheld")));
        assert!(!output.join("chapter.cbz").exists());
        assert!(output.join("gallery.html").exists());
    }

    fn write_test_png(path: &Path, color: [u8; 3]) {
        let image = RgbImage::from_pixel(16, 16, Rgb(color));
        image.save(path).unwrap();
    }
}
