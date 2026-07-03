use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Args, Parser, Subcommand};
use mangahue_api::{run_doctor_report, serve, ApiServerConfig, DoctorRequest, DoctorStatus};
use mangahue_config::{load_style_profile, EngineConfig};
use mangahue_core::{BoundingBox, ColorRgb, ImageSize, StyleProfile};
use mangahue_hints::{generate_hints, load_user_hints};
use mangahue_image::{
    collect_image_paths, export_chapter_archive, import_chapter_to_dir, load_image, save_gray,
    save_rgb, write_chapter_manifest, ChapterImportOptions,
};
use mangahue_inference::ModelMetadata;
use mangahue_ingest::{ingest_url, IngestRequest};
use mangahue_memory::{
    add_correction, create_character, create_project_directory, export_correction_hints,
    export_correction_training_manifest, export_corrections_jsonl, NewCorrection, ProjectMemory,
};
use mangahue_orchestrator::{ColorizeRequest, InspectRequest, MangaHueEngine};
use mangahue_preprocess::preprocess_page;
use rayon::prelude::*;
use serde::Serialize;
use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use sysinfo::{get_current_pid, ProcessRefreshKind, System};

#[derive(Debug, Parser)]
#[command(
    name = "mangahue",
    version,
    about = "MangaHue manga colorization engine"
)]
struct Cli {
    #[arg(long, global = true)]
    verbose: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Colorize(ColorizeArgs),
    Batch(BatchArgs),
    Inspect(InspectArgs),
    Preprocess(PreprocessArgs),
    MakeHints(MakeHintsArgs),
    Doctor(DoctorArgs),
    Link {
        #[command(subcommand)]
        command: LinkCommand,
    },
    Chapter {
        #[command(subcommand)]
        command: ChapterCommand,
    },
    Project {
        #[command(subcommand)]
        command: ProjectCommand,
    },
    Styles {
        #[command(subcommand)]
        command: StylesCommand,
    },
    Character {
        #[command(subcommand)]
        command: CharacterCommand,
    },
    Correction {
        #[command(subcommand)]
        command: CorrectionCommand,
    },
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Benchmark(BatchArgs),
    Serve(ServeArgs),
    Debug(ColorizeArgs),
}

#[derive(Debug, Args)]
struct ColorizeArgs {
    input: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    preset: Option<PathBuf>,
    #[arg(long)]
    style: Option<PathBuf>,
    #[arg(long)]
    hints: Option<PathBuf>,
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long = "character")]
    characters: Vec<String>,
    #[arg(long)]
    debug: Option<PathBuf>,
    #[arg(long, help = "Use deterministic preview backend for tests and demos")]
    mock: bool,
}

#[derive(Debug, Args)]
struct BatchArgs {
    input_dir: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    preset: Option<PathBuf>,
    #[arg(long)]
    style: Option<PathBuf>,
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long)]
    mock: bool,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(
        long = "jobs",
        short = 'j',
        help = "Parallel worker count (0 = rayon's default, 1 = serial)"
    )]
    jobs: Option<usize>,
}

#[derive(Debug, Args)]
struct InspectArgs {
    input: PathBuf,
    #[arg(long, short)]
    output: Option<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    preset: Option<PathBuf>,
    #[arg(long)]
    debug: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct PreprocessArgs {
    input: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long)]
    config: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct MakeHintsArgs {
    input: PathBuf,
    #[arg(long)]
    hints: Option<PathBuf>,
    #[arg(long)]
    style: Option<PathBuf>,
    #[arg(long = "output-map")]
    output_map: PathBuf,
    #[arg(long = "output-mask")]
    output_mask: PathBuf,
}

#[derive(Debug, Args)]
struct DoctorArgs {
    #[arg(long, default_value = "configs/engine.default.yaml")]
    config: PathBuf,
    #[arg(long, default_value = "configs/styles")]
    styles_dir: PathBuf,
    #[arg(long, default_value = "models/model.mangahue.json")]
    metadata: PathBuf,
    #[arg(long, default_value = ".mangahue/jobs")]
    jobs_dir: PathBuf,
    #[arg(long, default_value = "web/studio")]
    studio_dir: PathBuf,
    #[arg(long, default_value = "examples/input/smoke_page.png")]
    sample_input: PathBuf,
    #[arg(long)]
    python: Option<String>,
    #[arg(long)]
    json: bool,
}

#[derive(Debug, Subcommand)]
enum LinkCommand {
    Fetch(LinkFetchArgs),
    Colorize(Box<LinkColorizeArgs>),
}

#[derive(Debug, Args)]
struct LinkFetchArgs {
    url: String,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long, default_value_t = 32)]
    max_pages: usize,
    #[arg(long, default_value_t = 33_554_432)]
    max_bytes_per_page: u64,
    #[arg(long)]
    user_agent: Option<String>,
}

#[derive(Debug, Args)]
struct LinkColorizeArgs {
    url: String,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long, default_value = ".mangahue/link_downloads")]
    downloads_dir: PathBuf,
    #[arg(long, default_value_t = 32)]
    max_pages: usize,
    #[arg(long, default_value_t = 33_554_432)]
    max_bytes_per_page: u64,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    preset: Option<PathBuf>,
    #[arg(long)]
    style: Option<PathBuf>,
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long)]
    hints: Option<PathBuf>,
    #[arg(long)]
    debug: Option<PathBuf>,
    #[arg(long)]
    mock: bool,
    #[arg(long)]
    user_agent: Option<String>,
    #[arg(long)]
    report: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
enum ChapterCommand {
    Import(ChapterImportArgs),
    Export(ChapterExportArgs),
    Colorize(Box<ChapterColorizeArgs>),
}

#[derive(Debug, Args)]
struct ChapterImportArgs {
    input: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long, default_value_t = 500)]
    max_pages: usize,
    #[arg(long, default_value_t = 52_428_800)]
    max_bytes_per_page: u64,
    #[arg(long)]
    no_overwrite: bool,
}

#[derive(Debug, Args)]
struct ChapterExportArgs {
    input_dir: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long)]
    manifest: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ChapterColorizeArgs {
    input: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long, default_value = ".mangahue/chapters")]
    work_dir: PathBuf,
    #[arg(long)]
    export: Option<PathBuf>,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    preset: Option<PathBuf>,
    #[arg(long)]
    style: Option<PathBuf>,
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long)]
    mock: bool,
    #[arg(long)]
    report: Option<PathBuf>,
    #[arg(long, short = 'j')]
    jobs: Option<usize>,
    #[arg(long, default_value_t = 500)]
    max_pages: usize,
    #[arg(long, default_value_t = 52_428_800)]
    max_bytes_per_page: u64,
}

#[derive(Debug, Subcommand)]
enum ProjectCommand {
    Create {
        name: String,
        #[arg(long, short)]
        output: PathBuf,
    },
    Status {
        #[arg(long)]
        project: PathBuf,
    },
    ExportCorrections {
        #[arg(long)]
        project: PathBuf,
        #[arg(long, short)]
        output: PathBuf,
    },
    ExportHints {
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        page: String,
        #[arg(long, short)]
        output: PathBuf,
    },
    ExportTrainingManifest {
        #[arg(long)]
        project: PathBuf,
        #[arg(long = "lineart-dir")]
        lineart_dir: PathBuf,
        #[arg(long = "color-dir")]
        color_dir: PathBuf,
        #[arg(long = "hints-dir")]
        hints_dir: PathBuf,
        #[arg(long, short)]
        output: PathBuf,
    },
}

#[derive(Debug, Subcommand)]
enum StylesCommand {
    List {
        #[arg(long, default_value = "configs/styles")]
        dir: PathBuf,
        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Subcommand)]
enum CharacterCommand {
    Create {
        #[arg(long)]
        project: PathBuf,
        #[arg(long)]
        name: String,
        #[arg(long)]
        hair: Option<String>,
        #[arg(long)]
        eyes: Option<String>,
        #[arg(long)]
        skin: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum CorrectionCommand {
    Add(CorrectionAddArgs),
}

#[derive(Debug, Args)]
struct CorrectionAddArgs {
    #[arg(long)]
    project: PathBuf,
    #[arg(long)]
    page: String,
    #[arg(long)]
    x: u32,
    #[arg(long)]
    y: u32,
    #[arg(long)]
    width: u32,
    #[arg(long)]
    height: u32,
    #[arg(long)]
    label: Option<String>,
    #[arg(long)]
    old: String,
    #[arg(long)]
    new: String,
    #[arg(long)]
    character: Option<String>,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    Validate {
        #[arg(long)]
        metadata: PathBuf,
    },
    ValidateOnnx {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        metadata: PathBuf,
        #[arg(long)]
        python: Option<String>,
    },
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, default_value = "127.0.0.1:7878")]
    addr: SocketAddr,
    #[arg(long, default_value = ".mangahue/jobs")]
    jobs_dir: PathBuf,
    #[arg(long, default_value = "configs/styles")]
    styles_dir: PathBuf,
    #[arg(long, default_value = "web/studio")]
    studio_dir: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    init_tracing(cli.verbose);

    match cli.command {
        Commands::Colorize(args) => run_colorize(args, false),
        Commands::Debug(args) => run_colorize(args, true),
        Commands::Batch(args) => run_batch(args, false),
        Commands::Benchmark(args) => run_batch(args, true),
        Commands::Inspect(args) => run_inspect(args),
        Commands::Preprocess(args) => run_preprocess(args),
        Commands::MakeHints(args) => run_make_hints(args),
        Commands::Doctor(args) => run_doctor(args),
        Commands::Link { command } => run_link(command),
        Commands::Chapter { command } => run_chapter(command),
        Commands::Project { command } => run_project(command),
        Commands::Styles { command } => run_styles(command),
        Commands::Character { command } => run_character(command),
        Commands::Correction { command } => run_correction(command),
        Commands::Model { command } => run_model(command),
        Commands::Serve(args) => run_serve(args),
    }
}

#[derive(Debug, Serialize)]
struct CliStyleSummary {
    id: String,
    name: String,
    path: PathBuf,
    notes: Option<String>,
}

fn run_inspect(args: InspectArgs) -> Result<()> {
    let inspection = MangaHueEngine::new().inspect_page(InspectRequest {
        input_path: args.input,
        config_path: args.config,
        preset_path: args.preset,
        debug_dir: args.debug,
    })?;
    if let Some(output) = args.output {
        if let Some(parent) = output.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&output, serde_json::to_string_pretty(&inspection)?)
            .with_context(|| format!("failed to write {}", output.display()))?;
        println!("saved {}", output.display());
    }
    println!(
        "classification: {} | size: {}x{} | panels: {} | tiles: {}",
        inspection.page_classification,
        inspection.size.width,
        inspection.size.height,
        inspection.panels.len(),
        inspection.tiles.len()
    );
    println!(
        "ink density: {:.4} | tone score: {:.4} | bubbles: {} | protected: {:.4}",
        inspection.preprocess.line_density,
        inspection.preprocess.screen_tone_score,
        inspection.preprocess.bubble_regions,
        inspection.preprocess.protected_coverage
    );
    Ok(())
}

fn run_colorize(args: ColorizeArgs, force_debug: bool) -> Result<()> {
    let debug_dir = if force_debug && args.debug.is_none() {
        Some(args.output.with_extension("debug"))
    } else {
        args.debug
    };
    let result = MangaHueEngine::new().colorize(ColorizeRequest {
        input_path: args.input,
        output_path: args.output,
        model_path: args.model,
        config_path: args.config,
        preset_path: args.preset,
        style_path: args.style,
        project_path: args.project,
        character_ids: args.characters,
        user_hints_path: args.hints,
        debug_dir,
        use_mock_backend: args.mock,
    })?;

    println!("saved {}", result.output_path.display());
    println!(
        "tiles: {}, total: {}ms, inference: {}ms",
        result.report.tiles_count, result.report.total_ms, result.report.inference_ms
    );
    Ok(())
}

#[derive(Debug, Serialize)]
struct BatchSummary {
    input_dir: PathBuf,
    output_dir: PathBuf,
    processed: usize,
    failed: usize,
    total_ms: u128,
    total_pages: usize,
    pages: Vec<BatchPageSummary>,
    metrics: BatchMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<BatchSystemSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    config: Option<EngineConfig>,
    started_at: String,
    finished_at: String,
}

#[derive(Debug, Default, Serialize)]
struct BatchPageSummary {
    input: PathBuf,
    output: PathBuf,
    total_ms: u128,
    preprocessing_ms: u128,
    layout_ms: u128,
    hint_ms: u128,
    inference_ms: u128,
    postprocess_ms: u128,
    tiles: usize,
    pixels: u64,
    backend: String,
    verdict: Option<String>,
    error: Option<String>,
}

#[derive(Debug, Default, Serialize)]
struct BatchMetrics {
    mean_page_ms: f64,
    median_page_ms: f64,
    min_page_ms: u128,
    max_page_ms: u128,
    stddev_page_ms: f64,
    throughput_pages_per_min: f64,
    throughput_megapixels_per_sec: f64,
    backend_counts: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize)]
struct BatchSystemSnapshot {
    peak_memory_bytes: u64,
    final_memory_bytes: u64,
    total_memory_bytes: u64,
    cpu_count: usize,
}

#[derive(Debug, Serialize)]
struct LinkColorizeSummary {
    source_url: String,
    downloads_dir: PathBuf,
    output_dir: PathBuf,
    processed: usize,
    failed: usize,
    total_ms: u128,
    pages: Vec<LinkColorizePageSummary>,
}

#[derive(Debug, Serialize)]
struct LinkColorizePageSummary {
    source_url: String,
    input: PathBuf,
    output: PathBuf,
    total_ms: u128,
    tiles: usize,
    verdict: Option<String>,
    error: Option<String>,
}

fn run_batch(args: BatchArgs, benchmark: bool) -> Result<()> {
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("failed to create {}", args.output.display()))?;

    let engine = Arc::new(MangaHueEngine::new());
    let jobs = args.jobs.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .max(1)
    });
    let jobs = jobs.max(1);

    let model_path = Arc::new(args.model.clone());
    let config_path = Arc::new(args.config.clone());
    let preset_path = Arc::new(args.preset.clone());
    let style_path = Arc::new(args.style.clone());
    let project_path = Arc::new(args.project.clone());
    let output_dir = Arc::new(args.output.clone());
    let use_mock = args.mock;

    let started_at = Utc::now().to_rfc3339();
    let batch_start = Instant::now();
    let peak_memory = Arc::new(AtomicU64::new(0));
    let stop_monitor = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let entries = collect_image_paths(&args.input_dir)?;

    let monitor_peak = Arc::clone(&peak_memory);
    let monitor_stop = Arc::clone(&stop_monitor);
    let monitor_handle = Arc::new(std::thread::spawn(move || {
        let mut system = System::new();
        let pid = match get_current_pid() {
            Ok(pid) => pid,
            Err(_) => return,
        };
        while !monitor_stop.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(150));
            system.refresh_processes_specifics(
                sysinfo::ProcessesToUpdate::Some(&[pid]),
                ProcessRefreshKind::everything(),
            );
            if let Some(process) = system.process(pid) {
                let mem = process.memory();
                let current = monitor_peak.load(Ordering::Relaxed);
                if mem > current {
                    monitor_peak.store(mem, Ordering::Relaxed);
                }
            } else {
                break;
            }
        }
    }));

    let pages: Vec<BatchPageSummary> = if entries.len() < 2 || jobs == 1 {
        let mut serial = Vec::with_capacity(entries.len());
        for input in entries {
            let summary = process_one_page(
                &engine,
                input,
                &output_dir,
                &model_path,
                &config_path,
                &preset_path,
                &style_path,
                &project_path,
                use_mock,
            );
            serial.push(summary);
        }
        serial
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .thread_name(|index| format!("mangahue-batch-{index}"))
            .build()
            .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().expect("rayon pool"));
        pool.install(|| {
            entries
                .par_iter()
                .map(|input| {
                    process_one_page(
                        &engine,
                        input.clone(),
                        &output_dir,
                        &model_path,
                        &config_path,
                        &preset_path,
                        &style_path,
                        &project_path,
                        use_mock,
                    )
                })
                .collect::<Vec<_>>()
        })
    };

    let processed = pages.iter().filter(|p| p.error.is_none()).count();
    let failed = pages.len() - processed;
    let total_ms = batch_start.elapsed().as_millis();

    if !benchmark && failed > 0 {
        if let Some(first_error) = pages.iter().find_map(|p| p.error.clone()) {
            stop_monitor.store(true, Ordering::Relaxed);
            if let Ok(handle) = Arc::try_unwrap(monitor_handle.clone()) {
                let _ = handle.join();
            }
            anyhow::bail!("batch failed: {first_error}");
        }
    }

    if let Some(report) = args.report.as_ref() {
        if let Some(parent) = report.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
    }

    let metrics = compute_batch_metrics(&pages, total_ms);
    let system = finalize_system_snapshot(&peak_memory, monitor_handle, &stop_monitor);
    let finished_at = Utc::now().to_rfc3339();
    let config_snapshot =
        load_engine_config_snapshot(args.config.as_deref(), args.preset.as_deref());

    let summary = BatchSummary {
        input_dir: args.input_dir,
        output_dir: args.output,
        processed,
        failed,
        total_ms,
        total_pages: pages.len(),
        pages,
        metrics,
        system,
        config: config_snapshot,
        started_at,
        finished_at,
    };

    if let Some(report) = args.report.as_ref() {
        std::fs::write(report, serde_json::to_string_pretty(&summary)?)
            .with_context(|| format!("failed to write {}", report.display()))?;
        println!("saved {}", report.display());
    }
    println!(
        "processed {} page(s), failed {}, total {}ms, throughput {:.2} pages/min",
        summary.processed,
        summary.failed,
        summary.total_ms,
        summary.metrics.throughput_pages_per_min
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_one_page(
    engine: &Arc<MangaHueEngine>,
    input: PathBuf,
    output_dir: &Arc<PathBuf>,
    model_path: &Arc<Option<PathBuf>>,
    config_path: &Arc<Option<PathBuf>>,
    preset_path: &Arc<Option<PathBuf>>,
    style_path: &Arc<Option<PathBuf>>,
    project_path: &Arc<Option<PathBuf>>,
    use_mock: bool,
) -> BatchPageSummary {
    let file_name = input.file_name().unwrap_or_default();
    let output = output_dir.join(file_name).with_extension("png");
    let page_start = Instant::now();
    let pixels = std::fs::metadata(&input)
        .ok()
        .and_then(|_| image_dimensions(&input))
        .unwrap_or((0, 0));
    let total_pixels = (pixels.0 as u64).saturating_mul(pixels.1 as u64);

    let result = engine.colorize(ColorizeRequest {
        input_path: input.clone(),
        output_path: output.clone(),
        model_path: model_path.as_ref().clone(),
        config_path: config_path.as_ref().clone(),
        preset_path: preset_path.as_ref().clone(),
        style_path: style_path.as_ref().clone(),
        project_path: project_path.as_ref().clone(),
        use_mock_backend: use_mock,
        ..ColorizeRequest::default()
    });
    match result {
        Ok(result) => BatchPageSummary {
            input,
            output,
            total_ms: page_start.elapsed().as_millis(),
            preprocessing_ms: result.report.preprocessing_ms,
            layout_ms: result.report.layout_ms,
            hint_ms: result.report.hint_ms,
            inference_ms: result.report.inference_ms,
            postprocess_ms: result.report.postprocess_ms,
            tiles: result.report.tiles_count,
            pixels: total_pixels,
            backend: result
                .report
                .stage_reports
                .iter()
                .find(|s| s.name == "inference")
                .and_then(|s| {
                    let prefix = "backend=";
                    s.notes
                        .iter()
                        .find_map(|note| note.strip_prefix(prefix).map(|s| s.to_string()))
                })
                .unwrap_or_else(|| "unknown".to_string()),
            verdict: result.report.director_verdict,
            error: None,
        },
        Err(error) => BatchPageSummary {
            input,
            output,
            total_ms: page_start.elapsed().as_millis(),
            preprocessing_ms: 0,
            layout_ms: 0,
            hint_ms: 0,
            inference_ms: 0,
            postprocess_ms: 0,
            tiles: 0,
            pixels: total_pixels,
            backend: "unknown".to_string(),
            verdict: None,
            error: Some(error.to_string()),
        },
    }
}

fn image_dimensions(path: &Path) -> Option<(u32, u32)> {
    image::image_dimensions(path).ok()
}

fn compute_batch_metrics(pages: &[BatchPageSummary], total_ms: u128) -> BatchMetrics {
    let successful: Vec<u128> = pages
        .iter()
        .filter(|p| p.error.is_none())
        .map(|p| p.total_ms)
        .collect();
    let mut backend_counts: HashMap<String, usize> = HashMap::new();
    for page in pages {
        *backend_counts.entry(page.backend.clone()).or_insert(0) += 1;
    }
    if successful.is_empty() {
        return BatchMetrics {
            mean_page_ms: 0.0,
            median_page_ms: 0.0,
            min_page_ms: 0,
            max_page_ms: 0,
            stddev_page_ms: 0.0,
            throughput_pages_per_min: 0.0,
            throughput_megapixels_per_sec: 0.0,
            backend_counts,
        };
    }
    let mut sorted = successful.clone();
    sorted.sort_unstable();
    let min_page_ms = *sorted.first().unwrap_or(&0);
    let max_page_ms = *sorted.last().unwrap_or(&0);
    let mean = sorted.iter().map(|v| *v as f64).sum::<f64>() / sorted.len() as f64;
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2] as f64
    } else {
        let a = sorted[sorted.len() / 2 - 1] as f64;
        let b = sorted[sorted.len() / 2] as f64;
        (a + b) / 2.0
    };
    let variance = sorted
        .iter()
        .map(|v| {
            let diff = *v as f64 - mean;
            diff * diff
        })
        .sum::<f64>()
        / sorted.len() as f64;
    let stddev = variance.sqrt();
    let total_seconds = (total_ms as f64) / 1000.0;
    let throughput_pages_per_min = if total_seconds > 0.0 {
        (sorted.len() as f64) * 60.0 / total_seconds
    } else {
        0.0
    };
    let total_megapixels: f64 = pages
        .iter()
        .filter(|p| p.error.is_none())
        .map(|p| p.pixels as f64)
        .sum::<f64>()
        / 1_000_000.0;
    let throughput_megapixels_per_sec = if total_seconds > 0.0 {
        total_megapixels / total_seconds
    } else {
        0.0
    };

    BatchMetrics {
        mean_page_ms: mean,
        median_page_ms: median,
        min_page_ms,
        max_page_ms,
        stddev_page_ms: stddev,
        throughput_pages_per_min,
        throughput_megapixels_per_sec,
        backend_counts,
    }
}

fn finalize_system_snapshot(
    peak_memory: &Arc<AtomicU64>,
    monitor_handle: Arc<std::thread::JoinHandle<()>>,
    stop_monitor: &Arc<std::sync::atomic::AtomicBool>,
) -> Option<BatchSystemSnapshot> {
    stop_monitor.store(true, Ordering::Relaxed);
    let _ = Arc::try_unwrap(monitor_handle).map(|handle| handle.join());

    let mut system = System::new();
    system.refresh_memory();
    if let Ok(pid) = get_current_pid() {
        system.refresh_processes_specifics(
            sysinfo::ProcessesToUpdate::Some(&[pid]),
            ProcessRefreshKind::everything(),
        );
    }
    let final_memory = system
        .process(sysinfo::get_current_pid().ok()?)
        .map(|process| process.memory())
        .unwrap_or(0);
    let peak_memory = peak_memory.load(Ordering::Relaxed);
    Some(BatchSystemSnapshot {
        peak_memory_bytes: peak_memory.max(final_memory),
        final_memory_bytes: final_memory,
        total_memory_bytes: system.total_memory(),
        cpu_count: system.cpus().len(),
    })
}

fn load_engine_config_snapshot(
    config_path: Option<&Path>,
    preset_path: Option<&Path>,
) -> Option<EngineConfig> {
    let paths: Vec<&Path> = [config_path, preset_path].into_iter().flatten().collect();
    if paths.is_empty() {
        None
    } else {
        EngineConfig::from_paths(paths).ok()
    }
}

fn run_preprocess(args: PreprocessArgs) -> Result<()> {
    let config = match args.config {
        Some(path) => EngineConfig::from_path(path)?,
        None => EngineConfig::default(),
    };
    let input = load_image(&args.input)?;
    let result = preprocess_page(&input.image, &config.preprocess)?;
    save_gray(&args.output, &result.clean_lineart)?;
    println!("saved {}", args.output.display());
    Ok(())
}

fn run_make_hints(args: MakeHintsArgs) -> Result<()> {
    let input = load_image(&args.input)?;
    let user_hints = match args.hints {
        Some(path) => Some(load_user_hints(path)?),
        None => None,
    };
    let style = match args.style {
        Some(path) => load_style_profile(path)?,
        None => StyleProfile::default(),
    };
    let config = EngineConfig::default();
    let size = ImageSize::new(input.info.size.width, input.info.size.height);
    let hints = generate_hints(size, user_hints.as_ref(), &style, &[], &config.hints);
    save_rgb(&args.output_map, &hints.hint_map)?;
    save_gray(&args.output_mask, &hints.hint_mask)?;
    println!(
        "saved {} and {}",
        args.output_map.display(),
        args.output_mask.display()
    );
    Ok(())
}

fn run_doctor(args: DoctorArgs) -> Result<()> {
    let report = run_doctor_report(DoctorRequest {
        config_path: args.config,
        styles_dir: args.styles_dir,
        metadata_path: args.metadata,
        jobs_dir: args.jobs_dir,
        studio_dir: args.studio_dir,
        sample_input: args.sample_input,
        python: args
            .python
            .or_else(|| std::env::var("MANGAHUE_PYTHON").ok())
            .unwrap_or_else(|| "python".to_string()),
    });
    if args.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!(
            "doctor: {} | {}",
            doctor_status_label(report.status),
            report.summary
        );
        for check in &report.checks {
            println!(
                "[{}] {} - {}",
                doctor_status_label(check.status),
                check.name,
                check.message
            );
        }
    }
    if report.status == DoctorStatus::Fail {
        anyhow::bail!("doctor found failing checks");
    }
    Ok(())
}

fn run_link(command: LinkCommand) -> Result<()> {
    match command {
        LinkCommand::Fetch(args) => {
            let report = ingest_url(IngestRequest {
                source_url: args.url,
                output_dir: args.output,
                max_pages: args.max_pages,
                max_bytes_per_page: args.max_bytes_per_page,
                user_agent: args.user_agent,
            })?;
            println!(
                "downloaded {} page image(s) to {}",
                report.pages.len(),
                report.output_dir.display()
            );
            println!("manifest: {}", report.manifest_path.display());
            if !report.skipped.is_empty() {
                println!("skipped {} asset(s)", report.skipped.len());
            }
        }
        LinkCommand::Colorize(args) => run_link_colorize(*args)?,
    }
    Ok(())
}

fn run_link_colorize(args: LinkColorizeArgs) -> Result<()> {
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("failed to create {}", args.output.display()))?;
    let work_dir =
        args.downloads_dir
            .join(format!("{}_{}", safe_url_slug(&args.url), unix_millis()));
    let ingest = ingest_url(IngestRequest {
        source_url: args.url.clone(),
        output_dir: work_dir.clone(),
        max_pages: args.max_pages,
        max_bytes_per_page: args.max_bytes_per_page,
        user_agent: args.user_agent,
    })?;
    let engine = MangaHueEngine::new();
    let mut pages = Vec::new();
    let mut processed = 0usize;
    let mut failed = 0usize;
    let total_start = Instant::now();

    for page in &ingest.pages {
        let page_start = Instant::now();
        let stem = page
            .local_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("page");
        let output = args.output.join(format!("{stem}.png"));
        let debug_dir = args.debug.as_ref().map(|debug| debug.join(stem));
        let result = engine.colorize(ColorizeRequest {
            input_path: page.local_path.clone(),
            output_path: output.clone(),
            model_path: args.model.clone(),
            config_path: args.config.clone(),
            preset_path: args.preset.clone(),
            style_path: args.style.clone(),
            project_path: args.project.clone(),
            user_hints_path: args.hints.clone(),
            debug_dir,
            use_mock_backend: args.mock,
            ..ColorizeRequest::default()
        });
        match result {
            Ok(result) => {
                processed += 1;
                pages.push(LinkColorizePageSummary {
                    source_url: page.source_url.clone(),
                    input: page.local_path.clone(),
                    output,
                    total_ms: page_start.elapsed().as_millis(),
                    tiles: result.report.tiles_count,
                    verdict: result.report.director_verdict,
                    error: None,
                });
            }
            Err(error) => {
                failed += 1;
                pages.push(LinkColorizePageSummary {
                    source_url: page.source_url.clone(),
                    input: page.local_path.clone(),
                    output,
                    total_ms: page_start.elapsed().as_millis(),
                    tiles: 0,
                    verdict: None,
                    error: Some(error.to_string()),
                });
            }
        }
    }

    let summary = LinkColorizeSummary {
        source_url: args.url,
        downloads_dir: work_dir,
        output_dir: args.output.clone(),
        processed,
        failed,
        total_ms: total_start.elapsed().as_millis(),
        pages,
    };
    let report_path = args
        .report
        .unwrap_or_else(|| args.output.join("mangahue_link_colorize_report.json"));
    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&report_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", report_path.display()))?;
    println!(
        "colorized {} page(s), failed {}, report {}",
        summary.processed,
        summary.failed,
        report_path.display()
    );
    Ok(())
}

fn run_chapter(command: ChapterCommand) -> Result<()> {
    match command {
        ChapterCommand::Import(args) => {
            let manifest = import_chapter_to_dir(
                &args.input,
                &args.output,
                ChapterImportOptions {
                    max_pages: args.max_pages,
                    max_bytes_per_page: args.max_bytes_per_page,
                    overwrite: !args.no_overwrite,
                },
            )?;
            let manifest_path = args
                .manifest
                .unwrap_or_else(|| args.output.join("mangahue_chapter_manifest.json"));
            write_chapter_manifest(&manifest, &manifest_path)?;
            println!(
                "imported {} page(s) to {}; manifest {}",
                manifest.pages.len(),
                manifest.root.display(),
                manifest_path.display()
            );
            if !manifest.warnings.is_empty() {
                println!("warnings: {}", manifest.warnings.len());
            }
        }
        ChapterCommand::Export(args) => {
            let manifest = export_chapter_archive(&args.input_dir, &args.output)?;
            if let Some(manifest_path) = args.manifest {
                write_chapter_manifest(&manifest, &manifest_path)?;
                println!("manifest {}", manifest_path.display());
            }
            println!(
                "exported {} page(s) from {} to {}",
                manifest.pages.len(),
                args.input_dir.display(),
                args.output.display()
            );
        }
        ChapterCommand::Colorize(args) => run_chapter_colorize(*args)?,
    }
    Ok(())
}

fn run_chapter_colorize(args: ChapterColorizeArgs) -> Result<()> {
    let chapter_id = format!(
        "{}_{}",
        safe_url_slug(&args.input.to_string_lossy()),
        unix_millis()
    );
    let import_dir = args.work_dir.join(chapter_id).join("imported");
    let manifest = import_chapter_to_dir(
        &args.input,
        &import_dir,
        ChapterImportOptions {
            max_pages: args.max_pages,
            max_bytes_per_page: args.max_bytes_per_page,
            overwrite: true,
        },
    )?;
    std::fs::create_dir_all(&args.output)
        .with_context(|| format!("failed to create {}", args.output.display()))?;
    let manifest_path = args.output.join("mangahue_import_manifest.json");
    write_chapter_manifest(&manifest, &manifest_path)?;
    run_batch(
        BatchArgs {
            input_dir: import_dir,
            output: args.output.clone(),
            model: args.model,
            config: args.config,
            preset: args.preset,
            style: args.style,
            project: args.project,
            mock: args.mock,
            report: args
                .report
                .or_else(|| Some(args.output.join("mangahue_chapter_colorize_report.json"))),
            jobs: args.jobs,
        },
        true,
    )?;
    if let Some(export_path) = args.export {
        let exported = export_chapter_archive(&args.output, &export_path)?;
        println!(
            "chapter export {} page(s) -> {}",
            exported.pages.len(),
            export_path.display()
        );
    }
    Ok(())
}

fn run_project(command: ProjectCommand) -> Result<()> {
    match command {
        ProjectCommand::Create { name, output } => {
            let path = create_project_directory(output, &name)?;
            println!("created {}", path.display());
        }
        ProjectCommand::Status { project } => {
            let project_memory = ProjectMemory::load(&project)?;
            println!("project: {}", project_memory.name);
            println!("id: {}", project_memory.id);
            println!("characters: {}", project_memory.characters.len());
            println!("processed pages: {}", project_memory.processed_pages.len());
            println!("corrections: {}", project_memory.corrections.len());
        }
        ProjectCommand::ExportCorrections { project, output } => {
            let count = export_corrections_jsonl(project, &output)?;
            println!("exported {count} correction(s) to {}", output.display());
        }
        ProjectCommand::ExportHints {
            project,
            page,
            output,
        } => {
            let count = export_correction_hints(project, &page, &output)?;
            println!(
                "exported {count} correction hint(s) for {page} to {}",
                output.display()
            );
        }
        ProjectCommand::ExportTrainingManifest {
            project,
            lineart_dir,
            color_dir,
            hints_dir,
            output,
        } => {
            let summary = export_correction_training_manifest(
                project,
                lineart_dir,
                color_dir,
                hints_dir,
                &output,
            )?;
            println!(
                "saved {} trainable row(s) to {} from {} corrected page(s)",
                summary.rows,
                output.display(),
                summary.pages_with_corrections
            );
            if !summary.missing_lineart.is_empty() {
                println!("missing lineart: {}", summary.missing_lineart.join(", "));
            }
            if !summary.missing_color.is_empty() {
                println!(
                    "missing color targets: {}",
                    summary.missing_color.join(", ")
                );
            }
        }
    }
    Ok(())
}

fn run_styles(command: StylesCommand) -> Result<()> {
    match command {
        StylesCommand::List { dir, json } => {
            let mut styles = Vec::new();
            if dir.exists() {
                for entry in std::fs::read_dir(&dir)
                    .with_context(|| format!("failed to read {}", dir.display()))?
                {
                    let entry = entry?;
                    let path = entry.path();
                    if path.extension().and_then(|value| value.to_str()) != Some("yaml") {
                        continue;
                    }
                    let style = load_style_profile(&path)?;
                    styles.push(CliStyleSummary {
                        id: style.id,
                        name: style.name,
                        path,
                        notes: style.notes,
                    });
                }
            }
            styles.sort_by(|a, b| a.id.cmp(&b.id));
            if json {
                println!("{}", serde_json::to_string_pretty(&styles)?);
            } else {
                for style in styles {
                    println!("{} | {} | {}", style.id, style.name, style.path.display());
                }
            }
        }
    }
    Ok(())
}

fn run_character(command: CharacterCommand) -> Result<()> {
    match command {
        CharacterCommand::Create {
            project,
            name,
            hair,
            eyes,
            skin,
        } => {
            let path = create_character(
                project,
                &name,
                parse_color_opt(hair)?,
                parse_color_opt(eyes)?,
                parse_color_opt(skin)?,
            )?;
            println!("created {}", path.display());
        }
    }
    Ok(())
}

fn run_correction(command: CorrectionCommand) -> Result<()> {
    match command {
        CorrectionCommand::Add(args) => {
            add_correction(
                &args.project,
                NewCorrection {
                    page: args.page,
                    region: BoundingBox::new(args.x, args.y, args.width, args.height),
                    label: args.label,
                    old_color: ColorRgb::from_str(&args.old)?,
                    new_color: ColorRgb::from_str(&args.new)?,
                    character_id: args.character,
                },
            )?;
            println!("saved correction to {}", args.project.display());
        }
    }
    Ok(())
}

fn run_model(command: ModelCommand) -> Result<()> {
    match command {
        ModelCommand::Validate { metadata } => {
            let metadata = ModelMetadata::load(metadata)?;
            println!(
                "model metadata ok: {} v{} ({}, {} input channels)",
                metadata.model_name, metadata.version, metadata.model_type, metadata.input_channels
            );
        }
        ModelCommand::ValidateOnnx {
            model,
            metadata,
            python,
        } => {
            let python = python
                .or_else(|| std::env::var("MANGAHUE_PYTHON").ok())
                .unwrap_or_else(|| "python".to_string());
            let output = Command::new(&python)
                .arg("-m")
                .arg("mangahue_train.validate_onnx")
                .arg("--model")
                .arg(&model)
                .arg("--metadata")
                .arg(&metadata)
                .output()
                .with_context(|| {
                    format!(
                        "failed to start Python ONNX validator with '{}'; install ./python package or set MANGAHUE_PYTHON",
                        python
                    )
                })?;
            if !output.status.success() {
                anyhow::bail!(
                    "ONNX validation failed:\n{}\n{}",
                    String::from_utf8_lossy(&output.stderr),
                    String::from_utf8_lossy(&output.stdout)
                );
            }
            print!("{}", String::from_utf8_lossy(&output.stdout));
        }
    }
    Ok(())
}

fn run_serve(args: ServeArgs) -> Result<()> {
    tokio::runtime::Runtime::new()?.block_on(serve(ApiServerConfig {
        addr: args.addr,
        jobs_dir: args.jobs_dir,
        styles_dir: args.styles_dir,
        studio_dir: args.studio_dir,
    }))
}

fn parse_color_opt(value: Option<String>) -> Result<Option<ColorRgb>> {
    value
        .as_deref()
        .map(ColorRgb::from_str)
        .transpose()
        .map_err(anyhow::Error::from)
}

fn doctor_status_label(status: DoctorStatus) -> &'static str {
    match status {
        DoctorStatus::Pass => "pass",
        DoctorStatus::Warn => "warn",
        DoctorStatus::Fail => "fail",
    }
}

fn safe_url_slug(value: &str) -> String {
    let slug = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect::<String>();
    let trimmed = slug.trim_matches('_');
    if trimmed.is_empty() {
        "link".to_string()
    } else {
        trimmed.chars().take(64).collect()
    }
}

fn unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn init_tracing(verbose: bool) {
    let level = if verbose { "debug" } else { "warn" };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(format!("mangahue={level}"))
        .try_init();
}
