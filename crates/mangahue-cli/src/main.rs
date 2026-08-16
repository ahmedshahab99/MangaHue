use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Args, Parser, Subcommand};
use mangahue_api::{run_doctor_report, serve, ApiServerConfig, DoctorRequest, DoctorStatus};
use mangahue_brain::{
    build_palette_memory, extract_chapter_palette, write_palette_memory,
    write_palette_memory_style, ChapterBrainReport, PaletteMemoryRequest,
};
use mangahue_chapter::{
    run_chapter_job, ChapterConsistencyRetry, ChapterJobRequest, ChapterSafetyGateConfig,
};
use mangahue_config::{load_style_profile, EngineConfig};
use mangahue_core::{BoundingBox, ColorRgb, ImageSize, StyleProfile};
use mangahue_eval::reference_match_score;
use mangahue_hints::{generate_hints, load_region_locks, load_user_hints, rasterize_region_locks};
use mangahue_image::{
    collect_image_paths, export_chapter_archive, import_chapter_to_dir, is_supported_image_path,
    load_image, save_gray, save_rgb, write_chapter_manifest, ChapterImportOptions,
};
use mangahue_inference::ModelMetadata;
use mangahue_ingest::{ingest_url, IngestRequest};
use mangahue_memory::{
    add_correction, create_character, create_project_directory, export_correction_hints,
    export_correction_training_manifest, export_corrections_jsonl, NewCorrection, ProjectMemory,
};
use mangahue_orchestrator::{ColorizeRequest, InspectRequest, MangaHueEngine};
use mangahue_platform::PlatformProfile;
use mangahue_preprocess::preprocess_page;
use mangahue_production::{sweep_retention, LocalObjectStore, ProductionStore, UserTier};
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
use walkdir::WalkDir;

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
    Brain {
        #[command(subcommand)]
        command: BrainCommand,
    },
    Palette {
        #[command(subcommand)]
        command: PaletteCommand,
    },
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Benchmark(BatchArgs),
    Serve(ServeArgs),
    Debug(ColorizeArgs),
    Platform(PlatformArgs),
    Score(ScoreArgs),
    Production {
        #[command(subcommand)]
        command: ProductionCommand,
    },
}

#[derive(Debug, Subcommand)]
enum ProductionCommand {
    Bootstrap {
        #[arg(long, default_value = ".mangahue/production.db")]
        database: PathBuf,
        #[arg(long)]
        email: String,
        #[arg(long)]
        name: String,
        #[arg(long, default_value = "admin")]
        tier: String,
    },
    Token {
        #[arg(long, default_value = ".mangahue/production.db")]
        database: PathBuf,
        #[arg(long)]
        user: String,
        #[arg(long, default_value = "cli")]
        label: String,
        #[arg(long)]
        expires_days: Option<i64>,
    },
    Quota {
        #[arg(long, default_value = ".mangahue/production.db")]
        database: PathBuf,
        #[arg(long)]
        user: String,
    },
    Backup {
        #[arg(long, default_value = ".mangahue/production.db")]
        database: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Retention {
        #[arg(long, default_value = ".mangahue/production.db")]
        database: PathBuf,
        #[arg(long, default_value = ".mangahue/objects")]
        object_store: PathBuf,
        #[arg(long, default_value_t = 1000)]
        limit: usize,
    },
}

#[derive(Debug, Args)]
struct PlatformArgs {
    #[arg(long, help = "Write platform-tuned config to this YAML path")]
    apply_to: Option<PathBuf>,
    #[arg(long, help = "Base config to load before applying platform profile")]
    config: Option<PathBuf>,
    #[arg(long, help = "Preset overlay applied on top of --config")]
    preset: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ColorizeArgs {
    input: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long, help = "Approved colored style or character reference image")]
    reference_image: Option<PathBuf>,
    #[arg(
        long = "ensemble-model",
        help = "Auxiliary color model; repeat to enable structure-preserving chroma fusion"
    )]
    ensemble_models: Vec<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    preset: Option<PathBuf>,
    #[arg(long)]
    style: Option<PathBuf>,
    #[arg(long)]
    hints: Option<PathBuf>,
    #[arg(long = "region-locks")]
    region_locks: Option<PathBuf>,
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long = "character")]
    characters: Vec<String>,
    #[arg(long)]
    debug: Option<PathBuf>,
    #[arg(long, help = "Use deterministic preview backend for tests and demos")]
    mock: bool,
    #[arg(long, help = "Enable multi-resolution pyramid inference")]
    pyramid: bool,
    #[arg(long, help = "Propagate user hints to similar line-art regions")]
    hint_propagate: bool,
    #[arg(
        long,
        help = "Force the legacy four-pass grading cascade (A/B benchmark)"
    )]
    legacy: bool,
    #[arg(
        long = "palette",
        help = "Artist palette file (configs/palettes/*.yaml) overriding the style palette"
    )]
    palette: Option<PathBuf>,
    #[arg(
        long = "auto-palette",
        help = "Automatically select the artist palette matching the page fingerprint"
    )]
    auto_palette: bool,
    #[arg(
        long = "from-draft",
        help = "Client-side draft: skip model inference and finish this colorized draft with the full postprocess"
    )]
    from_draft: Option<PathBuf>,
}

#[derive(Debug, Args)]
struct ScoreArgs {
    /// Colorized output image to score.
    output: PathBuf,
    /// Reference photo (the target look).
    reference: PathBuf,
    /// Optional line mask image; regions on ink are excluded from tint delta.
    #[arg(long)]
    line_mask: Option<PathBuf>,
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
    #[arg(long)]
    hint_propagate: bool,
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
    #[arg(long, help = "Approved colored style or character reference image")]
    reference_image: Option<PathBuf>,
    #[arg(
        long = "ensemble-model",
        help = "Auxiliary color model; repeat to enable structure-preserving chroma fusion"
    )]
    ensemble_models: Vec<PathBuf>,
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    preset: Option<PathBuf>,
    #[arg(long)]
    style: Option<PathBuf>,
    #[arg(long)]
    project: Option<PathBuf>,
    #[arg(long = "region-locks")]
    region_locks: Option<PathBuf>,
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
    gallery: Option<PathBuf>,
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long = "brain-report")]
    brain_report: Option<PathBuf>,
    #[arg(long = "locked-style")]
    locked_style: Option<PathBuf>,
    #[arg(long = "palette-memory")]
    palette_memory: Option<PathBuf>,
    #[arg(long = "palette-memory-style")]
    palette_memory_style: Option<PathBuf>,
    #[arg(long = "safety-report")]
    safety_report: Option<PathBuf>,
    #[arg(long = "no-safety-gate")]
    no_safety_gate: bool,
    #[arg(long = "enforce-safety-gate")]
    enforce_safety_gate: bool,
    #[arg(long)]
    resume: bool,
    #[arg(long = "consistency-retry")]
    consistency_retry: bool,
    #[arg(long = "retry-drift-threshold", default_value_t = 0.18)]
    retry_drift_threshold: f32,
    #[arg(long = "retry-tone-threshold", default_value_t = 0.34)]
    retry_tone_threshold: f32,
    #[arg(long = "retry-max-pages", default_value_t = 24)]
    retry_max_pages: usize,
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
    #[arg(long = "region-locks")]
    region_locks: Option<PathBuf>,
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

#[derive(Debug, Subcommand)]
enum BrainCommand {
    PaletteMemory(BrainPaletteMemoryArgs),
    RegionLocks(BrainRegionLocksArgs),
}

#[derive(Debug, Subcommand)]
enum PaletteCommand {
    Extract {
        #[arg(long)]
        chapter: PathBuf,
        #[arg(long, short)]
        output: PathBuf,
        #[arg(long, default_value = "8")]
        colors: usize,
    },
}

#[derive(Debug, Args)]
struct BrainPaletteMemoryArgs {
    #[arg(long = "brain-report")]
    brain_report: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
    #[arg(long = "style-output")]
    style_output: Option<PathBuf>,
    #[arg(long = "base-style")]
    base_style: Option<PathBuf>,
    #[arg(long)]
    id: Option<String>,
    #[arg(long)]
    name: Option<String>,
    #[arg(long = "max-entries", default_value_t = 12)]
    max_entries: usize,
    #[arg(long = "min-confidence", default_value_t = 0.18)]
    min_confidence: f32,
}

#[derive(Debug, Args)]
struct BrainRegionLocksArgs {
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    locks: PathBuf,
    #[arg(long, short)]
    output: PathBuf,
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
    #[arg(long, default_value = ".mangahue/production.db")]
    production_db: PathBuf,
    #[arg(
        long,
        help = "Expose only authenticated v1 routes; disable local operator APIs"
    )]
    public_api: bool,
    #[arg(
        long,
        help = "Additional CORS origin allowed to call the API (repeatable; loopback is always allowed)"
    )]
    cors_origin: Vec<String>,
    #[arg(
        long,
        help = "Root directory served by /api/file and job sandboxes (default: current dir, or the jobs dir in public mode)"
    )]
    file_root: Option<PathBuf>,
    #[arg(long, default_value_t = 1, env = "MANGAHUE_MAX_CONCURRENT_JOBS")]
    max_concurrent_jobs: usize,
    #[arg(long, default_value_t = 168, env = "MANGAHUE_JOB_RETENTION_HOURS")]
    job_retention_hours: u64,
    #[arg(
        long,
        default_value = "models/ddcolor-tiny-fp16.onnx",
        env = "MANGAHUE_DEFAULT_MODEL"
    )]
    default_model: PathBuf,
    #[arg(
        long,
        default_value = "configs/engine.default.yaml",
        env = "MANGAHUE_DEFAULT_CONFIG"
    )]
    default_config: PathBuf,
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
        Commands::Brain { command } => run_brain(command),
        Commands::Palette { command } => match command {
            PaletteCommand::Extract {
                chapter,
                output,
                colors,
            } => {
                let paths: Vec<PathBuf> = WalkDir::new(&chapter)
                    .into_iter()
                    .filter_map(|e| e.ok())
                    .filter(|e| is_supported_image_path(e.path()))
                    .map(|e| e.into_path())
                    .collect();
                let palette = extract_chapter_palette(&paths, colors, 500)?;
                std::fs::write(&output, serde_json::to_string_pretty(&palette)?)?;
                println!(
                    "extracted {} colors from {} pages -> {}",
                    palette.dominant_colors.len(),
                    palette.pages_sampled,
                    output.display()
                );
                Ok(())
            }
        },
        Commands::Model { command } => run_model(command),
        Commands::Serve(args) => run_serve(args),
        Commands::Platform(args) => run_platform(args),
        Commands::Score(args) => run_score(args),
        Commands::Production { command } => run_production(command),
    }
}

fn run_platform(args: PlatformArgs) -> Result<()> {
    let profile = PlatformProfile::detect();
    println!("{}", serde_json::to_string_pretty(&profile)?);
    if let Some(config_path) = args.apply_to {
        let mut cfg = match (args.config, args.preset) {
            (Some(c), Some(p)) => {
                let paths: Vec<&Path> = vec![c.as_path(), p.as_path()];
                EngineConfig::from_paths(paths)?
            }
            (Some(c), None) => EngineConfig::from_path(c)?,
            (None, Some(p)) => EngineConfig::from_path(p)?,
            (None, None) => EngineConfig::default(),
        };
        profile.apply_to_config(&mut cfg);
        cfg.validate()?;
        let yaml = serde_yaml::to_string(&cfg)?;
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        std::fs::write(&config_path, yaml)
            .with_context(|| format!("failed to write {}", config_path.display()))?;
        println!("applied platform profile to {}", config_path.display());
    }
    Ok(())
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
        reference_image_path: args.reference_image,
        enable_ensemble: !args.ensemble_models.is_empty(),
        ensemble_model_paths: args.ensemble_models,
        config_path: args.config,
        preset_path: args.preset,
        style_path: args.style,
        project_path: args.project,
        character_ids: args.characters,
        user_hints_path: args.hints,
        region_locks_path: args.region_locks,
        debug_dir,
        use_mock_backend: args.mock,
        pyramid_enabled: args.pyramid,
        hint_propagation_enabled: args.hint_propagate,
        palette_path: args.palette,
        auto_palette: args.auto_palette,
        draft_path: args.from_draft,
        legacy_grade: args.legacy,
        ..ColorizeRequest::default()
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
    let hint_propagate = args.hint_propagate;

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
                hint_propagate,
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
                        hint_propagate,
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
    hint_propagate: bool,
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
        hint_propagation_enabled: hint_propagate,
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
    let result = preprocess_page(&input.image, &config.preprocess, &Default::default())?;
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
                min_request_interval: std::time::Duration::from_millis(500),
                selectors: mangahue_ingest::default_selectors(),
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
        min_request_interval: std::time::Duration::from_millis(500),
        selectors: mangahue_ingest::default_selectors(),
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
            reference_image_path: args.reference_image.clone(),
            enable_ensemble: !args.ensemble_models.is_empty(),
            ensemble_model_paths: args.ensemble_models.clone(),
            config_path: args.config.clone(),
            preset_path: args.preset.clone(),
            style_path: args.style.clone(),
            project_path: args.project.clone(),
            user_hints_path: args.hints.clone(),
            region_locks_path: args.region_locks.clone(),
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
    if let Some(jobs) = args.jobs {
        if jobs > 1 {
            println!("chapter resume mode processes pages serially; use batch for parallel folder-only runs");
        }
    }
    let manifest_path = args
        .manifest
        .clone()
        .or_else(|| args.report.clone())
        .unwrap_or_else(|| args.output.join("mangahue_chapter_job.json"));
    let result = run_chapter_job(ChapterJobRequest {
        input_path: args.input,
        work_dir: args.work_dir,
        output_dir: args.output.clone(),
        export_path: args.export,
        gallery_path: args
            .gallery
            .or_else(|| Some(args.output.join("gallery.html"))),
        manifest_path: Some(manifest_path.clone()),
        brain_report_path: args
            .brain_report
            .or_else(|| Some(args.output.join("mangahue_chapter_brain.json"))),
        locked_style_path: args
            .locked_style
            .or_else(|| Some(args.output.join("mangahue_locked_style.json"))),
        palette_memory_path: args
            .palette_memory
            .or_else(|| Some(args.output.join("mangahue_palette_memory.json"))),
        palette_memory_style_path: args
            .palette_memory_style
            .or_else(|| Some(args.output.join("mangahue_palette_memory_style.json"))),
        safety_report_path: args
            .safety_report
            .or_else(|| Some(args.output.join("mangahue_safety_gate.json"))),
        colorize_template: ColorizeRequest {
            model_path: args.model,
            config_path: args.config,
            preset_path: args.preset,
            style_path: args.style,
            project_path: args.project,
            region_locks_path: args.region_locks,
            use_mock_backend: args.mock,
            ..ColorizeRequest::default()
        },
        import_options: ChapterImportOptions {
            max_pages: args.max_pages,
            max_bytes_per_page: args.max_bytes_per_page,
            overwrite: !args.resume,
        },
        safety_gate: ChapterSafetyGateConfig {
            enabled: !args.no_safety_gate,
            enforce: args.enforce_safety_gate,
            ..ChapterSafetyGateConfig::default()
        },
        consistency_retry: ChapterConsistencyRetry {
            enabled: args.consistency_retry,
            palette_drift_threshold: args.retry_drift_threshold,
            tone_drift_threshold: args.retry_tone_threshold,
            max_pages: args.retry_max_pages,
            max_rounds: mangahue_chapter::default_retry_max_rounds(),
        },
        resume: args.resume,
    })?;
    println!(
        "chapter job {} {:?}: {} complete, {} failed, {} skipped; manifest {}",
        result.id,
        result.status,
        result.processed,
        result.failed,
        result.skipped,
        manifest_path.display()
    );
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

fn run_brain(command: BrainCommand) -> Result<()> {
    match command {
        BrainCommand::PaletteMemory(args) => {
            let report: ChapterBrainReport = serde_json::from_slice(
                &std::fs::read(&args.brain_report)
                    .with_context(|| format!("failed to read {}", args.brain_report.display()))?,
            )
            .with_context(|| format!("failed to parse {}", args.brain_report.display()))?;
            let memory = build_palette_memory(
                &report,
                PaletteMemoryRequest {
                    id: args.id,
                    name: args.name,
                    source_report_path: Some(args.brain_report),
                    max_entries: args.max_entries,
                    min_confidence: args.min_confidence,
                    lock_entries: true,
                },
            );
            write_palette_memory(&memory, &args.output)?;
            if let Some(style_output) = args.style_output {
                let base_style = load_base_style(args.base_style.as_deref())?;
                write_palette_memory_style(&memory, &base_style, &style_output)?;
                println!(
                    "palette memory {} entries -> {}; style {}",
                    memory.entries.len(),
                    args.output.display(),
                    style_output.display()
                );
            } else {
                println!(
                    "palette memory {} entries -> {}",
                    memory.entries.len(),
                    args.output.display()
                );
            }
        }
        BrainCommand::RegionLocks(args) => {
            let page = load_image(&args.input)?;
            let size = ImageSize::new(page.image.width(), page.image.height());
            let locks = load_region_locks(&args.locks)?;
            let raster = rasterize_region_locks(&locks, size)?;
            std::fs::create_dir_all(&args.output)
                .with_context(|| format!("failed to create {}", args.output.display()))?;
            save_rgb(args.output.join("region_color_map.png"), &raster.color_map)?;
            save_gray(
                args.output.join("region_color_mask.png"),
                &raster.color_mask,
            )?;
            save_gray(
                args.output.join("region_preserve_mask.png"),
                &raster.preserve_mask,
            )?;
            save_gray(
                args.output.join("region_exclude_mask.png"),
                &raster.exclude_mask,
            )?;
            println!(
                "validated {} region lock(s), {} active pixel(s) -> {}",
                raster.applied_locks,
                raster.active_pixels,
                args.output.display()
            );
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

fn run_score(args: ScoreArgs) -> Result<()> {
    let output = load_image(&args.output)
        .with_context(|| format!("failed to load output {}", args.output.display()))?
        .image
        .to_rgb8();
    let reference = load_image(&args.reference)
        .with_context(|| format!("failed to load reference {}", args.reference.display()))?
        .image
        .to_rgb8();

    // Resize reference to the output size (or vice versa) for comparable stats.
    let (w, h) = (output.width(), output.height());
    let reference = if reference.dimensions() != (w, h) {
        image::imageops::resize(&reference, w, h, image::imageops::FilterType::Lanczos3)
    } else {
        reference
    };

    let line_mask = match args.line_mask {
        Some(path) => {
            let gray = load_image(&path)
                .with_context(|| format!("failed to load line mask {}", path.display()))?
                .image
                .to_luma8();
            Some(gray)
        }
        None => None,
    };

    let report = match &line_mask {
        Some(mask) => reference_match_score(&output, &reference, mask),
        None => {
            // No mask: synthesize one from output luminance (ink lines are dark).
            let mask = image::GrayImage::from_fn(w, h, |x, y| {
                let p = output.get_pixel(x, y);
                let luma = (p[0] as u32 + p[1] as u32 + p[2] as u32) / 3;
                if luma < 40 {
                    image::Luma([255])
                } else {
                    image::Luma([0])
                }
            });
            reference_match_score(&output, &reference, &mask)
        }
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    println!(
        "reference match: {:.1}/100 (lightness {:.3}, chroma {:.3}, palette {:.3}, region {:.1})",
        report.match_score,
        report.lab_lightness_distance,
        report.lab_chroma_distance,
        report.palette_distance,
        report.region_tint_delta
    );
    Ok(())
}

fn run_serve(args: ServeArgs) -> Result<()> {
    let file_root = args.file_root.clone().unwrap_or_else(|| {
        if args.public_api {
            args.jobs_dir.clone()
        } else {
            PathBuf::from(".")
        }
    });
    tokio::runtime::Runtime::new()?.block_on(serve(ApiServerConfig {
        addr: args.addr,
        jobs_dir: args.jobs_dir,
        styles_dir: args.styles_dir,
        production_db: args.production_db,
        public_api: args.public_api,
        file_root,
        allowed_origins: args.cors_origin,
        max_concurrent_jobs: args.max_concurrent_jobs,
        job_retention_hours: args.job_retention_hours,
        default_model_path: args.default_model,
        default_config_path: args.default_config,
        default_style_path: PathBuf::from("configs/styles/soft_cel_anime_v2.yaml"),
    }))
}

fn run_production(command: ProductionCommand) -> Result<()> {
    match command {
        ProductionCommand::Bootstrap {
            database,
            email,
            name,
            tier,
        } => {
            let store = ProductionStore::open(database)?;
            let tier = parse_user_tier(&tier)?;
            let user = store.create_user(email, name, tier)?;
            let session = store.issue_token(&user.id, "bootstrap", None)?;
            println!("user_id={}", user.id);
            println!("api_token={}", session.token);
        }
        ProductionCommand::Token {
            database,
            user,
            label,
            expires_days,
        } => {
            let store = ProductionStore::open(database)?;
            let expires_at = expires_days.map(|days| Utc::now().timestamp() + days.max(1) * 86_400);
            let session = store.issue_token(&user, &label, expires_at)?;
            println!("token_id={}", session.token_id);
            println!("api_token={}", session.token);
        }
        ProductionCommand::Quota { database, user } => {
            let store = ProductionStore::open(database)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&store.quota_snapshot(&user)?)?
            );
        }
        ProductionCommand::Backup { database, output } => {
            let store = ProductionStore::open(database)?;
            store.backup(&output)?;
            println!("backup={}", output.display());
        }
        ProductionCommand::Retention {
            database,
            object_store,
            limit,
        } => {
            let store = ProductionStore::open(database)?;
            let objects = LocalObjectStore::new(object_store)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&sweep_retention(&store, &objects, limit)?)?
            );
        }
    }
    Ok(())
}

fn parse_user_tier(value: &str) -> Result<UserTier> {
    match value.trim().to_ascii_lowercase().as_str() {
        "free" => Ok(UserTier::Free),
        "creator" => Ok(UserTier::Creator),
        "studio" => Ok(UserTier::Studio),
        "admin" => Ok(UserTier::Admin),
        _ => anyhow::bail!("tier must be free, creator, studio, or admin"),
    }
}

fn parse_color_opt(value: Option<String>) -> Result<Option<ColorRgb>> {
    value
        .as_deref()
        .map(ColorRgb::from_str)
        .transpose()
        .map_err(anyhow::Error::from)
}

fn load_base_style(path: Option<&Path>) -> Result<StyleProfile> {
    match path {
        Some(path) => load_style_profile(path),
        None => Ok(StyleProfile::default()),
    }
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
