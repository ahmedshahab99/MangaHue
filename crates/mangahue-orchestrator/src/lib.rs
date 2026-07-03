use anyhow::{anyhow, Context, Result};
use image::{GrayImage, Luma, Rgb, RgbImage};
use mangahue_config::{load_style_profile, EngineConfig};
use mangahue_core::{
    CharacterProfile, ImageSize, MangaPanel, ProcessingReport, StageReport, StyleProfile, TileInfo,
};
use mangahue_hints::{generate_hints, load_user_hints};
use mangahue_image::{crop_gray, crop_rgb, load_image, save_gray, save_rgb};
use mangahue_inference::{
    ColorizationBackend, MockColorizationBackend, ModelInput, OnnxColorizationBackend,
};
use mangahue_layout::{analyze_layout, merge_rgb_tiles};
use mangahue_memory::{load_project_characters, ProjectMemory};
use mangahue_postprocess::{finalize, PostprocessInput};
use mangahue_preprocess::preprocess_page;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColorizeRequest {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub model_path: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub preset_path: Option<PathBuf>,
    pub style_path: Option<PathBuf>,
    pub project_path: Option<PathBuf>,
    #[serde(default)]
    pub character_ids: Vec<String>,
    pub user_hints_path: Option<PathBuf>,
    pub debug_dir: Option<PathBuf>,
    pub use_mock_backend: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorizeResult {
    pub output_path: PathBuf,
    pub report: ProcessingReport,
    pub quality_report_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct InspectRequest {
    pub input_path: PathBuf,
    pub config_path: Option<PathBuf>,
    pub preset_path: Option<PathBuf>,
    pub debug_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageInspection {
    pub input_path: PathBuf,
    pub size: ImageSize,
    pub page_classification: String,
    pub preprocess: mangahue_preprocess::PreprocessReport,
    pub panels: Vec<MangaPanel>,
    pub tiles: Vec<TileInfo>,
    pub stage_reports: Vec<StageReport>,
    pub debug_files: Vec<PathBuf>,
}

#[derive(Debug, Default)]
pub struct MangaHueEngine;

impl MangaHueEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn colorize(&self, request: ColorizeRequest) -> Result<ColorizeResult> {
        let total_start = Instant::now();
        let config = load_config(
            request.config_path.as_deref(),
            request.preset_path.as_deref(),
        )?;
        let style = load_style(request.style_path.as_deref())?;

        let mut report = ProcessingReport {
            input_path: request.input_path.clone(),
            output_path: request.output_path.clone(),
            model_path: request.model_path.clone(),
            style_id: Some(style.id.clone()),
            character_ids: request.character_ids.clone(),
            ..ProcessingReport::default()
        };

        let manga_image = load_image(&request.input_path)?;
        let resized = mangahue_image::resize_to_max(
            &manga_image.image,
            config.preprocess.max_input_dimension,
        );
        if resized.dimensions() != manga_image.image.dimensions() {
            report.warn(format!(
                "input resized from {:?} to {:?}",
                manga_image.image.dimensions(),
                resized.dimensions()
            ));
        }

        let preprocess_start = Instant::now();
        let preprocess = preprocess_page(&resized, &config.preprocess)?;
        report.preprocessing_ms = preprocess_start.elapsed().as_millis();
        report.page_classification = Some(classify_page(
            size_from_gray(&preprocess.clean_lineart),
            &preprocess.report,
        ));
        report.add_stage_report(preprocess_stage_report(&preprocess.report));
        for warning in &preprocess.report.warnings {
            report.warn(warning.clone());
        }

        let layout_start = Instant::now();
        let layout = analyze_layout(&preprocess.clean_lineart, &config.layout)?;
        report.layout_ms = layout_start.elapsed().as_millis();
        report.tiles_count = layout.tiles.len();
        report.add_stage_report(
            StageReport::new("layout")
                .metric("panel_count", layout.panels.len() as f64)
                .metric("tile_count", layout.tiles.len() as f64)
                .metric(
                    "webtoon_like",
                    if layout.is_webtoon_like { 1.0 } else { 0.0 },
                ),
        );
        info!(
            tiles = layout.tiles.len(),
            panels = layout.panels.len(),
            "layout complete"
        );

        let hint_start = Instant::now();
        let user_hints = match request.user_hints_path.as_deref() {
            Some(path) => Some(load_user_hints(path)?),
            None => None,
        };
        let characters = load_characters(request.project_path.as_deref(), &request.character_ids)?;
        if characters.is_empty() && !request.character_ids.is_empty() {
            report.warn("requested character ids were not found in project memory");
        }
        let size = ImageSize::new(
            preprocess.clean_lineart.width(),
            preprocess.clean_lineart.height(),
        );
        let hints = generate_hints(
            size,
            user_hints.as_ref(),
            &style,
            &characters,
            &config.hints,
        );
        report.hint_ms = hint_start.elapsed().as_millis();
        report.add_stage_report(
            StageReport::new("hints")
                .metric("applied_hints", hints.applied_hints as f64)
                .metric("style_hints", hints.style_hints_applied as f64)
                .metric("memory_hints", hints.memory_hints_applied as f64)
                .metric("user_hints", hints.user_hints_applied as f64)
                .metric("hint_coverage", mask_coverage(&hints.hint_mask) as f64),
        );
        for note in hints.notes {
            report.warn(note);
        }

        let inference_start = Instant::now();
        let backend: Arc<dyn ColorizationBackend> =
            Arc::from(select_backend(&request, &config, &style)?);
        let parallel_jobs = resolve_parallel_jobs(config.engine.parallel_jobs);
        info!(
            backend = backend.name(),
            jobs = parallel_jobs,
            tiles = layout.tiles.len(),
            "running inference"
        );
        let colored_tiles: Vec<(TileInfo, RgbImage)> =
            if layout.tiles.len() < 2 || parallel_jobs == 1 {
                layout
                    .tiles
                    .iter()
                    .map(|tile| {
                        let lineart = crop_gray(&preprocess.clean_lineart, tile.bbox);
                        let hint_map = crop_rgb(&hints.hint_map, tile.bbox);
                        let hint_mask = crop_gray(&hints.hint_mask, tile.bbox);
                        backend
                            .colorize(&ModelInput {
                                lineart,
                                hint_map,
                                hint_mask,
                            })
                            .map(|output| (tile.clone(), output))
                            .map_err(|err| anyhow!(err))
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                let pool = build_pool(parallel_jobs);
                pool.install(|| {
                    let tiles = &layout.tiles;
                    let clean = &preprocess.clean_lineart;
                    let hint_map = &hints.hint_map;
                    let hint_mask = &hints.hint_mask;
                    let backend = Arc::clone(&backend);
                    tiles
                        .par_iter()
                        .map(|tile| {
                            let lineart = crop_gray(clean, tile.bbox);
                            let hint_map = crop_rgb(hint_map, tile.bbox);
                            let hint_mask = crop_gray(hint_mask, tile.bbox);
                            backend
                                .colorize(&ModelInput {
                                    lineart,
                                    hint_map,
                                    hint_mask,
                                })
                                .map(|output| (tile.clone(), output))
                                .map_err(|err| anyhow!(err))
                        })
                        .collect::<Result<Vec<_>>>()
                })?
            };
        let merged = merge_rgb_tiles(size, &colored_tiles)?;
        report.inference_ms = inference_start.elapsed().as_millis();
        report.add_stage_report(
            StageReport::new("inference")
                .note(format!("backend={}", backend.name()))
                .metric("parallel_jobs", parallel_jobs as f64)
                .metric("tiles_completed", colored_tiles.len() as f64),
        );

        let postprocess_start = Instant::now();
        let final_image = finalize(PostprocessInput {
            colorized: merged,
            original_grayscale: &preprocess.original_grayscale,
            line_mask: &preprocess.line_mask,
            bubble_mask: &preprocess.bubble_mask,
            style: &style,
            config: &config.postprocess,
        })?;
        report.postprocess_ms = postprocess_start.elapsed().as_millis();

        save_rgb(&request.output_path, &final_image)?;

        if let Some(project_path) = request.project_path.as_deref() {
            update_project_history(project_path, &manga_image.info.id).ok();
        }

        if config.engine.save_debug_outputs
            || config.output.write_intermediates
            || request.debug_dir.is_some()
        {
            let debug_dir = request
                .debug_dir
                .clone()
                .unwrap_or_else(|| sibling_debug_dir(&request.output_path));
            write_debug_outputs(
                &debug_dir,
                &preprocess,
                &hints.hint_map,
                &hints.hint_mask,
                &mut report,
            )?;
        }

        let quality = mangahue_eval::evaluate_output_with_guidance(
            &final_image,
            &preprocess.line_mask,
            &preprocess.bubble_mask,
            Some(&preprocess.protected_mask),
            Some(&hints.hint_map),
            Some(&hints.hint_mask),
        );
        report.director_verdict = Some(director_verdict(&quality, &preprocess.report));

        report.total_ms = total_start.elapsed().as_millis();
        let report_path = if config.output.write_report {
            let path = mangahue_image::sibling_with_suffix(&request.output_path, ".report", "json");
            fs::write(&path, serde_json::to_string_pretty(&report)?)
                .with_context(|| format!("failed to write report {}", path.display()))?;
            Some(path)
        } else {
            None
        };

        let quality_report_path = report_path.as_ref().map(|path| {
            let quality_path = path.with_file_name(format!(
                "{}.quality.json",
                path.file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("report")
            ));
            let _ = fs::write(
                &quality_path,
                serde_json::to_string_pretty(&quality).unwrap(),
            );
            quality_path
        });

        Ok(ColorizeResult {
            output_path: request.output_path,
            report,
            quality_report_path,
        })
    }

    pub fn inspect_page(&self, request: InspectRequest) -> Result<PageInspection> {
        let config = load_config(
            request.config_path.as_deref(),
            request.preset_path.as_deref(),
        )?;
        let manga_image = load_image(&request.input_path)?;
        let resized = mangahue_image::resize_to_max(
            &manga_image.image,
            config.preprocess.max_input_dimension,
        );
        let preprocess = preprocess_page(&resized, &config.preprocess)?;
        let layout = analyze_layout(&preprocess.clean_lineart, &config.layout)?;
        let page_classification = classify_page(
            size_from_gray(&preprocess.clean_lineart),
            &preprocess.report,
        );
        let mut debug_files = Vec::new();
        if let Some(debug_dir) = request.debug_dir.as_deref() {
            let empty_hint_map = RgbImage::from_pixel(
                preprocess.clean_lineart.width(),
                preprocess.clean_lineart.height(),
                Rgb([0, 0, 0]),
            );
            let empty_hint_mask = GrayImage::from_pixel(
                preprocess.clean_lineart.width(),
                preprocess.clean_lineart.height(),
                Luma([0]),
            );
            let mut report = ProcessingReport::default();
            write_debug_outputs(
                debug_dir,
                &preprocess,
                &empty_hint_map,
                &empty_hint_mask,
                &mut report,
            )?;
            debug_files = report.debug_files;
        }
        Ok(PageInspection {
            input_path: request.input_path,
            size: size_from_gray(&preprocess.clean_lineart),
            page_classification,
            preprocess: preprocess.report.clone(),
            stage_reports: vec![
                preprocess_stage_report(&preprocess.report),
                StageReport::new("layout")
                    .metric("panel_count", layout.panels.len() as f64)
                    .metric("tile_count", layout.tiles.len() as f64)
                    .metric(
                        "webtoon_like",
                        if layout.is_webtoon_like { 1.0 } else { 0.0 },
                    ),
            ],
            panels: layout.panels,
            tiles: layout.tiles,
            debug_files,
        })
    }
}

fn load_config(config_path: Option<&Path>, preset_path: Option<&Path>) -> Result<EngineConfig> {
    let mut paths = Vec::new();
    if let Some(path) = config_path {
        paths.push(path);
    }
    if let Some(path) = preset_path {
        paths.push(path);
    }
    if paths.is_empty() {
        let config = EngineConfig::default();
        config.validate()?;
        Ok(config)
    } else {
        EngineConfig::from_paths(paths)
    }
}

fn load_style(style_path: Option<&Path>) -> Result<StyleProfile> {
    match style_path {
        Some(path) => load_style_profile(path),
        None => Ok(StyleProfile::default()),
    }
}

fn load_characters(project_path: Option<&Path>, ids: &[String]) -> Result<Vec<CharacterProfile>> {
    match project_path {
        Some(path) => load_project_characters(path, ids),
        None => Ok(Vec::new()),
    }
}

fn select_backend(
    request: &ColorizeRequest,
    config: &EngineConfig,
    style: &StyleProfile,
) -> Result<Box<dyn ColorizationBackend>> {
    if request.use_mock_backend {
        return Ok(Box::new(MockColorizationBackend::new(
            style.palette.clone(),
        )));
    }

    let model_path = request
        .model_path
        .as_ref()
        .unwrap_or(&config.inference.model_path);
    if model_path.extension().and_then(|value| value.to_str()) == Some("placeholder") {
        return Err(anyhow!(
            "real colorization requires an ONNX model; pass --model path/to/model.onnx or use --mock for pipeline verification"
        ));
    }
    Ok(Box::new(OnnxColorizationBackend::new(
        model_path,
        &config.inference.model_metadata_path,
    )?))
}

fn update_project_history(project_path: &Path, page_id: &str) -> Result<()> {
    let mut project = ProjectMemory::load(project_path)?;
    project.mark_processed(page_id);
    project.save(project_path)
}

fn write_debug_outputs(
    debug_dir: &Path,
    preprocess: &mangahue_preprocess::PreprocessResult,
    hint_map: &image::RgbImage,
    hint_mask: &image::GrayImage,
    report: &mut ProcessingReport,
) -> Result<()> {
    fs::create_dir_all(debug_dir)
        .with_context(|| format!("failed to create debug dir {}", debug_dir.display()))?;
    let files = [
        (
            "clean_lineart.png",
            DebugImage::Gray(&preprocess.clean_lineart),
        ),
        ("line_mask.png", DebugImage::Gray(&preprocess.line_mask)),
        ("tone_mask.png", DebugImage::Gray(&preprocess.tone_mask)),
        ("bubble_mask.png", DebugImage::Gray(&preprocess.bubble_mask)),
        ("text_mask.png", DebugImage::Gray(&preprocess.text_mask)),
        (
            "protected_mask.png",
            DebugImage::Gray(&preprocess.protected_mask),
        ),
        ("hint_mask.png", DebugImage::Gray(hint_mask)),
        ("hint_map.png", DebugImage::Rgb(hint_map)),
    ];
    for (name, image) in files {
        let path = debug_dir.join(name);
        match image {
            DebugImage::Gray(image) => save_gray(&path, image)?,
            DebugImage::Rgb(image) => save_rgb(&path, image)?,
        }
        report.debug_files.push(path);
    }
    let overlay = build_director_overlay(preprocess, hint_map, hint_mask);
    let overlay_path = debug_dir.join("director_overlay.png");
    save_rgb(&overlay_path, &overlay)?;
    report.debug_files.push(overlay_path);
    Ok(())
}

enum DebugImage<'a> {
    Gray(&'a image::GrayImage),
    Rgb(&'a image::RgbImage),
}

fn sibling_debug_dir(output_path: &Path) -> PathBuf {
    let stem = output_path
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("output");
    output_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(format!("{stem}.debug"))
}

trait DynamicImageDimensions {
    fn dimensions(&self) -> (u32, u32);
}

impl DynamicImageDimensions for image::DynamicImage {
    fn dimensions(&self) -> (u32, u32) {
        image::GenericImageView::dimensions(self)
    }
}

fn preprocess_stage_report(report: &mangahue_preprocess::PreprocessReport) -> StageReport {
    let mut stage = StageReport::new("preprocess")
        .metric("background_level", report.background_level as f64)
        .metric("adaptive_threshold", report.adaptive_threshold as f64)
        .metric("line_density", report.line_density as f64)
        .metric("ink_components", report.ink_component_count as f64)
        .metric("large_ink_components", report.large_ink_components as f64)
        .metric(
            "speckle_pixels_removed",
            report.speckle_pixels_removed as f64,
        )
        .metric("bubble_regions", report.bubble_regions as f64)
        .metric("bubble_coverage", report.bubble_coverage as f64)
        .metric("text_ink_pixels", report.text_ink_pixels as f64)
        .metric("protected_coverage", report.protected_coverage as f64)
        .metric("screen_tone_score", report.screen_tone_score as f64)
        .metric("page_complexity_score", report.page_complexity_score as f64);
    for warning in &report.warnings {
        stage = stage.note(warning.clone());
    }
    stage
}

fn classify_page(size: ImageSize, report: &mangahue_preprocess::PreprocessReport) -> String {
    let format = if size.height > size.width.saturating_mul(3) {
        "webtoon_strip"
    } else if size.height > size.width {
        "manga_page_portrait"
    } else {
        "manga_spread_or_panel"
    };
    let density = if report.line_density > 0.32 {
        "heavy_ink"
    } else if report.line_density < 0.015 {
        "light_ink"
    } else {
        "balanced_ink"
    };
    let tone = if report.screen_tone_score > 0.12 {
        "tone_rich"
    } else {
        "clean_lineart"
    };
    format!("{format}:{density}:{tone}")
}

fn director_verdict(
    quality: &mangahue_eval::QualityReport,
    preprocess: &mangahue_preprocess::PreprocessReport,
) -> String {
    if quality.line_preservation_score < 0.92 {
        return "needs_line_preservation_attention".to_string();
    }
    if quality.protected_region_violation_score < 0.96 {
        return "needs_protected_region_cleanup".to_string();
    }
    if quality.hint_adherence_score < 0.72 {
        return "needs_stronger_hint_or_memory_guidance".to_string();
    }
    if preprocess.screen_tone_score > 0.18 {
        return "production_ready_with_tone_training_warning".to_string();
    }
    "production_ready_preview".to_string()
}

fn build_director_overlay(
    preprocess: &mangahue_preprocess::PreprocessResult,
    hint_map: &RgbImage,
    hint_mask: &GrayImage,
) -> RgbImage {
    let mut overlay = RgbImage::new(
        preprocess.clean_lineart.width(),
        preprocess.clean_lineart.height(),
    );
    for (x, y, pixel) in overlay.enumerate_pixels_mut() {
        let base = preprocess.original_grayscale.get_pixel(x, y)[0];
        *pixel = Rgb([base, base, base]);
        if preprocess.tone_mask.get_pixel(x, y)[0] > 0 {
            blend_pixel(pixel, Rgb([245, 171, 53]), 0.35);
        }
        if preprocess.bubble_mask.get_pixel(x, y)[0] > 0 {
            blend_pixel(pixel, Rgb([38, 190, 120]), 0.42);
        }
        if preprocess.text_mask.get_pixel(x, y)[0] > 0 {
            blend_pixel(pixel, Rgb([55, 128, 235]), 0.70);
        }
        if hint_mask.get_pixel(x, y)[0] > 0 {
            let hint = hint_map.get_pixel(x, y);
            blend_pixel(
                pixel,
                *hint,
                (hint_mask.get_pixel(x, y)[0] as f32 / 255.0) * 0.65,
            );
        }
        if preprocess.line_mask.get_pixel(x, y)[0] > 0 {
            *pixel = Rgb([0, 0, 0]);
        }
    }
    overlay
}

fn blend_pixel(pixel: &mut Rgb<u8>, color: Rgb<u8>, alpha: f32) {
    let alpha = alpha.clamp(0.0, 1.0);
    for channel in 0..3 {
        pixel[channel] = (pixel[channel] as f32 * (1.0 - alpha) + color[channel] as f32 * alpha)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

fn size_from_gray(image: &GrayImage) -> ImageSize {
    ImageSize::new(image.width(), image.height())
}

fn mask_coverage(mask: &GrayImage) -> f32 {
    let active = mask.pixels().filter(|pixel| pixel[0] > 0).count() as f32;
    active / (mask.width() as f32 * mask.height() as f32).max(1.0)
}

fn resolve_parallel_jobs(requested: usize) -> usize {
    if requested == 0 {
        std::thread::available_parallelism()
            .map(|value| value.get())
            .unwrap_or(1)
            .max(1)
    } else {
        requested.max(1)
    }
}

fn build_pool(jobs: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(jobs.max(1))
        .thread_name(|index| format!("mangahue-worker-{index}"))
        .build()
        .unwrap_or_else(|error| {
            warn!(error = %error, "failed to build custom thread pool, falling back to global pool");
            rayon::ThreadPoolBuilder::new()
                .build()
                .expect("rayon global pool must always build")
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn full_pipeline_runs_with_explicit_mock_backend() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("page.png");
        let output = dir.path().join("out.png");
        let mut page = GrayImage::from_pixel(96, 128, Luma([255]));
        for x in 8..88 {
            page.put_pixel(x, 20, Luma([0]));
            page.put_pixel(x, 100, Luma([0]));
        }
        for y in 20..100 {
            page.put_pixel(8, y, Luma([0]));
            page.put_pixel(87, y, Luma([0]));
        }
        page.save(&input).unwrap();

        let engine = MangaHueEngine::new();
        let result = engine
            .colorize(ColorizeRequest {
                input_path: input,
                output_path: output.clone(),
                use_mock_backend: true,
                ..ColorizeRequest::default()
            })
            .unwrap();

        assert!(output.exists());
        assert_eq!(result.report.tiles_count, 1);
    }
}
