use anyhow::{anyhow, Context, Result};
use image::{GrayImage, Luma, Rgb, RgbImage};
use mangahue_cache::{ArtifactCache, ArtifactKind, CacheKey};
use mangahue_config::{load_style_profile, EngineConfig};
use mangahue_core::{
    BoundingBox, CharacterProfile, ColorRgb, ImageSize, MangaPanel, ProcessingReport, StageReport,
    StyleProfile, TileInfo,
};
use mangahue_ensemble::{
    ConfidenceMap, ConsensusStrategy, EnsembleConfig, EnsembleReport, Ensembler,
};
use mangahue_hints::{
    apply_region_locks_to_hints, apply_region_style_hints, detect_semantic_regions,
    enforce_region_locks, generate_hints, load_region_locks, load_user_hints, propagate_hints,
    rasterize_face_hints, rasterize_face_mask, rasterize_region_locks, rasterize_semantic_hints,
};
use mangahue_image::{crop_gray, crop_rgb, load_image, save_gray, save_rgb};
use mangahue_inference::{ColorizationBackend, MockColorizationBackend, ModelInput};
use mangahue_layout::{analyze_layout, merge_rgb_tiles};
use mangahue_memory::{load_project_characters, ProjectMemory};
use mangahue_postprocess::{
    cross_tile_coherence, finalize, reconstruct_full_resolution, PostprocessInput,
};
use mangahue_preprocess::preprocess_page;
use mangahue_pyramid::{PyramidColorizer, PyramidConfig};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ColorizeRequest {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub model_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_image_path: Option<PathBuf>,
    pub config_path: Option<PathBuf>,
    pub preset_path: Option<PathBuf>,
    pub style_path: Option<PathBuf>,
    pub project_path: Option<PathBuf>,
    pub region_locks_path: Option<PathBuf>,
    #[serde(default)]
    pub character_ids: Vec<String>,
    pub user_hints_path: Option<PathBuf>,
    pub debug_dir: Option<PathBuf>,
    pub use_mock_backend: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ensemble_model_paths: Vec<PathBuf>,
    #[serde(default)]
    pub enable_ensemble: bool,
    #[serde(default, skip)]
    pub cancel_token: Option<Arc<AtomicBool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_dir: Option<PathBuf>,
    #[serde(default)]
    pub pyramid_enabled: bool,
    #[serde(default)]
    pub hint_propagation_enabled: bool,
    /// Artist palette override (configs/palettes/*.yaml). Overrides
    /// style.palette before scene re-weighting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palette_path: Option<PathBuf>,
    /// Automatically select the artist palette that best fits the page
    /// fingerprint (brightness/contrast) from the scene palettes.
    #[serde(default)]
    pub auto_palette: bool,
    /// Per-chapter accent palette extracted from the manga's own colored
    /// output. Merged into the style palette's accent slots (>= 9) before
    /// scene re-weighting, so each manga gets its own color universe while
    /// paper/skin/hair slots stay natural.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palette_override: Option<Vec<ColorRgb>>,
    /// Client-side draft: a colorized image produced OUTSIDE the engine
    /// (e.g. by the WebGPU web app). When set, the orchestrator skips model
    /// inference and uses the draft as the model output, then runs the full
    /// protected-layer + director + grading postprocess.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draft_path: Option<PathBuf>,
    #[serde(skip)]
    pub progress_sender: Option<tokio::sync::mpsc::UnboundedSender<ProgressUpdate>>,
}

/// Progressive rendering update emitted at each pipeline stage and after
/// every tile completes. Streamed to API clients via SSE so Studio can show
/// partial results as they arrive.
#[derive(Debug, Clone, Serialize)]
pub struct ProgressUpdate {
    pub stage: String,
    pub tiles_completed: usize,
    pub tiles_total: usize,
    pub tile_id: Option<String>,
    pub percent: f32,
}

fn emit_progress(
    sender: &Option<tokio::sync::mpsc::UnboundedSender<ProgressUpdate>>,
    stage: &str,
    completed: usize,
    total: usize,
    tile_id: Option<String>,
) {
    if let Some(tx) = sender {
        let percent = if total == 0 {
            100.0
        } else {
            (completed as f32 / total as f32) * 100.0
        };
        let _ = tx.send(ProgressUpdate {
            stage: stage.to_string(),
            tiles_completed: completed,
            tiles_total: total,
            tile_id,
            percent,
        });
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorizeResult {
    pub output_path: PathBuf,
    pub report: ProcessingReport,
    pub quality_report_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence_map_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idempotency_key: Option<String>,
    #[serde(default)]
    pub from_cache: bool,
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
        let cancel = request.cancel_token.as_ref().map(Arc::clone);

        if let Some(key) = request.idempotency_key.as_deref().map(|s| s.to_string()) {
            let ck = CacheKey(key.clone());
            if let Some(dir) = request.cache_dir.as_deref() {
                let cache = ArtifactCache::new(dir);
                // A real idempotency hit requires BOTH the colored page and
                // its report to be cached under the caller's key; the output
                // image is restored to the requested output path.
                if let Ok(Some(page_artifact)) =
                    cache.get_kind(&ck, mangahue_cache::ArtifactKind::ColoredPage)
                {
                    let report =
                        match cache.get_kind(&ck, mangahue_cache::ArtifactKind::QualityReport) {
                            Ok(Some(report_artifact)) => serde_json::from_slice(
                                &fs::read(&report_artifact.path).with_context(|| {
                                    format!(
                                        "failed to read cached report {}",
                                        report_artifact.path.display()
                                    )
                                })?,
                            )
                            .unwrap_or_default(),
                            _ => ProcessingReport::default(),
                        };
                    fs::copy(&page_artifact.path, &request.output_path).with_context(|| {
                        format!(
                            "failed to restore cached output to {}",
                            request.output_path.display()
                        )
                    })?;
                    info!(cache_key = %ck.0, "idempotency cache hit, returning cached result");
                    return Ok(ColorizeResult {
                        output_path: request.output_path,
                        report,
                        quality_report_path: None,
                        confidence_map_path: None,
                        idempotency_key: Some(key),
                        from_cache: true,
                    });
                }
            }
        }

        let total_start = Instant::now();
        let config = load_config(
            request.config_path.as_deref(),
            request.preset_path.as_deref(),
        )?;
        let style = load_style(request.style_path.as_deref())?;

        // Resolve pyramid / hint propagation flags. Request explicit opt-in
        // takes precedence; otherwise fall back to the config-derived value.
        let pyramid_enabled =
            request.pyramid_enabled || config.engine.pyramid_enabled || config.pyramid.enabled;
        let pyramid_config = config.pyramid.clone();
        let hint_propagation_enabled =
            request.hint_propagation_enabled || config.engine.hint_propagation_enabled;
        let progress = request.progress_sender.clone();
        emit_progress(&progress, "queued", 0, 1, None);

        let mut report = ProcessingReport {
            input_path: request.input_path.clone(),
            output_path: request.output_path.clone(),
            model_path: request.model_path.clone(),
            style_id: Some(style.id.clone()),
            character_ids: request.character_ids.clone(),
            ..ProcessingReport::default()
        };

        let manga_image = load_image(&request.input_path)?;

        // Already-colored pass-through: full-color western comics are
        // common, and re-colorizing them wastes inference time and distorts
        // the artist's original palette. Measured on the ORIGINAL input
        // before any preprocessing.
        if config.engine.skip_already_colored {
            let (mean_sat, colored_fraction) = measure_page_colorfulness(&manga_image.image);
            let already_colored = mean_sat >= config.engine.already_colored_sat_threshold
                && colored_fraction >= config.engine.already_colored_colored_fraction;
            report.add_stage_report(
                StageReport::new("color_metrics")
                    .metric("avg_saturation", mean_sat as f64)
                    .metric("colorfulness", (colored_fraction * 100.0) as f64)
                    .metric("already_colored", if already_colored { 1.0 } else { 0.0 }),
            );
            if already_colored {
                report.warn(format!(
                    "page already colored (mean sat {mean_sat:.3}, colored share {colored_fraction:.3}); passing through untouched"
                ));
                info!(
                    mean_sat,
                    colored_fraction, "already-colored page passed through without inference"
                );
                save_rgb(&request.output_path, &manga_image.image.to_rgb8())?;
                emit_progress(&progress, "done", 6, 6, None);
                report.total_ms = total_start.elapsed().as_millis();
                report.director_verdict = Some("already colored".to_string());
                report.add_stage_report(
                    StageReport::new("postprocess")
                        .note("skipped: page already colored, original passed through"),
                );
                if config.output.write_report {
                    let path = mangahue_image::sibling_with_suffix(
                        &request.output_path,
                        ".report",
                        "json",
                    );
                    fs::write(&path, serde_json::to_string_pretty(&report)?)
                        .with_context(|| format!("failed to write report {}", path.display()))?;
                }
                return Ok(ColorizeResult {
                    output_path: request.output_path,
                    report,
                    quality_report_path: None,
                    confidence_map_path: None,
                    idempotency_key: None,
                    from_cache: false,
                });
            }
        }

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
        emit_progress(&progress, "preprocess", 0, 6, None);
        let preprocess = preprocess_page(&resized, &config.preprocess, &Default::default())?;
        report.preprocessing_ms = preprocess_start.elapsed().as_millis();
        emit_progress(&progress, "preprocess", 1, 6, None);
        report.page_classification = Some(classify_page(
            size_from_gray(&preprocess.clean_lineart),
            &preprocess.report,
        ));
        // Scene intelligence: every manga page has its own lighting
        // environment (day / sunset / night / overcast / indoor). Classify
        // it from the page's grayscale and adapt the style palette to the
        // scene so the coloring matches the environment instead of forcing
        // one fixed look.
        let scene = mangahue_postprocess::analyze_scene(&preprocess.original_grayscale);
        // Artist palette: every series has its own color vision. Either an
        // explicit palette file (configs/palettes/*.yaml) or automatic
        // selection from the page fingerprint overrides the style palette
        // BEFORE the scene re-weighting applies.
        let mut palette_style = style.clone();
        if request.auto_palette {
            let name = mangahue_config::select_palette_name(scene.brightness, scene.contrast);
            let path = mangahue_config::default_palettes_dir().join(format!("{name}.yaml"));
            if let Ok(colors) = mangahue_config::load_palette(&path) {
                report.warn(format!("artist palette auto-selected: {name}"));
                palette_style.palette = merge_artist_accents(&style.palette, &colors);
            } else {
                report.warn(format!(
                    "auto palette {name} not found at {}; using style palette",
                    path.display()
                ));
            }
        } else if let Some(palette_path) = request.palette_path.as_deref() {
            let colors = mangahue_config::load_palette(palette_path)?;
            report.warn(format!(
                "artist palette override: {}",
                palette_path.display()
            ));
            palette_style.palette = colors;
        }
        if let Some(accents) = request.palette_override.as_deref() {
            if !accents.is_empty() {
                report.warn(format!(
                    "chapter palette accents merged: {} color(s)",
                    accents.len()
                ));
                palette_style.palette = merge_artist_accents(&palette_style.palette, accents);
            }
        }
        let scene_style = {
            let mut adapted = palette_style.clone();
            adapted.palette =
                mangahue_postprocess::scene_palette(&palette_style.palette, &scene, 14);
            adapted
        };
        report.warn(format!(
            "scene detected: {} (brightness {:.0}, contrast {:.0})",
            scene.lighting.label(),
            scene.brightness * 255.0,
            scene.contrast
        ));
        report.add_stage_report(
            StageReport::new("scene")
                .metric("brightness", scene.brightness as f64)
                .metric("contrast", scene.contrast as f64)
                .metric("warmth", scene.warmth as f64)
                .note(format!("lighting={}", scene.lighting.label())),
        );
        report.add_stage_report(preprocess_stage_report(&preprocess.report));
        for warning in &preprocess.report.warnings {
            report.warn(warning.clone());
        }

        let layout_start = Instant::now();
        emit_progress(&progress, "layout", 1, 6, None);
        let layout = analyze_layout(&preprocess.clean_lineart, &config.layout)?;
        report.layout_ms = layout_start.elapsed().as_millis();
        emit_progress(&progress, "layout", 2, 6, None);
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
        emit_progress(&progress, "hints", 2, 6, None);
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
        let mut hints = generate_hints(
            size,
            user_hints.as_ref(),
            &scene_style,
            &characters,
            &config.hints,
        );
        apply_region_style_hints(
            &mut hints,
            &preprocess.clean_lineart,
            &preprocess.bubble_mask,
            &scene_style,
            &config.hints,
        );
        // Semantic full-coverage hints: dense region painting (skin/hair/
        // clothing/background/bubbles) so the hint-U-Net sees strong,
        // locally-correct material hints instead of sparse dots. These
        // override the weaker grid/dot hints inside the same regions.
        if config.hints.region_style_hints_enabled {
            let regions = detect_semantic_regions(
                &preprocess.original_grayscale,
                &preprocess.line_mask,
                &preprocess.bubble_mask,
            );
            let painted = rasterize_semantic_hints(
                &mut hints.hint_map,
                &mut hints.hint_mask,
                &preprocess.original_grayscale,
                &regions,
            );
            if painted > 0 {
                hints
                    .notes
                    .push(format!("semantic region hints painted {painted} pixels"));
                hints.applied_hints += 1;
                hints.region_style_hints_applied += 1;
            }
        }

        // Face-driven hints: detect anime faces and paint precise skin/hair/
        // eye ellipses so the model colors characters correctly. Only runs
        // when a face cascade directory is configured (MANGAHUE_FACE_MODEL).
        let mut face_boxes: Vec<[u32; 4]> = Vec::new();
        if let Some(face_dir) = std::env::var("MANGAHUE_FACE_MODEL")
            .ok()
            .map(PathBuf::from)
            .filter(|dir| mangahue_inference::is_anime_face_model_dir(dir))
        {
            match mangahue_inference::AnimeFaceBackend::shared(face_dir) {
                Ok(face_backend) => {
                    match face_backend.detect_faces(&preprocess.original_grayscale) {
                        Ok(faces) if !faces.is_empty() => {
                            face_boxes = faces;
                            // Skin/hair/eye colors from the style palette by
                            // semantic slots; the slot contract (0-1 paper,
                            // 2-5 skin, 6-8 hair) only holds for full
                            // >= 9-color palettes, so short palettes fall
                            // back to natural defaults instead of reading an
                            // arbitrary hue (e.g. slot 2 is BLUE in the
                            // 3-color default style).
                            let palette_slots_ok = style.palette.len() >= 9;
                            let skin = if palette_slots_ok {
                                style.palette.get(2).map(|c| [c.r, c.g, c.b])
                            } else {
                                None
                            }
                            .unwrap_or([242, 208, 182]);
                            let hair = if palette_slots_ok {
                                style.palette.get(6).map(|c| [c.r, c.g, c.b])
                            } else {
                                None
                            }
                            .unwrap_or([38, 28, 34]);
                            let eyes = [46, 40, 54];
                            let face_painted = rasterize_face_hints(
                                &mut hints.hint_map,
                                &mut hints.hint_mask,
                                &face_boxes,
                                skin,
                                hair,
                                eyes,
                            );
                            if face_painted > 0 {
                                hints.notes.push(format!(
                                    "face-driven hints painted {face_painted} pixels across {} face(s)",
                                    face_boxes.len()
                                ));
                                hints.applied_hints += 1;
                                hints.region_style_hints_applied += 1;
                            }
                        }
                        Ok(_) => {}
                        Err(error) => {
                            tracing::warn!(error = %error, "face detection failed; skipping face hints");
                        }
                    }
                }
                Err(error) => {
                    tracing::warn!(error = %error, "face backend failed to start; skipping face hints");
                }
            }
        }
        // Face mask for postprocess: guarantees detected faces are painted
        // peach even where the model outputs near-white.
        let face_mask = if face_boxes.is_empty() {
            None
        } else {
            Some(rasterize_face_mask(
                &face_boxes,
                preprocess.original_grayscale.width(),
                preprocess.original_grayscale.height(),
            ))
        };
        // The bubble detector mistakes bright faces (large bright region
        // bounded by dark contours) for speech bubbles. That would make every
        // postprocess stage skip/whiten the face â€” the white-face bug. The
        // face is authoritative skin: zero the bubble mask wherever the face
        // mask is strong, so exclusion, grading and bubble cleanup all leave
        // the face alone.
        let bubble_mask = if let Some(ref mask) = face_mask {
            let mut bm = preprocess.bubble_mask.clone();
            for ((_, _, bpx), mpx) in bm.enumerate_pixels_mut().zip(mask.pixels()) {
                if mpx[0] > 128 {
                    bpx[0] = 0;
                }
            }
            bm
        } else {
            preprocess.bubble_mask.clone()
        };
        if let (Some(mask), Some(debug_dir)) = (face_mask.as_ref(), request.debug_dir.as_deref()) {
            let painted = mask.pixels().filter(|p| p[0] > 0).count();
            report.warn(format!(
                "face mask rasterized: {painted} pixels, box={:?}",
                face_boxes
            ));
            let path = debug_dir.join("face_mask.png");
            save_gray(&path, mask)
                .with_context(|| format!("failed to save face mask {}", path.display()))?;
            report.debug_files.push(path);
        }
        report.hint_ms = hint_start.elapsed().as_millis();
        report.add_stage_report(
            StageReport::new("hints")
                .metric("applied_hints", hints.applied_hints as f64)
                .metric("style_hints", hints.style_hints_applied as f64)
                .metric(
                    "region_style_hints",
                    hints.region_style_hints_applied as f64,
                )
                .metric("memory_hints", hints.memory_hints_applied as f64)
                .metric("user_hints", hints.user_hints_applied as f64)
                .metric("hint_coverage", mask_coverage(&hints.hint_mask) as f64),
        );
        for note in &hints.notes {
            report.warn(note.clone());
        }

        if hint_propagation_enabled {
            let propagation_radius = config.hints.propagation_radius;
            let similarity_threshold = config.hints.propagation_similarity_threshold;
            let (propagated_map, propagated_mask) = propagate_hints(
                &hints.hint_map,
                &hints.hint_mask,
                &preprocess.clean_lineart,
                propagation_radius,
                similarity_threshold,
            );
            hints.hint_map = propagated_map;
            hints.hint_mask = propagated_mask;
            report.warn(format!(
                "hints propagated with radius {}",
                propagation_radius
            ));
        }

        let region_lock_raster = request
            .region_locks_path
            .as_deref()
            .map(load_region_locks)
            .transpose()?
            .map(|locks| rasterize_region_locks(&locks, size))
            .transpose()?;
        if let Some(raster) = &region_lock_raster {
            let lock_report =
                apply_region_locks_to_hints(&mut hints.hint_map, &mut hints.hint_mask, raster);
            report.warn(format!(
                "region locks applied: {} lock(s), {} constrained pixel(s), {} excluded hint pixel(s)",
                lock_report.applied_locks,
                lock_report.colorized_pixels + lock_report.preserved_pixels,
                lock_report.excluded_hint_pixels
            ));
            if let Some(debug_dir) = request.debug_dir.as_deref() {
                save_gray(debug_dir.join("region_color_mask.png"), &raster.color_mask)?;
                save_gray(
                    debug_dir.join("region_preserve_mask.png"),
                    &raster.preserve_mask,
                )?;
                save_gray(
                    debug_dir.join("region_exclude_mask.png"),
                    &raster.exclude_mask,
                )?;
            }
        }

        let inference_start = Instant::now();
        let parallel_jobs = resolve_parallel_jobs(config.engine.parallel_jobs);
        let use_ensemble = request.enable_ensemble && !request.ensemble_model_paths.is_empty();
        let tiles_total = layout.tiles.len();
        emit_progress(&progress, "inference_started", 0, tiles_total, None);

        let colored_tiles: Vec<(TileInfo, RgbImage)>;
        let mut ensemble_report: Option<mangahue_ensemble::EnsembleReport> = None;
        let mut confidence_map: Option<ConfidenceMap> = None;
        let backend_name: String;

        if let Some(draft_path) = request.draft_path.as_deref() {
            // Client-side draft mode (WebGPU web app): the model output was
            // produced on the CLIENT's GPU and uploaded here. Model
            // inference is skipped; the full protected-layer, region
            // director and grading postprocess still runs on the draft.
            let draft = load_image(draft_path)?.image.to_rgb8();
            let draft = if draft.dimensions() != (size.width, size.height) {
                image::imageops::resize(
                    &draft,
                    size.width,
                    size.height,
                    image::imageops::FilterType::Lanczos3,
                )
            } else {
                draft
            };
            backend_name = "client-webgpu-draft".to_string();
            info!(draft = %draft_path.display(), "using client-side draft");
            colored_tiles = vec![(
                TileInfo {
                    id: "tile_0001".to_string(),
                    bbox: BoundingBox::new(0, 0, size.width, size.height),
                    overlap: 0,
                    panel_id: None,
                },
                draft,
            )];
        } else if use_ensemble {
            backend_name = "ensemble".to_string();
            let ens = build_ensembler(&request, &config, &style)?;
            info!(
                ensemble_variants = 1usize + request.ensemble_model_paths.len(),
                jobs = parallel_jobs,
                tiles = layout.tiles.len(),
                "running ensemble inference"
            );
            let cancel = cancel.clone();
            // DDColor-family variants need the original grayscale; the
            // ensemble feeds every variant the same source, so the most
            // demanding variant decides.
            let use_original = ens.prefers_original_grayscale();
            let source_gray = if use_original {
                &preprocess.original_grayscale
            } else {
                &preprocess.clean_lineart
            };
            if layout.tiles.len() < 2 || parallel_jobs == 1 {
                let mut acc = Vec::with_capacity(layout.tiles.len());
                for tile in &layout.tiles {
                    if check_cancel(&cancel) {
                        return Err(anyhow!("colorization cancelled before tile {}", tile.id));
                    }
                    let (merged, conf, _report) = ens
                        .colorize(&ModelInput {
                            lineart: crop_gray(source_gray, tile.bbox),
                            hint_map: crop_rgb(&hints.hint_map, tile.bbox),
                            hint_mask: crop_gray(&hints.hint_mask, tile.bbox),
                        })
                        .map_err(|err| anyhow!(err))?;
                    ensemble_report = Some(_report);
                    if confidence_map.is_none() {
                        confidence_map = Some(conf.clone());
                    }
                    acc.push((tile.clone(), merged));
                    emit_progress(
                        &progress,
                        "inference_tile",
                        acc.len(),
                        tiles_total,
                        Some(tile.id.clone()),
                    );
                }
                colored_tiles = acc;
            } else {
                let pool = build_pool(parallel_jobs);
                let raw: Vec<Result<(TileInfo, RgbImage, ConfidenceMap, EnsembleReport)>> = pool
                    .install(|| {
                        let tiles = &layout.tiles;
                        let clean = source_gray;
                        let hint_map = &hints.hint_map;
                        let hint_mask = &hints.hint_mask;
                        let cancel = cancel.clone();
                        tiles
                            .par_iter()
                            .map(|tile| {
                                if check_cancel(&cancel) {
                                    return Err(anyhow!(
                                        "cancelled during ensemble tile {}",
                                        tile.id
                                    ));
                                }
                                ens.colorize(&ModelInput {
                                    lineart: crop_gray(clean, tile.bbox),
                                    hint_map: crop_rgb(hint_map, tile.bbox),
                                    hint_mask: crop_gray(hint_mask, tile.bbox),
                                })
                                .map(|(merged, conf, report)| (tile.clone(), merged, conf, report))
                                .map_err(|err| anyhow!(err))
                            })
                            .collect()
                    });
                let mut colored: Vec<(TileInfo, RgbImage)> = Vec::with_capacity(raw.len());
                for entry in raw {
                    let (tile, merged, conf, report) = entry?;
                    if ensemble_report.is_none() {
                        ensemble_report = Some(report);
                    }
                    if confidence_map.is_none() {
                        confidence_map = Some(conf);
                    }
                    colored.push((tile, merged));
                }
                colored_tiles = colored;
            }
            emit_progress(&progress, "inference_done", 4, 6, None);
        } else {
            let raw_backend: Arc<dyn ColorizationBackend> =
                Arc::from(select_backend(&request, &config, &style)?);
            let backend: Arc<dyn ColorizationBackend> = if pyramid_enabled {
                let config = if pyramid_config.levels.is_empty() {
                    PyramidConfig::default()
                } else {
                    PyramidConfig {
                        enabled: true,
                        levels: pyramid_config.levels.clone(),
                        blend_weights: if pyramid_config.blend_weights.len()
                            == pyramid_config.levels.len()
                        {
                            pyramid_config.blend_weights.clone()
                        } else {
                            // Equal weights when the count mismatches.
                            let n = pyramid_config.levels.len() as f32;
                            vec![1.0 / n; pyramid_config.levels.len()]
                        },
                        ..PyramidConfig::default()
                    }
                };
                Arc::new(
                    PyramidColorizer::new(raw_backend, config)
                        .with_cancel(request.cancel_token.clone()),
                )
            } else {
                raw_backend
            };
            backend_name = backend.name().to_string();
            info!(
                backend = backend.name(),
                jobs = parallel_jobs,
                tiles = layout.tiles.len(),
                "running inference"
            );
            let cancel = cancel.clone();
            let use_original = backend.prefers_original_grayscale();
            colored_tiles = if layout.tiles.len() < 2 || parallel_jobs == 1 {
                let mut acc = Vec::with_capacity(layout.tiles.len());
                for tile in &layout.tiles {
                    if check_cancel(&cancel) {
                        return Err(anyhow!("colorization cancelled before tile {}", tile.id));
                    }
                    let source_gray = if use_original {
                        &preprocess.original_grayscale
                    } else {
                        &preprocess.clean_lineart
                    };
                    let output = backend
                        .colorize(&ModelInput {
                            lineart: crop_gray(source_gray, tile.bbox),
                            hint_map: crop_rgb(&hints.hint_map, tile.bbox),
                            hint_mask: crop_gray(&hints.hint_mask, tile.bbox),
                        })
                        .map_err(|err| anyhow!(err))?;
                    acc.push((tile.clone(), output));
                    emit_progress(
                        &progress,
                        "inference_tile",
                        acc.len(),
                        tiles_total,
                        Some(tile.id.clone()),
                    );
                }
                acc
            } else {
                let pool = build_pool(parallel_jobs);
                let raw: Vec<Result<(TileInfo, RgbImage)>> = pool.install(|| {
                    let tiles = &layout.tiles;
                    let source_gray = if use_original {
                        &preprocess.original_grayscale
                    } else {
                        &preprocess.clean_lineart
                    };
                    let hint_map = &hints.hint_map;
                    let hint_mask = &hints.hint_mask;
                    let backend = Arc::clone(&backend);
                    let cancel = cancel.clone();
                    tiles
                        .par_iter()
                        .map(|tile| {
                            if check_cancel(&cancel) {
                                return Err(anyhow!("cancelled during tile {}", tile.id));
                            }
                            backend
                                .colorize(&ModelInput {
                                    lineart: crop_gray(source_gray, tile.bbox),
                                    hint_map: crop_rgb(hint_map, tile.bbox),
                                    hint_mask: crop_gray(hint_mask, tile.bbox),
                                })
                                .map(|output| (tile.clone(), output))
                                .map_err(|err| anyhow!(err))
                        })
                        .collect()
                });
                raw.into_iter().collect::<Result<Vec<_>>>()?
            };
            emit_progress(&progress, "inference_done", 4, 6, None);
        }

        if check_cancel(&cancel) {
            return Err(anyhow!("colorization cancelled after inference"));
        }

        let mut merged = merge_rgb_tiles(size, &colored_tiles)?;
        report.inference_ms = inference_start.elapsed().as_millis();

        // The model's own chroma signal: mean opponent magnitude of the
        // inference output, measured before any postprocess. The smart
        // governor uses it as this page's color budget.
        let chroma_signal = {
            let (mw, mh) = merged.dimensions();
            let total = (mw as usize).saturating_mul(mh as usize);
            let step = (total / 100_000).max(1);
            let mut sum = 0.0f64;
            let mut n = 0u64;
            for (x, y, pixel) in merged.enumerate_pixels() {
                let index = y as usize * mw as usize + x as usize;
                if index % step != 0 {
                    continue;
                }
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;
                let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                sum += ((r - luma).abs() + (b - luma).abs()) as f64;
                n += 1;
            }
            if n > 0 {
                Some((sum / n as f64) as f32)
            } else {
                None
            }
        };

        cross_tile_coherence(
            &mut merged,
            &preprocess.line_mask,
            config.layout.tile_size,
            config.layout.tile_overlap,
        );
        let mut inference_report = StageReport::new("inference")
            .note(format!("backend={}", backend_name))
            .metric("parallel_jobs", parallel_jobs as f64)
            .metric("tiles_completed", colored_tiles.len() as f64);
        if let Some(ref er) = ensemble_report {
            inference_report = inference_report
                .metric("ensemble_variants", er.variant_count as f64)
                .metric("divergence_score", er.divergence_score as f64)
                .metric("semantic_rescue_pixels", er.semantic_rescue_pixels as f64);
        }
        report.add_stage_report(inference_report);

        let confidence_map_path: Option<PathBuf> = if let Some(ref cm) = confidence_map {
            let conf_path =
                mangahue_image::sibling_with_suffix(&request.output_path, ".confidence", "png");
            save_gray(&conf_path, &cm.to_gray_image()).with_context(|| {
                format!("failed to save confidence map {}", conf_path.display())
            })?;
            Some(conf_path)
        } else {
            None
        };

        if let Some(debug_dir) = request.debug_dir.as_deref() {
            fs::create_dir_all(debug_dir)?;
            let path = debug_dir.join("ensemble_merged.png");
            save_rgb(&path, &merged)?;
            report.debug_files.push(path);
        }

        let postprocess_start = Instant::now();
        emit_progress(&progress, "postprocess", 5, 6, None);
        let reference_image = request.reference_image_path.as_deref().and_then(|path| {
            match mangahue_image::load_image(path) {
                Ok(m) => {
                    let loaded = m.image.to_rgb8();
                    // Match the reference to the WORKING page size (the size
                    // the pipeline actually operates at). Resizing against the
                    // original full-res size caused a second resample inside
                    // the transfer and blurred the reference statistics.
                    let (w, h) = preprocess.original_grayscale.dimensions();
                    Some(if loaded.dimensions() != (w, h) {
                        image::imageops::resize(
                            &loaded,
                            w,
                            h,
                            image::imageops::FilterType::Lanczos3,
                        )
                    } else {
                        loaded
                    })
                }
                Err(err) => {
                    report.warn(format!(
                        "reference image {} could not be loaded: {err:#}",
                        path.display()
                    ));
                    None
                }
            }
        });
        let (mut final_image, postprocess_report) = finalize(PostprocessInput {
            colorized: merged,
            original_grayscale: &preprocess.original_grayscale,
            line_mask: &preprocess.line_mask,
            bubble_mask: &bubble_mask,
            tone_mask: Some(&preprocess.tone_mask),
            protected_mask: Some(&preprocess.protected_mask),
            face_mask: face_mask.as_ref(),
            style: &scene_style,
            config: &config.postprocess,
            reference_image: reference_image.as_ref(),
            // Semantic region hints finally reach the output through the
            // hint-guided chroma correction pass (DDColor never sees them
            // itself).
            hint_map: Some(&hints.hint_map),
            hint_mask: Some(&hints.hint_mask),
            chroma_signal,
        })?;
        if check_cancel(&cancel) {
            return Err(anyhow!("colorization cancelled during postprocess"));
        }
        report.add_stage_report(
            StageReport::new("postprocess")
                .metric(
                    "chroma_pixels_completed",
                    postprocess_report.chroma_pixels_completed as f64,
                )
                .metric(
                    "material_pixels_refined",
                    postprocess_report.material_pixels_refined as f64,
                )
                .metric(
                    "material_regions",
                    postprocess_report.material_regions as f64,
                )
                .metric(
                    "seeded_material_regions",
                    postprocess_report.seeded_material_regions as f64,
                )
                .metric(
                    "inferred_material_regions",
                    postprocess_report.inferred_material_regions as f64,
                )
                .metric(
                    "semantic_skin_regions",
                    postprocess_report.semantic_skin_regions as f64,
                )
                .metric(
                    "face_cavity_pixels",
                    postprocess_report.face_cavity_pixels as f64,
                )
                .metric(
                    "region_pixels_rendered",
                    postprocess_report.region_pixels_rendered as f64,
                )
                .metric(
                    "reference_pixels_finished",
                    postprocess_report.reference_pixels_finished as f64,
                )
                .metric(
                    "reference_chroma_pixels_expanded",
                    postprocess_report.reference_chroma_pixels_expanded as f64,
                )
                .metric(
                    "reference_shadow_pixels_shaped",
                    postprocess_report.reference_shadow_pixels_shaped as f64,
                )
                .metric(
                    "face_pixels_painted",
                    postprocess_report.face_pixels_painted as f64,
                )
                .metric(
                    "semantic_chroma_restored",
                    postprocess_report.semantic_chroma_restored as f64,
                )
                .metric(
                    "hint_guided_pixels",
                    postprocess_report.hint_guided_pixels as f64,
                )
                .metric(
                    "bubble_pixels_restored",
                    postprocess_report.bubble_pixels_restored as f64,
                )
                .metric(
                    "protected_pixels_restored",
                    postprocess_report.protected_pixels_restored as f64,
                )
                .metric(
                    "tone_pixels_preserved",
                    postprocess_report.tone_pixels_preserved as f64,
                )
                .metric(
                    "model_pixels_overridden",
                    postprocess_report.model_pixels_overridden as f64,
                )
                .metric(
                    "palette_fill_pixels",
                    postprocess_report.palette_fill_pixels as f64,
                )
                .metric(
                    "model_chroma_retention",
                    postprocess_report.model_chroma_retention as f64,
                )
                .metric(
                    "mean_hue_shift_deg",
                    postprocess_report.mean_hue_shift_deg as f64,
                ),
        );
        // Standard page color metrics (avg RGB, brightness, saturation,
        // colorfulness, contrast) measured on the final working-res image.
        let color_metrics = mangahue_eval::analyze_color_metrics(&final_image);
        report.add_stage_report(
            StageReport::new("color_metrics")
                .metric("avg_r", color_metrics.avg_rgb[0] as f64)
                .metric("avg_g", color_metrics.avg_rgb[1] as f64)
                .metric("avg_b", color_metrics.avg_rgb[2] as f64)
                .metric("brightness", color_metrics.brightness as f64)
                .metric("avg_saturation", color_metrics.avg_saturation as f64)
                .metric("colorfulness", color_metrics.colorfulness as f64)
                .metric("contrast", color_metrics.contrast as f64)
                .metric("warmth", color_metrics.warmth as f64)
                .metric(
                    "chroma_signal",
                    chroma_signal.map(|v| v as f64).unwrap_or(0.0),
                ),
        );
        if let Some(raster) = &region_lock_raster {
            let lock_report = enforce_region_locks(
                &mut final_image,
                &preprocess.original_grayscale,
                &preprocess.line_mask,
                raster,
            );
            report.warn(format!(
                "region lock enforcement completed: {} color pixel(s), {} preserved pixel(s)",
                lock_report.colorized_pixels, lock_report.preserved_pixels
            ));
        }
        // Evaluate quality at WORKING resolution. All guidance masks
        // (line/bubble/protected/hint) are working-resolution; evaluating the
        // full-res image against them produced dimension-mismatched metrics
        // (hint adherence forced to 0.0, misaligned sampling) on every page
        // that exceeded max_input_dimension.
        let quality = mangahue_eval::evaluate_output_with_guidance(
            &final_image,
            &preprocess.line_mask,
            &preprocess.bubble_mask,
            Some(&preprocess.protected_mask),
            Some(&hints.hint_map),
            Some(&hints.hint_mask),
        );
        report.director_verdict = Some(director_verdict(&quality, &preprocess.report, &config));

        if final_image.dimensions() != manga_image.image.dimensions() {
            let original_gray = manga_image.image.to_luma8();
            final_image = reconstruct_full_resolution(&final_image, &original_gray);
            report.warn(format!(
                "full-resolution color reconstructed at {}x{}",
                final_image.width(),
                final_image.height()
            ));
        }
        report.postprocess_ms = postprocess_start.elapsed().as_millis();

        save_rgb(&request.output_path, &final_image)?;
        emit_progress(&progress, "done", 6, 6, None);

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
            let qpath = path.with_file_name(format!(
                "{}.quality.json",
                path.file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("report")
            ));
            if let Err(err) = fs::write(
                &qpath,
                serde_json::to_string_pretty(&quality).unwrap_or_default(),
            ) {
                report.warn(format!(
                    "failed to write quality report {}: {err:#}",
                    qpath.display()
                ));
            }
            qpath
        });

        let idempotency_key =
            compute_idempotency_key(&request, &preprocess, &hints, &config, &style);
        if let Some(dir) = request.cache_dir.as_deref() {
            let cache = ArtifactCache::new(dir);
            // Content-addressed entries (dedup) for the report.
            if let Some(report_path) = report_path.as_ref() {
                let _ = cache.put_file(
                    report_path,
                    ArtifactKind::QualityReport,
                    Some("application/json".to_string()),
                );
            }
            // Idempotency entries under the caller's key: the colored page
            // AND the report, so a later identical request can be served
            // entirely from cache (output image restored, not just a report).
            if let Some(user_key) = request.idempotency_key.as_deref() {
                let ck = CacheKey(user_key.to_string());
                let _ = cache.put_file_as(
                    &request.output_path,
                    &ck,
                    ArtifactKind::ColoredPage,
                    Some("image/png".to_string()),
                );
                if let Some(report_path) = report_path.as_ref() {
                    let _ = cache.put_file_as(
                        report_path,
                        &ck,
                        ArtifactKind::QualityReport,
                        Some("application/json".to_string()),
                    );
                }
            }
        }

        Ok(ColorizeResult {
            output_path: request.output_path,
            report,
            quality_report_path,
            confidence_map_path,
            idempotency_key: Some(idempotency_key),
            from_cache: false,
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
        let preprocess = preprocess_page(&resized, &config.preprocess, &Default::default())?;
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

fn merge_artist_accents(base: &[ColorRgb], accents: &[ColorRgb]) -> Vec<ColorRgb> {
    if base.len() < 9 {
        return accents.to_vec();
    }
    let mut merged = Vec::with_capacity(base.len() + accents.len());
    for color in base
        .iter()
        .take(9)
        .chain(accents)
        .chain(base.iter().skip(9))
    {
        if !merged.contains(color) {
            merged.push(*color);
        }
    }
    merged
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
    select_backend_for_model(model_path, config, request.reference_image_path.as_deref())
}

fn select_backend_for_model(
    model_path: &Path,
    config: &EngineConfig,
    reference_image_path: Option<&Path>,
) -> Result<Box<dyn ColorizationBackend>> {
    if mangahue_inference::is_anime_face_model_dir(model_path) {
        return Ok(Box::new(mangahue_inference::AnimeFaceBackend::shared(
            model_path,
        )?));
    }
    if mangahue_inference::is_reference_cgan_model_dir(model_path) {
        let reference_path = reference_image_path.ok_or_else(|| {
            anyhow!(
                "reference cGAN requires a reference image; pass --reference-image path/to/style.jpg"
            )
        })?;
        return Ok(Box::new(mangahue_inference::ReferenceCganBackend::shared(
            model_path,
            reference_path,
        )?));
    }
    #[cfg(feature = "native-onnx")]
    if mangahue_inference::is_style2paints_onnx_model(model_path) {
        // Style2Paints V3: full-RGB cel output, native Rust session.
        return Ok(Box::new(
            mangahue_inference::Style2PaintsV3OnnxBackend::shared(model_path)?,
        ));
    }
    #[cfg(not(feature = "native-onnx"))]
    if is_style2paints_model_name(model_path) {
        anyhow::bail!(
            "Style2Paints models require the native-onnx build (mangahue-telegram/mangahue-cli are built with it)"
        );
    }
    if mangahue_inference::is_ddcolor_onnx_model(model_path) {
        #[cfg(feature = "native-onnx")]
        {
            // Native Rust inference: no Python sidecar, no IPC. Sessions are
            // shared across pages so chapters never reload the model.
            return Ok(Box::new(
                mangahue_inference::NativeDdColorOnnxBackend::shared(model_path)?,
            ));
        }
        #[cfg(not(feature = "native-onnx"))]
        {
            return Ok(Box::new(mangahue_inference::DdColorOnnxBackend::shared(
                model_path,
            )?));
        }
    }
    if mangahue_inference::is_ddcolor_model_dir(model_path) {
        return Ok(Box::new(mangahue_inference::DdColorBackend::shared(
            model_path,
            &config.inference.device,
            512,
            "large",
        )?));
    }
    if mangahue_inference::is_manga_ninja_model_dir(model_path) {
        return Ok(Box::new(mangahue_inference::MangaNinjaBackend::shared(
            model_path,
            &config.inference.device,
            50,
        )?));
    }
    if mangahue_inference::is_pretrained_model_dir(model_path) {
        return Ok(Box::new(
            mangahue_inference::PretrainedWorkerBackend::shared(
                model_path,
                &config.inference.device,
                config.inference.input_size,
            )?,
        ));
    }
    if model_path.extension().and_then(|value| value.to_str()) == Some("placeholder") {
        return Err(anyhow!(
            "real colorization requires an ONNX model; pass --model path/to/model.onnx or use --mock for pipeline verification"
        ));
    }

    #[cfg(feature = "native-onnx")]
    {
        // Native inference is the ONLY path in this build: if the session
        // cannot be built (missing metadata, broken ORT_DYLIB_PATH, corrupt
        // model), fail loudly instead of silently falling back to a Python
        // sidecar that may not even exist on the host.
        let backend = mangahue_inference::RustOnnxBackend::shared(
            model_path,
            &config.inference.model_metadata_path,
        )?;
        return Ok(Box::new(backend));
    }
    #[cfg(not(feature = "native-onnx"))]
    {
        Ok(Box::new(mangahue_inference::OnnxColorizationBackend::new(
            model_path,
            &config.inference.model_metadata_path,
        )?))
    }
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
    config: &EngineConfig,
) -> String {
    if quality.line_preservation_score < config.eval.line_preservation_threshold {
        return "needs_line_preservation_attention".to_string();
    }
    if quality.protected_region_violation_score < config.eval.protected_region_violation_threshold {
        return "needs_protected_region_cleanup".to_string();
    }
    if quality.hint_adherence_score < config.eval.hint_adherence_threshold {
        return "needs_stronger_hint_or_memory_guidance".to_string();
    }
    if preprocess.screen_tone_score > config.eval.screen_tone_warning_threshold {
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

/// Style2Paints ONNX models are identified by file name; the native-onnx
/// build also verifies the file exists (see mangahue-inference). Kept here
/// so non-native builds can still produce a clear error for them.
fn is_style2paints_model_name(path: &Path) -> bool {
    path.extension().and_then(|value| value.to_str()) == Some("onnx")
        && path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|name| name.to_ascii_lowercase().contains("style2paints"))
}

/// Measure how colorful a page already is: (mean HSV saturation over
/// non-paper/non-ink pixels, share of those pixels with real color). A
/// grayscale scan sits near (0.00-0.03, 0.00); a colored page is typically
/// (0.15-0.45, 0.4-0.9). Sampled (<= 200k pixels) so the check costs
/// milliseconds even on huge pages.
fn measure_page_colorfulness(image: &image::DynamicImage) -> (f32, f32) {
    let rgb = image.to_rgb8();
    let (w, h) = rgb.dimensions();
    let total = w as usize * h as usize;
    let step = (total / 200_000).max(1);
    let mut sat_sum = 0.0f64;
    let mut sat_n = 0u64;
    let mut colored = 0u64;
    for (x, y, pixel) in rgb.enumerate_pixels() {
        let index = y as usize * w as usize + x as usize;
        if index % step != 0 {
            continue;
        }
        let r = pixel[0] as f32 / 255.0;
        let g = pixel[1] as f32 / 255.0;
        let b = pixel[2] as f32 / 255.0;
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        // Skip pure paper (white) and pure ink (black): they dominate
        // every manga page and would dilute the color signal.
        if !(0.10..0.96).contains(&luma) {
            continue;
        }
        let max = r.max(g).max(b);
        let min = r.min(g).min(b);
        let sat = if max > 0.0 { (max - min) / max } else { 0.0 };
        sat_sum += sat as f64;
        sat_n += 1;
        if sat > 0.15 {
            colored += 1;
        }
    }
    if sat_n == 0 {
        return (0.0, 0.0);
    }
    (
        (sat_sum / sat_n as f64) as f32,
        colored as f32 / sat_n as f32,
    )
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
                .unwrap_or_else(|fallback_error| {
                    warn!(
                        error = %fallback_error,
                        "failed to build any thread pool; running single-threaded"
                    );
                    rayon::ThreadPoolBuilder::new()
                        .num_threads(1)
                        .build()
                        .expect("a single-threaded rayon pool cannot fail")
                })
        })
}

fn build_ensembler(
    request: &ColorizeRequest,
    config: &EngineConfig,
    style: &StyleProfile,
) -> Result<Ensembler> {
    let primary: Arc<dyn ColorizationBackend> = Arc::from(select_backend(request, config, style)?);
    let has_reference = request.reference_image_path.is_some();
    let auxiliaries = request
        .ensemble_model_paths
        .iter()
        .filter_map(|path| {
            // Reference-cGAN auxiliaries require a reference image; skip them
            // gracefully (warn, don't fail the whole job) when none is set.
            if mangahue_inference::is_reference_cgan_model_dir(path) && !has_reference {
                tracing::warn!(
                    model = %path.display(),
                    "skipping reference-cGAN ensemble model (no reference image set)"
                );
                return None;
            }
            Some(
                select_backend_for_model(path, config, request.reference_image_path.as_deref())
                    .map(Arc::<dyn ColorizationBackend>::from)
                    .with_context(|| format!("failed to load ensemble model {}", path.display())),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let ensemble_config = EnsembleConfig {
        enabled: true,
        fallback_mock_count: 0,
        consensus_strategy: match config.ensemble.strategy {
            mangahue_config::EnsembleStrategy::MajorityWeighted => {
                ConsensusStrategy::MajorityWeighted
            }
            mangahue_config::EnsembleStrategy::AverageBlend => ConsensusStrategy::AverageBlend,
            mangahue_config::EnsembleStrategy::DivergenceReject => {
                ConsensusStrategy::DivergenceReject
            }
            mangahue_config::EnsembleStrategy::ChromaComplement => {
                ConsensusStrategy::ChromaComplement
            }
        },
        weight_divergence: true,
        cielab_distance: config.ensemble.cielab_distance,
        divergence_reject_threshold: config.ensemble.divergence_reject_threshold,
    };
    let weights = vec![config.ensemble.primary_weight; auxiliaries.len() + 1];
    Ok(Ensembler::with_weights(
        primary,
        auxiliaries,
        weights,
        ensemble_config,
    ))
}

fn check_cancel(cancel: &Option<Arc<AtomicBool>>) -> bool {
    cancel
        .as_ref()
        .is_some_and(|toggle| toggle.load(Ordering::Relaxed))
}

fn compute_idempotency_key(
    request: &ColorizeRequest,
    preprocess: &mangahue_preprocess::PreprocessResult,
    hints: &mangahue_hints::HintResult,
    config: &EngineConfig,
    style: &StyleProfile,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(request.input_path.to_string_lossy().as_bytes());
    hasher.update(b"\x00");
    hasher.update(preprocess.clean_lineart.width().to_le_bytes());
    hasher.update(preprocess.clean_lineart.height().to_le_bytes());
    hasher.update(preprocess.clean_lineart.as_raw());
    hasher.update(hints.applied_hints.to_le_bytes());
    hasher.update(hints.hint_map.as_raw());
    hasher.update(hints.hint_mask.as_raw());
    hasher.update(b"\x00");
    hasher.update(style.id.as_bytes());
    if let Ok(style_json) = serde_json::to_vec(style) {
        hasher.update(style_json);
    }
    if let Ok(config_json) = serde_json::to_vec(config) {
        hasher.update(config_json);
    }
    if let Some(model_path) = request.model_path.as_deref() {
        hasher.update(model_path.to_string_lossy().as_bytes());
        if let Ok(metadata) = fs::metadata(model_path) {
            hasher.update(metadata.len().to_le_bytes());
            if let Ok(modified) = metadata.modified() {
                if let Ok(elapsed) = modified.duration_since(std::time::UNIX_EPOCH) {
                    hasher.update(elapsed.as_nanos().to_le_bytes());
                }
            }
        }
    }
    hasher.update([
        request.use_mock_backend as u8,
        request.pyramid_enabled as u8,
        request.hint_propagation_enabled as u8,
        request.enable_ensemble as u8,
    ]);
    for model in &request.ensemble_model_paths {
        hasher.update(model.to_string_lossy().as_bytes());
        hasher.update(b"\x00");
    }
    let mode_val: u8 = match config.engine.mode {
        mangahue_config::EngineMode::Fast => 0,
        mangahue_config::EngineMode::Balanced => 1,
        mangahue_config::EngineMode::Quality => 2,
        mangahue_config::EngineMode::Debug => 3,
    };
    hasher.update([mode_val]);
    hasher.update(config.layout.tile_size.to_le_bytes());
    let digest = hasher.finalize();
    base64_encode_hash(&digest)
}

fn base64_encode_hash(digest: &[u8]) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    URL_SAFE_NO_PAD.encode(digest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn automatic_palette_preserves_semantic_roles() {
        let base = (0..12)
            .map(|index| ColorRgb::new(index, 40 + index, 80 + index))
            .collect::<Vec<_>>();
        let accents = vec![ColorRgb::new(200, 20, 40), ColorRgb::new(20, 80, 200)];

        let merged = merge_artist_accents(&base, &accents);

        assert_eq!(&merged[..9], &base[..9]);
        assert_eq!(&merged[9..11], accents.as_slice());
    }

    #[test]
    fn colorfulness_detects_grayscale_vs_colored_pages() {
        use image::RgbImage;
        // A pure grayscale page: near-zero saturation everywhere.
        let mut gray = RgbImage::new(64, 64);
        for (x, y, p) in gray.enumerate_pixels_mut() {
            let v = (40 + (x * 3 + y) % 160) as u8;
            p[0] = v;
            p[1] = v;
            p[2] = v;
        }
        let (mean_sat, colored_fraction) =
            measure_page_colorfulness(&image::DynamicImage::ImageRgb8(gray));
        assert!(mean_sat < 0.02, "grayscale mean sat: {mean_sat}");
        assert!(colored_fraction < 0.02);

        // A colored page: skin-tone blocks + saturated background.
        let mut colored = RgbImage::new(64, 64);
        for y in 0..64 {
            for x in 0..64 {
                let p = if x < 32 {
                    Rgb([242, 208, 182]) // skin
                } else if y < 32 {
                    Rgb([56, 104, 214]) // blue sky
                } else {
                    Rgb([120, 200, 90]) // green field
                };
                colored.put_pixel(x, y, p);
            }
        }
        let (mean_sat, colored_fraction) =
            measure_page_colorfulness(&image::DynamicImage::ImageRgb8(colored));
        assert!(mean_sat > 0.15, "colored page mean sat: {mean_sat}");
        assert!(
            colored_fraction > 0.5,
            "colored page share: {colored_fraction}"
        );
    }

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
