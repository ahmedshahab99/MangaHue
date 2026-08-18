mod natural_grade;
mod reference_finish;
mod region_renderer;
mod reinhard_transfer;
pub mod scene;
mod unified_grade;

pub use scene::{analyze_scene, scene_palette, SceneLighting, SceneProfile};

use anyhow::Result;
use image::{GrayImage, Rgb, RgbImage};
use mangahue_config::PostprocessSection;
use mangahue_core::StyleProfile;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct PostprocessInput<'a> {
    pub colorized: RgbImage,
    pub original_grayscale: &'a GrayImage,
    pub line_mask: &'a GrayImage,
    pub bubble_mask: &'a GrayImage,
    pub tone_mask: Option<&'a GrayImage>,
    pub protected_mask: Option<&'a GrayImage>,
    pub face_mask: Option<&'a GrayImage>,
    pub style: &'a StyleProfile,
    pub config: &'a PostprocessSection,
    pub reference_image: Option<&'a RgbImage>,
    /// Semantic region hints (skin/hair/clothing/background painted with
    /// palette colors): only consumed by the hint-guided chroma correction
    /// pass — the DDColor backend itself never sees them.
    pub hint_map: Option<&'a RgbImage>,
    pub hint_mask: Option<&'a GrayImage>,
    /// The model's own mean chroma on this page (opponent magnitude
    /// |R-L|+|B-L|, 0-255 units), measured right after inference. The
    /// smart governor uses it as the page's color budget so grading
    /// follows what the model judged the page to be.
    pub chroma_signal: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostprocessReport {
    pub bubble_pixels_restored: u64,
    pub protected_pixels_restored: u64,
    pub bleed_pixels_corrected: u64,
    pub tone_pixels_preserved: u64,
    #[serde(default)]
    pub chroma_pixels_completed: u64,
    #[serde(default)]
    pub material_pixels_refined: u64,
    #[serde(default)]
    pub material_regions: u64,
    #[serde(default)]
    pub seeded_material_regions: u64,
    #[serde(default)]
    pub inferred_material_regions: u64,
    #[serde(default)]
    pub semantic_skin_regions: u64,
    #[serde(default)]
    pub face_cavity_pixels: u64,
    #[serde(default)]
    pub region_pixels_rendered: u64,
    #[serde(default)]
    pub reference_pixels_finished: u64,
    #[serde(default)]
    pub reference_chroma_pixels_expanded: u64,
    #[serde(default)]
    pub reference_shadow_pixels_shaped: u64,
    #[serde(default)]
    pub face_pixels_painted: u64,
    #[serde(default)]
    pub semantic_chroma_restored: u64,
    /// Pixels whose hue was corrected toward the hint map (semantic region
    /// hints reaching the DDColor output).
    #[serde(default)]
    pub hint_guided_pixels: u64,
    pub sharpen_radius: f32,
    pub sharpen_amount: f32,
    /// Purity metrics (model-first audit): how much of the model's own
    /// chroma survived the finishing passes unchanged.
    #[serde(default)]
    pub model_pixels_overridden: u64,
    #[serde(default)]
    pub palette_fill_pixels: u64,
    #[serde(default)]
    pub model_chroma_retention: f32,
    #[serde(default)]
    pub mean_hue_shift_deg: f32,
    /// Which grading architecture ran: "unified" (two-stage: authoring +
    /// one measured grade) or "legacy" (the old four-pass cascade).
    #[serde(default)]
    pub grade_stage: String,
    /// Unified grade telemetry (measured page statistics + the single
    /// computed decision), so tuning has per-page ground truth.
    #[serde(default)]
    pub unified_gain: f32,
    #[serde(default)]
    pub unified_measured_saturation: f32,
    #[serde(default)]
    pub unified_measured_colorfulness: f32,
    #[serde(default)]
    pub unified_measured_warmth: f32,
    #[serde(default)]
    pub unified_warmth_shift: f32,
    #[serde(default)]
    pub unified_pixels_graded: u64,
}

pub fn finalize(input: PostprocessInput<'_>) -> Result<(RgbImage, PostprocessReport)> {
    let mut output = input.colorized;
    let semantic_evidence = output.clone();

    let mut report = PostprocessReport {
        bubble_pixels_restored: 0,
        protected_pixels_restored: 0,
        bleed_pixels_corrected: 0,
        tone_pixels_preserved: 0,
        chroma_pixels_completed: 0,
        material_pixels_refined: 0,
        material_regions: 0,
        seeded_material_regions: 0,
        inferred_material_regions: 0,
        semantic_skin_regions: 0,
        face_cavity_pixels: 0,
        region_pixels_rendered: 0,
        reference_pixels_finished: 0,
        reference_chroma_pixels_expanded: 0,
        reference_shadow_pixels_shaped: 0,
        face_pixels_painted: 0,
        semantic_chroma_restored: 0,
        hint_guided_pixels: 0,
        sharpen_radius: 1.0,
        sharpen_amount: input.config.final_sharpening,
        model_pixels_overridden: 0,
        palette_fill_pixels: 0,
        model_chroma_retention: 1.0,
        mean_hue_shift_deg: 0.0,
        grade_stage: String::new(),
        unified_gain: 1.0,
        unified_measured_saturation: 0.0,
        unified_measured_colorfulness: 0.0,
        unified_measured_warmth: 0.0,
        unified_warmth_shift: 0.0,
        unified_pixels_graded: 0,
    };

    // Hint-guided chroma correction: the semantic region hints (skin, hair,
    // clothing, background painted with palette colors) finally reach the
    // DDColor output here. Where the hint map is confident and the model's
    // chroma is weak (near-gray guess), the pixel's hue snaps toward the
    // hint color while the model's value keeps the shading — the model stays
    // the artist, the hints fix its blind spots. Runs FIRST so every
    // downstream pass (region renderer, grading) sees corrected colors.
    if input.config.hint_chroma_strength > 0.0 {
        if let (Some(hint_map), Some(hint_mask)) = (input.hint_map, input.hint_mask) {
            report.hint_guided_pixels = hint_guided_chroma_correction(
                &mut output,
                hint_map,
                hint_mask,
                input.line_mask,
                input.config.hint_chroma_strength,
            ) as u64;
        }
    }

    if input.config.bleeding_correction && !input.config.palette_correction {
        // When the region graph renders, it re-paints the ink-adjacent band
        // with strong chroma, undoing any bleed removal that ran before it.
        // Run the bleed pass AFTER region rendering instead (see below); this
        // branch only covers the no-region-renderer configuration.
        reduce_color_near_ink(
            &mut output,
            input.line_mask,
            &mut report.bleed_pixels_corrected,
        );
    }

    if input.config.palette_correction {
        let lut = MangaColorLUT::new(input.style);
        apply_lut_grading(&mut output, &lut, input.line_mask, input.original_grayscale);
        apply_palette_harmony(
            &mut output,
            input.style,
            input.line_mask,
            input.config.palette_harmony_strength,
            input.config.model_first,
        );
        complete_material_chroma(
            &mut output,
            input.original_grayscale,
            input.line_mask,
            None,
            input.style,
            &mut report.chroma_pixels_completed,
            &mut report.material_pixels_refined,
        );
        let region_report = region_renderer::render_region_graph(
            &mut output,
            input.original_grayscale,
            input.line_mask,
            Some(input.bubble_mask),
            input.style,
            input.config.soft_coverage,
            input.config.soft_strength,
            input.config.soft_confidence,
            input.config.region_chroma_separation,
            input.config.model_first,
            input.config.seed_chroma_bar_bright,
            input.config.seed_chroma_bar_mid,
            input.config.model_chroma_bar_bright,
            input.config.model_chroma_bar_mid,
            input.config.model_chroma_gain,
        );
        report.material_regions = region_report.regions;
        report.seeded_material_regions = region_report.seeded_regions;
        report.inferred_material_regions = region_report.inferred_regions;
        report.semantic_skin_regions = region_report.semantic_skin_regions;
        report.face_cavity_pixels = region_report.face_cavity_pixels;
        report.region_pixels_rendered = region_report.pixels_rendered;
    }

    if input.config.bleeding_correction {
        // Chroma bleed removal runs AFTER the region renderer so the pass
        // removes what the renderer just painted instead of being undone by
        // it: the 3px band hugging ink lines stays paper-clean.
        reduce_color_near_ink(
            &mut output,
            input.line_mask,
            &mut report.bleed_pixels_corrected,
        );
    }

    if input.config.reference_finishing {
        let finish_report = reference_finish::finish_reference_style(
            &mut output,
            input.original_grayscale,
            input.line_mask,
            input.bubble_mask,
            input.protected_mask,
            input.face_mask,
            input.style,
            input.config,
        );
        report.reference_pixels_finished = finish_report.pixels_finished;
        report.reference_chroma_pixels_expanded = finish_report.chroma_pixels_expanded;
        report.reference_shadow_pixels_shaped = finish_report.shadow_pixels_shaped;
        report.face_pixels_painted = finish_report.face_pixels_painted;
        report.protected_pixels_restored = finish_report.protected_pixels_skipped;
    }

    if input.config.reference_transfer {
        if let Some(reference) = input.reference_image {
            reinhard_transfer::apply_reinhard_transfer(
                &mut output,
                reference,
                input.line_mask,
                1.0,
            );
        }
    }

    // ------------------------------------------------------------------
    // Stage 2 — the grade. The two-stage architecture runs ONE measured,
    // bounded pass (unified_grade) that merges the natural grade's value
    // shaping and skin warm-up, the muted grade's per-luma saturation
    // profile and hue shifts, and the smart governor's measured gain and
    // warmth compression into a single coherent decision. The legacy
    // cascade (unified_grade: false) keeps the old order exactly.
    // ------------------------------------------------------------------
    if input.config.unified_grade {
        if input.config.muted_target_grade
            || input.config.natural_grading
            || input.config.smart_governor
        {
            let grade = unified_grade::unified_grade(
                &mut output,
                input.original_grayscale,
                input.line_mask,
                input.bubble_mask,
                input.protected_mask,
                input.hint_mask,
                input.config,
                input.chroma_signal,
            );
            report.grade_stage = "unified".to_string();
            report.unified_gain = grade.gain;
            report.unified_measured_saturation = grade.measured_saturation;
            report.unified_measured_colorfulness = grade.measured_colorfulness;
            report.unified_measured_warmth = grade.measured_warmth;
            report.unified_warmth_shift = grade.warmth_shift;
            report.unified_pixels_graded = grade.pixels_graded;
        } else {
            // No grading sub-behavior enabled: nothing to run.
            report.grade_stage = "unified".to_string();
        }
    } else {
        // Natural grading AFTER reference transfer: the transfer re-normalizes
        // the LAB stats toward the reference, which would otherwise wash out
        // the saturation curve. Running grading last keeps the richer look.
        if input.config.natural_grading {
            natural_grade::natural_grade(
                &mut output,
                input.original_grayscale,
                input.line_mask,
                input.bubble_mask,
            );
        }

        restore_semantic_chroma(
            &mut output,
            &semantic_evidence,
            input.line_mask,
            input.face_mask,
            &mut report.semantic_chroma_restored,
        );

        // The muted target grade runs LAST of the color passes: the semantic
        // chroma restore would otherwise re-saturate what the grade just
        // desaturated (its washed-out gate fires after any desaturation).
        if input.config.muted_target_grade {
            natural_grade::muted_target_grade(
                &mut output,
                input.original_grayscale,
                input.line_mask,
                input.bubble_mask,
                input.hint_mask,
                input.config.muted_strength,
                input.config.muted_hue_shift_deg,
                input.config.muted_yellow_shift_deg,
                input.config.muted_dark_pull,
                input.config.muted_dark_sat,
                input.config.muted_mid_sat,
                input.config.muted_bright_sat,
                input.config.muted_paper_sat,
                input.config.muted_vivid_preserve,
            );
        }

        // Smart governor runs last of the color passes: measured page metrics
        // drive a bounded chroma gain toward the preset targets, scaled by the
        // model's own chroma signal (the page's color budget).
        if input.config.smart_governor {
            natural_grade::smart_color_governor(
                &mut output,
                input.line_mask,
                input.config.governor_target_saturation,
                input.config.governor_target_colorfulness,
                input.config.governor_min_gain,
                input.config.governor_max_gain,
                input.config.governor_target_warmth,
                input.chroma_signal,
                input.hint_mask,
            );
        }
        report.grade_stage = "legacy".to_string();
    }

    if let Some(tone_mask) = input.tone_mask {
        preserve_screentones(
            &mut output,
            input.original_grayscale,
            tone_mask,
            input.config.line_strength * 0.55,
            &mut report.tone_pixels_preserved,
        );
    }

    edge_aware_smooth(&mut output, input.line_mask);

    if input.config.bubble_cleanup {
        clean_speech_bubbles(
            &mut output,
            input.original_grayscale,
            input.bubble_mask,
            &mut report.bubble_pixels_restored,
        );
    }

    if input.config.restore_lines {
        restore_lines_adaptive(
            &mut output,
            input.line_mask,
            input.config.line_strength * input.style.line_strength,
        );
    }

    if input.config.final_sharpening > 0.0 {
        sharpen_lightly(&mut output, input.config.final_sharpening);
    }

    // Model speckle cleanup: DDColor-family models scatter isolated
    // strong-color pixels (measured ~11.5k per page, mostly warm) on
    // textures and midtones — the "colored dots" complaint. Runs after
    // every color-producing stage so nothing re-creates them, and after
    // the sharpener so its chroma ringing around dots is cleaned too.
    if input.config.despeckle_color_specks {
        remove_color_specks(&mut output, input.line_mask);
    }
    if let Some(protected_mask) = input.protected_mask {
        report.protected_pixels_restored = restore_protected_paper_and_ink(
            &mut output,
            input.original_grayscale,
            input.line_mask,
            input.bubble_mask,
            protected_mask,
        );
    }
    enforce_strong_ink(&mut output, input.line_mask);
    measure_purity(
        &output,
        &semantic_evidence,
        input.line_mask,
        &mut report.model_pixels_overridden,
        &mut report.palette_fill_pixels,
        &mut report.model_chroma_retention,
        &mut report.mean_hue_shift_deg,
    );
    Ok((output, report))
}

/// Model-first purity audit. Compares the final output against the pure
/// model evidence (captured before any postprocess) on pixels where the
/// model was confident (opponent chroma >= 16), excluding ink. Reports how
/// many of those pixels were overridden (hue rotated > 15 deg or chroma
/// changed by > 40%%), how many near-neutral pixels received palette fill,
/// the mean chroma retention ratio and the mean hue shift in degrees.
fn measure_purity(
    output: &RgbImage,
    evidence: &RgbImage,
    line_mask: &GrayImage,
    overridden: &mut u64,
    palette_fill: &mut u64,
    retention: &mut f32,
    mean_hue_shift: &mut f32,
) {
    let (width, height) = output.dimensions();
    if width == 0 || height == 0 || evidence.dimensions() != (width, height) {
        return;
    }
    let mut confident = 0u64;
    let mut retention_sum = 0.0f64;
    let mut hue_shift_sum = 0.0f64;
    let mut overridden_count = 0u64;
    let mut fill_count = 0u64;
    for y in 0..height {
        for x in 0..width {
            if line_mask.get_pixel(x, y)[0] > 180 {
                continue;
            }
            let e = *evidence.get_pixel(x, y);
            let o = *output.get_pixel(x, y);
            let el = 0.2126 * e[0] as f32 + 0.7152 * e[1] as f32 + 0.0722 * e[2] as f32;
            let e_mag = (e[0] as f32 - el).abs() + (e[2] as f32 - el).abs();
            let ol = 0.2126 * o[0] as f32 + 0.7152 * o[1] as f32 + 0.0722 * o[2] as f32;
            let o_mag = (o[0] as f32 - ol).abs() + (o[2] as f32 - ol).abs();
            if e_mag >= 16.0 {
                confident += 1;
                if o_mag > 0.5 {
                    retention_sum += (o_mag / e_mag).min(3.0) as f64;
                }
                // Opponent-space hue angle difference.
                let e_angle = e_mag.atan2(e[2] as f32 - el);
                let o_angle = o_mag.atan2(o[2] as f32 - ol);
                let mut delta = (e_angle - o_angle).abs();
                while delta > std::f32::consts::PI {
                    delta -= std::f32::consts::PI * 2.0;
                }
                hue_shift_sum += (delta * 180.0 / std::f32::consts::PI) as f64;
                let chroma_ratio = o_mag / e_mag.max(0.01);
                if delta * 180.0 / std::f32::consts::PI > 15.0
                    || !(0.6..=1.67).contains(&chroma_ratio)
                {
                    overridden_count += 1;
                }
            } else if e_mag < 8.0 && o_mag > 10.0 {
                fill_count += 1;
            }
        }
    }
    if confident > 0 {
        *overridden = overridden_count;
        *retention = (retention_sum / confident as f64) as f32;
        *mean_hue_shift = (hue_shift_sum / confident as f64) as f32;
    }
    *palette_fill = fill_count;
}

pub fn cross_tile_coherence(
    image: &mut RgbImage,
    line_mask: &GrayImage,
    tile_size: u32,
    overlap: u32,
) {
    let width = image.width();
    let height = image.height();
    let step = tile_size.saturating_sub(overlap);

    for ty in (0..height).step_by(step as usize) {
        for tx in (0..width).step_by(step as usize) {
            let tile_end_x = (tx + tile_size).min(width);
            let tile_end_y = (ty + tile_size).min(height);

            if tx > 0 && overlap > 0 {
                let blend_width = overlap.min(tile_end_x - tx);
                for y in ty..tile_end_y {
                    for dx in 0..blend_width {
                        let x = tx + dx;
                        if x >= width {
                            continue;
                        }
                        let t = dx as f32 / blend_width as f32;
                        // The left neighbour always covers [tx-overlap, tx),
                        // which is exactly this blend band.
                        let left_x = x - overlap;
                        let left = *image.get_pixel(left_x, y);
                        let right = *image.get_pixel(x, y);
                        let line = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
                        let line_weight = (1.0 - line * 0.7).clamp(0.3, 1.0);
                        let blend_t = t * line_weight;
                        let blended = [
                            (left[0] as f32 + (right[0] as f32 - left[0] as f32) * blend_t) as u8,
                            (left[1] as f32 + (right[1] as f32 - left[1] as f32) * blend_t) as u8,
                            (left[2] as f32 + (right[2] as f32 - left[2] as f32) * blend_t) as u8,
                        ];
                        image.put_pixel(x, y, Rgb(blended));
                    }
                }
            }

            if ty > 0 && overlap > 0 {
                let blend_height = overlap.min(tile_end_y - ty);
                for x in tx..tile_end_x {
                    for dy in 0..blend_height {
                        let y = ty + dy;
                        if y >= height {
                            continue;
                        }
                        let t = dy as f32 / blend_height as f32;
                        // The top neighbour always covers [ty-overlap, ty).
                        let top_y = y - overlap;
                        let top = *image.get_pixel(x, top_y);
                        let bottom = *image.get_pixel(x, y);
                        let line = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
                        let line_weight = (1.0 - line * 0.7).clamp(0.3, 1.0);
                        let blend_t = t * line_weight;
                        let blended = [
                            (top[0] as f32 + (bottom[0] as f32 - top[0] as f32) * blend_t) as u8,
                            (top[1] as f32 + (bottom[1] as f32 - top[1] as f32) * blend_t) as u8,
                            (top[2] as f32 + (bottom[2] as f32 - top[2] as f32) * blend_t) as u8,
                        ];
                        image.put_pixel(x, y, Rgb(blended));
                    }
                }
            }
        }
    }
}

fn restore_semantic_chroma(
    output: &mut RgbImage,
    evidence: &RgbImage,
    line_mask: &GrayImage,
    face_mask: Option<&GrayImage>,
    restored: &mut u64,
) {
    let (width, height) = output.dimensions();
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        if line_mask.get_pixel(x, y)[0] > 172 {
            continue;
        }
        // Face pixels are already force-painted with deep peach by
        // reference_finish; re-injecting the model's flat chroma here would
        // wash them back to pale. The face paint is authoritative.
        if let Some(mask) = face_mask {
            if mask.dimensions() == (width, height) && mask.get_pixel(x, y)[0] > 128 {
                continue;
            }
        }
        let source = evidence.get_pixel(x, y);
        let source_max = source[0].max(source[1]).max(source[2]);
        let source_min = source[0].min(source[1]).min(source[2]);
        let source_chroma = (source_max - source_min) as f32;
        // Only clearly visible model colors are candidates for restoration,
        // and only when the output is genuinely washed out (below half the
        // model's chroma): the region director's deliberate hues must not
        // be overwritten by the model's casts.
        if !(40.0..=112.0).contains(&source_chroma) {
            continue;
        }
        let output_max = pixel[0].max(pixel[1]).max(pixel[2]);
        let output_min = pixel[0].min(pixel[1]).min(pixel[2]);
        let output_chroma = (output_max - output_min) as f32;
        // Restore only on genuine wash-out (below HALF the model's chroma):
        // keeps the director's deliberate hues while still re-coloring
        // truly flat gray areas.
        if output_chroma >= source_chroma * 0.5 {
            continue;
        }

        let source_luma =
            0.2126 * source[0] as f32 + 0.7152 * source[1] as f32 + 0.0722 * source[2] as f32;
        let output_luma =
            0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
        // Skip very dark pixels so shadows keep their graded deep chroma.
        if source_luma < 42.0 || output_luma < 38.0 {
            continue;
        }
        let strength = (0.55 + (source_chroma - output_chroma) / 220.0).clamp(0.55, 0.75);
        let red = (source[0] as f32 - source_luma) * strength;
        let blue = (source[2] as f32 - source_luma) * strength;
        *pixel = rgb_from_luma_chroma(output_luma, red, blue);
        *restored += 1;
    }
}

pub fn reconstruct_full_resolution(colorized: &RgbImage, original_gray: &GrayImage) -> RgbImage {
    if colorized.dimensions() == original_gray.dimensions() {
        return colorized.clone();
    }

    let (width, height) = original_gray.dimensions();
    let resized = image::imageops::resize(
        colorized,
        width,
        height,
        image::imageops::FilterType::Lanczos3,
    );
    let pixel_count = (width as usize) * (height as usize);
    let mut red_chroma = vec![0.0_f32; pixel_count];
    let mut blue_chroma = vec![0.0_f32; pixel_count];
    let mut predicted_luma = vec![0.0_f32; pixel_count];

    for (index, pixel) in resized.pixels().enumerate() {
        let luma = rgb_luma(pixel);
        predicted_luma[index] = luma;
        red_chroma[index] = pixel[0] as f32 - luma;
        blue_chroma[index] = pixel[2] as f32 - luma;
    }

    let range_weights: Vec<f32> = (0..=255)
        .map(|difference| {
            let normalized = difference as f32 / 22.0;
            (-0.5 * normalized * normalized).exp()
        })
        .collect();
    let spatial_weights = [0.18_f32, 0.58, 1.0, 0.58, 0.18];
    let mut scratch_red = vec![0.0_f32; pixel_count];
    let mut scratch_blue = vec![0.0_f32; pixel_count];

    joint_bilateral_pass(
        &red_chroma,
        &blue_chroma,
        &mut scratch_red,
        &mut scratch_blue,
        original_gray,
        &range_weights,
        &spatial_weights,
        true,
    );
    joint_bilateral_pass(
        &scratch_red,
        &scratch_blue,
        &mut red_chroma,
        &mut blue_chroma,
        original_gray,
        &range_weights,
        &spatial_weights,
        false,
    );

    let mut output = RgbImage::new(width, height);
    for (x, y, pixel) in output.enumerate_pixels_mut() {
        let index = (y as usize) * (width as usize) + x as usize;
        let original = original_gray.get_pixel(x, y)[0] as f32;
        let left = original_gray.get_pixel(x.saturating_sub(1), y)[0] as f32;
        let right = original_gray.get_pixel((x + 1).min(width - 1), y)[0] as f32;
        let top = original_gray.get_pixel(x, y.saturating_sub(1))[0] as f32;
        let bottom = original_gray.get_pixel(x, (y + 1).min(height - 1))[0] as f32;
        let local_min = original.min(left).min(right).min(top).min(bottom);
        let gradient = (right - left).abs() + (bottom - top).abs();
        let edge = ((gradient - 20.0) / 100.0).clamp(0.0, 1.0);
        let ink = ((112.0 - original) / 112.0).clamp(0.0, 1.0).powf(1.7);
        let adjacent_ink = ((124.0 - local_min) / 124.0).clamp(0.0, 1.0);
        let chroma_energy = red_chroma[index]
            .abs()
            .max(blue_chroma[index].abs())
            .clamp(0.0, 64.0);
        let prediction_weight = 0.44 + chroma_energy / 64.0 * 0.24;
        let target_luma =
            original * (1.0 - prediction_weight) + predicted_luma[index] * prediction_weight;
        let midtone = (1.0 - ((target_luma - 132.0).abs() / 132.0)).clamp(0.0, 1.0);
        let edge_ink = edge * adjacent_ink;
        let mut chroma_boost = 1.12 + midtone * 0.18;
        // Magenta-band pixels (red AND blue above the luma, green below)
        // are the model's mauve cast, already capped by the unified grade
        // at the working resolution — the reconstruction's chroma boost
        // would amplify them back over the visibility threshold (measured
        // 0.28 -> 0.34). No boost for the band: the cap survives.
        if red_chroma[index] > 0.0 && blue_chroma[index] > 0.0 {
            chroma_boost = 1.0;
        }
        let red = red_chroma[index] * chroma_boost;
        let blue = blue_chroma[index] * chroma_boost;
        let green = -(0.2126 * red + 0.0722 * blue) / 0.7152;
        let structural_attenuation = (1.0 - ink * 0.78) * (1.0 - edge_ink * 0.38);
        let gamut_scale =
            chroma_gamut_scale(target_luma, [red, green, blue]) * structural_attenuation;

        pixel[0] = (target_luma + red * gamut_scale).round().clamp(0.0, 255.0) as u8;
        pixel[1] = (target_luma + green * gamut_scale)
            .round()
            .clamp(0.0, 255.0) as u8;
        pixel[2] = (target_luma + blue * gamut_scale).round().clamp(0.0, 255.0) as u8;

        let ink_anchor = ink * 0.94;
        for channel in 0..3 {
            pixel[channel] = (pixel[channel] as f32 * (1.0 - ink_anchor) + original * ink_anchor)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        // Final mauve cap AT FULL RESOLUTION: the working-res cap (unified
        // grade) survives the removed chroma boost, but the luma
        // re-anchoring toward the original gray raises the perceived
        // saturation of midtone magenta pixels (measured 0.28 -> 0.34) and
        // the JBU spread adds more at region edges. Cap the band here, at
        // the exact resolution the user sees — this is the last chroma
        // shaping step in the pipeline.
        let mx = pixel[0].max(pixel[1]).max(pixel[2]) as f32;
        let mn = pixel[0].min(pixel[1]).min(pixel[2]) as f32;
        if mx > mn && mx > 0.0 {
            let hue = if mx == pixel[0] as f32 {
                60.0 * (((pixel[1] as f32 - pixel[2] as f32) / (mx - mn)) % 6.0)
            } else if mx == pixel[1] as f32 {
                60.0 * ((pixel[2] as f32 - pixel[0] as f32) / (mx - mn) + 2.0)
            } else {
                60.0 * ((pixel[0] as f32 - pixel[1] as f32) / (mx - mn) + 4.0)
            };
            let hue_deg = ((hue % 360.0) + 360.0) % 360.0;
            if (280.0..360.0).contains(&hue_deg) && (mx - mn) / mx > 0.28 {
                let new_min = mx * (1.0 - 0.28);
                for channel in 0..3 {
                    let c = pixel[channel] as f32;
                    if c < mx {
                        let t = if mx > mn { (c - mn) / (mx - mn) } else { 0.0 };
                        pixel[channel] =
                            (new_min + t * (mx - new_min)).round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn joint_bilateral_pass(
    source_red: &[f32],
    source_blue: &[f32],
    target_red: &mut [f32],
    target_blue: &mut [f32],
    guide: &GrayImage,
    range_weights: &[f32],
    spatial_weights: &[f32; 5],
    horizontal: bool,
) {
    let (width, height) = guide.dimensions();
    let stride = width as usize;
    for y in 0..height {
        for x in 0..width {
            let mut red_sum = 0.0_f32;
            let mut blue_sum = 0.0_f32;
            let mut weight_sum = 0.0_f32;
            for offset in -2_i32..=2 {
                let sample_x = if horizontal {
                    (x as i32 + offset).clamp(0, width as i32 - 1) as u32
                } else {
                    x
                };
                let sample_y = if horizontal {
                    y
                } else {
                    (y as i32 + offset).clamp(0, height as i32 - 1) as u32
                };
                let path_difference =
                    guide_path_difference(guide, x, y, sample_x, sample_y, horizontal);
                let spatial = spatial_weights[(offset + 2) as usize];
                let weight = spatial * range_weights[path_difference];
                let sample_index = sample_y as usize * stride + sample_x as usize;
                red_sum += source_red[sample_index] * weight;
                blue_sum += source_blue[sample_index] * weight;
                weight_sum += weight;
            }
            let index = y as usize * stride + x as usize;
            target_red[index] = red_sum / weight_sum.max(f32::EPSILON);
            target_blue[index] = blue_sum / weight_sum.max(f32::EPSILON);
        }
    }
}

fn guide_path_difference(
    guide: &GrayImage,
    start_x: u32,
    start_y: u32,
    end_x: u32,
    end_y: u32,
    horizontal: bool,
) -> usize {
    let mut current_x = start_x;
    let mut current_y = start_y;
    let mut previous = guide.get_pixel(current_x, current_y)[0];
    let mut accumulated = 0_u16;

    while current_x != end_x || current_y != end_y {
        if horizontal {
            current_x = if end_x > current_x {
                current_x + 1
            } else {
                current_x - 1
            };
        } else {
            current_y = if end_y > current_y {
                current_y + 1
            } else {
                current_y - 1
            };
        }
        let next = guide.get_pixel(current_x, current_y)[0];
        accumulated = accumulated.saturating_add(previous.abs_diff(next) as u16);
        previous = next;
    }

    accumulated.min(255) as usize
}

fn rgb_luma(pixel: &Rgb<u8>) -> f32 {
    0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32
}

fn chroma_gamut_scale(luma: f32, chroma: [f32; 3]) -> f32 {
    let mut scale = 1.0_f32;
    for value in chroma {
        if value > 0.0 {
            scale = scale.min((255.0 - luma) / value);
        } else if value < 0.0 {
            scale = scale.min(luma / -value);
        }
    }
    scale.clamp(0.0, 1.0)
}

pub fn restore_lines(image: &mut RgbImage, line_mask: &GrayImage, strength: f32) {
    let strength = strength.clamp(0.0, 2.0);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
        if ink <= 0.0 {
            continue;
        }
        let keep = (1.0 - ink * strength).clamp(0.0, 1.0);
        pixel[0] = (pixel[0] as f32 * keep).round() as u8;
        pixel[1] = (pixel[1] as f32 * keep).round() as u8;
        pixel[2] = (pixel[2] as f32 * keep).round() as u8;
    }
}

pub fn clean_speech_bubbles(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    bubble_mask: &GrayImage,
    bubble_pixels_restored: &mut u64,
) {
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if bubble_mask.get_pixel(x, y)[0] == 0 {
            continue;
        }
        let original = original_gray.get_pixel(x, y)[0];
        let chroma = pixel[0].max(pixel[1]).max(pixel[2]) - pixel[0].min(pixel[1]).min(pixel[2]);
        let luma = (pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3;
        if chroma >= 14 && luma < 225 {
            continue;
        }
        *bubble_pixels_restored += 1;
        if original < 200 {
            continue;
        }
        let paper = original.max(250);
        *pixel = Rgb([paper, paper, paper]);
    }
}

/// Last-mile safety layer. It runs after every color-producing pass and only
/// restores protected pixels that are evidenced as bubble paper, bright
/// paper, or ink. A broad/uncertain protected mask therefore cannot erase
/// legitimate midtone material color.
fn restore_protected_paper_and_ink(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
    protected_mask: &GrayImage,
) -> u64 {
    let mut restored = 0;
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if protected_mask.get_pixel(x, y)[0] == 0 {
            continue;
        }
        let original = original_gray.get_pixel(x, y)[0];
        let bubble = bubble_mask.get_pixel(x, y)[0] > 0;
        let ink = line_mask.get_pixel(x, y)[0] > 0 || original < 72;
        let paper = original >= 228;
        if !bubble && !ink && !paper {
            continue;
        }
        let neutral = if bubble {
            let chroma =
                pixel[0].max(pixel[1]).max(pixel[2]) - pixel[0].min(pixel[1]).min(pixel[2]);
            let luma = (pixel[0] as u16 + pixel[1] as u16 + pixel[2] as u16) / 3;
            if original >= 72 && chroma >= 14 && luma < 225 {
                continue;
            }
            original.max(250)
        } else {
            original
        };
        *pixel = Rgb([neutral, neutral, neutral]);
        restored += 1;
    }
    restored
}

/// Re-apply the non-negotiable ink, text, bubble-paper, and bright-paper
/// invariants after full-resolution reconstruction and any delivery-time
/// region locks. This wrapper is intentionally narrow: protected midtone
/// artwork is left colored unless it is evidenced as paper or ink.
pub fn restore_delivery_protected_regions(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
    protected_mask: &GrayImage,
) -> u64 {
    let restored = restore_protected_paper_and_ink(
        image,
        original_gray,
        line_mask,
        bubble_mask,
        protected_mask,
    );
    enforce_strong_ink(image, line_mask);
    restored
}

/// Strong manga ink is an invariant, not a style choice. The adaptive pass
/// keeps anti-aliased edges, while this final floor removes residual colored
/// ink left by inference or sharpening.
fn enforce_strong_ink(image: &mut RgbImage, line_mask: &GrayImage) {
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
        if ink >= 0.37 {
            // Proportional enforcement: solid ink (ink=1.0) → full black,
            // soft edges (ink=0.37) → partially darkened, anti-aliased look preserved
            let blend = (ink * 1.5).clamp(0.0, 1.0);
            let target = 32.0 + (1.0 - ink) * 180.0;
            for c in 0..3 {
                let v = pixel[c] as f32;
                pixel[c] = (v * (1.0 - blend) + target * blend).round() as u8;
            }
        }
    }
}

pub fn reduce_color_near_ink(
    image: &mut RgbImage,
    line_mask: &GrayImage,
    bleed_pixels_corrected: &mut u64,
) {
    let (w, h) = image.dimensions();
    if w < 3 || h < 3 {
        return;
    }
    let w_i32 = w as i32;
    let h_i32 = h as i32;

    let source: Vec<Rgb<u8>> = image.pixels().copied().collect();

    let radius: i32 = 3;
    let spatial_sigma: f64 = 1.5;
    let range_sigma: f64 = 50.0;

    for y in 0..h_i32 {
        for x in 0..w_i32 {
            let idx = (y * w_i32 + x) as usize;
            if line_mask.get_pixel(x as u32, y as u32)[0] > 0 {
                continue;
            }

            let mut near_ink = false;
            'outer: for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w_i32 || ny >= h_i32 {
                        continue;
                    }
                    if line_mask.get_pixel(nx as u32, ny as u32)[0] > 0 {
                        near_ink = true;
                        break 'outer;
                    }
                }
            }
            if !near_ink {
                continue;
            }

            let center = source[idx];
            let mut sum = [0.0f64; 3];
            let mut total_weight: f64 = 0.0;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = x + dx;
                    let ny = y + dy;
                    if nx < 0 || ny < 0 || nx >= w_i32 || ny >= h_i32 {
                        continue;
                    }
                    let nidx = (ny * w_i32 + nx) as usize;
                    if line_mask.get_pixel(nx as u32, ny as u32)[0] > 0 {
                        continue;
                    }
                    let neighbor = source[nidx];
                    let spatial_dist = ((dx * dx + dy * dy) as f64).sqrt();
                    let spatial_weight = (-0.5 * (spatial_dist / spatial_sigma).powi(2)).exp();

                    let mut color_dist_sq: f64 = 0.0;
                    for c in 0..3 {
                        let diff = center[c] as f64 - neighbor[c] as f64;
                        color_dist_sq += diff * diff;
                    }
                    let color_dist = color_dist_sq.sqrt();
                    let range_weight = (-0.5 * (color_dist / range_sigma).powi(2)).exp();

                    let weight = spatial_weight * range_weight;
                    for c in 0..3 {
                        sum[c] += neighbor[c] as f64 * weight;
                    }
                    total_weight += weight;
                }
            }

            if total_weight > 0.0 {
                image.put_pixel(
                    x as u32,
                    y as u32,
                    Rgb([
                        (sum[0] / total_weight).round().clamp(0.0, 255.0) as u8,
                        (sum[1] / total_weight).round().clamp(0.0, 255.0) as u8,
                        (sum[2] / total_weight).round().clamp(0.0, 255.0) as u8,
                    ]),
                );
                *bleed_pixels_corrected += 1;
            }
        }
    }
}

// ── 3D LUT Manga Color Grading ──────────────────────────────────────────
const LUT_SIZE: usize = 17;

struct MangaColorLUT {
    table: Vec<[[Rgb<u8>; LUT_SIZE]; LUT_SIZE]>,
}

impl MangaColorLUT {
    fn new(style: &StyleProfile) -> Self {
        let contrast = style.contrast.clamp(0.9, 1.15);
        let sat = style.saturation.clamp(0.7, 1.3);
        let bright = style.brightness * 15.0;
        let mut table = vec![[[Rgb([0; 3]); LUT_SIZE]; LUT_SIZE]; LUT_SIZE];

        for (ri, red_slice) in table.iter_mut().enumerate() {
            for (gi, green_slice) in red_slice.iter_mut().enumerate() {
                for (bi, entry) in green_slice.iter_mut().enumerate() {
                    let r = ri as f32 * 16.0;
                    let g = gi as f32 * 16.0;
                    let b = bi as f32 * 16.0;

                    // Subtle desaturate toward grayscale (natural manga look)
                    let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                    let sr = gray + (r - gray) * sat;
                    let sg = gray + (g - gray) * sat;
                    let sb = gray + (b - gray) * sat;

                    // Gentle contrast around 128
                    let cr = ((sr - 128.0) * contrast + 128.0 + bright).clamp(0.0, 255.0) as u8;
                    let cg = ((sg - 128.0) * contrast + 128.0 + bright).clamp(0.0, 255.0) as u8;
                    let cb = ((sb - 128.0) * contrast + 128.0 + bright).clamp(0.0, 255.0) as u8;

                    *entry = Rgb([cr, cg, cb]);
                }
            }
        }

        MangaColorLUT { table }
    }
}

fn apply_palette_harmony(
    image: &mut RgbImage,
    style: &StyleProfile,
    line_mask: &GrayImage,
    strength: f32,
    model_first: bool,
) {
    if style.palette.is_empty() {
        return;
    }
    let strength = strength.clamp(0.0, 0.5);
    let palette = style
        .palette
        .iter()
        .map(|color| {
            let r = color.r as f32;
            let g = color.g as f32;
            let b = color.b as f32;
            let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            (luma, b - luma, r - luma, r.max(g).max(b) - r.min(g).min(b))
        })
        .collect::<Vec<_>>();

    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
        if ink > 0.72 {
            continue;
        }
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let cb = b - luma;
        let cr = r - luma;
        let saturation = r.max(g).max(b) - r.min(g).min(b);
        let source_is_neutral = saturation < 8.0;

        let target = palette
            .iter()
            .filter(|entry| (entry.3 < 10.0) == source_is_neutral)
            .min_by(|left, right| {
                let distance = |entry: &&(f32, f32, f32, f32)| {
                    let dl = (luma - entry.0) * 0.22;
                    let dcb = cb - entry.1;
                    let dcr = cr - entry.2;
                    dl * dl + dcb * dcb + dcr * dcr
                };
                distance(left).total_cmp(&distance(right))
            })
            .or_else(|| {
                palette.iter().min_by(|left, right| {
                    let distance = |entry: &&(f32, f32, f32, f32)| {
                        let dl = (luma - entry.0) * 0.22;
                        let dcb = cb - entry.1;
                        let dcr = cr - entry.2;
                        dl * dl + dcb * dcb + dcr * dcr
                    };
                    distance(left).total_cmp(&distance(right))
                })
            });
        let Some((_, target_cb, target_cr, _)) = target else {
            continue;
        };

        let mut pull = strength * (1.0 - ink).powi(2);
        // Model-first: the palette pull yields to existing chroma — neutral
        // pixels keep the harmony nudge, model-colored pixels stay untouched.
        if model_first {
            let chroma_gate = ((saturation - 8.0) / 18.0).clamp(0.0, 1.0);
            pull *= 1.0 - chroma_gate;
        }
        let harmonized_cb = cb + (target_cb - cb) * pull;
        let harmonized_cr = cr + (target_cr - cr) * pull;
        pixel[0] = (luma + harmonized_cr).round().clamp(0.0, 255.0) as u8;
        pixel[2] = (luma + harmonized_cb).round().clamp(0.0, 255.0) as u8;
        pixel[1] = ((luma - 0.2126 * pixel[0] as f32 - 0.0722 * pixel[2] as f32) / 0.7152)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
}

/// Complete sparse model chroma inside manga regions without changing geometry.
///
/// The model's confident colors are treated as seeds. Chroma diffuses through
/// similar grayscale pixels, while ink and strong luminance transitions act as
/// barriers. Remaining neutral material receives a restrained luminance-matched
/// palette prior. Original luminance is retained, so line art and screentones
/// remain available to later reconstruction stages.
pub fn complete_material_chroma(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    exclusion_mask: Option<&GrayImage>,
    style: &StyleProfile,
    completed: &mut u64,
    _refined: &mut u64,
) {
    let (width, height) = image.dimensions();
    if width == 0
        || height == 0
        || original_gray.dimensions() != (width, height)
        || line_mask.dimensions() != (width, height)
    {
        return;
    }

    let len = width as usize * height as usize;
    let source_pixels = image.pixels().copied().collect::<Vec<_>>();
    let mut luma = vec![0.0f32; len];
    let mut red_chroma = vec![0.0f32; len];
    let mut blue_chroma = vec![0.0f32; len];
    let mut confidence = vec![0.0f32; len];
    let mut fixed_seed = vec![false; len];
    let mut blocked = vec![false; len];
    let mut gray = vec![0u8; len];

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let pixel = source_pixels[index];
            let y_value =
                0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
            let chroma =
                pixel[0].max(pixel[1]).max(pixel[2]) - pixel[0].min(pixel[1]).min(pixel[2]);
            let ink = line_mask.get_pixel(x, y)[0];
            let excluded = exclusion_mask
                .map(|mask| mask.get_pixel(x, y)[0])
                .unwrap_or(0);
            let is_blocked = ink > 112 || excluded > 96;
            let seed_confidence = ((chroma as f32 - 5.0) / 24.0).clamp(0.0, 1.0);

            luma[index] = y_value;
            red_chroma[index] = pixel[0] as f32 - y_value;
            blue_chroma[index] = pixel[2] as f32 - y_value;
            confidence[index] = if is_blocked { 0.0 } else { seed_confidence };
            fixed_seed[index] = seed_confidence >= 0.48 && !is_blocked;
            blocked[index] = is_blocked;
            gray[index] = original_gray.get_pixel(x, y)[0];
        }
    }

    let similarity = (0..=255)
        .map(|difference| (-(difference as f32) / 20.0).exp())
        .collect::<Vec<_>>();
    let mut next_red = red_chroma.clone();
    let mut next_blue = blue_chroma.clone();
    let mut next_confidence = confidence.clone();
    const ITERATIONS: usize = 18;
    const NEIGHBORS: [(i32, i32); 8] = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (1, -1),
        (-1, 1),
        (1, 1),
    ];

    for _ in 0..ITERATIONS {
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                if blocked[index] || fixed_seed[index] {
                    next_red[index] = red_chroma[index];
                    next_blue[index] = blue_chroma[index];
                    next_confidence[index] = confidence[index];
                    continue;
                }

                let mut red_sum = 0.0f32;
                let mut blue_sum = 0.0f32;
                let mut weight_sum = 0.0f32;
                let mut support_sum = 0.0f32;
                for (dx, dy) in NEIGHBORS {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    let neighbor = (ny as u32 * width + nx as u32) as usize;
                    if blocked[neighbor] || confidence[neighbor] <= 0.005 {
                        continue;
                    }
                    let difference = gray[index].abs_diff(gray[neighbor]) as usize;
                    let diagonal = if dx != 0 && dy != 0 { 0.707 } else { 1.0 };
                    let weight = confidence[neighbor] * similarity[difference] * diagonal;
                    red_sum += red_chroma[neighbor] * weight;
                    blue_sum += blue_chroma[neighbor] * weight;
                    weight_sum += weight;
                    support_sum += similarity[difference] * diagonal;
                }

                if weight_sum > 0.01 {
                    let proposal_red = red_sum / weight_sum;
                    let proposal_blue = blue_sum / weight_sum;
                    let proposal_confidence = (weight_sum / support_sum.max(0.01)).clamp(0.0, 1.0);
                    let spread =
                        (proposal_confidence * (1.0 - confidence[index]) * 0.58).clamp(0.0, 0.58);
                    next_red[index] =
                        red_chroma[index] + (proposal_red - red_chroma[index]) * spread;
                    next_blue[index] =
                        blue_chroma[index] + (proposal_blue - blue_chroma[index]) * spread;
                    next_confidence[index] = confidence[index].max(proposal_confidence * 0.82);
                } else {
                    next_red[index] = red_chroma[index];
                    next_blue[index] = blue_chroma[index];
                    next_confidence[index] = confidence[index];
                }
            }
        }
        std::mem::swap(&mut red_chroma, &mut next_red);
        std::mem::swap(&mut blue_chroma, &mut next_blue);
        std::mem::swap(&mut confidence, &mut next_confidence);
    }

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if blocked[index] {
                continue;
            }

            let old_chroma = source_pixels[index][0]
                .max(source_pixels[index][1])
                .max(source_pixels[index][2])
                - source_pixels[index][0]
                    .min(source_pixels[index][1])
                    .min(source_pixels[index][2]);
            let propagated = confidence[index];
            let target_red = red_chroma[index];
            let target_blue = blue_chroma[index];

            let apply_strength = if fixed_seed[index] {
                1.0
            } else {
                (0.28 + propagated * 0.72).clamp(0.28, 0.92)
            };
            let red = rgb_from_luma_chroma(
                luma[index],
                target_red * apply_strength,
                target_blue * apply_strength,
            );
            image.put_pixel(x, y, red);

            let new_pixel = image.get_pixel(x, y);
            let new_chroma = new_pixel[0].max(new_pixel[1]).max(new_pixel[2])
                - new_pixel[0].min(new_pixel[1]).min(new_pixel[2]);
            if old_chroma < 10 && new_chroma >= 10 {
                *completed += 1;
            }
        }
    }

    refine_skin_and_material_light(image, original_gray, line_mask, exclusion_mask, style);
}

fn rgb_from_luma_chroma(luma: f32, red_chroma: f32, blue_chroma: f32) -> Rgb<u8> {
    let green_chroma = (-0.2126 * red_chroma - 0.0722 * blue_chroma) / 0.7152;
    let mut scale = 1.0f32;
    for chroma in [red_chroma, green_chroma, blue_chroma] {
        if chroma > 0.0 {
            scale = scale.min((255.0 - luma) / chroma);
        } else if chroma < 0.0 {
            scale = scale.min(luma / -chroma);
        }
    }
    let red = luma + red_chroma * scale.clamp(0.0, 1.0);
    let blue = luma + blue_chroma * scale.clamp(0.0, 1.0);
    let green = luma + green_chroma * scale.clamp(0.0, 1.0);
    Rgb([
        red.round().clamp(0.0, 255.0) as u8,
        green.round().clamp(0.0, 255.0) as u8,
        blue.round().clamp(0.0, 255.0) as u8,
    ])
}

fn refine_skin_and_material_light(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    exclusion_mask: Option<&GrayImage>,
    style: &StyleProfile,
) {
    let (width, height) = image.dimensions();
    let source = image.clone();
    for y in 0..height {
        for x in 0..width {
            let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
            let excluded = exclusion_mask
                .map(|mask| mask.get_pixel(x, y)[0])
                .unwrap_or(0);
            if ink > 0.44 || excluded > 96 {
                continue;
            }
            let pixel = source.get_pixel(x, y);
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let y_value = 0.2126 * r + 0.7152 * g + 0.0722 * b;
            let red_delta = r - g;
            let blue_delta = g - b;
            let skin = y_value > 72.0
                && y_value < 246.0
                && red_delta > 6.0
                && red_delta < 82.0
                && blue_delta > -4.0
                && blue_delta < 58.0;

            let original = original_gray.get_pixel(x, y)[0] as f32;
            let shadow = ((154.0 - original) / 154.0).clamp(0.0, 1.0);
            let highlight = ((original - 205.0) / 50.0).clamp(0.0, 1.0);
            let shade =
                -shadow * style.shadow_strength * 5.0 + highlight * style.highlight_strength * 2.5;
            let mut out = [r + shade, g + shade, b + shade];

            if skin {
                let warmth = (1.0 - shadow * 0.45) * (1.0 - ink) * 2.8;
                out[0] += warmth;
                out[1] += warmth * 0.28;
                out[2] -= warmth * 0.38;
            } else if y_value < 108.0 {
                out[0] += style.shadow_strength * 1.1;
                out[1] += style.shadow_strength * 0.15;
                out[2] += style.shadow_strength * 0.7;
            }

            image.put_pixel(
                x,
                y,
                Rgb([
                    out[0].round().clamp(0.0, 255.0) as u8,
                    out[1].round().clamp(0.0, 255.0) as u8,
                    out[2].round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
}

fn lut_lookup(lut: &MangaColorLUT, r: u8, g: u8, b: u8) -> Rgb<u8> {
    let rf = r as f32 / 16.0;
    let gf = g as f32 / 16.0;
    let bf = b as f32 / 16.0;
    let r0 = (rf.floor() as usize).min(LUT_SIZE - 2);
    let g0 = (gf.floor() as usize).min(LUT_SIZE - 2);
    let b0 = (bf.floor() as usize).min(LUT_SIZE - 2);
    let rd = rf - r0 as f32;
    let gd = gf - g0 as f32;
    let bd = bf - b0 as f32;

    let c000 = lut.table[r0][g0][b0];
    let c001 = lut.table[r0][g0][b0 + 1];
    let c010 = lut.table[r0][g0 + 1][b0];
    let c011 = lut.table[r0][g0 + 1][b0 + 1];
    let c100 = lut.table[r0 + 1][g0][b0];
    let c101 = lut.table[r0 + 1][g0][b0 + 1];
    let c110 = lut.table[r0 + 1][g0 + 1][b0];
    let c111 = lut.table[r0 + 1][g0 + 1][b0 + 1];

    let lerp = |left: f32, right: f32, amount: f32| left + (right - left) * amount;
    let channel = |index: usize| {
        let r0g0 = lerp(c000[index] as f32, c001[index] as f32, bd);
        let r0g1 = lerp(c010[index] as f32, c011[index] as f32, bd);
        let r1g0 = lerp(c100[index] as f32, c101[index] as f32, bd);
        let r1g1 = lerp(c110[index] as f32, c111[index] as f32, bd);
        let r0 = lerp(r0g0, r0g1, gd);
        let r1 = lerp(r1g0, r1g1, gd);
        lerp(r0, r1, rd).round().clamp(0.0, 255.0) as u8
    };

    Rgb([channel(0), channel(1), channel(2)])
}

fn apply_lut_grading(
    image: &mut RgbImage,
    lut: &MangaColorLUT,
    line_mask: &GrayImage,
    original: &GrayImage,
) {
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
        if ink > 0.85 {
            continue;
        }

        let mapped = lut_lookup(lut, pixel[0], pixel[1], pixel[2]);
        let orig_luma = original.get_pixel(x, y)[0] as f32;
        let new_luma =
            0.2126 * mapped[0] as f32 + 0.7152 * mapped[1] as f32 + 0.0722 * mapped[2] as f32;
        let anchor = 0.10;
        let diff = orig_luma - new_luma;

        pixel[0] = (mapped[0] as f32 + diff * anchor).clamp(0.0, 255.0) as u8;
        pixel[1] = (mapped[1] as f32 + diff * anchor).clamp(0.0, 255.0) as u8;
        pixel[2] = (mapped[2] as f32 + diff * anchor).clamp(0.0, 255.0) as u8;
    }
}

// ── Edge-Aware Color Smoothness ──────────────────────────────────────────
fn edge_aware_smooth(image: &mut RgbImage, line_mask: &GrayImage) {
    let (w, h) = image.dimensions();
    let src: Vec<Rgb<u8>> = image.pixels().copied().collect();
    let spatial_sigma = 2.5f32;
    let range_sigma = 35.0f32;
    let radius: i32 = 3;

    let spatial_w: Vec<f32> = (0..=radius)
        .map(|d| (-(d * d) as f32 / (2.0 * spatial_sigma * spatial_sigma)).exp())
        .collect();

    for y in 0..h {
        for x in 0..w {
            let idx = (y * w + x) as usize;
            let ink = line_mask.get_pixel(x, y)[0];

            if ink > 200 {
                continue;
            }

            let center = src[idx];
            let center_luma =
                0.2126 * center[0] as f32 + 0.7152 * center[1] as f32 + 0.0722 * center[2] as f32;
            let mut sum = [0.0f32; 3];
            let mut total_w = 0.0f32;

            for dy in -radius..=radius {
                for dx in -radius..=radius {
                    let nx = (x as i32 + dx).clamp(0, w as i32 - 1) as u32;
                    let ny = (y as i32 + dy).clamp(0, h as i32 - 1) as u32;
                    let n_id = (ny * w + nx) as usize;

                    let n_ink = line_mask.get_pixel(nx, ny)[0] as f32 / 255.0;
                    if n_ink > 0.85 {
                        continue;
                    }
                    let ink_w = (1.0 - n_ink).powi(3);

                    let n = src[n_id];
                    let n_luma = 0.2126 * n[0] as f32 + 0.7152 * n[1] as f32 + 0.0722 * n[2] as f32;

                    let dist = dx.abs().max(dy.abs()) as usize;
                    let sw = spatial_w[dist];

                    let r_diff = (center_luma - n_luma).abs();
                    let rw = (-r_diff * r_diff / (2.0 * range_sigma * range_sigma)).exp();

                    let weight = sw * rw * ink_w;
                    sum[0] += n[0] as f32 * weight;
                    sum[1] += n[1] as f32 * weight;
                    sum[2] += n[2] as f32 * weight;
                    total_w += weight;
                }
            }

            if total_w > 0.0 {
                image.put_pixel(
                    x,
                    y,
                    Rgb([
                        (sum[0] / total_w).clamp(0.0, 255.0) as u8,
                        (sum[1] / total_w).clamp(0.0, 255.0) as u8,
                        (sum[2] / total_w).clamp(0.0, 255.0) as u8,
                    ]),
                );
            }
        }
    }
}

/// Blend original grayscale back into screentone regions so the dot/line
/// patterns remain visible instead of being painted over with flat color.
/// Dark-zone chroma control lives in the unified grade (per-luma targets),
/// so this pass only restores the dot pattern's luma.
pub fn preserve_screentones(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    tone_mask: &GrayImage,
    strength: f32,
    count: &mut u64,
) {
    let blend = strength.clamp(0.2, 1.0);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        if tone_mask.get_pixel(x, y)[0] == 0 {
            continue;
        }
        let original_luma = original_gray.get_pixel(x, y)[0] as f32;
        let colored_luma =
            0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
        let tone_strength = blend * (0.52 + 0.16 * (1.0 - original_luma / 255.0));
        let target_luma =
            colored_luma + (original_luma - colored_luma) * tone_strength.clamp(0.0, 0.72);
        let chroma = [
            pixel[0] as f32 - colored_luma,
            pixel[1] as f32 - colored_luma,
            pixel[2] as f32 - colored_luma,
        ];
        let mut gamut_scale = 1.0f32;
        for value in chroma {
            if value > 0.0 {
                gamut_scale = gamut_scale.min((255.0 - target_luma) / value);
            } else if value < 0.0 {
                gamut_scale = gamut_scale.min(target_luma / -value);
            }
        }
        let gamut_scale = gamut_scale.clamp(0.0, 1.0);
        // For grayscale manga the original has zero chroma in screentone
        // regions.  Dampen the coloured image's chroma proportionally to
        // the luma-restoration strength so model-invented colour noise
        // (dots, squares) fades toward the original grey while strong
        // legitimate chroma from dense coloured regions survives at
        // reduced intensity.  The 0.45 factor keeps the dampening gentle
        // (at max tone_strength 0.72 the chroma is reduced by ~32 %).
        let chroma_scale = 1.0 - tone_strength.clamp(0.0, 0.72) * 0.45;
        for channel in 0..3 {
            pixel[channel] = (target_luma + chroma[channel] * gamut_scale * chroma_scale)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        *count += 1;
    }
}

/// Remove color speckle noise (the model's dot artifacts): small connected
/// components of strong color in sparse neighborhoods are blended into their
/// surroundings. Dense patterns (halftone/screentone fields, real colored
/// regions) are left untouched — the dot TEXTURE and its warm tint survive.
pub fn remove_color_specks(image: &mut RgbImage, line_mask: &GrayImage) {
    let (w, h) = image.dimensions();
    if w < 3 || h < 3 || line_mask.dimensions() != (w, h) {
        return;
    }
    const DENSITY_WINDOW: u32 = 15;
    const MAX_DENSITY: f32 = 0.18;
    let idx = |x: u32, y: u32| (y * w + x) as usize;

    // Strong-color mask (saturation > 0.35). Colored pixels ON ink are
    // included on purpose: the source is grayscale manga, so any strong
    // chroma sitting on an ink line is the model's invented noise — the
    // "dots on the lines" — and must be cleaned like any other speck (the
    // neutral ink around it becomes the replacement background). The old
    // exemption ("colored pixels on ink are art") is what let colored
    // specks survive ON every line.
    let mut strong = vec![false; (w * h) as usize];
    for (i, pixel) in image.pixels().enumerate() {
        let mx = pixel[0].max(pixel[1]).max(pixel[2]) as u16;
        let mn = pixel[0].min(pixel[1]).min(pixel[2]) as u16;
        strong[i] = mx > 0 && (mx - mn) as f32 / mx as f32 > 0.35;
    }
    // Label strong-color components (8-connectivity, BFS).
    let mut labels = vec![0u32; (w * h) as usize];
    let mut sizes: Vec<u32> = Vec::new();
    let mut comp_on_ink: Vec<bool> = Vec::new();
    let mut next_label = 1u32;
    for start in 0..(w * h) as usize {
        if !strong[start] || labels[start] != 0 {
            continue;
        }
        let mut queue = vec![start];
        labels[start] = next_label;
        let mut size = 0u32;
        let mut touches_ink = false;
        while let Some(cur) = queue.pop() {
            size += 1;
            let (cx, cy) = (cur % w as usize, cur / w as usize);
            if line_mask.get_pixel(cx as u32, cy as u32)[0] > 184 {
                touches_ink = true;
            }
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (cx as i32 + dx, cy as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nidx = idx(nx as u32, ny as u32);
                    if strong[nidx] && labels[nidx] == 0 {
                        labels[nidx] = next_label;
                        queue.push(nidx);
                    }
                }
            }
        }
        sizes.push(size);
        comp_on_ink.push(touches_ink);
        next_label += 1;
    }
    // Mean color and luma of each component.
    let mut component_mean: Vec<[f32; 3]> = vec![[0.0; 3]; sizes.len()];
    for (i, &label) in labels.iter().enumerate() {
        if label == 0 {
            continue;
        }
        let p = image.get_pixel((i % w as usize) as u32, (i / w as usize) as u32);
        let mean = &mut component_mean[(label - 1) as usize];
        mean[0] += p[0] as f32;
        mean[1] += p[1] as f32;
        mean[2] += p[2] as f32;
    }
    for (label, mean) in component_mean.iter_mut().enumerate() {
        let n = sizes[label].max(1) as f32;
        mean[0] /= n;
        mean[1] /= n;
        mean[2] /= n;
    }
    let component_luma = |label: usize| -> f32 {
        let m = component_mean[label];
        0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]
    };
    // A component is "small" when it is tiny, or when it lives in the dark
    // range where the model's chroma noise forms short chains (dark shading
    // specks); real dark features (hair, shadows) are far larger. Halftone
    // fields are mid-tone and always fail the density gate anyway. Ink-bound
    // components may form longer chains along a line (the dot trace), so
    // they get a longer leash — their color is still checked against the
    // (neutral) ink background. Midtone components up to 100px are also
    // examined: the model scatters isolated purple/red specks across the
    // page's midtones (measured: 0.2% of a page in 7-100px components, 60%
    // of them foreign to their background), and the density/foreign gates
    // decide — dense texture and legit colored details survive, isolated
    // specks are blended into their background.
    let small = |label: u32| -> bool {
        let size = sizes[(label - 1) as usize];
        let luma = component_luma((label - 1) as usize);
        let limit = if comp_on_ink[(label - 1) as usize] {
            80
        } else if luma < 90.0 {
            40
        } else {
            100
        };
        size <= limit
    };

    // A speck component is removed when it does not belong to its
    // neighborhood: either the strong-color density around it is low
    // (sparse noise on paper), or its mean color differs strongly from the
    // local background (a foreign dot inside a colored region). Halftone
    // fields are dense AND match their surroundings — they survive intact.
    let mut speck = vec![false; (w * h) as usize];
    let mut speck_bg = vec![[0.0f32; 3]; (w * h) as usize];
    let mut speck_bg_n = vec![0u32; (w * h) as usize];
    for (i, &label) in labels.iter().enumerate() {
        if label == 0 || !small(label) {
            continue;
        }
        let (cx, cy) = (i % w as usize, i / w as usize);
        let half = (DENSITY_WINDOW / 2) as i32;
        let (x0, x1) = (
            (cx as i32 - half).max(0),
            (cx as i32 + half).min(w as i32 - 1),
        );
        let (y0, y1) = (
            (cy as i32 - half).max(0),
            (cy as i32 + half).min(h as i32 - 1),
        );
        let mut colored = 0u32;
        let mut total = 0u32;
        let mut bg = [0.0f32; 3];
        let mut bg_n = 0u32;
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                total += 1;
                if strong[idx(xx as u32, yy as u32)] {
                    colored += 1;
                } else {
                    let p = image.get_pixel(xx as u32, yy as u32);
                    bg[0] += p[0] as f32;
                    bg[1] += p[1] as f32;
                    bg[2] += p[2] as f32;
                    bg_n += 1;
                }
            }
        }
        if total == 0 {
            continue;
        }
        let density = colored as f32 / (total as f32);
        let mean = component_mean[(label - 1) as usize];
        // Dark components tolerate denser speck fields; midtone keeps the
        // strict gate.
        let dark = component_luma((label - 1) as usize) < 90.0;
        let max_density = if dark { 0.30 } else { MAX_DENSITY };
        let bg_mean = if bg_n > 0 {
            [
                bg[0] / bg_n as f32,
                bg[1] / bg_n as f32,
                bg[2] / bg_n as f32,
            ]
        } else {
            mean
        };
        let mut foreign = false;
        let mut dist = 0.0f32;
        let mut threshold = 0.0f32;
        if bg_n > 0 {
            dist = ((mean[0] - bg_mean[0]).powi(2)
                + (mean[1] - bg_mean[1]).powi(2)
                + (mean[2] - bg_mean[2]).powi(2))
            .sqrt();
            // Foreign threshold scales with background luma (dark fields
            // need less distance to count as foreign).
            let bg_luma = 0.2126 * bg_mean[0] + 0.7152 * bg_mean[1] + 0.0722 * bg_mean[2];
            threshold = 55.0 * ((bg_luma + 30.0) / 120.0).clamp(0.4, 1.3);
            foreign = dist > threshold;
        }
        let (cx, cy) = (i % w as usize, i / w as usize);
        let on_line = line_mask.get_pixel(cx as u32, cy as u32)[0] > 184;
        let mut remove = false;
        if on_line && bg_n > 0 {
            // Chroma sitting ON an ink line is the model's invention (the
            // source page is grayscale), so the density gate protects
            // nothing — the neighbors are other specks on the same line.
            // The only question is whether the color matches the line's
            // own neutral color: a tight foreign threshold (40%% of the
            // normal one, floored) removes the dim colored chains while
            // legit colored regions keep their line-edge pixels.
            remove = dist > (threshold * 0.4).max(18.0);
        } else if density < max_density || foreign {
            remove = true;
        }
        if remove {
            speck[i] = true;
            speck_bg[i] = bg_mean;
            speck_bg_n[i] = bg_n;
        }
    }

    // Pass 2 — ON-LINE pixels, independently of component size. A speck
    // attached to a legit colored region (8-connected through the strong
    // mask) rides the region's giant component past the small() gate
    // above, so it is never examined. Chroma on an ink line is invented
    // (grayscale source): examine every strong on-line pixel directly and
    // remove it when it does not match the line's own neutral color.
    for (i, &is_strong) in strong.iter().enumerate() {
        if !is_strong || speck[i] {
            continue;
        }
        let (cx, cy) = (i % w as usize, i / w as usize);
        if line_mask.get_pixel(cx as u32, cy as u32)[0] <= 184 {
            continue;
        }
        let half = (DENSITY_WINDOW / 2) as i32;
        let (x0, x1) = (
            (cx as i32 - half).max(0),
            (cx as i32 + half).min(w as i32 - 1),
        );
        let (y0, y1) = (
            (cy as i32 - half).max(0),
            (cy as i32 + half).min(h as i32 - 1),
        );
        let mut bg = [0.0f32; 3];
        let mut bg_n = 0u32;
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let nidx = idx(xx as u32, yy as u32);
                if strong[nidx] {
                    continue;
                }
                let p = image.get_pixel(xx as u32, yy as u32);
                bg[0] += p[0] as f32;
                bg[1] += p[1] as f32;
                bg[2] += p[2] as f32;
                bg_n += 1;
            }
        }
        if bg_n == 0 {
            // Fully colored neighborhood on the line: the invented chroma
            // is dense, so blend the pixel toward the line (the 3x3
            // fallback in the replacement step resolves the target).
            speck[i] = true;
            speck_bg_n[i] = 0;
            continue;
        }
        let bg_mean = [
            bg[0] / bg_n as f32,
            bg[1] / bg_n as f32,
            bg[2] / bg_n as f32,
        ];
        let p = image.get_pixel(cx as u32, cy as u32);
        let dist = ((p[0] as f32 - bg_mean[0]).powi(2)
            + (p[1] as f32 - bg_mean[1]).powi(2)
            + (p[2] as f32 - bg_mean[2]).powi(2))
        .sqrt();
        let bg_luma = 0.2126 * bg_mean[0] + 0.7152 * bg_mean[1] + 0.0722 * bg_mean[2];
        let threshold = 55.0 * ((bg_luma + 30.0) / 120.0).clamp(0.4, 1.3);
        if dist > (threshold * 0.4).max(12.0) {
            speck[i] = true;
            speck_bg[i] = bg_mean;
            speck_bg_n[i] = bg_n;
        }
    }
    // Replace speck pixels with the window background mean (the same value
    // used for the foreign check, so the replacement cannot re-create a
    // strong pixel). The 3x3 non-strong mean is only a fallback when the
    // window had no background at all.
    let mut work = image.clone();
    for (i, &is_speck) in speck.iter().enumerate() {
        if !is_speck {
            continue;
        }
        let (cx, cy) = (i % w as usize, i / w as usize);
        let mut target = speck_bg[i];
        if speck_bg_n[i] == 0 {
            let mut sum = [0.0f32; 3];
            let mut n = 0u32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (cx as i32 + dx, cy as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nidx = idx(nx as u32, ny as u32);
                    if strong[nidx] {
                        continue; // non-strong neighbors only
                    }
                    let neighbor = image.get_pixel(nx as u32, ny as u32);
                    sum[0] += neighbor[0] as f32;
                    sum[1] += neighbor[1] as f32;
                    sum[2] += neighbor[2] as f32;
                    n += 1;
                }
            }
            if n > 0 {
                target = [sum[0] / n as f32, sum[1] / n as f32, sum[2] / n as f32];
            }
        }
        let p = work.get_pixel_mut(cx as u32, cy as u32);
        for channel in 0..3 {
            p[channel] = target[channel].round().clamp(0.0, 255.0) as u8;
        }
    }
    *image = work;
}

/// Remove printed halftone dot artifacts from colorized/printed pages: small
/// saturated warm-color components (dark red/orange print dots on lighter
/// paper, measured on the user's colored scan: 1-8px clusters, hue 0-60 /
/// 250-360, strongly saturated) are replaced with the local background
/// (median RGB of the 9x9 neighborhood). The engine's ordinary despeckle
/// handles sparse model noise; this pass handles source print dots that can
/// survive inference. Black ink carries no strong chroma so it is never in
/// the mask; large colored regions (solid fills, gradients) are not small
/// components and survive exactly. Returns the number of replaced pixels.
pub fn remove_halftone_dots(image: &mut RgbImage) -> u64 {
    let (w, h) = image.dimensions();
    if w < 3 || h < 3 {
        return 0;
    }
    const MAX_DOT_SIZE: u32 = 8;
    const SAT_THRESHOLD: f32 = 0.35;
    const MIN_LUMA: f32 = 40.0;
    const WINDOW: i32 = 4; // 9x9 median window
    let idx = |x: u32, y: u32| (y * w + x) as usize;

    // Strong warm/magenta chroma mask (red/orange/pink/purple specks).
    let mut strong = vec![false; (w * h) as usize];
    for (i, pixel) in image.pixels().enumerate() {
        let (r, g, b) = (pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        if mx <= 0.0 {
            continue;
        }
        if (mx - mn) / mx <= SAT_THRESHOLD {
            continue;
        }
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        if luma <= MIN_LUMA {
            continue;
        }
        // No hue filter: a speck is a small isolated region of unexpected
        // color regardless of hue. The spatial checks (component size,
        // isolation from neighbors) already do the real work; a hue gate
        // here would only miss blue/green/cyan specks.
        strong[i] = true;
    }

    // Label strong-color components (8-connectivity, BFS).
    let mut labels = vec![0u32; (w * h) as usize];
    let mut sizes: Vec<u32> = Vec::new();
    let mut next_label = 1u32;
    for start in 0..(w * h) as usize {
        if !strong[start] || labels[start] != 0 {
            continue;
        }
        let mut queue = vec![start];
        labels[start] = next_label;
        let mut size = 0u32;
        while let Some(cur) = queue.pop() {
            size += 1;
            let (cx, cy) = (cur % w as usize, cur / w as usize);
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let (nx, ny) = (cx as i32 + dx, cy as i32 + dy);
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        continue;
                    }
                    let nidx = idx(nx as u32, ny as u32);
                    if strong[nidx] && labels[nidx] == 0 {
                        labels[nidx] = next_label;
                        queue.push(nidx);
                    }
                }
            }
        }
        sizes.push(size);
        next_label += 1;
    }

    // Small components are halftone dots: replace with the neighborhood
    // median color (the local paper/background).
    let mut speck = vec![false; (w * h) as usize];
    let mut speck_count = 0u64;
    for (i, &label) in labels.iter().enumerate() {
        if label != 0 && sizes[(label - 1) as usize] <= MAX_DOT_SIZE {
            speck[i] = true;
            speck_count += 1;
        }
    }
    if speck_count == 0 {
        return 0;
    }

    // Median replacement per speck pixel (per-channel median of the
    // 9x9 window; sorting a tiny u8 slice per dot is far cheaper than a
    // full-image median filter).
    let mut work = image.clone();
    let mut window = Vec::with_capacity(81);
    for (i, &is_speck) in speck.iter().enumerate() {
        if !is_speck {
            continue;
        }
        let (cx, cy) = (i % w as usize, i / w as usize);
        let (x0, x1) = (
            (cx as i32 - WINDOW).max(0),
            (cx as i32 + WINDOW).min(w as i32 - 1),
        );
        let (y0, y1) = (
            (cy as i32 - WINDOW).max(0),
            (cy as i32 + WINDOW).min(h as i32 - 1),
        );
        let mut target = [0u8; 3];
        for (channel, value) in target.iter_mut().enumerate() {
            window.clear();
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    window.push(image.get_pixel(xx as u32, yy as u32)[channel]);
                }
            }
            window.sort_unstable();
            *value = window[window.len() / 2];
        }
        let p = work.get_pixel_mut(cx as u32, cy as u32);
        *p = Rgb(target);
    }
    *image = work;
    speck_count
}

/// Fast exact median filter (per-channel, square window) via sliding
/// per-column histograms. Returns a new plane; the input is untouched.
fn median_filter_plane(plane: &[u8], w: usize, h: usize, radius: usize) -> Vec<u8> {
    let n = radius * 2 + 1;
    let half_n: u32 = ((n * n) / 2 + 1) as u32;
    let mut out = vec![0u8; w * h];
    // Per-column histograms of the vertical window.
    let mut col_hists: Vec<[u32; 256]> = vec![[0u32; 256]; w];
    let mut combined: [u32; 256] = [0; 256];

    // Replicate the boundary pixels so every window contains exactly n*n
    // samples. The old implementation used a partial window but compared
    // it against the full-window median index and also inserted row/column
    // `radius` twice. That biased the result and returned zero near page
    // edges, which is why dense dot fields survived the descreen pass.
    for offset in 0..n {
        let yy = offset.saturating_sub(radius).min(h - 1);
        let row = &plane[yy * w..(yy + 1) * w];
        for (c, hist) in col_hists.iter_mut().enumerate() {
            hist[row[c] as usize] += 1;
        }
    }

    for y in 0..h {
        // Slide the replicated vertical window by one row.
        if y > 0 {
            let drop = y.saturating_sub(radius + 1).min(h - 1);
            let add = (y + radius).min(h - 1);
            let drop_row = &plane[drop * w..(drop + 1) * w];
            let add_row = &plane[add * w..(add + 1) * w];
            for (c, hist) in col_hists.iter_mut().enumerate() {
                hist[drop_row[c] as usize] -= 1;
                hist[add_row[c] as usize] += 1;
            }
        }

        // Build the replicated x=0 window.
        for offset in 0..n {
            let xx = offset.saturating_sub(radius).min(w - 1);
            let hist = &col_hists[xx];
            for (v, count) in hist.iter().enumerate() {
                combined[v] += count;
            }
        }
        // Horizontal slide of the combined histogram.
        let row_out = &mut out[y * w..(y + 1) * w];
        for (x, output_value) in row_out.iter_mut().enumerate().take(w) {
            if x > 0 {
                let drop = x.saturating_sub(radius + 1).min(w - 1);
                let add = (x + radius).min(w - 1);
                for (v, count) in col_hists[drop].iter().enumerate() {
                    combined[v] -= count;
                }
                for (v, count) in col_hists[add].iter().enumerate() {
                    combined[v] += count;
                }
            }
            // Median = first index with cumulative >= half_n.
            let mut cum = 0u32;
            for (v, &count) in combined.iter().enumerate() {
                cum += count;
                if cum >= half_n {
                    *output_value = v as u8;
                    break;
                }
            }
        }
        // Reset combined for the next row (rebuilt at x=0).
        combined = [0; 256];
    }
    out
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ChromaGridArtifactReport {
    pub candidate_fraction: f32,
    pub grid_score: f32,
    pub period_x: u32,
    pub period_y: u32,
    pub needs_cleanup: bool,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ChromaGridCleanupReport {
    pub changed_pixels: u64,
    pub before: ChromaGridArtifactReport,
    pub after: ChromaGridArtifactReport,
}

const CHROMA_OFFSET: f32 = 128.0;
// Scale the full legal R-Y / B-Y ranges into u8 without clipping. Storing
// raw opponent values with an offset loses yellow/orange B-Y values below
// -128, making vivid grid squares indistinguishable from their median.
const CR_SCALE: f32 = 0.635;
const CB_SCALE: f32 = 0.539;

fn chroma_candidate_mask(image: &RgbImage, radius: usize) -> Vec<bool> {
    let (w, h) = (image.width() as usize, image.height() as usize);
    let mut luma = Vec::with_capacity(w * h);
    let mut saturation = Vec::with_capacity(w * h);
    let mut cr = Vec::with_capacity(w * h);
    let mut cb = Vec::with_capacity(w * h);
    for p in image.pixels() {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        luma.push(l);
        saturation.push(if mx > 0.0 { (mx - mn) / mx } else { 0.0 });
        cr.push(
            ((r - l) * CR_SCALE + CHROMA_OFFSET)
                .round()
                .clamp(0.0, 255.0) as u8,
        );
        cb.push(
            ((b - l) * CB_SCALE + CHROMA_OFFSET)
                .round()
                .clamp(0.0, 255.0) as u8,
        );
    }
    let cr_median = median_filter_plane(&cr, w, h, radius);
    let cb_median = median_filter_plane(&cb, w, h, radius);
    (0..w * h)
        .map(|i| {
            let dr = (cr[i] as f32 - cr_median[i] as f32) / CR_SCALE;
            let db = (cb[i] as f32 - cb_median[i] as f32) / CB_SCALE;
            let residual = (dr * dr + db * db).sqrt();
            residual > 8.0 && saturation[i] > 0.14 && luma[i] > 40.0 && luma[i] < 245.0
        })
        .collect()
}

fn best_projection_period(values: &[f32]) -> (f32, u32) {
    if values.len() < 27 {
        return (0.0, 0);
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    let centered: Vec<f32> = values.iter().map(|value| value - mean).collect();
    let max_lag = 24usize.min(values.len() / 3);
    let mut best = (0.0f32, 0u32);
    for lag in 8..=max_lag {
        let mut numerator = 0.0f32;
        let mut left_energy = 0.0f32;
        let mut right_energy = 0.0f32;
        for i in 0..values.len() - lag {
            numerator += centered[i] * centered[i + lag];
            left_energy += centered[i] * centered[i];
            right_energy += centered[i + lag] * centered[i + lag];
        }
        let denominator = (left_energy * right_energy).sqrt();
        let score = if denominator > 1e-9 {
            numerator / denominator
        } else {
            0.0
        };
        if score > best.0 {
            best = (score, lag as u32);
        }
    }
    best
}

/// Detect repeated square opponent-chroma outliers. Detection runs on a
/// bounded 512px preview, making it cheap enough to gate every delivered
/// page. Requiring matching horizontal and vertical periods separates the
/// decoder/halftone grid from ordinary colored edges and small details.
pub fn detect_chroma_grid_artifacts(image: &RgbImage) -> ChromaGridArtifactReport {
    if image.width() < 32 || image.height() < 32 {
        return ChromaGridArtifactReport::default();
    }
    let longest = image.width().max(image.height());
    let preview = if longest > 512 {
        let scale = 512.0 / longest as f32;
        image::imageops::resize(
            image,
            (image.width() as f32 * scale).round().max(1.0) as u32,
            (image.height() as f32 * scale).round().max(1.0) as u32,
            image::imageops::FilterType::Triangle,
        )
    } else {
        image.clone()
    };
    let (w, h) = (preview.width() as usize, preview.height() as usize);
    let radius = w.max(h).div_ceil(105).clamp(3, 6);
    let mask = chroma_candidate_mask(&preview, radius);
    let candidate_count = mask.iter().filter(|&&value| value).count();
    let candidate_fraction = candidate_count as f32 / (w * h).max(1) as f32;
    let mut x_projection = vec![0.0f32; w];
    let mut y_projection = vec![0.0f32; h];
    for y in 0..h {
        for x in 0..w {
            if mask[y * w + x] {
                x_projection[x] += 1.0 / h as f32;
                y_projection[y] += 1.0 / w as f32;
            }
        }
    }
    let (x_score, period_x) = best_projection_period(&x_projection);
    let (y_score, period_y) = best_projection_period(&y_projection);
    let periods_align = period_x.abs_diff(period_y) <= 2;
    let grid_score = x_score.min(y_score) * if periods_align { 1.0 } else { 0.65 };
    ChromaGridArtifactReport {
        candidate_fraction,
        grid_score,
        period_x,
        period_y,
        needs_cleanup: candidate_fraction >= 0.035 && grid_score >= 0.58,
    }
}

/// Remove dense square color noise without blurring the whole page. The
/// page-level periodicity detector must opt in first, then only local
/// opponent-chroma outliers are pulled toward an adaptive median. A bounded
/// amount of luma descreening is applied to those same pixels so colored
/// squares do not survive as bright/dark stamps; black ink, white paper,
/// speech text and all pixels outside the artifact mask remain untouched.
pub fn remove_chroma_grid_artifacts(image: &mut RgbImage) -> ChromaGridCleanupReport {
    let before = detect_chroma_grid_artifacts(image);
    if !before.needs_cleanup {
        return ChromaGridCleanupReport {
            changed_pixels: 0,
            before,
            after: before,
        };
    }
    let (w, h) = (image.width() as usize, image.height() as usize);
    let radius = w.max(h).div_ceil(105).clamp(5, 24);
    let mut luma = Vec::with_capacity(w * h);
    let mut saturation = Vec::with_capacity(w * h);
    let mut cr = Vec::with_capacity(w * h);
    let mut cb = Vec::with_capacity(w * h);
    for p in image.pixels() {
        let (r, g, b) = (p[0] as f32, p[1] as f32, p[2] as f32);
        let l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        let mx = r.max(g).max(b);
        let mn = r.min(g).min(b);
        luma.push(l);
        saturation.push(if mx > 0.0 { (mx - mn) / mx } else { 0.0 });
        cr.push(
            ((r - l) * CR_SCALE + CHROMA_OFFSET)
                .round()
                .clamp(0.0, 255.0) as u8,
        );
        cb.push(
            ((b - l) * CB_SCALE + CHROMA_OFFSET)
                .round()
                .clamp(0.0, 255.0) as u8,
        );
    }
    let cr_median = median_filter_plane(&cr, w, h, radius);
    let cb_median = median_filter_plane(&cb, w, h, radius);
    let luma_u8: Vec<u8> = luma
        .iter()
        .map(|value| value.round().clamp(0.0, 255.0) as u8)
        .collect();
    let luma_median = median_filter_plane(&luma_u8, w, h, radius);
    let mut changed_pixels = 0u64;
    let mut output = image.clone();
    for i in 0..w * h {
        let dr = (cr[i] as f32 - cr_median[i] as f32) / CR_SCALE;
        let db = (cb[i] as f32 - cb_median[i] as f32) / CB_SCALE;
        let residual = (dr * dr + db * db).sqrt();
        if residual <= 8.0 || saturation[i] <= 0.14 || luma[i] <= 40.0 || luma[i] >= 245.0 {
            continue;
        }
        let alpha = ((residual - 5.0) / 14.0).clamp(0.0, 1.0);
        let corrected_luma = luma[i] * (1.0 - alpha * 0.72) + luma_median[i] as f32 * alpha * 0.72;
        let source_cr = (cr[i] as f32 - CHROMA_OFFSET) / CR_SCALE;
        let source_cb = (cb[i] as f32 - CHROMA_OFFSET) / CB_SCALE;
        let median_cr = (cr_median[i] as f32 - CHROMA_OFFSET) / CR_SCALE;
        let median_cb = (cb_median[i] as f32 - CHROMA_OFFSET) / CB_SCALE;
        let corrected_cr = source_cr * (1.0 - alpha) + median_cr * alpha;
        let corrected_cb = source_cb * (1.0 - alpha) + median_cb * alpha;
        let corrected_r = corrected_luma + corrected_cr;
        let corrected_b = corrected_luma + corrected_cb;
        let corrected_g = (corrected_luma - 0.2126 * corrected_r - 0.0722 * corrected_b) / 0.7152;
        let corrected = Rgb([
            corrected_r.round().clamp(0.0, 255.0) as u8,
            corrected_g.round().clamp(0.0, 255.0) as u8,
            corrected_b.round().clamp(0.0, 255.0) as u8,
        ]);
        let x = (i % w) as u32;
        let y = (i / w) as u32;
        if output.get_pixel(x, y) != &corrected {
            output.put_pixel(x, y, corrected);
            changed_pixels += 1;
        }
    }
    *image = output;
    ChromaGridCleanupReport {
        changed_pixels,
        before,
        after: detect_chroma_grid_artifacts(image),
    }
}

/// Compatibility wrapper used by existing callers.
pub fn descreen_halftone_regions(image: &mut RgbImage) -> u64 {
    remove_chroma_grid_artifacts(image).changed_pixels
}

/// Adaptive line restoration: stronger in dark ink areas, gentler near edges,
/// preserving colored details in thin-line regions.
pub fn restore_lines_adaptive(image: &mut RgbImage, line_mask: &GrayImage, strength: f32) {
    let strength = strength.clamp(0.0, 2.0);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
        if ink <= 0.01 {
            continue;
        }

        // Progressive darkening: heavier ink = more black, lighter ink = subtle darkening
        let factor = ink.powf(1.8) * strength;
        let keep = (1.0 - factor).clamp(0.0, 1.0);

        pixel[0] = (pixel[0] as f32 * keep).round() as u8;
        pixel[1] = (pixel[1] as f32 * keep).round() as u8;
        pixel[2] = (pixel[2] as f32 * keep).round() as u8;
    }
}

/// Hint-guided chroma correction: where the hint map is confident and the
/// model's own chroma is weak, blend the pixel's color toward the hint
/// hue while keeping the model's LUMA (shading) untouched. The hint color
/// is first re-targeted onto the model's luma, then lerped in RGB — luma
/// is linear, so the blend preserves shading exactly. The blend is
/// proportional to the hint mask strength and inversely proportional to
/// the model's saturation, so confident model colors are never overridden
/// (model-first) and unpainted/paper regions are skipped.
fn hint_guided_chroma_correction(
    output: &mut RgbImage,
    hint_map: &RgbImage,
    hint_mask: &GrayImage,
    line_mask: &GrayImage,
    strength: f32,
) -> usize {
    let (w, h) = output.dimensions();
    if hint_map.dimensions() != (w, h)
        || hint_mask.dimensions() != (w, h)
        || line_mask.dimensions() != (w, h)
    {
        return 0;
    }
    let strength = strength.clamp(0.0, 1.0);
    if strength <= 0.0 {
        return 0;
    }
    // Model saturation above which the model's guess is treated as
    // confident and left alone.
    const CONFIDENT_SAT: f32 = 0.30;
    // Rec. 709 luma weights (the perceptually fair "shading" measure).
    const LUM_W: [f32; 3] = [0.2126, 0.7152, 0.0722];
    let luma = |p: &Rgb<u8>| -> f32 {
        LUM_W[0] * p[0] as f32 + LUM_W[1] * p[1] as f32 + LUM_W[2] * p[2] as f32
    };
    let mut changed = 0usize;
    for y in 0..h {
        for x in 0..w {
            if line_mask.get_pixel(x, y)[0] > 184 {
                continue; // ink
            }
            let mask = hint_mask.get_pixel(x, y)[0] as f32 / 255.0;
            if mask <= 0.15 {
                continue;
            }
            let model = *output.get_pixel(x, y);
            let hint = *hint_map.get_pixel(x, y);
            // Model saturation: near-gray guesses are the blind spots the
            // hints may correct; confident colors stay untouched.
            let m_max = model[0].max(model[1]).max(model[2]) as f32;
            let m_min = model[0].min(model[1]).min(model[2]) as f32;
            if m_max <= 0.0 {
                continue;
            }
            let ms = (m_max - m_min) / m_max;
            if ms >= CONFIDENT_SAT {
                continue; // model is confident: keep its color
            }
            // The hint must carry real chroma: unpainted/paper/black hints
            // have nothing to guide with.
            let h_max = hint[0].max(hint[1]).max(hint[2]) as f32;
            let h_min = hint[0].min(hint[1]).min(hint[2]) as f32;
            let hs = (h_max - h_min) / h_max.max(1.0);
            if hs < 0.08 {
                continue;
            }
            let model_luma = luma(&model);
            let hint_luma = luma(&hint);
            if hint_luma < 1.0 {
                continue; // near-black hint: nothing to guide with
            }
            let blend = mask * strength * (1.0 - ms / CONFIDENT_SAT);
            if blend <= 0.02 {
                continue;
            }
            // Re-target the hint color onto the model's luma, then lerp in
            // RGB: luma stays exactly the model's, hue moves toward the
            // hint, and the blend carries the hint's saturation.
            let scale = model_luma / hint_luma;
            let mut out = [0u8; 3];
            for c in 0..3 {
                let target = hint[c] as f32 * scale;
                out[c] = (model[c] as f32 + (target - model[c] as f32) * blend)
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
            *output.get_pixel_mut(x, y) = Rgb(out);
            changed += 1;
        }
    }
    changed
}

pub fn sharpen_lightly(image: &mut RgbImage, amount: f32) {
    if amount <= 0.0 {
        return;
    }
    let (w, h) = image.dimensions();
    if w < 3 || h < 3 {
        return;
    }
    let blurred = gaussian_blur_3x3(image);
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let b = blurred.get_pixel(x, y);
        for c in 0..3 {
            let v = pixel[c] as f32 + (pixel[c] as f32 - b[c] as f32) * amount;
            pixel[c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

fn gaussian_blur_3x3(image: &RgbImage) -> RgbImage {
    let (w, h) = image.dimensions();
    let mut blurred = RgbImage::new(w, h);
    let kernel: [(i32, i32, f64); 9] = [
        (-1, -1, 1.0),
        (0, -1, 2.0),
        (1, -1, 1.0),
        (-1, 0, 2.0),
        (0, 0, 4.0),
        (1, 0, 2.0),
        (-1, 1, 1.0),
        (0, 1, 2.0),
        (1, 1, 1.0),
    ];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut sum = [0.0f64; 3];
            for (dx, dy, weight) in &kernel {
                let px = image.get_pixel((x as i32 + dx) as u32, (y as i32 + dy) as u32);
                for c in 0..3 {
                    sum[c] += px[c] as f64 * weight;
                }
            }
            blurred.put_pixel(
                x,
                y,
                Rgb([
                    (sum[0] / 16.0).round().clamp(0.0, 255.0) as u8,
                    (sum[1] / 16.0).round().clamp(0.0, 255.0) as u8,
                    (sum[2] / 16.0).round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    for y in 0..h {
        for x in 0..w {
            if x == 0 || x == w - 1 || y == 0 || y == h - 1 {
                blurred.put_pixel(x, y, *image.get_pixel(x, y));
            }
        }
    }
    blurred
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, Rgb, RgbImage};

    #[test]
    fn bubble_cleanup_whitens_bright_regions_and_preserves_text() {
        let mut image = RgbImage::from_pixel(2, 1, Rgb([120, 120, 120]));
        let mut original = GrayImage::from_pixel(2, 1, Luma([255]));
        original.put_pixel(1, 0, Luma([0]));
        let mask = GrayImage::from_pixel(2, 1, Luma([255]));
        let mut restored = 0;

        clean_speech_bubbles(&mut image, &original, &mask, &mut restored);

        assert_eq!(restored, 2);
        assert_eq!(image.get_pixel(0, 0).0, [255, 255, 255]);
        assert_eq!(image.get_pixel(1, 0).0, [120, 120, 120]);
    }

    #[test]
    fn bubble_cleanup_does_not_erase_confident_material_color() {
        let mut image = RgbImage::from_pixel(2, 1, Rgb([238, 198, 181]));
        image.put_pixel(1, 0, Rgb([228, 224, 222]));
        let original = GrayImage::from_pixel(2, 1, Luma([244]));
        let mask = GrayImage::from_pixel(2, 1, Luma([255]));
        let mut restored = 0;

        clean_speech_bubbles(&mut image, &original, &mask, &mut restored);

        assert_eq!(image.get_pixel(0, 0).0, [238, 198, 181]);
        assert!(image.get_pixel(1, 0)[0] > 245);
        assert_eq!(restored, 1);
    }

    #[test]
    fn semantic_evidence_restores_lost_skin_chroma_without_changing_luma() {
        let evidence = RgbImage::from_pixel(1, 1, Rgb([240, 194, 178]));
        let mut output = RgbImage::from_pixel(1, 1, Rgb([214, 212, 211]));
        let lines = GrayImage::from_pixel(1, 1, Luma([0]));
        let before = rgb_luma(output.get_pixel(0, 0));
        let mut restored = 0;

        restore_semantic_chroma(&mut output, &evidence, &lines, None, &mut restored);

        let pixel = output.get_pixel(0, 0);
        assert_eq!(restored, 1);
        assert!(pixel[0] > pixel[1] && pixel[1] > pixel[2]);
        assert!((rgb_luma(pixel) - before).abs() < 1.5);
    }

    #[test]
    fn color_speckle_removal_kills_isolated_dots_keeps_dense_blobs() {
        // A single strong-red pixel on a gray field is model dot noise.
        let mut image = RgbImage::from_pixel(61, 61, Rgb([150, 150, 150]));
        image.put_pixel(55, 55, Rgb([230, 40, 30]));
        // A coherent 20x20 red region (dense enough to fill the density
        // window) placed far from the dot must stay.
        for y in 4..24 {
            for x in 4..24 {
                image.put_pixel(x, y, Rgb([220, 60, 50]));
            }
        }
        // A sparse 4x4 blob (a midtone speck — the visible purple/red
        // dots) is dense enough to survive as a region only when its
        // neighborhood carries it; on paper it is noise and is removed.
        for y in 34..38 {
            for x in 34..38 {
                image.put_pixel(x, y, Rgb([200, 60, 180])); // purple speck
            }
        }
        let lines = GrayImage::from_pixel(61, 61, Luma([0]));

        remove_color_specks(&mut image, &lines);

        let dot = image.get_pixel(55, 55);
        let region = image.get_pixel(12, 12);
        let speck = image.get_pixel(36, 36);
        // The isolated dot is blended toward the gray field...
        assert!(
            (dot[0] as i32 - dot[2] as i32) < 40,
            "isolated dot desaturated, got {:?}",
            dot
        );
        // ...and so is the sparse purple speck (the dot complaint)...
        assert!(
            (speck[0] as i32 - speck[2] as i32) < 60,
            "sparse midtone speck removed, got {:?}",
            speck
        );
        // ...while the dense 20x20 region keeps its color.
        assert!(region[0] as i32 - region[2] as i32 > 80, "region preserved");
    }

    #[test]
    fn color_speckle_removal_cleans_specks_on_ink() {
        // A saturated red speck sitting ON an ink line: the source page is
        // grayscale, so strong chroma on ink is the model's invented noise
        // (the "dots on the lines" complaint) and must be blended into the
        // dark neutral line instead of being exempted.
        let mut image = RgbImage::from_pixel(9, 9, Rgb([40, 40, 40]));
        image.put_pixel(4, 4, Rgb([230, 40, 30]));
        let mut lines = GrayImage::from_pixel(9, 9, Luma([0]));
        lines.put_pixel(4, 4, Luma([255]));

        remove_color_specks(&mut image, &lines);

        let pixel = image.get_pixel(4, 4).0;
        let chroma = pixel.iter().max().unwrap() - pixel.iter().min().unwrap();
        assert!(
            chroma < 40,
            "on-ink speck desaturated toward the line, got {chroma} ({pixel:?})"
        );
        // The line stays dark.
        assert!(pixel.iter().all(|c| *c < 110), "line stays dark: {pixel:?}");
    }

    #[test]
    fn median_filter_replicates_boundaries_without_zero_bias() {
        let plane = vec![73u8; 25];
        let filtered = median_filter_plane(&plane, 5, 5, 2);

        assert_eq!(filtered, plane);

        let varied: Vec<u8> = (0..42)
            .map(|i| ((i * 37 + (i / 7) * 19) % 251) as u8)
            .collect();
        let actual = median_filter_plane(&varied, 7, 6, 2);
        let mut expected = vec![0u8; varied.len()];
        for y in 0..6usize {
            for x in 0..7usize {
                let mut window = Vec::with_capacity(25);
                for dy in -2i32..=2 {
                    for dx in -2i32..=2 {
                        let xx = (x as i32 + dx).clamp(0, 6) as usize;
                        let yy = (y as i32 + dy).clamp(0, 5) as usize;
                        window.push(varied[yy * 7 + xx]);
                    }
                }
                window.sort_unstable();
                expected[y * 7 + x] = window[12];
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn periodic_chroma_squares_are_removed_without_ink_or_paper_damage() {
        let mut image = RgbImage::from_pixel(384, 384, Rgb([204, 96, 104]));
        // A dense 16px-period decoder grid, matching the colored-square
        // failure after scaling to the detector's bounded preview.
        for y0 in (4..384).step_by(16) {
            for x0 in (4..384).step_by(16) {
                for y in y0..(y0 + 6).min(384) {
                    for x in x0..(x0 + 6).min(384) {
                        // Exact orange family from the reported page. Its
                        // B-Y value falls below -128, which the old raw-u8
                        // opponent encoding clipped and therefore missed.
                        image.put_pixel(x, y, Rgb([235, 125, 4]));
                    }
                }
            }
        }
        // Representative ink and paper must remain byte-for-byte stable.
        for y in 0..384 {
            image.put_pixel(190, y, Rgb([18, 18, 18]));
        }
        for y in 300..360 {
            for x in 250..350 {
                image.put_pixel(x, y, Rgb([252, 252, 252]));
            }
        }
        let original = image.clone();
        let before = detect_chroma_grid_artifacts(&image);
        assert!(before.needs_cleanup, "expected grid detection: {before:?}");

        let cleanup = remove_chroma_grid_artifacts(&mut image);

        assert!(
            cleanup.changed_pixels > 1_000,
            "cleanup report: {cleanup:?}"
        );
        assert!(
            !cleanup.after.needs_cleanup,
            "residual grid must block delivery: {cleanup:?}"
        );
        assert!(cleanup.after.candidate_fraction < cleanup.before.candidate_fraction);
        for y in 0..384 {
            assert_eq!(image.get_pixel(190, y), original.get_pixel(190, y));
        }
        for &(x, y) in &[(260, 320), (320, 340)] {
            assert_eq!(image.get_pixel(x, y), original.get_pixel(x, y));
        }
        // The square's brightness is descreened toward its local material,
        // while the byte-stability checks above protect ink and paper.
        let before_luma = rgb_luma(original.get_pixel(6, 6));
        let after_luma = rgb_luma(image.get_pixel(6, 6));
        let local_luma = rgb_luma(original.get_pixel(2, 2));
        assert!((after_luma - local_luma).abs() < (before_luma - local_luma).abs());
        assert!(
            image.get_pixel(6, 6)[2] > 70,
            "clipped orange square must recover local blue chroma: {:?}",
            image.get_pixel(6, 6)
        );
    }

    #[test]
    fn coherent_color_regions_do_not_trigger_grid_cleanup() {
        let mut image = RgbImage::from_pixel(256, 256, Rgb([224, 174, 142]));
        for y in 0..256 {
            for x in 128..256 {
                image.put_pixel(x, y, Rgb([42, 105, 190]));
            }
        }
        let original = image.clone();
        let report = remove_chroma_grid_artifacts(&mut image);

        assert!(!report.before.needs_cleanup, "false grid: {report:?}");
        assert_eq!(report.changed_pixels, 0);
        assert_eq!(image, original);
    }

    #[test]
    fn broad_protected_mask_does_not_erase_generated_color() {
        let colorized = RgbImage::from_pixel(3, 3, Rgb([210, 132, 104]));
        let original = GrayImage::from_pixel(3, 3, Luma([180]));
        let lines = GrayImage::from_pixel(3, 3, Luma([0]));
        let bubbles = GrayImage::from_pixel(3, 3, Luma([0]));
        let protected = GrayImage::from_pixel(3, 3, Luma([255]));
        let style = StyleProfile::default();
        let config = PostprocessSection {
            restore_lines: false,
            line_strength: 1.0,
            bubble_cleanup: true,
            despeckle_color_specks: true,
            descreen_halftone: true,
            bleeding_correction: false,
            palette_correction: false,
            final_sharpening: 0.0,
            reference_finishing: false,
            target_colorfulness: 1.0,
            midtone_depth: 0.08,
            local_contrast: 0.12,
            semantic_relighting: false,
            relighting_strength: 0.25,
            natural_grading: true,
            reference_transfer: true,
            soft_coverage: true,
            region_chroma_separation: 34.0,
            muted_target_grade: false,
            muted_strength: 0.95,
            muted_hue_shift_deg: 16.0,
            muted_dark_pull: 0.92,
            soft_strength: 1.0,
            soft_confidence: 1.0,
            muted_dark_sat: 0.19,
            muted_mid_sat: 0.27,
            muted_bright_sat: 0.22,
            muted_paper_sat: 0.034,
            muted_vivid_preserve: 0.0,
            muted_yellow_shift_deg: 0.0,
            model_first: false,
            seed_chroma_bar_bright: 48.0,
            seed_chroma_bar_mid: 24.0,
            model_chroma_bar_bright: 40.0,
            model_chroma_bar_mid: 20.0,
            model_chroma_gain: 1.3,
            palette_harmony_strength: 0.16,
            face_paint_strength: 0.92,
            ambient_occlusion_strength: 0.0,
            smart_governor: false,
            governor_target_saturation: 0.20,
            governor_target_colorfulness: 36.0,
            governor_min_gain: 0.8,
            governor_max_gain: 1.25,
            governor_target_warmth: 45.0,
            hint_chroma_strength: 0.0,
            unified_grade: true,
        };

        let (output, report) = finalize(PostprocessInput {
            colorized,
            original_grayscale: &original,
            line_mask: &lines,
            bubble_mask: &bubbles,
            tone_mask: None,
            protected_mask: Some(&protected),
            face_mask: None,
            style: &style,
            config: &config,
            reference_image: None,
            hint_map: None,
            hint_mask: None,
            chroma_signal: None,
        })
        .unwrap();

        assert_eq!(output.get_pixel(1, 1).0, [210, 132, 104]);
        assert_eq!(report.protected_pixels_restored, 0);
    }

    #[test]
    fn hint_guided_correction_snaps_weak_model_color_to_hint_hue() {
        // Model output is a weak gray-green; the hint says skin (warm hue).
        let mut output = RgbImage::from_pixel(2, 1, Rgb([150, 152, 148]));
        output.put_pixel(1, 0, Rgb([140, 60, 50])); // confident model color
        let hint_map = RgbImage::from_pixel(2, 1, Rgb([242, 208, 182]));
        let hint_mask = GrayImage::from_pixel(2, 1, Luma([255]));
        let lines = GrayImage::from_pixel(2, 1, Luma([0]));

        let changed =
            hint_guided_chroma_correction(&mut output, &hint_map, &hint_mask, &lines, 1.0);

        assert_eq!(changed, 1, "only the weak-chroma pixel is corrected");
        let corrected = output.get_pixel(0, 0);
        // Warm skin hue: R must clearly dominate B, and the model's value
        // (shading) must be preserved (luma stays close to the original).
        assert!(
            corrected[0] as i32 - corrected[2] as i32 > 25,
            "hue snapped toward warm skin, got {:?}",
            corrected
        );
        let luma = |p: &Rgb<u8>| 0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32;
        assert!(
            (luma(corrected) - luma(&Rgb([150, 152, 148]))).abs() < 3.0,
            "shading preserved"
        );
        // The confident model color is untouched.
        assert_eq!(output.get_pixel(1, 0).0, [140, 60, 50]);
    }

    #[test]
    fn full_resolution_reconstruction_restores_dimensions_and_dark_ink() {
        let colorized = RgbImage::from_pixel(2, 2, Rgb([220, 150, 130]));
        let mut original = GrayImage::from_pixel(4, 4, Luma([230]));
        original.put_pixel(1, 1, Luma([0]));

        let restored = reconstruct_full_resolution(&colorized, &original);

        assert_eq!(restored.dimensions(), (4, 4));
        let ink = restored.get_pixel(1, 1);
        assert!(ink[0] < 20 && ink[1] < 20 && ink[2] < 20);
        let color = restored.get_pixel(3, 3);
        assert!(color[0] > color[1] && color[1] > color[2]);
    }

    #[test]
    fn full_resolution_reconstruction_suppresses_edge_chroma() {
        let colorized = RgbImage::from_pixel(2, 2, Rgb([230, 70, 40]));
        let mut original = GrayImage::from_pixel(4, 4, Luma([230]));
        for y in 0..4 {
            original.put_pixel(0, y, Luma([40]));
            original.put_pixel(1, y, Luma([40]));
        }

        let restored = reconstruct_full_resolution(&colorized, &original);
        let edge = restored.get_pixel(2, 2);
        let interior = restored.get_pixel(3, 2);
        let edge_chroma = edge.0.iter().max().unwrap() - edge.0.iter().min().unwrap();
        let interior_chroma = interior.0.iter().max().unwrap() - interior.0.iter().min().unwrap();

        assert!(edge_chroma < interior_chroma);
    }

    #[test]
    fn full_resolution_reconstruction_retains_material_chroma() {
        let colorized = RgbImage::from_pixel(2, 2, Rgb([220, 150, 130]));
        let original = GrayImage::from_pixel(8, 8, Luma([220]));

        let restored = reconstruct_full_resolution(&colorized, &original);
        let pixel = restored.get_pixel(4, 4);
        let chroma = pixel.0.iter().max().unwrap() - pixel.0.iter().min().unwrap();

        assert!(chroma >= 75);
        assert!(pixel[0] > pixel[1] && pixel[1] > pixel[2]);
    }

    #[test]
    fn joint_bilateral_chroma_does_not_cross_an_ink_boundary() {
        let guide = GrayImage::from_raw(5, 1, vec![225, 225, 24, 225, 225]).unwrap();
        let source_red = vec![42.0, 42.0, 0.0, -38.0, -38.0];
        let source_blue = vec![-24.0, -24.0, 0.0, 35.0, 35.0];
        let mut target_red = vec![0.0; 5];
        let mut target_blue = vec![0.0; 5];
        let range_weights: Vec<f32> = (0..=255)
            .map(|difference| {
                let normalized = difference as f32 / 22.0;
                (-0.5 * normalized * normalized).exp()
            })
            .collect();

        joint_bilateral_pass(
            &source_red,
            &source_blue,
            &mut target_red,
            &mut target_blue,
            &guide,
            &range_weights,
            &[0.18, 0.58, 1.0, 0.58, 0.18],
            true,
        );

        assert!(target_red[1] > 35.0 && target_blue[1] < -18.0);
        assert!(target_red[3] < -30.0 && target_blue[3] > 28.0);
    }

    #[test]
    fn neutral_lut_is_continuous_and_preserves_channel_order() {
        let style = StyleProfile {
            saturation: 1.0,
            contrast: 1.0,
            brightness: 0.0,
            ..StyleProfile::default()
        };
        let lut = MangaColorLUT::new(&style);

        let low = lut_lookup(&lut, 100, 140, 180);
        let high = lut_lookup(&lut, 101, 141, 181);

        assert!(low[0] < low[1] && low[1] < low[2]);
        for channel in 0..3 {
            assert!((high[channel] as i16 - low[channel] as i16).abs() <= 2);
        }
    }

    #[test]
    fn palette_harmony_preserves_neutral_pixels_and_guides_colored_pixels() {
        let style = StyleProfile {
            palette: vec![
                mangahue_core::ColorRgb::new(238, 199, 174),
                mangahue_core::ColorRgb::new(245, 245, 242),
            ],
            ..StyleProfile::default()
        };
        let mut image = RgbImage::from_pixel(2, 1, Rgb([210, 120, 150]));
        image.put_pixel(1, 0, Rgb([240, 240, 240]));
        let lines = GrayImage::from_pixel(2, 1, Luma([0]));

        apply_palette_harmony(&mut image, &style, &lines, 0.16, false);

        let colored = image.get_pixel(0, 0);
        assert!(colored[1] > 120);
        assert!(colored[0] - colored[1] < 90);
        assert_eq!(image.get_pixel(1, 0).0, [240, 240, 240]);
    }

    #[test]
    fn screentone_restoration_recovers_luma() {
        // The source page is grayscale, so tone chroma is the model's
        // invention — the pass restores the dot pattern's luma toward the
        // original gray while keeping the hue direction.
        let mut image = RgbImage::from_pixel(1, 1, Rgb([180, 80, 60]));
        let original = GrayImage::from_pixel(1, 1, Luma([180]));
        let tone_mask = GrayImage::from_pixel(1, 1, Luma([255]));
        let before_luma: f32 = 0.2126 * 180.0 + 0.7152 * 80.0 + 0.0722 * 60.0;
        let mut restored = 0;

        preserve_screentones(&mut image, &original, &tone_mask, 1.0, &mut restored);

        let pixel = image.get_pixel(0, 0);
        let after_luma =
            0.2126 * pixel[0] as f32 + 0.7152 * pixel[1] as f32 + 0.0722 * pixel[2] as f32;
        assert_eq!(restored, 1);
        assert!((180.0f32 - after_luma).abs() < (180.0f32 - before_luma).abs());
        // The warm hue direction survives (r > g > b).
        assert!(pixel[0] > pixel[1] && pixel[1] > pixel[2], "warm tint kept");
    }

    #[test]
    fn chroma_completion_fills_region_but_respects_ink_wall() {
        let mut image = RgbImage::from_pixel(15, 5, Rgb([150, 150, 150]));
        image.put_pixel(2, 2, Rgb([205, 112, 88]));
        let original = GrayImage::from_pixel(15, 5, Luma([150]));
        let mut lines = GrayImage::from_pixel(15, 5, Luma([0]));
        for y in 0..5 {
            lines.put_pixel(7, y, Luma([255]));
        }
        let protected = GrayImage::from_pixel(15, 5, Luma([0]));
        let style = StyleProfile {
            palette: Vec::new(),
            ..StyleProfile::default()
        };
        let mut completed = 0;
        let mut refined = 0;

        complete_material_chroma(
            &mut image,
            &original,
            &lines,
            Some(&protected),
            &style,
            &mut completed,
            &mut refined,
        );

        let left = image.get_pixel(5, 2);
        let right = image.get_pixel(11, 2);
        assert!(left[0] > left[1] && left[1] > left[2]);
        assert_eq!(right.0, [150, 150, 150]);
        assert!(completed > 0);
        assert_eq!(refined, 0);
    }

    #[test]
    fn local_completion_does_not_invent_color_without_evidence() {
        let mut image = RgbImage::from_pixel(9, 9, Rgb([170, 170, 170]));
        let original = GrayImage::from_pixel(9, 9, Luma([170]));
        let lines = GrayImage::from_pixel(9, 9, Luma([0]));
        let protected = GrayImage::from_pixel(9, 9, Luma([0]));
        let style = StyleProfile::default();
        let mut completed = 0;
        let mut refined = 0;

        complete_material_chroma(
            &mut image,
            &original,
            &lines,
            Some(&protected),
            &style,
            &mut completed,
            &mut refined,
        );

        let pixel = image.get_pixel(4, 4);
        assert!((pixel[0] as i16 - pixel[2] as i16).abs() < 5);
        assert_eq!(refined, 0);
    }
}
