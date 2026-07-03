use anyhow::{Context, Result};
use chrono::Utc;
use mangahue_core::{ColorRgb, ImageSize, StyleProfile};
use mangahue_eval::QualityReport;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterBrainRequest {
    #[serde(default)]
    pub pages: Vec<BrainPageInput>,
    #[serde(default = "default_max_palette_colors")]
    pub max_palette_colors: usize,
    #[serde(default = "default_max_sampled_pixels")]
    pub max_sampled_pixels: u64,
}

impl Default for ChapterBrainRequest {
    fn default() -> Self {
        Self {
            pages: Vec::new(),
            max_palette_colors: default_max_palette_colors(),
            max_sampled_pixels: default_max_sampled_pixels(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainPageInput {
    pub index: usize,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterBrainReport {
    pub generated_at: String,
    pub total_pages: usize,
    pub analyzed_pages: usize,
    pub consistency_score: f32,
    pub director_verdict: String,
    #[serde(default)]
    pub global_palette: Vec<PaletteSwatch>,
    #[serde(default)]
    pub palette_locks: Vec<PaletteLock>,
    #[serde(default)]
    pub pages: Vec<PageBrainReport>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageBrainReport {
    pub index: usize,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub size: ImageSize,
    pub sampled_pixels: u64,
    pub chromatic_pixel_ratio: f32,
    pub dark_ink_ratio: f32,
    pub white_region_ratio: f32,
    pub mean_luma: f32,
    pub mean_saturation: f32,
    pub colorfulness: f32,
    pub palette_drift_score: f32,
    pub tone_drift_score: f32,
    pub consistency_score: f32,
    #[serde(default)]
    pub palette: Vec<PaletteSwatch>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaletteSwatch {
    pub color: ColorRgb,
    pub coverage: f32,
    pub weight: f32,
    pub pixels: u64,
    pub support_pages: usize,
    pub luma: f32,
    pub saturation: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaletteLock {
    pub id: String,
    pub color: ColorRgb,
    pub tolerance: f32,
    pub confidence: f32,
    pub coverage: f32,
    pub support_pages: usize,
    pub suggested_role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyGateRequest {
    #[serde(default)]
    pub pages: Vec<SafetyGatePageInput>,
    #[serde(default)]
    pub thresholds: SafetyGateThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyGatePageInput {
    pub index: usize,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub quality_report_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyGateThresholds {
    #[serde(default = "default_min_line_preservation")]
    pub min_line_preservation: f32,
    #[serde(default = "default_min_bubble_cleanliness")]
    pub min_bubble_cleanliness: f32,
    #[serde(default = "default_min_protected_region")]
    pub min_protected_region: f32,
    #[serde(default = "default_min_hint_adherence")]
    pub min_hint_adherence: f32,
}

impl Default for SafetyGateThresholds {
    fn default() -> Self {
        Self {
            min_line_preservation: default_min_line_preservation(),
            min_bubble_cleanliness: default_min_bubble_cleanliness(),
            min_protected_region: default_min_protected_region(),
            min_hint_adherence: default_min_hint_adherence(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChapterSafetyReport {
    pub generated_at: String,
    pub pass: bool,
    pub verdict: String,
    pub audited_pages: usize,
    pub failed_pages: usize,
    pub min_line_preservation_score: f32,
    pub min_bubble_cleanliness_score: f32,
    pub min_protected_region_score: f32,
    pub min_hint_adherence_score: f32,
    #[serde(default)]
    pub pages: Vec<PageSafetyReport>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageSafetyReport {
    pub index: usize,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub quality_report_path: Option<PathBuf>,
    pub pass: bool,
    pub severity: f32,
    pub line_preservation_score: f32,
    pub bubble_cleanliness_score: f32,
    pub protected_region_violation_score: f32,
    pub hint_adherence_score: f32,
    pub output_sharpness_estimate: f32,
    #[serde(default)]
    pub violations: Vec<String>,
}

#[derive(Debug, Default)]
struct Bucket {
    pixels: u64,
    r: u64,
    g: u64,
    b: u64,
}

#[derive(Debug)]
struct GlobalBucket {
    color: ColorRgb,
    coverage: f32,
    weight: f32,
    pixels: u64,
    pages: BTreeSet<usize>,
}

pub fn analyze_chapter_brain(request: ChapterBrainRequest) -> Result<ChapterBrainReport> {
    let max_palette_colors = request.max_palette_colors.clamp(3, 24);
    let mut warnings = Vec::new();
    let mut pages = Vec::new();

    for page in &request.pages {
        if !page.output_path.exists() {
            warnings.push(format!(
                "page {:03} missing output {}",
                page.index,
                page.output_path.display()
            ));
            continue;
        }
        pages.push(analyze_page_palette(
            page,
            max_palette_colors,
            request.max_sampled_pixels,
        )?);
    }

    let global_palette = build_global_palette(&pages, max_palette_colors);
    let palette_locks = build_palette_locks(&global_palette, pages.len());
    let mean_luma = mean_by(&pages, |page| page.mean_luma);
    let mean_saturation = mean_by(&pages, |page| page.mean_saturation);

    for page in &mut pages {
        page.palette_drift_score = palette_drift_score(page, &palette_locks);
        page.tone_drift_score = tone_drift_score(page, mean_luma, mean_saturation);
        page.consistency_score = (1.0
            - (page.palette_drift_score * 0.74 + page.tone_drift_score * 0.26))
            .clamp(0.0, 1.0);
        if page.chromatic_pixel_ratio < 0.012 {
            page.warnings
                .push("low chromatic signal; page may be mostly line art or bubbles".to_string());
        }
        if page.palette_drift_score > 0.34 {
            page.warnings
                .push("palette drift above chapter lock tolerance".to_string());
        }
        if page.tone_drift_score > 0.32 {
            page.warnings
                .push("tone drift above chapter average".to_string());
        }
    }

    let consistency_score = mean_by(&pages, |page| page.consistency_score);
    if palette_locks.len() < 3 {
        warnings.push("chapter brain found fewer than three stable palette locks".to_string());
    }
    for page in &pages {
        for warning in &page.warnings {
            warnings.push(format!("page {:03}: {warning}", page.index));
        }
    }

    let director_verdict = if pages.is_empty() {
        "brain_no_completed_pages".to_string()
    } else if consistency_score >= 0.88 {
        "chapter_palette_locked".to_string()
    } else if consistency_score >= 0.74 {
        "chapter_consistent_with_minor_drift".to_string()
    } else {
        "needs_palette_lock_guided_retry".to_string()
    };

    Ok(ChapterBrainReport {
        generated_at: Utc::now().to_rfc3339(),
        total_pages: request.pages.len(),
        analyzed_pages: pages.len(),
        consistency_score,
        director_verdict,
        global_palette,
        palette_locks,
        pages,
        warnings,
    })
}

pub fn write_chapter_brain_report(
    report: &ChapterBrainReport,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(output_path, serde_json::to_string_pretty(report)?)
        .with_context(|| format!("failed to write {}", output_path.display()))
}

pub fn locked_style_profile(report: &ChapterBrainReport, base: &StyleProfile) -> StyleProfile {
    let mut style = base.clone();
    let palette = report
        .palette_locks
        .iter()
        .map(|lock| lock.color)
        .chain(report.global_palette.iter().map(|swatch| swatch.color))
        .take(16)
        .collect::<Vec<_>>();
    if !palette.is_empty() {
        style.palette = palette;
    }
    style.id = "mangahue_chapter_palette_lock".to_string();
    style.name = "MangaHue Chapter Palette Lock".to_string();
    style.notes = Some(format!(
        "Generated by MangaHue brain at {}; consistency_score={:.3}; verdict={}",
        report.generated_at, report.consistency_score, report.director_verdict
    ));
    style
}

pub fn write_locked_style_profile(
    report: &ChapterBrainReport,
    base: &StyleProfile,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(
        output_path,
        serde_json::to_string_pretty(&locked_style_profile(report, base))?,
    )
    .with_context(|| format!("failed to write {}", output_path.display()))
}

pub fn analyze_safety_gate(request: SafetyGateRequest) -> Result<ChapterSafetyReport> {
    let thresholds = request.thresholds;
    let mut warnings = Vec::new();
    let mut pages = Vec::new();

    for page in &request.pages {
        let Some(path) = page
            .quality_report_path
            .as_ref()
            .filter(|path| path.exists())
        else {
            warnings.push(format!("page {:03} missing quality report", page.index));
            continue;
        };
        let quality: QualityReport = serde_json::from_slice(
            &fs::read(path).with_context(|| format!("failed to read {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse {}", path.display()))?;
        pages.push(page_safety_report(page, &quality, &thresholds));
    }

    let failed_pages = pages.iter().filter(|page| !page.pass).count();
    let pass = !pages.is_empty() && failed_pages == 0 && warnings.is_empty();
    let min_line_preservation_score =
        min_by(&pages, |page| page.line_preservation_score).unwrap_or(0.0);
    let min_bubble_cleanliness_score =
        min_by(&pages, |page| page.bubble_cleanliness_score).unwrap_or(0.0);
    let min_protected_region_score =
        min_by(&pages, |page| page.protected_region_violation_score).unwrap_or(0.0);
    let min_hint_adherence_score = min_by(&pages, |page| page.hint_adherence_score).unwrap_or(0.0);
    let verdict = if pages.is_empty() {
        "safety_no_quality_reports".to_string()
    } else if pass {
        "safety_gate_pass".to_string()
    } else if pages.iter().any(|page| {
        page.bubble_cleanliness_score < thresholds.min_bubble_cleanliness
            || page.protected_region_violation_score < thresholds.min_protected_region
    }) {
        "safety_gate_text_or_bubble_risk".to_string()
    } else {
        "safety_gate_soft_quality_risk".to_string()
    };

    Ok(ChapterSafetyReport {
        generated_at: Utc::now().to_rfc3339(),
        pass,
        verdict,
        audited_pages: pages.len(),
        failed_pages,
        min_line_preservation_score,
        min_bubble_cleanliness_score,
        min_protected_region_score,
        min_hint_adherence_score,
        pages,
        warnings,
    })
}

pub fn write_safety_gate_report(
    report: &ChapterSafetyReport,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(output_path, serde_json::to_string_pretty(report)?)
        .with_context(|| format!("failed to write {}", output_path.display()))
}

fn page_safety_report(
    input: &SafetyGatePageInput,
    quality: &QualityReport,
    thresholds: &SafetyGateThresholds,
) -> PageSafetyReport {
    let mut violations = Vec::new();
    let mut severity = 0.0f32;
    add_violation(
        &mut violations,
        &mut severity,
        "line_preservation",
        thresholds.min_line_preservation,
        quality.line_preservation_score,
        0.86,
    );
    add_violation(
        &mut violations,
        &mut severity,
        "bubble_cleanliness",
        thresholds.min_bubble_cleanliness,
        quality.bubble_cleanliness_score,
        1.0,
    );
    add_violation(
        &mut violations,
        &mut severity,
        "protected_region",
        thresholds.min_protected_region,
        quality.protected_region_violation_score,
        0.96,
    );
    add_violation(
        &mut violations,
        &mut severity,
        "hint_adherence",
        thresholds.min_hint_adherence,
        quality.hint_adherence_score,
        0.42,
    );

    PageSafetyReport {
        index: input.index,
        input_path: input.input_path.clone(),
        output_path: input.output_path.clone(),
        quality_report_path: input.quality_report_path.clone(),
        pass: violations.is_empty(),
        severity: severity.clamp(0.0, 1.0),
        line_preservation_score: quality.line_preservation_score,
        bubble_cleanliness_score: quality.bubble_cleanliness_score,
        protected_region_violation_score: quality.protected_region_violation_score,
        hint_adherence_score: quality.hint_adherence_score,
        output_sharpness_estimate: quality.output_sharpness_estimate,
        violations,
    }
}

fn add_violation(
    violations: &mut Vec<String>,
    severity: &mut f32,
    name: &str,
    threshold: f32,
    score: f32,
    weight: f32,
) {
    if score >= threshold {
        return;
    }
    let deficit = (threshold - score).max(0.0);
    *severity += deficit * weight;
    violations.push(format!("{name}:{score:.3}<{threshold:.3}"));
}

fn analyze_page_palette(
    input: &BrainPageInput,
    max_palette_colors: usize,
    max_sampled_pixels: u64,
) -> Result<PageBrainReport> {
    let image = image::open(&input.output_path)
        .with_context(|| format!("failed to open {}", input.output_path.display()))?
        .to_rgb8();
    let (width, height) = image.dimensions();
    let sampled_pixels = (width as u64 * height as u64).max(1);
    let stride = ((sampled_pixels as f64 / max_sampled_pixels.max(1) as f64)
        .sqrt()
        .ceil() as u32)
        .max(1);

    let mut buckets: HashMap<u16, Bucket> = HashMap::new();
    let mut sampled = 0u64;
    let mut chromatic = 0u64;
    let mut dark_ink = 0u64;
    let mut white_region = 0u64;
    let mut luma_sum = 0.0f32;
    let mut saturation_sum = 0.0f32;
    let mut rg_sum = 0.0f32;
    let mut yb_sum = 0.0f32;
    let mut rg_sq_sum = 0.0f32;
    let mut yb_sq_sum = 0.0f32;

    for y in (0..height).step_by(stride as usize) {
        for x in (0..width).step_by(stride as usize) {
            let pixel = image.get_pixel(x, y);
            let color = ColorRgb::new(pixel[0], pixel[1], pixel[2]);
            let luma = color_luma(color);
            let saturation = color_saturation(color);
            let chroma = color_chroma(color);
            sampled += 1;
            luma_sum += luma;
            saturation_sum += saturation;

            let rg = pixel[0] as f32 - pixel[1] as f32;
            let yb = 0.5 * (pixel[0] as f32 + pixel[1] as f32) - pixel[2] as f32;
            rg_sum += rg;
            yb_sum += yb;
            rg_sq_sum += rg * rg;
            yb_sq_sum += yb * yb;

            if luma > 242.0 && chroma < 18.0 {
                white_region += 1;
                continue;
            }
            if luma < 28.0 && chroma < 22.0 {
                dark_ink += 1;
                continue;
            }
            if chroma < 9.0 {
                continue;
            }
            chromatic += 1;
            let bucket = buckets.entry(quantize_color(color)).or_default();
            bucket.pixels += 1;
            bucket.r += color.r as u64;
            bucket.g += color.g as u64;
            bucket.b += color.b as u64;
        }
    }

    let sampled = sampled.max(1);
    let mut palette = buckets
        .into_values()
        .filter(|bucket| bucket.pixels > 0)
        .map(|bucket| {
            let color = ColorRgb::new(
                (bucket.r / bucket.pixels).min(255) as u8,
                (bucket.g / bucket.pixels).min(255) as u8,
                (bucket.b / bucket.pixels).min(255) as u8,
            );
            let saturation = color_saturation(color);
            PaletteSwatch {
                color,
                coverage: bucket.pixels as f32 / sampled as f32,
                weight: bucket.pixels as f32 * (0.35 + saturation),
                pixels: bucket.pixels,
                support_pages: 1,
                luma: color_luma(color),
                saturation,
            }
        })
        .collect::<Vec<_>>();
    palette.sort_by(|left, right| right.weight.total_cmp(&left.weight));
    palette = merge_swatches(palette, max_palette_colors, 34.0);

    let rg_mean = rg_sum / sampled as f32;
    let yb_mean = yb_sum / sampled as f32;
    let rg_var = (rg_sq_sum / sampled as f32 - rg_mean * rg_mean).max(0.0);
    let yb_var = (yb_sq_sum / sampled as f32 - yb_mean * yb_mean).max(0.0);
    let colorfulness = (rg_var.sqrt() * rg_var.sqrt() + yb_var.sqrt() * yb_var.sqrt()).sqrt()
        + 0.3 * (rg_mean * rg_mean + yb_mean * yb_mean).sqrt();

    Ok(PageBrainReport {
        index: input.index,
        input_path: input.input_path.clone(),
        output_path: input.output_path.clone(),
        size: ImageSize::new(width, height),
        sampled_pixels: sampled,
        chromatic_pixel_ratio: chromatic as f32 / sampled as f32,
        dark_ink_ratio: dark_ink as f32 / sampled as f32,
        white_region_ratio: white_region as f32 / sampled as f32,
        mean_luma: luma_sum / sampled as f32,
        mean_saturation: saturation_sum / sampled as f32,
        colorfulness,
        palette_drift_score: 0.0,
        tone_drift_score: 0.0,
        consistency_score: 0.0,
        palette,
        warnings: Vec::new(),
    })
}

fn build_global_palette(
    pages: &[PageBrainReport],
    max_palette_colors: usize,
) -> Vec<PaletteSwatch> {
    let mut buckets: Vec<GlobalBucket> = Vec::new();
    for page in pages {
        for swatch in &page.palette {
            if let Some(bucket) = buckets
                .iter_mut()
                .min_by(|left, right| {
                    left.color
                        .distance(swatch.color)
                        .total_cmp(&right.color.distance(swatch.color))
                })
                .filter(|bucket| bucket.color.distance(swatch.color) <= 42.0)
            {
                let total = bucket.weight + swatch.weight;
                bucket.color =
                    weighted_color(bucket.color, bucket.weight, swatch.color, swatch.weight);
                bucket.coverage += swatch.coverage;
                bucket.weight = total;
                bucket.pixels += swatch.pixels;
                bucket.pages.insert(page.index);
            } else {
                let mut support = BTreeSet::new();
                support.insert(page.index);
                buckets.push(GlobalBucket {
                    color: swatch.color,
                    coverage: swatch.coverage,
                    weight: swatch.weight,
                    pixels: swatch.pixels,
                    pages: support,
                });
            }
        }
    }
    let mut swatches = buckets
        .into_iter()
        .map(|bucket| {
            let support_pages = bucket.pages.len();
            let saturation = color_saturation(bucket.color);
            PaletteSwatch {
                color: bucket.color,
                coverage: bucket.coverage / pages.len().max(1) as f32,
                weight: bucket.weight * (1.0 + support_pages as f32 * 0.18),
                pixels: bucket.pixels,
                support_pages,
                luma: color_luma(bucket.color),
                saturation,
            }
        })
        .collect::<Vec<_>>();
    swatches.sort_by(|left, right| {
        right
            .support_pages
            .cmp(&left.support_pages)
            .then_with(|| right.weight.total_cmp(&left.weight))
    });
    swatches.truncate(max_palette_colors);
    swatches
}

fn build_palette_locks(global_palette: &[PaletteSwatch], total_pages: usize) -> Vec<PaletteLock> {
    global_palette
        .iter()
        .take(12)
        .enumerate()
        .map(|(index, swatch)| {
            let page_support = swatch.support_pages as f32 / total_pages.max(1) as f32;
            let coverage_strength = (swatch.coverage * 8.0).clamp(0.0, 1.0);
            let confidence = (page_support * 0.72 + coverage_strength * 0.28).clamp(0.0, 1.0);
            PaletteLock {
                id: format!(
                    "lock_{:02}_{}",
                    index + 1,
                    swatch.color.to_hex().trim_start_matches('#')
                ),
                color: swatch.color,
                tolerance: (24.0 + (1.0 - confidence) * 34.0).clamp(18.0, 58.0),
                confidence,
                coverage: swatch.coverage,
                support_pages: swatch.support_pages,
                suggested_role: suggested_role(swatch.color),
            }
        })
        .collect()
}

fn palette_drift_score(page: &PageBrainReport, locks: &[PaletteLock]) -> f32 {
    if page.palette.is_empty() || locks.is_empty() {
        return 1.0;
    }
    let mut total_weight = 0.0f32;
    let mut total_distance = 0.0f32;
    let mut outside_tolerance = 0.0f32;
    let mut unstable_lock = 0.0f32;
    for swatch in &page.palette {
        let Some(nearest) = locks.iter().min_by(|left, right| {
            left.color
                .distance(swatch.color)
                .total_cmp(&right.color.distance(swatch.color))
        }) else {
            continue;
        };
        let distance = nearest.color.distance(swatch.color);
        let weight = swatch.coverage.max(0.002);
        total_weight += weight;
        total_distance += distance * weight;
        if distance > nearest.tolerance {
            outside_tolerance += weight;
        }
        unstable_lock += (1.0 - nearest.confidence) * weight;
    }
    if total_weight <= f32::EPSILON {
        return 1.0;
    }
    let normalized_distance = (total_distance / total_weight / 148.0).clamp(0.0, 1.0);
    let miss_penalty = (outside_tolerance / total_weight).clamp(0.0, 1.0);
    let stability_penalty = (unstable_lock / total_weight).clamp(0.0, 1.0);
    (normalized_distance * 0.34 + miss_penalty * 0.16 + stability_penalty * 0.50).clamp(0.0, 1.0)
}

fn tone_drift_score(page: &PageBrainReport, mean_luma: f32, mean_saturation: f32) -> f32 {
    let luma = ((page.mean_luma - mean_luma).abs() / 92.0).clamp(0.0, 1.0);
    let saturation = ((page.mean_saturation - mean_saturation).abs() / 0.55).clamp(0.0, 1.0);
    (luma * 0.56 + saturation * 0.44).clamp(0.0, 1.0)
}

fn merge_swatches(
    mut swatches: Vec<PaletteSwatch>,
    limit: usize,
    threshold: f32,
) -> Vec<PaletteSwatch> {
    let mut merged: Vec<PaletteSwatch> = Vec::new();
    for swatch in swatches.drain(..) {
        if let Some(existing) = merged
            .iter_mut()
            .find(|existing| existing.color.distance(swatch.color) <= threshold)
        {
            let total = existing.weight + swatch.weight;
            existing.color =
                weighted_color(existing.color, existing.weight, swatch.color, swatch.weight);
            existing.coverage += swatch.coverage;
            existing.weight = total;
            existing.pixels += swatch.pixels;
            existing.luma = color_luma(existing.color);
            existing.saturation = color_saturation(existing.color);
        } else {
            merged.push(swatch);
        }
        if merged.len() >= limit {
            break;
        }
    }
    merged
}

fn weighted_color(
    left: ColorRgb,
    left_weight: f32,
    right: ColorRgb,
    right_weight: f32,
) -> ColorRgb {
    let total = (left_weight + right_weight).max(f32::EPSILON);
    let mix = |a: u8, b: u8| {
        ((a as f32 * left_weight + b as f32 * right_weight) / total)
            .round()
            .clamp(0.0, 255.0) as u8
    };
    ColorRgb::new(
        mix(left.r, right.r),
        mix(left.g, right.g),
        mix(left.b, right.b),
    )
}

fn quantize_color(color: ColorRgb) -> u16 {
    ((color.r as u16 >> 4) << 8) | ((color.g as u16 >> 4) << 4) | (color.b as u16 >> 4)
}

fn color_luma(color: ColorRgb) -> f32 {
    0.2126 * color.r as f32 + 0.7152 * color.g as f32 + 0.0722 * color.b as f32
}

fn color_chroma(color: ColorRgb) -> f32 {
    let max = color.r.max(color.g).max(color.b) as f32;
    let min = color.r.min(color.g).min(color.b) as f32;
    max - min
}

fn color_saturation(color: ColorRgb) -> f32 {
    let max = color.r.max(color.g).max(color.b) as f32;
    if max <= f32::EPSILON {
        0.0
    } else {
        color_chroma(color) / max
    }
}

fn suggested_role(color: ColorRgb) -> String {
    let luma = color_luma(color);
    let saturation = color_saturation(color);
    if luma < 62.0 {
        "shadow_or_ink_tint".to_string()
    } else if color.r > color.g && color.g > color.b && luma > 118.0 && saturation < 0.48 {
        "skin_candidate".to_string()
    } else if saturation > 0.56 {
        "accent_candidate".to_string()
    } else if luma > 206.0 {
        "highlight_candidate".to_string()
    } else {
        "base_tone".to_string()
    }
}

fn mean_by(pages: &[PageBrainReport], metric: impl Fn(&PageBrainReport) -> f32) -> f32 {
    if pages.is_empty() {
        return 0.0;
    }
    pages.iter().map(metric).sum::<f32>() / pages.len() as f32
}

fn min_by(pages: &[PageSafetyReport], metric: impl Fn(&PageSafetyReport) -> f32) -> Option<f32> {
    pages.iter().map(metric).min_by(f32::total_cmp)
}

fn default_max_palette_colors() -> usize {
    12
}

fn default_max_sampled_pixels() -> u64 {
    260_000
}

fn default_min_line_preservation() -> f32 {
    0.92
}

fn default_min_bubble_cleanliness() -> f32 {
    0.97
}

fn default_min_protected_region() -> f32 {
    0.96
}

fn default_min_hint_adherence() -> f32 {
    0.62
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn locks_stable_chapter_palette() {
        let dir = tempfile::tempdir().unwrap();
        let pages = (1..=3)
            .map(|index| {
                let path = dir.path().join(format!("page_{index:03}.png"));
                write_blocks(&path, [238, 176, 142], [56, 104, 214]);
                BrainPageInput {
                    index,
                    input_path: path.clone(),
                    output_path: path,
                }
            })
            .collect::<Vec<_>>();

        let report = analyze_chapter_brain(ChapterBrainRequest {
            pages,
            ..ChapterBrainRequest::default()
        })
        .unwrap();

        assert!(report.consistency_score > 0.86);
        assert!(report.palette_locks.len() >= 2);
        assert_eq!(report.director_verdict, "chapter_palette_locked");
    }

    #[test]
    fn detects_palette_drift() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("page_001.png");
        let second = dir.path().join("page_002.png");
        write_blocks(&first, [238, 176, 142], [56, 104, 214]);
        write_blocks(&second, [88, 222, 190], [232, 72, 88]);

        let report = analyze_chapter_brain(ChapterBrainRequest {
            pages: vec![
                BrainPageInput {
                    index: 1,
                    input_path: first.clone(),
                    output_path: first,
                },
                BrainPageInput {
                    index: 2,
                    input_path: second.clone(),
                    output_path: second,
                },
            ],
            ..ChapterBrainRequest::default()
        })
        .unwrap();

        assert!(report
            .pages
            .iter()
            .any(|page| page.palette_drift_score > 0.12));
    }

    #[test]
    fn safety_gate_flags_bubble_risk() {
        let dir = tempfile::tempdir().unwrap();
        let quality_path = dir.path().join("page_001.quality.json");
        let quality = QualityReport {
            line_preservation_score: 0.98,
            bubble_cleanliness_score: 0.72,
            protected_region_violation_score: 0.70,
            hint_adherence_score: 0.90,
            output_sharpness_estimate: 0.40,
        };
        std::fs::write(&quality_path, serde_json::to_string(&quality).unwrap()).unwrap();

        let report = analyze_safety_gate(SafetyGateRequest {
            pages: vec![SafetyGatePageInput {
                index: 1,
                input_path: dir.path().join("input.png"),
                output_path: dir.path().join("output.png"),
                quality_report_path: Some(quality_path),
            }],
            thresholds: SafetyGateThresholds::default(),
        })
        .unwrap();

        assert!(!report.pass);
        assert_eq!(report.failed_pages, 1);
        assert_eq!(report.verdict, "safety_gate_text_or_bubble_risk");
    }

    fn write_blocks(path: &Path, skin: [u8; 3], accent: [u8; 3]) {
        let mut image = RgbImage::from_pixel(96, 96, Rgb([255, 255, 255]));
        for y in 8..48 {
            for x in 8..48 {
                image.put_pixel(x, y, Rgb(skin));
            }
        }
        for y in 48..88 {
            for x in 48..88 {
                image.put_pixel(x, y, Rgb(accent));
            }
        }
        for x in 0..96 {
            image.put_pixel(x, 0, Rgb([0, 0, 0]));
            image.put_pixel(x, 95, Rgb([0, 0, 0]));
        }
        image.save(path).unwrap();
    }
}
