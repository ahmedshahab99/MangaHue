use anyhow::{anyhow, Result};
use image::{imageops::FilterType, DynamicImage, GrayImage, Rgb, RgbImage};
use mangahue_core::EngineError;
use mangahue_inference::{ColorizationBackend, ModelInput};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyramidConfig {
    pub enabled: bool,
    pub levels: Vec<u32>,
    pub blend_weights: Vec<f32>,
    pub upsample_filter: UpsampleFilter,
}

impl Default for PyramidConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            levels: vec![256, 512, 768],
            blend_weights: vec![0.25, 0.35, 0.4],
            upsample_filter: UpsampleFilter::Lanczos3,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum UpsampleFilter {
    Nearest,
    Bilinear,
    Lanczos3,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PyramidReport {
    pub levels_used: usize,
    pub per_level_ms: Vec<u128>,
    pub total_ms: u128,
    pub blend_weights: Vec<f32>,
}

pub struct PyramidColorizer {
    backend: Arc<dyn ColorizationBackend>,
    config: PyramidConfig,
    cancel: Option<Arc<AtomicBool>>,
}

impl PyramidColorizer {
    pub fn new(backend: Arc<dyn ColorizationBackend>, config: PyramidConfig) -> Self {
        Self {
            backend,
            config,
            cancel: None,
        }
    }

    pub fn with_cancel(mut self, cancel: Option<Arc<AtomicBool>>) -> Self {
        self.cancel = cancel;
        self
    }

    pub fn colorize_with_report(
        &self,
        lineart: &GrayImage,
        hint_map: &RgbImage,
        hint_mask: &GrayImage,
    ) -> Result<(RgbImage, PyramidReport)> {
        let total_start = Instant::now();
        let target_w = lineart.width();
        let target_h = lineart.height();
        let target_dim = target_w.max(target_h);
        if target_dim == 0 {
            return Err(anyhow!("cannot colorize an empty pyramid input"));
        }

        let mut levels = self
            .config
            .levels
            .iter()
            .copied()
            .filter(|level| *level <= target_dim)
            .collect::<Vec<_>>();
        if levels.is_empty() {
            levels.push(target_dim);
            info!(
                resolution = target_dim,
                "using native-size pyramid fallback for a small tile"
            );
        }

        let mut accumulated = RgbImage::new(0, 0);
        let mut per_level_ms = Vec::new();
        // Index of the previous APPLIED level (levels larger than the tile
        // are skipped and must not consume a blend weight).
        let mut applied_count = 0usize;

        for (i, &level) in levels.iter().enumerate() {
            if let Some(cancel) = &self.cancel {
                if cancel.load(Ordering::Acquire) {
                    return Err(anyhow!(
                        "colorization cancelled before pyramid level {i} ({level}px)"
                    ));
                }
            }

            let level_start = Instant::now();

            let scale = level as f32 / target_dim as f32;
            let lw = (target_w as f32 * scale).round().max(1.0) as u32;
            let lh = (target_h as f32 * scale).round().max(1.0) as u32;

            let lineart_scaled = DynamicImage::ImageLuma8(lineart.clone())
                .resize_exact(lw, lh, FilterType::Lanczos3)
                .to_luma8();
            let hint_map_scaled = DynamicImage::ImageRgb8(hint_map.clone())
                .resize_exact(lw, lh, FilterType::Lanczos3)
                .to_rgb8();
            let hint_mask_scaled = DynamicImage::ImageLuma8(hint_mask.clone())
                .resize_exact(lw, lh, FilterType::Lanczos3)
                .to_luma8();

            let input = ModelInput {
                lineart: lineart_scaled,
                hint_map: hint_map_scaled,
                hint_mask: hint_mask_scaled,
            };

            let level_output = self
                .backend
                .colorize(&input)
                .map_err(|e| anyhow!("pyramid level {i} ({}px) failed: {e}", level))?;

            let upsampled = DynamicImage::ImageRgb8(level_output)
                .resize_exact(target_w, target_h, filter_for(&self.config.upsample_filter))
                .to_rgb8();

            if accumulated.width() == 0 {
                accumulated = upsampled;
            } else {
                let weight = self
                    .config
                    .blend_weights
                    .get(applied_count)
                    .copied()
                    .unwrap_or(1.0 / (applied_count + 1) as f32);
                accumulated = blend_images(&accumulated, &upsampled, weight);
            }
            applied_count += 1;

            per_level_ms.push(level_start.elapsed().as_millis());
            info!(
                level = i,
                resolution = level,
                ms = per_level_ms.last().unwrap(),
                "pyramid level complete"
            );
        }

        if accumulated.width() == 0 {
            return Err(anyhow!(
                "no pyramid levels were applicable for tile {}x{}",
                target_w,
                target_h
            ));
        }

        let total_ms = total_start.elapsed().as_millis();
        Ok((
            accumulated,
            PyramidReport {
                levels_used: applied_count,
                per_level_ms,
                total_ms,
                blend_weights: self.config.blend_weights.clone(),
            },
        ))
    }
}

impl ColorizationBackend for PyramidColorizer {
    fn name(&self) -> &str {
        "pyramid"
    }

    fn prefers_original_grayscale(&self) -> bool {
        // DDColor-family models must receive the ORIGINAL grayscale (not the
        // whitened clean lineart) to produce chroma; the pyramid wrapper must
        // forward the underlying backend's requirement or the orchestrator
        // feeds it lineart that produces near-zero chroma.
        self.backend.prefers_original_grayscale()
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        self.colorize_with_report(&input.lineart, &input.hint_map, &input.hint_mask)
            .map(|(image, _report)| image)
            .map_err(|e| EngineError::Inference(e.to_string()))
    }
}

fn filter_for(filter: &UpsampleFilter) -> FilterType {
    match filter {
        UpsampleFilter::Nearest => FilterType::Nearest,
        UpsampleFilter::Bilinear => FilterType::Triangle,
        UpsampleFilter::Lanczos3 => FilterType::Lanczos3,
    }
}

fn blend_images(base: &RgbImage, overlay: &RgbImage, overlay_weight: f32) -> RgbImage {
    let mut out = RgbImage::new(base.width(), base.height());
    let weight = overlay_weight.clamp(0.0, 1.0);
    let base_w = 1.0 - weight;
    for y in 0..base.height() {
        for x in 0..base.width() {
            let b = base.get_pixel(x, y);
            let o = overlay.get_pixel(x, y);
            out.put_pixel(
                x,
                y,
                Rgb([
                    (b[0] as f32 * base_w + o[0] as f32 * weight)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                    (b[1] as f32 * base_w + o[1] as f32 * weight)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                    (b[2] as f32 * base_w + o[2] as f32 * weight)
                        .round()
                        .clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn pyramid_blends_levels() {
        let backend: Arc<dyn ColorizationBackend> =
            Arc::new(mangahue_inference::MockColorizationBackend::default());
        let config = PyramidConfig::default();
        let colorizer = PyramidColorizer::new(backend, config);

        let w = 768u32;
        let h = 512u32;
        let lineart = GrayImage::from_pixel(w, h, Luma([200]));
        let hint_map = RgbImage::from_pixel(w, h, Rgb([0, 0, 0]));
        let hint_mask = GrayImage::from_pixel(w, h, Luma([0]));

        let (result, report) = colorizer
            .colorize_with_report(&lineart, &hint_map, &hint_mask)
            .unwrap();

        assert_eq!(result.dimensions(), (w, h));
        assert_eq!(report.levels_used, 3);
        assert!(!report.per_level_ms.is_empty());
    }

    #[test]
    fn pyramid_uses_native_fallback_for_small_tiles() {
        let backend: Arc<dyn ColorizationBackend> =
            Arc::new(mangahue_inference::MockColorizationBackend::default());
        let config = PyramidConfig {
            levels: vec![768],
            blend_weights: vec![1.0],
            ..PyramidConfig::default()
        };
        let colorizer = PyramidColorizer::new(backend, config);
        let lineart = GrayImage::from_pixel(384, 512, Luma([200]));
        let hint_map = RgbImage::from_pixel(384, 512, Rgb([0, 0, 0]));
        let hint_mask = GrayImage::from_pixel(384, 512, Luma([0]));

        let (result, report) = colorizer
            .colorize_with_report(&lineart, &hint_map, &hint_mask)
            .unwrap();

        assert_eq!(result.dimensions(), (384, 512));
        assert_eq!(report.levels_used, 1);
    }
}
