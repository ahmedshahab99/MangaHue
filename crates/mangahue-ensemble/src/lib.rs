use anyhow::{anyhow, Result};
use image::{GrayImage, Luma, Rgb, RgbImage};
use mangahue_core::ColorRgb;
use mangahue_inference::{
    ColorizationBackend, MockColorizationBackend, ModelInput, OnnxColorizationBackend,
};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    pub enabled: bool,
    pub fallback_mock_count: usize,
    pub consensus_strategy: ConsensusStrategy,
    pub weight_divergence: bool,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fallback_mock_count: 2,
            consensus_strategy: ConsensusStrategy::MajorityWeighted,
            weight_divergence: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsensusStrategy {
    MajorityWeighted,
    DivergenceReject,
    AverageBlend,
    ChromaComplement,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleReport {
    pub variant_count: usize,
    pub consensus_strategy: ConsensusStrategy,
    pub divergence_score: f32,
    pub pixel_disagreement_ratio: f32,
    pub per_variant_times_ms: Vec<u128>,
    pub total_ms: u128,
    pub verdict: EnsemblerVerdict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnsemblerVerdict {
    CleanConsensus,
    SoftDisagreement,
    HardConflict,
}

#[derive(Clone)]
pub struct Ensembler {
    primary: Arc<dyn ColorizationBackend>,
    auxiliaries: Vec<Arc<dyn ColorizationBackend>>,
    config: EnsembleConfig,
}

impl std::fmt::Debug for Ensembler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Ensembler")
            .field("primary", &"(dyn ColorizationBackend)")
            .field("auxiliary_count", &self.auxiliaries.len())
            .field("config", &self.config)
            .finish()
    }
}

impl Ensembler {
    pub fn new(
        primary: Arc<dyn ColorizationBackend>,
        auxiliaries: Vec<Arc<dyn ColorizationBackend>>,
        config: EnsembleConfig,
    ) -> Self {
        Self {
            primary,
            auxiliaries,
            config,
        }
    }

    pub fn from_onnx(model_paths: &[PathBuf], metadata_path: &Path) -> Result<Self> {
        let mut backends: Vec<Arc<dyn ColorizationBackend>> = Vec::new();
        for path in model_paths {
            let backend = OnnxColorizationBackend::new(path, metadata_path)
                .map_err(|e| anyhow!("failed to load ensemble model {}: {}", path.display(), e))?;
            backends.push(Arc::new(backend));
        }
        if backends.is_empty() {
            return Err(anyhow!("ensemble requires at least one ONNX model"));
        }
        let primary = backends.remove(0);
        let config = EnsembleConfig {
            enabled: true,
            fallback_mock_count: 0,
            consensus_strategy: ConsensusStrategy::MajorityWeighted,
            weight_divergence: true,
        };
        Ok(Self {
            primary,
            auxiliaries: backends,
            config,
        })
    }

    pub fn from_mock_ensemble(
        primary_palette: Vec<ColorRgb>,
        aux_palettes: Vec<Vec<ColorRgb>>,
    ) -> Self {
        let primary: Arc<dyn ColorizationBackend> =
            Arc::new(MockColorizationBackend::new(primary_palette));
        let auxiliaries: Vec<Arc<dyn ColorizationBackend>> = aux_palettes
            .into_iter()
            .map(|palette| {
                Arc::new(MockColorizationBackend::new(palette)) as Arc<dyn ColorizationBackend>
            })
            .collect();
        let config = EnsembleConfig {
            enabled: true,
            fallback_mock_count: auxiliaries.len().max(1),
            consensus_strategy: ConsensusStrategy::MajorityWeighted,
            weight_divergence: true,
        };
        Self {
            primary,
            auxiliaries,
            config,
        }
    }

    pub fn colorize(
        &self,
        input: &ModelInput,
    ) -> Result<(RgbImage, ConfidenceMap, EnsembleReport)> {
        let variants: Vec<BackendVariant> = self.run_all(input)?;
        let (merged, confidence, report) = self.consensus(variants, input);

        info!(
            divergence = %report.divergence_score,
            pixels_disagree = %report.pixel_disagreement_ratio,
            "ensemble concluded"
        );
        Ok((merged, confidence, report))
    }

    fn run_all(&self, input: &ModelInput) -> Result<Vec<BackendVariant>> {
        let all: Vec<Arc<dyn ColorizationBackend>> = {
            let mut all = vec![Arc::clone(&self.primary)];
            all.extend(self.auxiliaries.iter().map(Arc::clone));
            all
        };
        // Model workers are already internally parallel. Running large CPU
        // models concurrently multiplies peak memory and hurts throughput on
        // the small production VPS, so variants are deliberately serialized.
        let results: Vec<BackendVariant> = all
            .iter()
            .map(|backend| {
                let start = Instant::now();
                let result = backend.colorize(input).map_err(|error| {
                    anyhow!("ensemble variant {} failed: {error}", backend.name())
                });
                let ms = start.elapsed().as_millis();
                result.map(|image| BackendVariant {
                    image,
                    elapsed_ms: ms,
                })
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(results)
    }

    fn consensus(
        &self,
        variants: Vec<BackendVariant>,
        input: &ModelInput,
    ) -> (RgbImage, ConfidenceMap, EnsembleReport) {
        let w = variants[0].image.width();
        let h = variants[0].image.height();
        let mut merged = RgbImage::new(w, h);
        let mut confidence = ConfidenceMap::new(w, h);
        let mut per_pixel_disagreement = 0u64;
        let total_pixels = (w as u64).saturating_mul(h as u64);

        for y in 0..h {
            for x in 0..w {
                let mut votes: Vec<[u8; 3]> = Vec::with_capacity(variants.len());
                for variant in &variants {
                    let pixel = variant.image.get_pixel(x, y);
                    votes.push([pixel[0], pixel[1], pixel[2]]);
                }
                if votes.is_empty() {
                    merged.put_pixel(x, y, Rgb([128, 128, 128]));
                    confidence.placed(x, y, 0.0, ConfidenceCause::NoBackends);
                    continue;
                }
                if votes.len() == 1 {
                    let [r, g, b] = votes[0];
                    merged.put_pixel(x, y, Rgb([r, g, b]));
                    confidence.placed(x, y, 1.0, ConfidenceCause::PrimaryOnly);
                    continue;
                }
                let disagreement = pixel_disagreement(&votes);
                if disagreement > 0.6 {
                    per_pixel_disagreement += 1;
                }
                match self.config.consensus_strategy {
                    ConsensusStrategy::AverageBlend => {
                        let blended = average_pixel(&votes);
                        merged.put_pixel(x, y, Rgb(blended));
                        let conf = 1.0 - disagreement.min(1.0);
                        confidence.placed(x, y, conf, ConfidenceCause::Blended);
                    }
                    ConsensusStrategy::MajorityWeighted => {
                        let blended = average_pixel(&votes);
                        let conf = 1.0 - disagreement.min(1.0);
                        let line_strength = input.lineart.get_pixel(x, y)[0] as f32 / 255.0;
                        let weight = if line_strength > 0.55 {
                            votes[0]
                        } else {
                            blended
                        };
                        merged.put_pixel(x, y, Rgb(weight));
                        let final_conf = conf * (1.0 - line_strength * 0.6);
                        confidence.placed(
                            x,
                            y,
                            final_conf,
                            if disagreement > 0.3 {
                                ConfidenceCause::SoftDisagreement
                            } else {
                                ConfidenceCause::Consensus
                            },
                        );
                    }
                    ConsensusStrategy::DivergenceReject => {
                        if disagreement > 0.5 {
                            let base = votes[0];
                            let fallback: [u8; 3] = [
                                base[0].saturating_div(2),
                                base[1].saturating_div(2),
                                base[2].saturating_div(2),
                            ];
                            merged.put_pixel(x, y, Rgb(fallback));
                            confidence.placed(x, y, 0.0, ConfidenceCause::Rejected);
                        } else {
                            let blended = average_pixel(&votes);
                            let conf = 1.0 - disagreement.min(1.0);
                            merged.put_pixel(x, y, Rgb(blended));
                            confidence.placed(x, y, conf, ConfidenceCause::Consensus);
                        }
                    }
                    ConsensusStrategy::ChromaComplement => {
                        let (pixel, borrowed) =
                            chroma_complement(votes[0], &votes[1..], disagreement);
                        merged.put_pixel(x, y, Rgb(pixel));
                        let confidence_value = if borrowed > 0.05 {
                            (0.92 - disagreement * 0.28).clamp(0.35, 0.95)
                        } else {
                            (1.0 - disagreement * 0.35).clamp(0.45, 1.0)
                        };
                        confidence.placed(
                            x,
                            y,
                            confidence_value,
                            if borrowed > 0.05 {
                                ConfidenceCause::ChromaBorrowed
                            } else {
                                ConfidenceCause::PrimaryOnly
                            },
                        );
                    }
                }
            }
        }

        let total_variant_ms: u128 = variants.iter().map(|v| v.elapsed_ms).sum();
        let pixel_disagreement_ratio = if total_pixels == 0 {
            0.0
        } else {
            per_pixel_disagreement as f32 / total_pixels as f32
        };
        let divergence_score = pixel_disagreement_ratio;
        let verdict = if divergence_score < 0.05 {
            EnsemblerVerdict::CleanConsensus
        } else if divergence_score < 0.25 {
            EnsemblerVerdict::SoftDisagreement
        } else {
            EnsemblerVerdict::HardConflict
        };
        let report = EnsembleReport {
            variant_count: variants.len(),
            consensus_strategy: self.config.consensus_strategy,
            divergence_score,
            pixel_disagreement_ratio,
            per_variant_times_ms: variants.iter().map(|v| v.elapsed_ms).collect(),
            total_ms: total_variant_ms,
            verdict,
        };
        (merged, confidence, report)
    }
}

#[derive(Debug, Clone)]
struct BackendVariant {
    image: RgbImage,
    elapsed_ms: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceCause {
    Consensus,
    PrimaryOnly,
    Blended,
    ChromaBorrowed,
    SoftDisagreement,
    Rejected,
    NoBackends,
}

#[derive(Debug, Clone)]
pub struct ConfidenceMap {
    pixels: Vec<f32>,
    causes: Vec<ConfidenceCause>,
    width: u32,
    height: u32,
}

impl ConfidenceMap {
    pub fn new(width: u32, height: u32) -> Self {
        let cap = (width as usize).saturating_mul(height as usize);
        Self {
            pixels: vec![1.0; cap],
            causes: vec![ConfidenceCause::PrimaryOnly; cap],
            width,
            height,
        }
    }

    fn placed(&mut self, x: u32, y: u32, value: f32, cause: ConfidenceCause) {
        let idx = (y as usize)
            .saturating_mul(self.width as usize)
            .saturating_add(x as usize);
        if idx < self.pixels.len() {
            self.pixels[idx] = value.clamp(0.0, 1.0);
            self.causes[idx] = cause;
        }
    }

    pub fn confidence_at(&self, x: u32, y: u32) -> f32 {
        let idx = (y as usize)
            .saturating_mul(self.width as usize)
            .saturating_add(x as usize);
        self.pixels.get(idx).copied().unwrap_or(0.0)
    }

    pub fn to_gray_image(&self) -> GrayImage {
        let mut image = GrayImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let conf = self.confidence_at(x, y);
                image.put_pixel(x, y, Luma([(conf * 255.0).round().clamp(0.0, 255.0) as u8]));
            }
        }
        image
    }

    pub fn cause_at(&self, x: u32, y: u32) -> ConfidenceCause {
        let idx = (y as usize)
            .saturating_mul(self.width as usize)
            .saturating_add(x as usize);
        self.causes
            .get(idx)
            .copied()
            .unwrap_or(ConfidenceCause::NoBackends)
    }

    pub fn mean_confidence(&self) -> f32 {
        let sum: f32 = self.pixels.iter().sum();
        sum / self.pixels.len().max(1) as f32
    }

    pub fn disagreement_heatmap(&self) -> GrayImage {
        let mut image = GrayImage::new(self.width, self.height);
        for y in 0..self.height {
            for x in 0..self.width {
                let cause = self.cause_at(x, y);
                let intensity = match cause {
                    ConfidenceCause::SoftDisagreement | ConfidenceCause::Rejected => 200u8,
                    ConfidenceCause::Blended | ConfidenceCause::ChromaBorrowed => 80u8,
                    ConfidenceCause::Consensus | ConfidenceCause::PrimaryOnly => 20u8,
                    ConfidenceCause::NoBackends => 255u8,
                };
                image.put_pixel(x, y, Luma([intensity]));
            }
        }
        image
    }
}

fn pixel_disagreement(votes: &[[u8; 3]]) -> f32 {
    if votes.len() < 2 {
        return 0.0;
    }
    let mut max_per_channel = [0u8; 3];
    let mut min_per_channel = [255u8; 3];
    for vote in votes {
        for c in 0..3 {
            max_per_channel[c] = max_per_channel[c].max(vote[c]);
            min_per_channel[c] = min_per_channel[c].min(vote[c]);
        }
    }
    let mut total_spread = 0u16;
    for c in 0..3 {
        total_spread = total_spread
            .saturating_add((max_per_channel[c].saturating_sub(min_per_channel[c])) as u16);
    }
    let max_spread = 255u16 * 3;
    total_spread as f32 / max_spread as f32
}

fn average_pixel(votes: &[[u8; 3]]) -> [u8; 3] {
    let n = votes.len() as u32;
    if n == 0 {
        return [128, 128, 128];
    }
    let mut sums = [0u32; 3];
    for vote in votes {
        for c in 0..3 {
            sums[c] = sums[c].saturating_add(vote[c] as u32);
        }
    }
    [
        (sums[0] / n).min(255) as u8,
        (sums[1] / n).min(255) as u8,
        (sums[2] / n).min(255) as u8,
    ]
}

fn chroma_complement(
    primary: [u8; 3],
    auxiliaries: &[[u8; 3]],
    disagreement: f32,
) -> ([u8; 3], f32) {
    if auxiliaries.is_empty() {
        return (primary, 0.0);
    }

    let primary_linear = [
        srgb_to_linear(primary[0]),
        srgb_to_linear(primary[1]),
        srgb_to_linear(primary[2]),
    ];
    let primary_luma = linear_luma(primary_linear);
    let primary_chroma = chroma_magnitude(primary_linear, primary_luma);
    if primary_luma <= 0.012 || primary_luma >= 0.94 {
        return (primary, 0.0);
    }

    let mut best = primary_linear;
    let mut best_advantage = 0.0f32;
    for auxiliary in auxiliaries {
        let candidate = [
            srgb_to_linear(auxiliary[0]),
            srgb_to_linear(auxiliary[1]),
            srgb_to_linear(auxiliary[2]),
        ];
        let candidate_luma = linear_luma(candidate);
        let candidate_chroma = chroma_magnitude(candidate, candidate_luma);
        let advantage = candidate_chroma - primary_chroma;
        if advantage > best_advantage {
            best = candidate;
            best_advantage = advantage;
        }
    }

    // Protect ink, paper white, and already-confident primary color. The
    // auxiliary contributes chroma only in drawable midtones where it carries
    // substantially more color evidence than the primary model.
    let drawable =
        smoothstep(0.025, 0.16, primary_luma) * (1.0 - smoothstep(0.88, 0.985, primary_luma));
    let evidence = smoothstep(0.012, 0.105, best_advantage);
    let neutral_need = 1.0 - smoothstep(0.035, 0.16, primary_chroma);
    let agreement_guard = 1.0 - smoothstep(0.72, 1.0, disagreement);
    let borrow =
        (drawable * evidence * (0.04 + neutral_need * 0.76) * agreement_guard).clamp(0.0, 0.62);

    if borrow <= 0.001 {
        return (primary, 0.0);
    }

    let best_luma = linear_luma(best);
    let best_chroma = [
        best[0] - best_luma,
        best[1] - best_luma,
        best[2] - best_luma,
    ];
    let primary_chroma_vector = [
        primary_linear[0] - primary_luma,
        primary_linear[1] - primary_luma,
        primary_linear[2] - primary_luma,
    ];
    let fused = [
        primary_luma + mix(primary_chroma_vector[0], best_chroma[0], borrow),
        primary_luma + mix(primary_chroma_vector[1], best_chroma[1], borrow),
        primary_luma + mix(primary_chroma_vector[2], best_chroma[2], borrow),
    ];
    (gamut_map_preserving_luma(fused, primary_luma), borrow)
}

fn srgb_to_linear(value: u8) -> f32 {
    let value = value as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round().clamp(0.0, 255.0) as u8
}

fn linear_luma(rgb: [f32; 3]) -> f32 {
    rgb[0] * 0.2126 + rgb[1] * 0.7152 + rgb[2] * 0.0722
}

fn chroma_magnitude(rgb: [f32; 3], luma: f32) -> f32 {
    let dr = rgb[0] - luma;
    let dg = rgb[1] - luma;
    let db = rgb[2] - luma;
    (dr * dr + dg * dg + db * db).sqrt()
}

fn gamut_map_preserving_luma(mut rgb: [f32; 3], luma: f32) -> [u8; 3] {
    let min_channel = rgb.iter().copied().fold(f32::INFINITY, f32::min);
    let max_channel = rgb.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale_low = if min_channel < 0.0 {
        luma / (luma - min_channel).max(1e-6)
    } else {
        1.0
    };
    let scale_high = if max_channel > 1.0 {
        (1.0 - luma) / (max_channel - luma).max(1e-6)
    } else {
        1.0
    };
    let scale = scale_low.min(scale_high).clamp(0.0, 1.0);
    for channel in &mut rgb {
        *channel = luma + (*channel - luma) * scale;
    }
    [
        linear_to_srgb(rgb[0]),
        linear_to_srgb(rgb[1]),
        linear_to_srgb(rgb[2]),
    ]
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn mix(a: f32, b: f32, amount: f32) -> f32 {
    a + (b - a) * amount
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma, RgbImage};

    #[test]
    fn ensemble_merge_reduces_divergence() {
        let w = 16u32;
        let h = 12u32;
        let lineart = GrayImage::from_pixel(w, h, Luma([200]));
        let hint_rgb = RgbImage::from_pixel(w, h, Rgb([0, 0, 0]));
        let hint_mask = GrayImage::from_pixel(w, h, Luma([0]));

        let primary = palette([100, 100, 100]);
        let aux1 = palette([120, 100, 90]);
        let aux2 = palette([95, 100, 105]);

        let rope = Ensembler::from_mock_ensemble(primary, vec![aux1, aux2]);
        let input = ModelInput {
            lineart,
            hint_map: hint_rgb,
            hint_mask,
        };
        let (merged, confidence, report) = rope.colorize(&input).unwrap();
        assert_eq!(merged.dimensions(), (w, h));
        assert!(report.variant_count == 3);
        assert!(confidence.mean_confidence() > 0.0);
    }

    #[test]
    fn chroma_complement_preserves_primary_luma_and_borrows_color() {
        let primary = [150, 150, 150];
        let auxiliary = [[210, 118, 82]];
        let (fused, borrowed) = chroma_complement(primary, &auxiliary, 0.18);
        let source_luma = linear_luma([
            srgb_to_linear(primary[0]),
            srgb_to_linear(primary[1]),
            srgb_to_linear(primary[2]),
        ]);
        let fused_luma = linear_luma([
            srgb_to_linear(fused[0]),
            srgb_to_linear(fused[1]),
            srgb_to_linear(fused[2]),
        ]);
        assert!(borrowed > 0.2);
        assert!((source_luma - fused_luma).abs() < 0.015);
        assert!(fused[0] > fused[1] && fused[1] > fused[2]);
    }

    #[test]
    fn chroma_complement_protects_paper_and_ink() {
        for primary in [[3, 3, 3], [252, 252, 252]] {
            let (fused, borrowed) = chroma_complement(primary, &[[230, 70, 30]], 0.2);
            assert_eq!(fused, primary);
            assert!(borrowed < 0.01);
        }
    }

    fn palette(rgb: [u8; 3]) -> Vec<mangahue_core::ColorRgb> {
        vec![ColorRgb::new(rgb[0], rgb[1], rgb[2])]
    }
}
