use anyhow::{anyhow, Result};
use image::{GrayImage, Luma, Rgb, RgbImage};
use mangahue_core::ColorRgb;
use mangahue_inference::{
    ColorizationBackend, MockColorizationBackend, ModelInput, OnnxColorizationBackend,
};
use rayon::prelude::*;
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
        let results: Vec<BackendVariant> = all
            .par_iter()
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
                    ConfidenceCause::Blended => 80u8,
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

    fn palette(rgb: [u8; 3]) -> Vec<mangahue_core::ColorRgb> {
        vec![ColorRgb::new(rgb[0], rgb[1], rgb[2])]
    }
}
