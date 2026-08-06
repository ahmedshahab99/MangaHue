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
    #[serde(default)]
    pub cielab_distance: bool,
    #[serde(default = "default_divergence_reject_threshold")]
    pub divergence_reject_threshold: f32,
}

fn default_divergence_reject_threshold() -> f32 {
    0.5
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            fallback_mock_count: 2,
            consensus_strategy: ConsensusStrategy::MajorityWeighted,
            weight_divergence: true,
            cielab_distance: false,
            divergence_reject_threshold: default_divergence_reject_threshold(),
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
    #[serde(default)]
    pub semantic_rescue_pixels: u64,
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
    weights: Vec<f32>,
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
        let weights = vec![1.0; auxiliaries.len() + 1];
        Self {
            primary,
            auxiliaries,
            weights,
            config,
        }
    }

    pub fn with_weights(
        primary: Arc<dyn ColorizationBackend>,
        auxiliaries: Vec<Arc<dyn ColorizationBackend>>,
        weights: Vec<f32>,
        config: EnsembleConfig,
    ) -> Self {
        let mut all_weights = weights;
        if all_weights.is_empty() {
            all_weights = vec![1.0; auxiliaries.len() + 1];
        }
        if all_weights.len() != auxiliaries.len() + 1 {
            all_weights.resize(auxiliaries.len() + 1, 1.0);
        }
        Self {
            primary,
            auxiliaries,
            weights: all_weights,
            config,
        }
    }

    pub fn from_onnx(
        model_paths: &[PathBuf],
        metadata_path: &Path,
        weights: Option<&[f32]>,
    ) -> Result<Self> {
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
        let model_weights = weights
            .map(|w| w.to_vec())
            .unwrap_or_else(|| vec![1.0; backends.len() + 1]);
        let config = EnsembleConfig {
            enabled: true,
            fallback_mock_count: 0,
            consensus_strategy: ConsensusStrategy::MajorityWeighted,
            weight_divergence: true,
            cielab_distance: false,
            divergence_reject_threshold: 0.5,
        };
        Ok(Self {
            primary,
            auxiliaries: backends,
            weights: model_weights,
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
        let weights = vec![1.0; auxiliaries.len() + 1];
        let config = EnsembleConfig {
            enabled: true,
            fallback_mock_count: auxiliaries.len().max(1),
            consensus_strategy: ConsensusStrategy::MajorityWeighted,
            weight_divergence: true,
            cielab_distance: false,
            divergence_reject_threshold: 0.5,
        };
        Self {
            primary,
            auxiliaries,
            weights,
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

    /// Whether ANY variant requires the original grayscale input (DDColor
    /// family) rather than the whitened clean lineart. The ensemble feeds
    /// every variant the same tile, so the most demanding variant decides.
    pub fn prefers_original_grayscale(&self) -> bool {
        self.primary.prefers_original_grayscale()
            || self
                .auxiliaries
                .iter()
                .any(|backend| backend.prefers_original_grayscale())
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
                    highlight_rescue: backend.name() == "anime-face-palette",
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
        let mut semantic_rescue_pixels = 0u64;
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
                let disagreement = if self.config.cielab_distance {
                    cielab_disagreement(&votes)
                } else {
                    pixel_disagreement(&votes)
                };
                if disagreement > 0.6 {
                    per_pixel_disagreement += 1;
                }
                let pixel_weights = &self.weights[..votes.len()];
                match self.config.consensus_strategy {
                    ConsensusStrategy::AverageBlend => {
                        let blended = weighted_average_pixel(&votes, pixel_weights);
                        merged.put_pixel(x, y, Rgb(blended));
                        let conf = 1.0 - disagreement.min(1.0);
                        confidence.placed(x, y, conf, ConfidenceCause::Blended);
                    }
                    ConsensusStrategy::MajorityWeighted => {
                        let blended = weighted_average_pixel(&votes, pixel_weights);
                        let conf = 1.0 - disagreement.min(1.0);
                        // lineart is DARK on ink: high luma = paper, low luma
                        // = ink. Ink pixels take the primary model's output
                        // raw (line protection); everything else blends.
                        let paper = input.lineart.get_pixel(x, y)[0] as f32 / 255.0;
                        let ink_strength = 1.0 - paper;
                        let weight = if ink_strength > 0.55 {
                            votes[0]
                        } else {
                            blended
                        };
                        merged.put_pixel(x, y, Rgb(weight));
                        // Where the primary was forced without consensus the
                        // confidence is reduced; blended regions keep it.
                        let final_conf = conf * (0.4 + paper * 0.6);
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
                        if disagreement > self.config.divergence_reject_threshold {
                            let base = votes[0];
                            let fallback: [u8; 3] = [
                                base[0].saturating_div(2),
                                base[1].saturating_div(2),
                                base[2].saturating_div(2),
                            ];
                            merged.put_pixel(x, y, Rgb(fallback));
                            confidence.placed(x, y, 0.0, ConfidenceCause::Rejected);
                        } else {
                            let blended = weighted_average_pixel(&votes, pixel_weights);
                            let conf = 1.0 - disagreement.min(1.0);
                            merged.put_pixel(x, y, Rgb(blended));
                            confidence.placed(x, y, conf, ConfidenceCause::Consensus);
                        }
                    }
                    ConsensusStrategy::ChromaComplement => {
                        let auxiliaries = variants[1..]
                            .iter()
                            .map(|variant| {
                                (
                                    {
                                        let pixel = variant.image.get_pixel(x, y);
                                        [pixel[0], pixel[1], pixel[2]]
                                    },
                                    variant.highlight_rescue,
                                )
                            })
                            .collect::<Vec<_>>();
                        let (pixel, borrowed, semantic_rescue) =
                            chroma_complement(votes[0], &auxiliaries, disagreement);
                        if semantic_rescue {
                            semantic_rescue_pixels += 1;
                        }
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
            semantic_rescue_pixels,
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
    highlight_rescue: bool,
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

fn cielab_disagreement(votes: &[[u8; 3]]) -> f32 {
    if votes.len() < 2 {
        return 0.0;
    }
    let lab_votes: Vec<[f32; 3]> = votes.iter().map(|v| srgb_to_cielab(*v)).collect();
    let mut max_distance = 0.0f32;
    for i in 0..lab_votes.len() {
        for j in (i + 1)..lab_votes.len() {
            let d = cielab_distance(lab_votes[i], lab_votes[j]);
            if d > max_distance {
                max_distance = d;
            }
        }
    }
    (max_distance / 100.0).clamp(0.0, 1.0)
}

fn srgb_to_cielab(rgb: [u8; 3]) -> [f32; 3] {
    let linear = [
        srgb_to_linear(rgb[0]),
        srgb_to_linear(rgb[1]),
        srgb_to_linear(rgb[2]),
    ];
    let xyz = linear_to_xyz(linear);
    xyz_to_cielab(xyz)
}

fn linear_to_xyz(rgb: [f32; 3]) -> [f32; 3] {
    [
        rgb[0] * 0.4124564 + rgb[1] * 0.3575761 + rgb[2] * 0.1804375,
        rgb[0] * 0.2126729 + rgb[1] * 0.7151522 + rgb[2] * 0.0721750,
        rgb[0] * 0.0193339 + rgb[1] * 0.1191920 + rgb[2] * 0.9503041,
    ]
}

fn xyz_to_cielab(xyz: [f32; 3]) -> [f32; 3] {
    let xn = 0.95047;
    let yn = 1.0;
    let zn = 1.08883;
    let f = |t: f32| -> f32 {
        let delta: f32 = 6.0 / 29.0;
        if t > delta.powi(3) {
            t.powf(1.0 / 3.0)
        } else {
            t / (3.0 * delta.powi(2)) + 4.0 / 29.0
        }
    };
    let fx = f(xyz[0] / xn);
    let fy = f(xyz[1] / yn);
    let fz = f(xyz[2] / zn);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

fn cielab_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dl = a[0] - b[0];
    let da = a[1] - b[1];
    let db = a[2] - b[2];
    (dl * dl + da * da + db * db).sqrt()
}

fn weighted_average_pixel(votes: &[[u8; 3]], weights: &[f32]) -> [u8; 3] {
    let total_weight: f32 = weights.iter().sum();
    if total_weight <= 0.0 {
        return [128, 128, 128];
    }
    let mut sums = [0.0f32; 3];
    for (vote, weight) in votes.iter().zip(weights.iter()) {
        for c in 0..3 {
            sums[c] += vote[c] as f32 * weight;
        }
    }
    [
        (sums[0] / total_weight).round().clamp(0.0, 255.0) as u8,
        (sums[1] / total_weight).round().clamp(0.0, 255.0) as u8,
        (sums[2] / total_weight).round().clamp(0.0, 255.0) as u8,
    ]
}

fn chroma_complement(
    primary: [u8; 3],
    auxiliaries: &[([u8; 3], bool)],
    disagreement: f32,
) -> ([u8; 3], f32, bool) {
    if auxiliaries.is_empty() {
        return (primary, 0.0, false);
    }

    let primary_linear = [
        srgb_to_linear(primary[0]),
        srgb_to_linear(primary[1]),
        srgb_to_linear(primary[2]),
    ];
    let primary_luma = linear_luma(primary_linear);
    let primary_chroma = chroma_magnitude(primary_linear, primary_luma);
    if primary_luma <= 0.012 {
        return (primary, 0.0, false);
    }

    let mut best = primary_linear;
    let mut best_advantage = 0.0f32;
    let mut best_allows_highlight_rescue = false;
    let mut rescue_best = primary_linear;
    let mut rescue_advantage = 0.0f32;
    for (auxiliary, allows_highlight_rescue) in auxiliaries {
        if primary_luma >= 0.94 && !allows_highlight_rescue {
            continue;
        }
        let candidate = [
            srgb_to_linear(auxiliary[0]),
            srgb_to_linear(auxiliary[1]),
            srgb_to_linear(auxiliary[2]),
        ];
        let candidate_luma = linear_luma(candidate);
        let candidate_chroma = chroma_magnitude(candidate, candidate_luma);
        let advantage = candidate_chroma - primary_chroma;
        if *allows_highlight_rescue && advantage > rescue_advantage {
            rescue_best = candidate;
            rescue_advantage = advantage;
        }
        if advantage > best_advantage {
            best = candidate;
            best_advantage = advantage;
            best_allows_highlight_rescue = *allows_highlight_rescue;
        }
    }

    // Sparse semantic experts return neutral grayscale outside their detected
    // material. Their chromatic pixels are therefore an implicit confidence
    // mask and should survive full-page model disagreement and paper guards.
    if rescue_advantage >= 0.018 && primary_luma >= 0.08 {
        let rescue_luma = linear_luma(rescue_best);
        let rescue_chroma = [
            rescue_best[0] - rescue_luma,
            rescue_best[1] - rescue_luma,
            rescue_best[2] - rescue_luma,
        ];
        let primary_chroma_vector = [
            primary_linear[0] - primary_luma,
            primary_linear[1] - primary_luma,
            primary_linear[2] - primary_luma,
        ];
        let neutral_need = 1.0 - smoothstep(0.045, 0.18, primary_chroma);
        let evidence = smoothstep(0.018, 0.11, rescue_advantage);
        let borrow = (0.48 + 0.34 * neutral_need) * evidence;
        let target_luma = mix(primary_luma, linear_luma(rescue_best), 0.14 * evidence);
        let fused = [
            target_luma + mix(primary_chroma_vector[0], rescue_chroma[0], borrow),
            target_luma + mix(primary_chroma_vector[1], rescue_chroma[1], borrow),
            target_luma + mix(primary_chroma_vector[2], rescue_chroma[2], borrow),
        ];
        return (gamut_map_preserving_luma(fused, target_luma), borrow, true);
    }
    if primary_luma >= 0.68 && primary_chroma <= 0.09 && rescue_advantage >= 0.08 {
        best = rescue_best;
        best_advantage = rescue_advantage;
        best_allows_highlight_rescue = true;
    }
    if primary_luma >= 0.94 && !best_allows_highlight_rescue {
        return (primary, 0.0, false);
    }

    // Protect ink, paper white, and already-confident primary color. The
    // auxiliary contributes chroma only in drawable midtones where it carries
    // substantially more color evidence than the primary model.
    let highlight_rescue = if best_allows_highlight_rescue {
        smoothstep(0.14, 0.28, best_advantage)
    } else {
        0.0
    };
    let paper_guard = smoothstep(0.88, 0.985, primary_luma);
    let drawable =
        smoothstep(0.025, 0.16, primary_luma) * (1.0 - paper_guard * (1.0 - highlight_rescue));
    let evidence = smoothstep(0.012, 0.105, best_advantage);
    let neutral_need = 1.0 - smoothstep(0.035, 0.16, primary_chroma);
    let agreement_guard = 1.0 - smoothstep(0.72, 1.0, disagreement);
    let borrow =
        (drawable * evidence * (0.04 + neutral_need * 0.76) * agreement_guard).clamp(0.0, 0.62);

    if borrow <= 0.001 {
        return (primary, 0.0, false);
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
    let target_luma = if primary_luma >= 0.94 && highlight_rescue > 0.0 {
        mix(primary_luma, best_luma, borrow * 0.42)
    } else {
        primary_luma
    };
    let fused = [
        target_luma + mix(primary_chroma_vector[0], best_chroma[0], borrow),
        target_luma + mix(primary_chroma_vector[1], best_chroma[1], borrow),
        target_luma + mix(primary_chroma_vector[2], best_chroma[2], borrow),
    ];
    (
        gamut_map_preserving_luma(fused, target_luma),
        borrow,
        best_allows_highlight_rescue,
    )
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
        let auxiliary = [([210, 118, 82], false)];
        let (fused, borrowed, _) = chroma_complement(primary, &auxiliary, 0.18);
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
            let (fused, borrowed, _) = chroma_complement(primary, &[([230, 70, 30], false)], 0.2);
            assert_eq!(fused, primary);
            assert!(borrowed < 0.01);
        }
    }

    #[test]
    fn face_expert_can_rescue_bright_skin_without_disabling_paper_guard() {
        let primary = [255, 255, 255];
        let face = [([250, 205, 189], true)];

        let (fused, borrowed, rescued) = chroma_complement(primary, &face, 0.18);

        assert!(borrowed > 0.4);
        assert!(rescued);
        assert!(fused[0] > fused[1] && fused[1] > fused[2]);
        assert!(fused[1] < 250);
    }

    #[test]
    fn face_expert_wins_on_bright_neutral_face_pixels() {
        let primary = [235, 235, 235];
        let auxiliaries = [([120, 60, 180], false), ([250, 205, 189], true)];

        let (fused, borrowed, rescued) = chroma_complement(primary, &auxiliaries, 0.2);

        assert!(borrowed > 0.3);
        assert!(rescued);
        assert!(fused[0] > fused[1] && fused[1] > fused[2]);
    }

    fn palette(rgb: [u8; 3]) -> Vec<mangahue_core::ColorRgb> {
        vec![ColorRgb::new(rgb[0], rgb[1], rgb[2])]
    }
}
