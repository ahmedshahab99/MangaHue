//! Pure-Rust anime face detection — a faithful port of the OpenCV LBP
//! cascade path the Python worker used (`cv2.CascadeClassifier` +
//! `detectMultiScale`), so face hints need no Python/OpenCV at runtime.
//!
//! Semantics replicated exactly from OpenCV 4.x (`cascadedetect.cpp`,
//! `cascadedetect.hpp`):
//!   - LBP feature value: 8-bit code comparing the 3x3 block neighborhood
//!     around the feature rect against the center block (>= center).
//!   - Categorical stump: `subsets[c >> 5] & (1 << (c & 31))` picks
//!     left/right leaf; stage fails when the accumulated leaves fall
//!     below `stageThreshold` (minus the 1e-5 read epsilon).
//!   - Scan: scale factor 1.08, window step 1 if scale >= 2 else 2,
//!     window fully inside the scaled image, `INTER_LINEAR_EXACT` resize
//!     per scale, 32-bit integral image.
//!   - Grouping: `groupRectangles(..., minNeighbors = 3, eps = 0.2)` with
//!     the default (count-based) weights.

use anyhow::{Context, Result};
use image::{GrayImage, Luma, Rgb, RgbImage};
use std::path::Path;
use std::sync::Arc;

use crate::{EngineError, ModelInput};

/// Scale factor of the detection pyramid (the worker's `scaleFactor`).
const PYRAMID_SCALE: f64 = 1.08;
/// Minimum neighbors for `groupRectangles` (the worker's `minNeighbors`).
const MIN_NEIGHBORS: usize = 3;
/// Grouping similarity epsilon (OpenCV's `GROUP_EPS`).
const GROUP_EPS: f64 = 0.2;
/// Maximum input dimension the worker resized pages to before scanning.
const MAX_DETECT_DIM: u32 = 1024;
/// LBP category count: 256 codes -> 8 x 32-bit subset masks per stump.
const SUBSET_WORDS: usize = 8;

/// A parsed LBP cascade (`lbpcascade_animeface.xml`).
pub struct LbpCascade {
    win_w: u32,
    win_h: u32,
    /// Stage thresholds already shifted by OpenCV's read epsilon.
    stages: Vec<Stage>,
    /// Per-stage contiguous stump tables.
    stumps: Vec<Stump>,
    /// Per-stump contiguous 8-word category subsets.
    subsets: Vec<[u32; SUBSET_WORDS]>,
    /// Feature rects (x, y, w, h) in the 24x24 window.
    features: Vec<[u32; 4]>,
}

struct Stage {
    threshold: f32,
    ntrees: usize,
}

struct Stump {
    feature_idx: usize,
    left: f32,
    right: f32,
}

/// Extract the text between the first `open` and the following `close`.
fn between<'a>(text: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = text.find(open)? + open.len();
    let end = text[start..].find(close)? + start;
    Some(&text[start..end])
}

fn split_ints(text: &str) -> Vec<i64> {
    text.split_whitespace()
        .filter_map(|token| token.parse::<i64>().ok())
        .collect()
}

fn split_floats(text: &str) -> Vec<f32> {
    text.split_whitespace()
        .filter_map(|token| token.parse::<f32>().ok())
        .collect()
}

/// OpenCV reads stage thresholds with a 1e-5 epsilon subtracted so an
/// exactly-equal accumulation passes.
const STAGE_THRESHOLD_EPS: f32 = 1e-5;

impl LbpCascade {
    /// Parse a cascade from its OpenCV storage XML.
    pub fn load(path: &Path) -> Result<Self> {
        let xml = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read cascade {}", path.display()))?;
        let stages_xml = between(&xml, "<stages>", "</stages>")
            .with_context(|| format!("no <stages> block in {}", path.display()))?;
        let features_xml = between(&xml, "<features>", "</features>")
            .with_context(|| format!("no <features> block in {}", path.display()))?;
        let width = between(&xml, "<width>", "</width>")
            .and_then(|v| v.trim().parse::<u32>().ok())
            .context("missing <width>")?;
        let height = between(&xml, "<height>", "</height>")
            .and_then(|v| v.trim().parse::<u32>().ok())
            .context("missing <height>")?;

        // Stage thresholds, in order.
        let thresholds: Vec<f32> = stages_xml
            .split("<stageThreshold>")
            .skip(1)
            .filter_map(|block| {
                let end = block.find("</stageThreshold>")?;
                block[..end].trim().parse::<f32>().ok()
            })
            .collect();
        // Weak classifier blocks, in order (one per stage).
        let weak_blocks: Vec<&str> = stages_xml
            .split("<weakClassifiers>")
            .skip(1)
            .filter_map(|block| {
                let end = block.find("</weakClassifiers>")?;
                Some(&block[..end])
            })
            .collect();
        if weak_blocks.len() != thresholds.len() {
            return Err(anyhow::Error::msg(format!(
                "cascade stage mismatch: {} thresholds, {} weak blocks",
                thresholds.len(),
                weak_blocks.len()
            )));
        }

        // Features: one <rect> per feature.
        let mut features = Vec::new();
        for block in features_xml.split("<rect>").skip(1) {
            let end = block.find("</rect>").context("unterminated <rect>")?;
            let values = split_ints(&block[..end]);
            if values.len() != 4 {
                return Err(anyhow::Error::msg(format!(
                    "feature rect with {} values",
                    values.len()
                )));
            }
            features.push([
                values[0] as u32,
                values[1] as u32,
                values[2] as u32,
                values[3] as u32,
            ]);
        }

        // Stumps + subsets per stage (a stump owns one node => one subset).
        let mut stages = Vec::with_capacity(thresholds.len());
        let mut stumps = Vec::new();
        let mut subsets = Vec::new();
        for (threshold, weak) in thresholds.iter().zip(&weak_blocks) {
            let mut ntrees = 0usize;
            let mut cursor = *weak;
            while let Some(pos) = cursor.find("<internalNodes>") {
                let tail = &cursor[pos + "<internalNodes>".len()..];
                let end = tail
                    .find("</internalNodes>")
                    .context("unterminated internalNodes")?;
                let values = split_ints(&tail[..end]);
                if values.len() != 3 + SUBSET_WORDS {
                    return Err(anyhow::Error::msg(format!(
                        "internalNodes with {} values (expected {})",
                        values.len(),
                        3 + SUBSET_WORDS
                    )));
                }
                let feature_idx = values[2] as usize;
                if feature_idx >= features.len() {
                    return Err(anyhow::Error::msg(format!(
                        "feature index {feature_idx} out of range"
                    )));
                }
                let mut subset = [0u32; SUBSET_WORDS];
                for (slot, value) in subset.iter_mut().zip(&values[3..]) {
                    *slot = *value as u32; // two's-complement bit pattern, like OpenCV's int
                }
                // Leaf values follow the internal node block.
                let leaf_tail = &tail[end + "</internalNodes>".len()..];
                let leaf_pos = leaf_tail.find("<leafValues>");
                let leaf_end = leaf_tail.find("</leafValues>");
                let (left, right) = match (leaf_pos, leaf_end) {
                    (Some(lp), Some(le)) if le > lp => {
                        let values = split_floats(&leaf_tail[lp + "<leafValues>".len()..le]);
                        if values.len() != 2 {
                            return Err(anyhow::Error::msg(format!(
                                "leafValues with {} values",
                                values.len()
                            )));
                        }
                        (values[0], values[1])
                    }
                    _ => return Err(anyhow::Error::msg("missing <leafValues> for a stump")),
                };
                stumps.push(Stump {
                    feature_idx,
                    left,
                    right,
                });
                subsets.push(subset);
                ntrees += 1;
                cursor = &leaf_tail[leaf_end.unwrap() + "</leafValues>".len()..];
            }
            if ntrees == 0 {
                return Err(anyhow::Error::msg("stage with no weak classifiers"));
            }
            stages.push(Stage {
                threshold: threshold - STAGE_THRESHOLD_EPS,
                ntrees,
            });
        }

        Ok(Self {
            win_w: width,
            win_h: height,
            stages,
            stumps,
            subsets,
            features,
        })
    }

    /// Detect faces in a grayscale page; returns (x, y, w, h) boxes in
    /// full-page coordinates, largest first (the worker's contract).
    pub fn detect(&self, gray: &GrayImage) -> Vec<[u32; 4]> {
        let (width, height) = gray.dimensions();
        if width < self.win_w || height < self.win_h {
            return Vec::new();
        }
        // Worker parity: pages larger than 1024 on the longest side are
        // pre-shrunk with area averaging.
        let pre_scale = (MAX_DETECT_DIM as f64 / width.max(height) as f64).min(1.0);
        let base: GrayImage = if pre_scale < 1.0 {
            resize_area(
                gray,
                ((width as f64 * pre_scale).round() as u32).max(1),
                ((height as f64 * pre_scale).round() as u32).max(1),
            )
        } else {
            gray.clone()
        };
        let (bw, bh) = base.dimensions();
        let minimum = 20.max(((bw.min(bh) as f64 * 0.045).round()) as u32);

        // Build the scale pyramid (window sizes, like OpenCV).
        let mut factor = 1.0f64;
        let mut scales: Vec<f64> = Vec::new();
        loop {
            let win = (self.win_w as f64 * factor).round() as u32;
            if win > bw || win > bh {
                break;
            }
            if win >= minimum {
                scales.push(factor);
            }
            factor *= PYRAMID_SCALE;
        }

        let mut candidates: Vec<(u32, u32, u32, u32)> = Vec::new();
        for &scale in &scales {
            let win = (self.win_w as f64 * scale).round() as u32;
            let (rw, rh) = (
                ((bw as f64) / scale).round() as u32,
                ((bh as f64) / scale).round() as u32,
            );
            let scaled = if (rw, rh) == (bw, bh) {
                base.clone()
            } else {
                resize_bilinear(&base, rw, rh)
            };
            let sum = integral32(&scaled);
            let iw = rw + 1;
            let step = if scale >= 2.0 { 1u32 } else { 2u32 };
            let mut y = 0u32;
            while y + self.win_h <= rh {
                let mut x = 0u32;
                while x + self.win_w <= rw {
                    if self.window_passes(&sum, iw, x, y) {
                        candidates.push((
                            ((x as f64) * scale).round() as u32,
                            ((y as f64) * scale).round() as u32,
                            win,
                            win,
                        ));
                    }
                    x += step;
                }
                y += step;
            }
        }

        let mut grouped = group_rectangles(candidates, MIN_NEIGHBORS, GROUP_EPS);
        // Map back to full-page coordinates (worker parity: round, clamp to
        // the page, sort by area descending).
        let inv = 1.0 / pre_scale;
        let mut faces: Vec<[u32; 4]> = grouped
            .drain(..)
            .map(|(x, y, w, h)| {
                [
                    ((x as f64) * inv).round() as u32,
                    ((y as f64) * inv).round() as u32,
                    (((w as f64) * inv).round() as u32).min(width),
                    (((h as f64) * inv).round() as u32).min(height),
                ]
            })
            .collect();
        faces.sort_by(|a, b| (b[2] * b[3]).cmp(&(a[2] * a[3])));
        faces
    }

    /// Evaluate the whole cascade at window (wx, wy) of the scaled image.
    fn window_passes(&self, sum: &[i32], iw: u32, wx: u32, wy: u32) -> bool {
        let mut ofs = 0usize;
        for stage in &self.stages {
            let mut acc = 0.0f32;
            for wi in 0..stage.ntrees {
                let stump = &self.stumps[ofs + wi];
                let code = lbp_code(sum, iw, wx, wy, self.features[stump.feature_idx]);
                let subset = &self.subsets[ofs + wi];
                let in_left = subset[code >> 5] & (1u32 << (code & 31)) != 0;
                acc += if in_left { stump.left } else { stump.right };
            }
            if acc < stage.threshold {
                return false;
            }
            ofs += stage.ntrees;
        }
        true
    }
}

/// LBP code for one feature at window (wx, wy): the 8 neighbors of the
/// feature's 3x3 block neighborhood compared against the center block,
/// bits 7..0 in OpenCV's order (TL, TC, TR, MR, BR, BC, BL, ML).
#[inline]
fn lbp_code(sum: &[i32], iw: u32, wx: u32, wy: u32, feature: [u32; 4]) -> usize {
    let (x, y, w, h) = (wx + feature[0], wy + feature[1], feature[2], feature[3]);
    let center = block_sum(sum, iw, x + w, y + h, w, h);
    let neighbors: [[u32; 2]; 8] = [
        [x, y],
        [x + w, y],
        [x + 2 * w, y],
        [x + 2 * w, y + h],
        [x + 2 * w, y + 2 * h],
        [x + w, y + 2 * h],
        [x, y + 2 * h],
        [x, y + h],
    ];
    let mut code = 0usize;
    for (i, [bx, by]) in neighbors.iter().enumerate() {
        if block_sum(sum, iw, *bx, *by, w, h) >= center {
            code |= 1 << (7 - i);
        }
    }
    code
}

/// Integral sum over the block [bx, bx+bw) x [by, by+bh) in a 32-bit
/// integral image of row stride `iw`.
#[inline]
fn block_sum(sum: &[i32], iw: u32, bx: u32, by: u32, bw: u32, bh: u32) -> i32 {
    let at = |x: u32, y: u32| (y * iw + x) as usize;
    sum[at(bx, by)] - sum[at(bx + bw, by)] - sum[at(bx, by + bh)] + sum[at(bx + bw, by + bh)]
}

/// 32-bit integral image with a zero first row/column (OpenCV CV_32S).
fn integral32(gray: &GrayImage) -> Vec<i32> {
    let (w, h) = gray.dimensions();
    let iw = w + 1;
    let mut sum = vec![0i32; ((w + 1) * (h + 1)) as usize];
    for y in 0..h {
        let mut row = 0i32;
        for x in 0..w {
            row += gray.get_pixel(x, y)[0] as i32;
            sum[((y + 1) * iw + x + 1) as usize] = sum[(y * iw + x + 1) as usize] + row;
        }
    }
    sum
}

/// Bilinear resize matching OpenCV's `INTER_LINEAR_EXACT` geometry
/// (half-pixel-centered sampling, edge clamping).
fn resize_bilinear(src: &GrayImage, dw: u32, dh: u32) -> GrayImage {
    let (sw, sh) = src.dimensions();
    if (sw, sh) == (dw, dh) {
        return src.clone();
    }
    let sx = sw as f64 / dw as f64;
    let sy = sh as f64 / dh as f64;
    let mut dst = GrayImage::new(dw, dh);
    for y in 0..dh {
        let fy = (y as f64 + 0.5) * sy - 0.5;
        let y0 = fy.floor().max(0.0) as u32;
        let y1 = (y0 + 1).min(sh - 1);
        let wy = (fy - y0 as f64).clamp(0.0, 1.0);
        for x in 0..dw {
            let fx = (x as f64 + 0.5) * sx - 0.5;
            let x0 = fx.floor().max(0.0) as u32;
            let x1 = (x0 + 1).min(sw - 1);
            let wx = (fx - x0 as f64).clamp(0.0, 1.0);
            let p00 = src.get_pixel(x0, y0)[0] as f64;
            let p10 = src.get_pixel(x1, y0)[0] as f64;
            let p01 = src.get_pixel(x0, y1)[0] as f64;
            let p11 = src.get_pixel(x1, y1)[0] as f64;
            let top = p00 * (1.0 - wx) + p10 * wx;
            let bottom = p01 * (1.0 - wx) + p11 * wx;
            let value = (top * (1.0 - wy) + bottom * wy).round();
            dst.put_pixel(x, y, Luma([value.clamp(0.0, 255.0) as u8]));
        }
    }
    dst
}

/// Box-average shrink approximating OpenCV's `INTER_AREA` for the worker's
/// pre-scale step (only reached for pages larger than 1024).
fn resize_area(src: &GrayImage, dw: u32, dh: u32) -> GrayImage {
    let (sw, sh) = src.dimensions();
    let sx = sw as f64 / dw as f64;
    let sy = sh as f64 / dh as f64;
    let mut dst = GrayImage::new(dw, dh);
    for y in 0..dh {
        let y0 = (y as f64 * sy).floor() as u32;
        let y1 = (((y as f64 + 1.0) * sy).ceil() as u32).min(sh);
        for x in 0..dw {
            let x0 = (x as f64 * sx).floor() as u32;
            let x1 = (((x as f64 + 1.0) * sx).ceil() as u32).min(sw);
            let mut acc = 0u64;
            let mut n = 0u64;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    acc += src.get_pixel(xx, yy)[0] as u64;
                    n += 1;
                }
            }
            dst.put_pixel(x, y, Luma([if n > 0 { (acc / n) as u8 } else { 0 }]));
        }
    }
    dst
}

/// OpenCV `groupRectangles` (default weights): union-find over
/// `SimilarRects(eps)`, class averaging, keep classes with more than
/// `group_threshold` members, then drop small boxes nested inside larger
/// classes.
fn group_rectangles(
    rects: Vec<(u32, u32, u32, u32)>,
    group_threshold: usize,
    eps: f64,
) -> Vec<(u32, u32, u32, u32)> {
    let n = rects.len();
    if n == 0 {
        return Vec::new();
    }
    let similar = |a: (u32, u32, u32, u32), b: (u32, u32, u32, u32)| -> bool {
        let delta = eps * (a.2.min(b.2) as f64 + a.3.min(b.3) as f64) * 0.5;
        (a.0 as f64 - b.0 as f64).abs() <= delta
            && (a.1 as f64 - b.1 as f64).abs() <= delta
            && ((a.0 + a.2) as f64 - (b.0 + b.2) as f64).abs() <= delta
            && ((a.1 + a.3) as f64 - (b.1 + b.3) as f64).abs() <= delta
    };
    // Union-find.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], i: usize) -> usize {
        let mut root = i;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = i;
        while parent[cur] != cur {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }
    for (i, &a) in rects.iter().enumerate() {
        for (j, &b) in rects.iter().enumerate().skip(i + 1) {
            if similar(a, b) {
                let ri = find(&mut parent, i);
                let rj = find(&mut parent, j);
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }
    // Class stats.
    let mut sums: Vec<(u64, u64, u64, u64, usize)> = vec![(0, 0, 0, 0, 0); n];
    for (i, &r) in rects.iter().enumerate() {
        let root = find(&mut parent, i);
        let entry = &mut sums[root];
        entry.0 += r.0 as u64;
        entry.1 += r.1 as u64;
        entry.2 += r.2 as u64;
        entry.3 += r.3 as u64;
        entry.4 += 1;
    }
    // Average each class (OpenCV saturating cast, counts kept).
    let mut classes: Vec<((u32, u32, u32, u32), usize)> = Vec::new();
    for (sx, sy, sw, sh, count) in sums.into_iter() {
        if count > group_threshold {
            classes.push((
                (
                    (sx as f64 / count as f64).round() as u32,
                    (sy as f64 / count as f64).round() as u32,
                    (sw as f64 / count as f64).round() as u32,
                    (sh as f64 / count as f64).round() as u32,
                ),
                count,
            ));
        }
    }
    // Drop small boxes nested inside larger classes.
    let mut kept: Vec<(u32, u32, u32, u32)> = Vec::new();
    'outer: for (i, (r1, n1)) in classes.iter().enumerate() {
        for (j, (r2, n2)) in classes.iter().enumerate() {
            if i == j {
                continue;
            }
            let dx = (r2.2 as f64 * eps).round() as u32;
            let dy = (r2.3 as f64 * eps).round() as u32;
            if r1.0 >= r2.0.saturating_sub(dx)
                && r1.1 >= r2.1.saturating_sub(dy)
                && r1.0 + r1.2 <= r2.0 + r2.2 + dx
                && r1.1 + r1.3 <= r2.1 + r2.3 + dy
                && (*n2 > 3.max(*n1) || *n1 < 3)
            {
                continue 'outer;
            }
        }
        kept.push(*r1);
    }
    kept
}

/// Face palette candidate image (the worker's `_face_palette_candidate`):
/// peach skin blobs with blush on detected faces, shading preserved from
/// the page luma. Produced only when this backend is used as a full
/// colorization backend.
pub fn face_palette_candidate(gray: &GrayImage, faces: &[[u32; 4]]) -> RgbImage {
    let (width, height) = gray.dimensions();
    let mut luma = vec![0.0f32; (width * height) as usize];
    for (i, pixel) in gray.pixels().enumerate() {
        luma[i] = pixel[0] as f32;
    }
    let mut red = vec![0.0f32; luma.len()];
    let mut blue = vec![0.0f32; luma.len()];
    let mut confidence = vec![0.0f32; luma.len()];
    for face in faces {
        let [fx, fy, fw, fh] = *face;
        if fw == 0 || fh == 0 {
            continue;
        }
        let center_x = fx as f32 + fw as f32 * 0.5;
        let center_y = fy as f32 + fh as f32 * 0.57;
        let radius_x = (fw as f32 * 0.43).max(8.0);
        let radius_y = (fh as f32 * 0.43).max(8.0);
        let cheek_y = fy as f32 + fh as f32 * 0.62;
        let cheek_rx = (fw as f32 * 0.12).max(4.0);
        let cheek_ry = (fh as f32 * 0.07).max(3.0);
        let left_cx = fx as f32 + fw as f32 * 0.31;
        let right_cx = fx as f32 + fw as f32 * 0.69;
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let l = luma[idx];
                let dx = (x as f32 - center_x) / radius_x;
                let dy = (y as f32 - center_y) / radius_y;
                let radial = (1.0 - dx * dx - dy * dy).clamp(0.0, 1.0);
                let drawable = ((l - 92.0) / 84.0).clamp(0.0, 1.0);
                let face_weight = radial.sqrt() * drawable;
                let lc = (-2.5
                    * (((x as f32 - left_cx) / cheek_rx).powi(2)
                        + ((y as f32 - cheek_y) / cheek_ry).powi(2)))
                .exp();
                let rc = (-2.5
                    * (((x as f32 - right_cx) / cheek_rx).powi(2)
                        + ((y as f32 - cheek_y) / cheek_ry).powi(2)))
                .exp();
                let blush = lc.max(rc) * face_weight;
                if face_weight > confidence[idx] {
                    red[idx] = 38.0 + blush * 13.0;
                    blue[idx] = -26.0 - blush * 5.0;
                    confidence[idx] = face_weight;
                }
            }
        }
    }
    let mut out = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let c = confidence[idx];
            let l = luma[idx];
            let r = red[idx] * c;
            let b = blue[idx] * c;
            let g = (-0.2126 * r - 0.0722 * b) / 0.7152;
            let channels = [r, g, b];
            let mut positive = 1.0f32;
            let mut negative = 1.0f32;
            for channel in channels {
                if channel > 0.0 {
                    positive = positive.min((255.0 - l) / channel.max(1e-6));
                } else if channel < 0.0 {
                    negative = negative.min(l / (-channel).max(1e-6));
                }
            }
            let scale = positive.min(negative).min(1.0);
            let palette_luma = l * (1.0 - c) + l.min(212.0) * c;
            let mut rgb = [
                (palette_luma + channels[0] * scale).clamp(0.0, 255.0),
                (palette_luma + channels[1] * scale).clamp(0.0, 255.0),
                (palette_luma + channels[2] * scale).clamp(0.0, 255.0),
            ];
            let ink = ((88.0 - l) / 68.0).clamp(0.0, 1.0);
            for channel in &mut rgb {
                *channel = (*channel * (1.0 - ink) + l * ink).clamp(0.0, 255.0);
            }
            out.put_pixel(
                x,
                y,
                Rgb([
                    rgb[0].round() as u8,
                    rgb[1].round() as u8,
                    rgb[2].round() as u8,
                ]),
            );
        }
    }
    out
}

/// In-process anime face backend: parses the LBP cascade once and detects
/// faces directly — no Python worker, no IPC, no temp files.
pub struct AnimeFaceBackend {
    cascade: Arc<LbpCascade>,
}

impl AnimeFaceBackend {
    pub fn new(model_dir: impl AsRef<Path>) -> Result<Self> {
        let model_dir = model_dir.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve anime face model {}",
                model_dir.as_ref().display()
            )
        })?;
        if !crate::is_anime_face_model_dir(&model_dir) {
            return Err(anyhow::Error::msg(format!(
                "anime face cascade missing in {}",
                model_dir.display()
            )));
        }
        let cascade = LbpCascade::load(&model_dir.join("lbpcascade_animeface.xml"))?;
        Ok(Self {
            cascade: Arc::new(cascade),
        })
    }

    pub fn shared(model_dir: impl AsRef<Path>) -> Result<Arc<Self>> {
        static WORKERS: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashMap<String, Arc<AnimeFaceBackend>>>,
        > = std::sync::OnceLock::new();
        let canonical = model_dir.as_ref().canonicalize().with_context(|| {
            format!(
                "cannot resolve anime face model {}",
                model_dir.as_ref().display()
            )
        })?;
        let key = canonical.to_string_lossy().to_string();
        let workers =
            WORKERS.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
        let mut workers = workers.lock().map_err(|error| {
            anyhow::Error::msg(format!("anime face registry poisoned: {error}"))
        })?;
        if let Some(worker) = workers.get(&key) {
            return Ok(Arc::clone(worker));
        }
        let worker = Arc::new(Self::new(&canonical)?);
        if workers.len() >= 4 {
            if let Some(oldest) = workers.keys().next().cloned() {
                workers.remove(&oldest);
            }
        }
        workers.insert(key, Arc::clone(&worker));
        Ok(worker)
    }

    /// Detect anime faces in a grayscale page. Returns (x, y, w, h) boxes
    /// in full-page coordinates, largest first.
    pub fn detect_faces(
        &self,
        lineart: &GrayImage,
    ) -> std::result::Result<Vec<[u32; 4]>, EngineError> {
        Ok(self.cascade.detect(lineart))
    }
}

impl crate::ColorizationBackend for AnimeFaceBackend {
    fn name(&self) -> &str {
        "anime-face-palette"
    }

    fn colorize(&self, input: &ModelInput) -> std::result::Result<RgbImage, EngineError> {
        let faces = self.cascade.detect(&input.lineart);
        Ok(face_palette_candidate(&input.lineart, &faces))
    }

    fn prefers_original_grayscale(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::imageops::FilterType;

    fn cascade_path() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/anime-face/lbpcascade_animeface.xml")
    }

    #[test]
    fn parses_real_cascade_structure() {
        let path = cascade_path();
        if !path.is_file() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let cascade = LbpCascade::load(&path).expect("cascade loads");
        assert_eq!((cascade.win_w, cascade.win_h), (24, 24));
        assert_eq!(cascade.stages.len(), 20, "20 stages");
        assert_eq!(cascade.stumps.len(), 771, "771 weak classifiers");
        assert_eq!(cascade.features.len(), 642, "642 features");
        let total_trees: usize = cascade.stages.iter().map(|s| s.ntrees).sum();
        assert_eq!(total_trees, 771);
        assert_eq!(cascade.subsets.len(), 771);
    }

    #[test]
    fn detects_the_same_face_as_the_python_worker() {
        // Regression: the OpenCV worker found one face at ~(101,199,225,225)
        // on real_orig.jpg downscaled to 717x1023. The pure-Rust port must
        // reproduce it (the whole point of the port).
        let path = cascade_path();
        if !path.is_file() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let page = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../real_orig.jpg");
        if !page.is_file() {
            eprintln!("skipping: {} not present", page.display());
            return;
        }
        let image = image::open(&page).expect("page loads").to_luma8();
        let working = image::imageops::resize(&image, 717, 1023, FilterType::Triangle);
        let cascade = LbpCascade::load(&path).expect("cascade loads");
        let faces = cascade.detect(&working);
        assert!(!faces.is_empty(), "expected at least one face");
        let largest = faces[0];
        let close = |a: u32, b: u32| a.abs_diff(b) <= 6;
        assert!(
            close(largest[0], 101) && close(largest[1], 199),
            "face at unexpected position: {largest:?} (expected ~[101, 199, 225, 225])"
        );
        assert!(
            close(largest[2], 225) && close(largest[3], 225),
            "face with unexpected size: {largest:?}"
        );
    }

    #[test]
    fn constant_page_detects_nothing() {
        let path = cascade_path();
        if !path.is_file() {
            eprintln!("skipping: {} not present", path.display());
            return;
        }
        let cascade = LbpCascade::load(&path).expect("cascade loads");
        let gray = GrayImage::from_pixel(400, 400, Luma([255]));
        assert!(cascade.detect(&gray).is_empty(), "blank page has no faces");
        let black = GrayImage::from_pixel(400, 400, Luma([0]));
        assert!(cascade.detect(&black).is_empty(), "black page has no faces");
    }
}
