use anyhow::{anyhow, Result};
use image::{GrayImage, Rgb, RgbImage};
use mangahue_config::{LayoutSection, ReadingDirection};
use mangahue_core::{BoundingBox, ImageSize, MangaPanel, TileInfo};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutResult {
    pub panels: Vec<MangaPanel>,
    pub tiles: Vec<TileInfo>,
    pub is_webtoon_like: bool,
}

pub fn analyze_layout(lineart: &GrayImage, config: &LayoutSection) -> Result<LayoutResult> {
    let size = ImageSize::new(lineart.width(), lineart.height());
    let is_webtoon_like = webtoon_like(lineart.height(), lineart.width(), &config.webtoon_mode);
    let panels = if config.panel_detection {
        detect_panels_basic(lineart, config.reading_direction)
    } else {
        vec![MangaPanel {
            id: "panel_0001".to_string(),
            bbox: BoundingBox::new(0, 0, size.width, size.height),
            reading_order: 0,
            confidence: 1.0,
        }]
    };
    let tiles = split_tiles(size, config);
    validate_tile_coverage(size, &tiles)?;
    Ok(LayoutResult {
        panels,
        tiles,
        is_webtoon_like,
    })
}

/// Whether the page should be treated as a webtoon (long vertical strip).
/// Honors the `layout.webtoon_mode` config: "force"/"always"/"webtoon"
/// always returns true, "off"/"disabled"/"false"/"no" never, and anything
/// else ("auto") uses the geometric heuristic.
pub fn webtoon_like(height: u32, width: u32, mode: &str) -> bool {
    match mode.to_ascii_lowercase().as_str() {
        "force" | "always" | "webtoon" => true,
        "off" | "disabled" | "false" | "no" => false,
        _ => height > width.saturating_mul(3),
    }
}

pub fn split_tiles(size: ImageSize, config: &LayoutSection) -> Vec<TileInfo> {
    if !config.tiling_enabled
        || (size.width <= config.tile_size && size.height <= config.tile_size)
        || config.tile_size == 0
    {
        return vec![TileInfo {
            id: "tile_0001".to_string(),
            bbox: BoundingBox::new(0, 0, size.width, size.height),
            overlap: 0,
            panel_id: None,
        }];
    }

    let x_positions = positions(size.width, config.tile_size, config.tile_overlap);
    let y_positions = positions(size.height, config.tile_size, config.tile_overlap);
    let mut tiles = Vec::with_capacity(x_positions.len() * y_positions.len());
    for (row, y) in y_positions.iter().enumerate() {
        for (col, x) in x_positions.iter().enumerate() {
            let width = config.tile_size.min(size.width - *x);
            let height = config.tile_size.min(size.height - *y);
            tiles.push(TileInfo {
                id: format!("tile_{row:03}_{col:03}"),
                bbox: BoundingBox::new(*x, *y, width, height),
                overlap: config.tile_overlap,
                panel_id: None,
            });
        }
    }
    tiles
}

pub fn validate_tile_coverage(size: ImageSize, tiles: &[TileInfo]) -> Result<()> {
    if tiles.is_empty() {
        return Err(anyhow!("layout generated zero tiles"));
    }
    let mut covered = vec![false; (size.width as usize) * (size.height as usize)];
    for tile in tiles {
        let bbox = tile.bbox.clamp_to(size);
        for y in bbox.y..bbox.bottom() {
            for x in bbox.x..bbox.right() {
                covered[(y as usize) * (size.width as usize) + x as usize] = true;
            }
        }
    }
    let uncovered = covered.iter().filter(|covered| !**covered).count();
    if uncovered > 0 {
        return Err(anyhow!("layout left {uncovered} pixels uncovered"));
    }
    Ok(())
}

pub fn merge_rgb_tiles(size: ImageSize, tiles: &[(TileInfo, RgbImage)]) -> Result<RgbImage> {
    if tiles.is_empty() {
        return Err(anyhow!("cannot merge zero tiles"));
    }
    let pixels = (size.width as usize) * (size.height as usize);
    let mut r = vec![0.0f32; pixels];
    let mut g = vec![0.0f32; pixels];
    let mut b = vec![0.0f32; pixels];
    let mut weights = vec![0.0f32; pixels];

    for (tile, image) in tiles {
        if image.width() != tile.bbox.width || image.height() != tile.bbox.height {
            return Err(anyhow!(
                "tile {} image dimensions do not match bbox",
                tile.id
            ));
        }
        for ty in 0..image.height() {
            for tx in 0..image.width() {
                let x = tile.bbox.x + tx;
                let y = tile.bbox.y + ty;
                if x >= size.width || y >= size.height {
                    continue;
                }
                let idx = (y as usize) * (size.width as usize) + x as usize;
                let weight = tile_weight(tx, ty, image.width(), image.height(), tile.overlap);
                let pixel = image.get_pixel(tx, ty);
                r[idx] += pixel[0] as f32 * weight;
                g[idx] += pixel[1] as f32 * weight;
                b[idx] += pixel[2] as f32 * weight;
                weights[idx] += weight;
            }
        }
    }

    let mut output = RgbImage::new(size.width, size.height);
    for y in 0..size.height {
        for x in 0..size.width {
            let idx = (y as usize) * (size.width as usize) + x as usize;
            let weight = weights[idx].max(0.0001);
            output.put_pixel(
                x,
                y,
                Rgb([
                    (r[idx] / weight).round().clamp(0.0, 255.0) as u8,
                    (g[idx] / weight).round().clamp(0.0, 255.0) as u8,
                    (b[idx] / weight).round().clamp(0.0, 255.0) as u8,
                ]),
            );
        }
    }
    Ok(output)
}

pub fn detect_panels_basic(lineart: &GrayImage, direction: ReadingDirection) -> Vec<MangaPanel> {
    let width = lineart.width();
    let height = lineart.height();
    if width < 64 || height < 64 {
        return vec![single_panel(width, height)];
    }

    let white_rows = white_runs_rows(lineart);
    let white_cols = white_runs_cols(lineart);
    let row_cuts = significant_runs(&white_rows, height / 80 + 3);
    let col_cuts = significant_runs(&white_cols, width / 80 + 3);

    if row_cuts.is_empty() && col_cuts.is_empty() {
        return vec![single_panel(width, height)];
    }

    let y_ranges = ranges_from_cuts(height, &row_cuts);
    let x_ranges = ranges_from_cuts(width, &col_cuts);
    let x_len = x_ranges.len();
    let x_ranges_rev: Vec<(u32, u32)> = x_ranges.iter().rev().cloned().collect();
    let x_iter: &[(u32, u32)] = match direction {
        ReadingDirection::LeftToRight => &x_ranges,
        ReadingDirection::RightToLeft => &x_ranges_rev,
    };
    let mut panels = Vec::new();
    for (row, (y0, y1)) in y_ranges.iter().enumerate() {
        for (col, (x0, x1)) in x_iter.iter().enumerate() {
            let bbox = BoundingBox::new(*x0, *y0, x1 - x0, y1 - y0);
            if bbox.area() < (width as u64 * height as u64) / 200 {
                continue;
            }
            panels.push(MangaPanel {
                id: format!("panel_{:04}", panels.len() + 1),
                bbox,
                reading_order: row * x_len + col,
                confidence: 0.45,
            });
        }
    }
    if panels.is_empty() {
        vec![single_panel(width, height)]
    } else {
        panels
    }
}

fn single_panel(width: u32, height: u32) -> MangaPanel {
    MangaPanel {
        id: "panel_0001".to_string(),
        bbox: BoundingBox::new(0, 0, width, height),
        reading_order: 0,
        confidence: 1.0,
    }
}

fn positions(length: u32, tile_size: u32, overlap: u32) -> Vec<u32> {
    if length <= tile_size {
        return vec![0];
    }
    let step = tile_size.saturating_sub(overlap).max(1);
    let mut out = Vec::new();
    let mut pos = 0;
    loop {
        out.push(pos);
        if pos + tile_size >= length {
            break;
        }
        let next = pos + step;
        pos = if next + tile_size >= length {
            length - tile_size
        } else {
            next
        };
    }
    out.dedup();
    out
}

fn tile_weight(x: u32, y: u32, width: u32, height: u32, overlap: u32) -> f32 {
    if overlap == 0 {
        return 1.0;
    }
    let edge_x = x.min(width.saturating_sub(1).saturating_sub(x));
    let edge_y = y.min(height.saturating_sub(1).saturating_sub(y));
    let dist = edge_x.min(edge_y) as f32;
    if dist >= overlap as f32 {
        return 1.0;
    }
    let t = dist / overlap as f32;
    (0.5 - 0.5 * (std::f32::consts::PI * t).cos()).clamp(0.0, 1.0)
}

fn white_runs_rows(lineart: &GrayImage) -> Vec<(u32, u32)> {
    let mut runs = Vec::new();
    let mut start = None;
    for y in 0..lineart.height() {
        let white = (0..lineart.width())
            .filter(|x| lineart.get_pixel(*x, y)[0] > 245)
            .count() as f32
            / lineart.width() as f32;
        if white > 0.985 {
            start.get_or_insert(y);
        } else if let Some(s) = start.take() {
            runs.push((s, y));
        }
    }
    if let Some(s) = start {
        runs.push((s, lineart.height()));
    }
    runs
}

fn white_runs_cols(lineart: &GrayImage) -> Vec<(u32, u32)> {
    let mut runs = Vec::new();
    let mut start = None;
    for x in 0..lineart.width() {
        let white = (0..lineart.height())
            .filter(|y| lineart.get_pixel(x, *y)[0] > 245)
            .count() as f32
            / lineart.height() as f32;
        if white > 0.985 {
            start.get_or_insert(x);
        } else if let Some(s) = start.take() {
            runs.push((s, x));
        }
    }
    if let Some(s) = start {
        runs.push((s, lineart.width()));
    }
    runs
}

fn significant_runs(runs: &[(u32, u32)], min_width: u32) -> Vec<(u32, u32)> {
    runs.iter()
        .copied()
        .filter(|(start, end)| end.saturating_sub(*start) >= min_width)
        .collect()
}

fn ranges_from_cuts(length: u32, cuts: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut ranges = Vec::new();
    let mut cursor = 0;
    for (start, end) in cuts {
        if *start > cursor {
            ranges.push((cursor, *start));
        }
        cursor = *end;
    }
    if cursor < length {
        ranges.push((cursor, length));
    }
    ranges
        .into_iter()
        .filter(|(start, end)| end.saturating_sub(*start) > 8)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mangahue_config::EngineConfig;

    #[test]
    fn tiles_cover_full_image() {
        let config = EngineConfig::default();
        let size = ImageSize::new(1800, 1200);
        let tiles = split_tiles(size, &config.layout);
        assert!(tiles.len() > 1);
        validate_tile_coverage(size, &tiles).unwrap();
    }

    #[test]
    fn merge_preserves_dimensions() {
        let config = EngineConfig::default();
        let size = ImageSize::new(900, 900);
        let tiles = split_tiles(size, &config.layout);
        let colored: Vec<_> = tiles
            .into_iter()
            .map(|tile| {
                let image =
                    RgbImage::from_pixel(tile.bbox.width, tile.bbox.height, Rgb([10, 20, 30]));
                (tile, image)
            })
            .collect();
        let merged = merge_rgb_tiles(size, &colored).unwrap();
        assert_eq!(merged.dimensions(), (900, 900));
    }
}
