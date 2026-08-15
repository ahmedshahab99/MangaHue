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

    // Recursively split on gutters. A global row/column cross-product creates
    // phantom panels whenever different rows use different column layouts.
    let page = BoundingBox::new(0, 0, width, height);
    let mut boxes = Vec::new();
    split_on_gutters(lineart, page, 0, &mut boxes);
    if boxes.len() <= 1 {
        return vec![single_panel(width, height)];
    }
    boxes.sort_by(|a, b| {
        let row_tolerance = height / 40 + 4;
        if a.y.abs_diff(b.y) > row_tolerance {
            a.y.cmp(&b.y)
        } else {
            match direction {
                ReadingDirection::LeftToRight => a.x.cmp(&b.x),
                ReadingDirection::RightToLeft => b.x.cmp(&a.x),
            }
        }
    });
    boxes
        .into_iter()
        .enumerate()
        .map(|(index, bbox)| MangaPanel {
            id: format!("panel_{:04}", index + 1),
            bbox,
            reading_order: index,
            confidence: 0.72,
        })
        .collect()
}

#[derive(Clone, Copy)]
enum GutterAxis {
    Horizontal,
    Vertical,
}

fn split_on_gutters(
    image: &GrayImage,
    bbox: BoundingBox,
    depth: usize,
    output: &mut Vec<BoundingBox>,
) {
    let page_area = image.width() as u64 * image.height() as u64;
    if depth >= 8 || bbox.area() < page_area / 80 || bbox.width < 40 || bbox.height < 40 {
        output.push(bbox);
        return;
    }
    let horizontal = best_gutter(image, bbox, GutterAxis::Horizontal);
    let vertical = best_gutter(image, bbox, GutterAxis::Vertical);
    let selected = match (horizontal, vertical) {
        (Some(h), Some(v)) => {
            let h_score = (h.1 - h.0) as f32 / bbox.height as f32;
            let v_score = (v.1 - v.0) as f32 / bbox.width as f32;
            if h_score >= v_score {
                Some((GutterAxis::Horizontal, h))
            } else {
                Some((GutterAxis::Vertical, v))
            }
        }
        (Some(h), None) => Some((GutterAxis::Horizontal, h)),
        (None, Some(v)) => Some((GutterAxis::Vertical, v)),
        (None, None) => None,
    };
    let Some((axis, (start, end))) = selected else {
        output.push(bbox);
        return;
    };
    match axis {
        GutterAxis::Horizontal => {
            split_on_gutters(
                image,
                BoundingBox::new(bbox.x, bbox.y, bbox.width, start - bbox.y),
                depth + 1,
                output,
            );
            split_on_gutters(
                image,
                BoundingBox::new(bbox.x, end, bbox.width, bbox.bottom() - end),
                depth + 1,
                output,
            );
        }
        GutterAxis::Vertical => {
            split_on_gutters(
                image,
                BoundingBox::new(bbox.x, bbox.y, start - bbox.x, bbox.height),
                depth + 1,
                output,
            );
            split_on_gutters(
                image,
                BoundingBox::new(end, bbox.y, bbox.right() - end, bbox.height),
                depth + 1,
                output,
            );
        }
    }
}

fn best_gutter(image: &GrayImage, bbox: BoundingBox, axis: GutterAxis) -> Option<(u32, u32)> {
    let (start, end, cross_start, cross_end) = match axis {
        GutterAxis::Horizontal => (bbox.y, bbox.bottom(), bbox.x, bbox.right()),
        GutterAxis::Vertical => (bbox.x, bbox.right(), bbox.y, bbox.bottom()),
    };
    // Never treat page margins or the white padding just inside a detected
    // panel as another gutter. A real split must leave meaningful content on
    // both sides.
    let margin = ((end - start) / 8).max(8);
    let min_run = ((end - start) / 100).max(3);
    let mut runs = Vec::new();
    let mut run_start = None;
    for position in start + margin..end.saturating_sub(margin) {
        let white = (cross_start..cross_end)
            .filter(|cross| match axis {
                GutterAxis::Horizontal => image.get_pixel(*cross, position)[0] >= 238,
                GutterAxis::Vertical => image.get_pixel(position, *cross)[0] >= 238,
            })
            .count() as f32
            / (cross_end - cross_start).max(1) as f32;
        if white >= 0.94 {
            run_start.get_or_insert(position);
        } else if let Some(value) = run_start.take() {
            if position - value >= min_run {
                runs.push((value, position));
            }
        }
    }
    if let Some(value) = run_start {
        let run_end = end.saturating_sub(margin);
        if run_end - value >= min_run {
            runs.push((value, run_end));
        }
    }
    runs.into_iter().max_by_key(|(a, b)| b - a)
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

    #[test]
    fn mixed_layout_does_not_create_cross_product_panels() {
        let mut page = GrayImage::from_pixel(300, 300, image::Luma([255]));
        for y in 10..140 {
            for x in 10..140 {
                page.put_pixel(x, y, image::Luma([80]));
            }
            for x in 160..290 {
                page.put_pixel(x, y, image::Luma([80]));
            }
        }
        for y in 160..290 {
            for x in 10..290 {
                page.put_pixel(x, y, image::Luma([80]));
            }
        }

        let panels = detect_panels_basic(&page, ReadingDirection::RightToLeft);
        assert_eq!(panels.len(), 3);
        assert!(panels[0].bbox.x > panels[1].bbox.x);
        assert!(panels[2].bbox.width > panels[0].bbox.width);
    }
}
