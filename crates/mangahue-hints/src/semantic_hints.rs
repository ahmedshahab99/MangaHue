use image::{GrayImage, Luma, Rgb, RgbImage};
use std::collections::VecDeque;

/// Semantic region hints: dense full-coverage hints generated from the
/// grayscale page, mirroring what a colorist would assign:
/// skin -> warm peach, hair -> natural dark/medium, clothing -> varied,
/// background -> cool/warm by luma, bubbles -> white.
///
/// Unlike sparse dot hints, these are large flat regions with strong mask
/// values so the hint-U-Net (stride-8 bottleneck) can actually see them.
const SKIN: [u8; 3] = [244, 196, 166];
const SKIN_SHA: [u8; 3] = [194, 134, 104];
const HAIR_D: [u8; 3] = [42, 28, 36];
const HAIR_M: [u8; 3] = [74, 48, 42];
const CLOTH_L: [u8; 3] = [178, 168, 160];
const CLOTH_D: [u8; 3] = [48, 42, 60];
const BG_WARM: [u8; 3] = [228, 222, 215];
const BG_COOL: [u8; 3] = [195, 192, 200];
const WHITE: [u8; 3] = [255, 253, 250];

const REGION_STRENGTH: u8 = 192; // 0.75

#[derive(Debug, Default, Clone)]
pub struct SemanticMasks {
    pub skin: Vec<bool>,
    pub hair: Vec<bool>,
    pub clothing: Vec<bool>,
    pub background: Vec<bool>,
    pub bubble: Vec<bool>,
}

/// Detect semantic regions from the grayscale page (0-1 values in 0-255).
pub fn detect_semantic_regions(
    gray: &GrayImage,
    line_mask: &GrayImage,
    bubble_mask: &GrayImage,
) -> SemanticMasks {
    let (width, height) = gray.dimensions();
    let len = (width * height) as usize;

    // Per-pixel candidate predicates (scipy thresholds from onnx_worker).
    let mut skin_cand = vec![false; len];
    let mut hair_cand = vec![false; len];
    let mut ink = vec![false; len];
    for (i, p) in gray.pixels().enumerate() {
        let g = p[0] as u32;
        let bubble = bubble_mask.get_pixel(i as u32 % width, i as u32 / width)[0] > 72;
        let ink_px = line_mask.get_pixel(i as u32 % width, i as u32 / width)[0] > 100;
        ink[i] = ink_px;
        if !ink_px && !bubble {
            // Skin range extended to bright skin (98-250) so light faces and
            // chests still get skin hints; only true paper (>250) is not skin.
            if (98..=250).contains(&g) {
                skin_cand[i] = true;
            }
            if (20..=89).contains(&g) {
                hair_cand[i] = true;
            }
        }
    }

    // Connected components with area pruning (flood fill).
    let mut masks = SemanticMasks {
        skin: vec![false; len],
        hair: vec![false; len],
        clothing: vec![false; len],
        background: vec![false; len],
        bubble: vec![false; len],
    };
    let skin_min = 60usize;
    let hair_min = 50usize;

    let mut visited = vec![false; len];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            if visited[idx] {
                continue;
            }
            let is_skin = skin_cand[idx];
            let is_hair = hair_cand[idx];
            if !is_skin && !is_hair {
                visited[idx] = true;
                continue;
            }
            // Flood fill the connected class.
            let mut queue = VecDeque::from([(x, y)]);
            visited[idx] = true;
            let mut component = Vec::new();
            let class = if is_skin { 1u8 } else { 2u8 };
            while let Some((cx, cy)) = queue.pop_front() {
                let cidx = (cy * width + cx) as usize;
                component.push(cidx);
                for (dx, dy) in [(-1_i32, 0), (1, 0), (0, -1), (0, 1)] {
                    let nx = cx as i32 + dx;
                    let ny = cy as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= width as i32 || ny >= height as i32 {
                        continue;
                    }
                    let nidx = (ny as u32 * width + nx as u32) as usize;
                    if visited[nidx] {
                        continue;
                    }
                    // Continue same class only; skin/hair are exclusive ranges.
                    let same_skin = skin_cand[nidx];
                    let same_hair = hair_cand[nidx];
                    let same = if class == 1 { same_skin } else { same_hair };
                    if same {
                        visited[nidx] = true;
                        queue.push_back((nx as u32, ny as u32));
                    }
                }
            }
            // Area prune.
            if class == 1 && component.len() >= skin_min {
                for &ci in &component {
                    masks.skin[ci] = true;
                }
            } else if class == 2 && component.len() >= hair_min {
                for &ci in &component {
                    masks.hair[ci] = true;
                }
            }
        }
    }

    // Clothing = mid gray minus skin/hair/ink/bubble.
    for (i, p) in gray.pixels().enumerate() {
        let g = p[0] as u32;
        if (31..=191).contains(&g)
            && !masks.skin[i]
            && !masks.hair[i]
            && !ink[i]
            && bubble_mask.get_pixel(i as u32 % width, i as u32 / width)[0] <= 72
        {
            masks.clothing[i] = true;
        }
    }

    // Background = complement.
    for (i, _) in gray.pixels().enumerate() {
        let bubble = bubble_mask.get_pixel(i as u32 % width, i as u32 / width)[0] > 72;
        if !ink[i] && !masks.skin[i] && !masks.hair[i] && !masks.clothing[i] && !bubble {
            masks.background[i] = true;
        }
    }

    // Bubble hint mask (union existing bubbles with bright CCs).
    for (i, p) in gray.pixels().enumerate() {
        let g = p[0] as u32;
        let b = bubble_mask.get_pixel(i as u32 % width, i as u32 / width)[0] > 72;
        if b && g > 0 {
            masks.bubble[i] = true;
        }
    }

    masks
}

/// Paint dense semantic hints into the map/mask. Replaces prior region style
/// hints (semantic knowledge beats grid ambience). Mask uses max-composite.
pub fn rasterize_semantic_hints(
    hint_map: &mut RgbImage,
    hint_mask: &mut GrayImage,
    gray: &GrayImage,
    regions: &SemanticMasks,
) -> usize {
    let (width, height) = hint_map.dimensions();
    let len = (width * height) as usize;
    let mut painted = 0usize;
    for i in 0..len {
        let x = i as u32 % width;
        let y = i as u32 / width;
        let g = gray.get_pixel(x, y)[0] as u32;

        // Priority: bubble > skin > hair > clothing > background.
        let color: Option<[u8; 3]> = if regions.bubble[i] {
            Some(WHITE)
        } else if regions.skin[i] {
            Some(if g > 153 { SKIN } else { SKIN_SHA })
        } else if regions.hair[i] {
            Some(if g < 51 { HAIR_D } else { HAIR_M })
        } else if regions.clothing[i] {
            Some(if g > 115 { CLOTH_L } else { CLOTH_D })
        } else if regions.background[i] {
            Some(if g > 128 { BG_WARM } else { BG_COOL })
        } else {
            None
        };

        if let Some(c) = color {
            let existing = hint_mask.get_pixel(x, y)[0];
            if REGION_STRENGTH >= existing {
                hint_map.put_pixel(x, y, Rgb(c));
            }
            hint_mask.put_pixel(x, y, Luma([REGION_STRENGTH.max(existing)]));
            painted += 1;
        }
    }
    painted
}

/// Face-driven material hints: for each detected face box, paint a skin
/// ellipse, a hair ellipse above it, and two eye dots with high strength.
/// The model then colors faces with correct material hues.
pub fn rasterize_face_hints(
    hint_map: &mut RgbImage,
    hint_mask: &mut GrayImage,
    faces: &[[u32; 4]],
    skin_color: [u8; 3],
    hair_color: [u8; 3],
    eye_color: [u8; 3],
) -> usize {
    let (width, height) = hint_map.dimensions();
    let mut painted = 0usize;
    const FACE_STRENGTH: u8 = 204; // 0.80

    for &[x, y, w, h] in faces {
        let (x, y, w, h) = (x as f32, y as f32, w as f32, h as f32);
        if w < 8.0 || h < 8.0 {
            continue;
        }
        // Skin ellipse: centered at 0.57h (worker geometry), rx 0.43w.
        let skin_cx = x + w * 0.5;
        let skin_cy = y + h * 0.57;
        let skin_rx = (w * 0.43).max(6.0);
        let skin_ry = (h * 0.43).max(6.0);
        // Hair ellipse: above the face.
        let hair_cx = x + w * 0.5;
        let hair_cy = y + h * 0.18;
        let hair_rx = (w * 0.55).max(6.0);
        let hair_ry = (h * 0.45).max(6.0);
        // Eyes.
        let eye_y = y + h * 0.47;
        let eye_rx = (w * 0.05).max(2.0);
        let eye_x1 = x + w * 0.33;
        let eye_x2 = x + w * 0.67;

        let min_y = (y - h * 0.5).max(0.0) as u32;
        let max_y = ((y + h * 1.4).min(height as f32)) as u32;
        let min_x = (x - w * 0.4).max(0.0) as u32;
        let max_x = ((x + w * 1.4).min(width as f32)) as u32;

        for yy in min_y..max_y {
            for xx in min_x..max_x {
                let fx = xx as f32 + 0.5;
                let fy = yy as f32 + 0.5;
                let mut color: Option<[u8; 3]> = None;

                // Hair ellipse (drawn first so skin overrides).
                let hdx = (fx - hair_cx) / hair_rx;
                let hdy = (fy - hair_cy) / hair_ry;
                if hdx * hdx + hdy * hdy <= 1.0 {
                    color = Some(hair_color);
                }
                // Skin ellipse.
                let sdx = (fx - skin_cx) / skin_rx;
                let sdy = (fy - skin_cy) / skin_ry;
                if sdx * sdx + sdy * sdy <= 1.0 {
                    color = Some(skin_color);
                }
                // Eyes (small circles override skin).
                for ex in [eye_x1, eye_x2] {
                    let edx = (fx - ex) / eye_rx;
                    let edy = (fy - eye_y) / eye_rx;
                    if edx * edx + edy * edy <= 1.0 {
                        color = Some(eye_color);
                    }
                }

                if let Some(c) = color {
                    let existing = hint_mask.get_pixel(xx, yy)[0];
                    if FACE_STRENGTH >= existing {
                        hint_map.put_pixel(xx, yy, Rgb(c));
                    }
                    hint_mask.put_pixel(xx, yy, Luma([FACE_STRENGTH.max(existing)]));
                    painted += 1;
                }
            }
        }
    }
    painted
}

/// Face skin mask (full-page grayscale): soft ellipse per face box covering
/// the skin region only (hair and eyes excluded). Consumed by postprocess so
/// faces are force-painted peach even where the model outputs white.
pub fn rasterize_face_mask(faces: &[[u32; 4]], width: u32, height: u32) -> GrayImage {
    let mut mask = GrayImage::new(width, height);
    for &[x, y, w, h] in faces {
        let (x, y, w, h) = (x as f32, y as f32, w as f32, h as f32);
        if w < 8.0 || h < 8.0 {
            continue;
        }
        // Skin ellipse: covers forehead (0.30h) through chin (1.42h) and
        // full cheek width so the whole face gets painted.
        let skin_cx = x + w * 0.5;
        let skin_cy = y + h * 0.62;
        let skin_rx = (w * 0.95).max(8.0);
        let skin_ry = (h * 0.80).max(8.0);
        // Hair exclusion: NARROW bangs band at the top of the box only
        // (real hair is the top ~20-30%; a tall ellipse would block the
        // forehead and cheeks, leaving the face white).
        let hair_cx = x + w * 0.5;
        let hair_cy = y + h * 0.16;
        let hair_rx = (w * 0.75).max(6.0);
        let hair_ry = (h * 0.18).max(6.0);
        let eye_y = y + h * 0.47;
        let eye_rx = (w * 0.07).max(3.0);
        let eye_x1 = x + w * 0.33;
        let eye_x2 = x + w * 0.67;

        let min_y = (y - h * 0.35).max(0.0) as u32;
        let max_y = ((y + h * 1.45).min(height as f32)) as u32;
        let min_x = (x - w * 0.45).max(0.0) as u32;
        let max_x = ((x + w * 1.45).min(width as f32)) as u32;

        for yy in min_y..max_y {
            for xx in min_x..max_x {
                let fx = xx as f32 + 0.5;
                let fy = yy as f32 + 0.5;
                let hdx = (fx - hair_cx) / hair_rx;
                let hdy = (fy - hair_cy) / hair_ry;
                if hdx * hdx + hdy * hdy <= 1.0 {
                    continue;
                }
                let mut excluded = false;
                for ex in [eye_x1, eye_x2] {
                    let edx = (fx - ex) / eye_rx;
                    let edy = (fy - eye_y) / eye_rx;
                    if edx * edx + edy * edy <= 1.0 {
                        excluded = true;
                        break;
                    }
                }
                if excluded {
                    continue;
                }
                let sdx = (fx - skin_cx) / skin_rx;
                let sdy = (fy - skin_cy) / skin_ry;
                let d = sdx * sdx + sdy * sdy;
                if d <= 1.0 {
                    // Soft falloff at the ellipse edge (1.0 -> 255, 1.2 -> 0).
                    let v = ((1.0 - (d.sqrt() - 0.9) / 0.3).clamp(0.0, 1.0) * 255.0) as u8;
                    let existing = mask.get_pixel(xx, yy)[0];
                    mask.put_pixel(xx, yy, Luma([v.max(existing)]));
                }
            }
        }
    }
    mask
}
