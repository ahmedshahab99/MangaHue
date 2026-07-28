use image::{GrayImage, Rgb, RgbImage};
use mangahue_core::StyleProfile;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RegionRenderReport {
    pub regions: u64,
    pub seeded_regions: u64,
    pub inferred_regions: u64,
    pub semantic_skin_regions: u64,
    pub face_cavity_pixels: u64,
    pub pixels_rendered: u64,
}

#[derive(Debug, Clone, Copy)]
struct RegionStats {
    count: u32,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    luma_sum: f64,
    texture_sum: f64,
    seed_weight: f64,
    red_chroma_sum: f64,
    blue_chroma_sum: f64,
    existing_red_sum: f64,
    existing_blue_sum: f64,
    touches_border: bool,
}

impl RegionStats {
    fn new(x: u32, y: u32) -> Self {
        Self {
            count: 0,
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
            luma_sum: 0.0,
            texture_sum: 0.0,
            seed_weight: 0.0,
            red_chroma_sum: 0.0,
            blue_chroma_sum: 0.0,
            existing_red_sum: 0.0,
            existing_blue_sum: 0.0,
            touches_border: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct RegionColor {
    red: f32,
    blue: f32,
    confidence: f32,
    seeded: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct RegionSemantic {
    skin: f32,
    fabric: f32,
    skin_seed: bool,
}

struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len as u32).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, value: u32) -> u32 {
        let index = value as usize;
        if self.parent[index] != value {
            self.parent[index] = self.find(self.parent[index]);
        }
        self.parent[index]
    }

    fn union(&mut self, left: u32, right: u32) {
        let left_root = self.find(left);
        let right_root = self.find(right);
        if left_root == right_root {
            return;
        }
        let left_rank = self.rank[left_root as usize];
        let right_rank = self.rank[right_root as usize];
        if left_rank < right_rank {
            self.parent[left_root as usize] = right_root;
        } else {
            self.parent[right_root as usize] = left_root;
            if left_rank == right_rank {
                self.rank[left_root as usize] = left_rank.saturating_add(1);
            }
        }
    }
}

pub(crate) fn render_region_graph(
    image: &mut RgbImage,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
    style: &StyleProfile,
) -> RegionRenderReport {
    let (width, height) = image.dimensions();
    if width < 3
        || height < 3
        || original_gray.dimensions() != (width, height)
        || line_mask.dimensions() != (width, height)
    {
        return RegionRenderReport::default();
    }

    let len = width as usize * height as usize;
    let smooth_gray = box_blur_gray(original_gray, 2);
    let source = image.pixels().copied().collect::<Vec<_>>();
    let mut union_find = UnionFind::new(len);

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if line_mask.get_pixel(x, y)[0] > 118 {
                continue;
            }
            if x + 1 < width {
                maybe_union(
                    &mut union_find,
                    index,
                    index + 1,
                    smooth_gray.get_pixel(x, y)[0],
                    smooth_gray.get_pixel(x + 1, y)[0],
                    line_mask.get_pixel(x + 1, y)[0],
                );
            }
            if y + 1 < height {
                maybe_union(
                    &mut union_find,
                    index,
                    index + width as usize,
                    smooth_gray.get_pixel(x, y)[0],
                    smooth_gray.get_pixel(x, y + 1)[0],
                    line_mask.get_pixel(x, y + 1)[0],
                );
            }
        }
    }

    let mut roots = vec![0u32; len];
    let mut root_to_region = HashMap::<u32, usize>::new();
    let mut stats = Vec::<RegionStats>::new();
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let root = union_find.find(index as u32);
            roots[index] = root;
            if line_mask.get_pixel(x, y)[0] > 118 {
                continue;
            }
            let region = *root_to_region.entry(root).or_insert_with(|| {
                stats.push(RegionStats::new(x, y));
                stats.len() - 1
            });
            accumulate_region(
                &mut stats[region],
                source[index],
                original_gray.get_pixel(x, y)[0],
                smooth_gray.get_pixel(x, y)[0],
                x,
                y,
                (width, height),
            );
        }
    }

    let palette = style
        .palette
        .iter()
        .map(|color| {
            let r = color.r as f32;
            let g = color.g as f32;
            let b = color.b as f32;
            let luma = rgb_luma(r, g, b);
            (luma, r - luma, b - luma)
        })
        .collect::<Vec<_>>();

    let mut colors = stats
        .iter()
        .map(|region| infer_region_color(region, &palette))
        .collect::<Vec<_>>();
    let adjacency = build_adjacency(&roots, &root_to_region, &stats, line_mask, width, height);
    propagate_graph_colors(&mut colors, &stats, &adjacency);
    let semantic_adjacency =
        build_cross_ink_adjacency(&roots, &root_to_region, &stats, line_mask, width, height);
    let mut semantics = infer_region_semantics(&stats, &colors, original_gray, line_mask);
    propagate_region_semantics(&mut semantics, &stats, &semantic_adjacency);

    let illumination = box_blur_gray(original_gray, 6);
    let ink_distance = distance_to_ink(line_mask);
    let face_cavity = detect_face_cavity_field(original_gray, line_mask);
    let mut report = RegionRenderReport {
        regions: stats.iter().filter(|region| region.count >= 20).count() as u64,
        seeded_regions: colors.iter().filter(|color| color.seeded).count() as u64,
        inferred_regions: colors
            .iter()
            .filter(|color| !color.seeded && color.confidence > 0.08)
            .count() as u64,
        semantic_skin_regions: semantics
            .iter()
            .filter(|semantic| semantic.skin > 0.42)
            .count() as u64,
        face_cavity_pixels: face_cavity.iter().filter(|score| **score > 0.12).count() as u64,
        pixels_rendered: 0,
    };

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let ink = line_mask.get_pixel(x, y)[0] as f32 / 255.0;
            if ink > 0.47 {
                continue;
            }
            let Some(&region_index) = root_to_region.get(&roots[index]) else {
                continue;
            };
            let region = stats[region_index];
            if region.count < 20 {
                continue;
            }
            let color = colors[region_index];
            render_material_pixel(
                image.get_pixel_mut(x, y),
                original_gray.get_pixel(x, y)[0],
                illumination.get_pixel(x, y)[0],
                ink_distance[index],
                ink,
                x,
                y,
                region,
                color,
                semantics[region_index],
                face_cavity[index],
                style,
            );
            report.pixels_rendered += 1;
        }
    }

    report
}

#[derive(Debug, Clone, Copy)]
struct DarkComponent {
    count: u32,
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
    touches_border: bool,
}

fn detect_face_cavity_field(original_gray: &GrayImage, line_mask: &GrayImage) -> Vec<f32> {
    let (width, height) = original_gray.dimensions();
    let len = width as usize * height as usize;
    let mut union_find = UnionFind::new(len);
    let is_dark =
        |x: u32, y: u32| original_gray.get_pixel(x, y)[0] < 94 && line_mask.get_pixel(x, y)[0] > 18;
    for y in 0..height {
        for x in 0..width {
            if !is_dark(x, y) {
                continue;
            }
            let index = (y * width + x) as usize;
            if x + 1 < width && is_dark(x + 1, y) {
                union_find.union(index as u32, (index + 1) as u32);
            }
            if y + 1 < height && is_dark(x, y + 1) {
                union_find.union(index as u32, (index + width as usize) as u32);
            }
        }
    }

    let mut components = HashMap::<u32, DarkComponent>::new();
    for y in 0..height {
        for x in 0..width {
            if !is_dark(x, y) {
                continue;
            }
            let root = union_find.find(y * width + x);
            let component = components.entry(root).or_insert(DarkComponent {
                count: 0,
                min_x: x,
                min_y: y,
                max_x: x,
                max_y: y,
                touches_border: false,
            });
            component.count += 1;
            component.min_x = component.min_x.min(x);
            component.min_y = component.min_y.min(y);
            component.max_x = component.max_x.max(x);
            component.max_y = component.max_y.max(y);
            component.touches_border |= x == 0 || y == 0 || x + 1 == width || y + 1 == height;
        }
    }

    let mut field = vec![0.0_f32; len];
    for component in components.values() {
        let component_width = component.max_x - component.min_x + 1;
        let component_height = component.max_y - component.min_y + 1;
        let aspect = component_width as f32 / component_height.max(1) as f32;
        let fill =
            component.count as f32 / (component_width.max(1) * component_height.max(1)) as f32;
        if component.touches_border
            || component.count < 140
            || !(38..=210).contains(&component_width)
            || !(38..=250).contains(&component_height)
            || !(0.38..=1.85).contains(&aspect)
            || !(0.07..=0.72).contains(&fill)
        {
            continue;
        }

        let head_height = component_height.min(component_width * 6 / 5).max(1);
        let center_x = (component.min_x + component.max_x) as f32 * 0.5;
        let center_y = component.min_y as f32 + head_height as f32 * 0.58;
        let radius_x = (component_width as f32 * 0.31).max(12.0);
        let radius_y = (head_height as f32 * 0.35).max(14.0);
        let x0 = (center_x - radius_x).floor().max(0.0) as u32;
        let x1 = (center_x + radius_x).ceil().min((width - 1) as f32) as u32;
        let y0 = (center_y - radius_y).floor().max(0.0) as u32;
        let y1 = (center_y + radius_y).ceil().min((height - 1) as f32) as u32;
        let mut cavity_pixels = 0_u32;
        let mut bright_pixels = 0_u32;
        let mut dark_pixels = 0_u32;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let dx = (x as f32 - center_x) / radius_x;
                let dy = (y as f32 - center_y) / radius_y;
                if dx * dx + dy * dy > 1.0 {
                    continue;
                }
                cavity_pixels += 1;
                let gray = original_gray.get_pixel(x, y)[0];
                let ink = line_mask.get_pixel(x, y)[0];
                bright_pixels += u32::from(gray > 145 && ink < 108);
                dark_pixels += u32::from(gray < 88);
            }
        }
        let bright_ratio = bright_pixels as f32 / cavity_pixels.max(1) as f32;
        let dark_ratio = dark_pixels as f32 / cavity_pixels.max(1) as f32;
        if bright_ratio < 0.34 || dark_ratio > 0.48 {
            continue;
        }
        let validation =
            ((bright_ratio - 0.34) / 0.42).clamp(0.0, 1.0) * (1.0 - dark_ratio).clamp(0.4, 1.0);
        for y in y0..=y1 {
            for x in x0..=x1 {
                let gray = original_gray.get_pixel(x, y)[0];
                let ink = line_mask.get_pixel(x, y)[0];
                if gray < 112 || ink > 122 {
                    continue;
                }
                let dx = (x as f32 - center_x) / radius_x;
                let dy = (y as f32 - center_y) / radius_y;
                let radial = (1.0 - dx * dx - dy * dy).clamp(0.0, 1.0);
                let score = radial.sqrt() * validation;
                let index = (y * width + x) as usize;
                field[index] = field[index].max(score);
            }
        }
    }
    field
}

fn infer_region_semantics(
    stats: &[RegionStats],
    colors: &[RegionColor],
    original_gray: &GrayImage,
    line_mask: &GrayImage,
) -> Vec<RegionSemantic> {
    stats
        .iter()
        .zip(colors)
        .map(|(region, color)| {
            if region.count < 20 {
                return RegionSemantic::default();
            }
            let count = region.count as f64;
            let luma = (region.luma_sum / count) as f32;
            let texture = (region.texture_sum / count) as f32;
            let red = (region.existing_red_sum / count) as f32;
            let blue = (region.existing_blue_sum / count) as f32;
            let width = (region.max_x - region.min_x + 1).max(1) as f32;
            let height = (region.max_y - region.min_y + 1).max(1) as f32;
            let fill = region.count as f32 / (width * height);
            let warm_evidence =
                ((red - 3.0) / 18.0).clamp(0.0, 1.0) * ((-blue - 1.5) / 15.0).clamp(0.0, 1.0);
            let graph_warmth = ((color.red - 7.0) / 20.0).clamp(0.0, 1.0)
                * ((2.0 - color.blue) / 18.0).clamp(0.0, 1.0);
            let plausible_skin =
                !region.touches_border && luma > 72.0 && texture < 9.5 && fill > 0.2;
            let mut skin = if plausible_skin {
                warm_evidence.max(graph_warmth * color.confidence) * 0.94
            } else {
                0.0
            };
            let face_topology = face_topology_score(region, original_gray, line_mask);
            if face_topology > 0.34 {
                skin = skin.max(0.48 + face_topology * 0.42);
            }
            let elongated = width > height * 1.42 || height > width * 1.42;
            let large_bright_region = region.count > 900 && luma > 172.0 && elongated;
            let neutral = red.abs() + blue.abs() < 12.0;
            let fabric =
                if neutral && (region.touches_border || large_bright_region) && texture < 12.0 {
                    if region.touches_border {
                        0.98
                    } else {
                        0.78
                    }
                } else {
                    0.0
                };
            RegionSemantic {
                skin,
                fabric,
                skin_seed: skin > 0.34 || face_topology > 0.34,
            }
        })
        .collect()
}

fn face_topology_score(
    region: &RegionStats,
    original_gray: &GrayImage,
    line_mask: &GrayImage,
) -> f32 {
    let count = region.count.max(1) as f32;
    let mean_luma = (region.luma_sum / count as f64) as f32;
    let texture = (region.texture_sum / count as f64) as f32;
    let width = region.max_x - region.min_x + 1;
    let height = region.max_y - region.min_y + 1;
    let aspect = width as f32 / height.max(1) as f32;
    let fill = count / (width.max(1) * height.max(1)) as f32;
    if region.touches_border
        || region.count < 650
        || width < 28
        || height < 28
        || !(0.62..=1.52).contains(&aspect)
        || !(0.24..=0.76).contains(&fill)
        || !(168.0..=251.0).contains(&mean_luma)
        || texture > 9.8
    {
        return 0.0;
    }

    let top_end = region.min_y + (height * 36 / 100).max(1);
    let lower_start = region.min_y + height * 58 / 100;
    let mut top_dark = 0_u32;
    let mut top_total = 0_u32;
    let mut interior_ink = 0_u32;
    let mut interior_total = 0_u32;
    let mut lower_bright = 0_u32;
    let mut lower_total = 0_u32;
    for y in region.min_y..=region.max_y {
        for x in region.min_x..=region.max_x {
            let gray = original_gray.get_pixel(x, y)[0];
            let ink = line_mask.get_pixel(x, y)[0];
            interior_total += 1;
            interior_ink += u32::from(ink > 105 || gray < 82);
            if y <= top_end {
                top_total += 1;
                top_dark += u32::from(ink > 92 || gray < 105);
            }
            if y >= lower_start {
                lower_total += 1;
                lower_bright += u32::from(ink < 72 && gray > 155);
            }
        }
    }

    let top_ratio = top_dark as f32 / top_total.max(1) as f32;
    let ink_ratio = interior_ink as f32 / interior_total.max(1) as f32;
    let lower_ratio = lower_bright as f32 / lower_total.max(1) as f32;
    let dark_cap = ((top_ratio - 0.13) / 0.30).clamp(0.0, 1.0);
    let feature_density = ((ink_ratio - 0.045) / 0.15).clamp(0.0, 1.0);
    let open_lower = ((lower_ratio - 0.32) / 0.48).clamp(0.0, 1.0);
    (dark_cap * 0.48 + feature_density * 0.30 + open_lower * 0.22)
        * dark_cap.min(feature_density + 0.25)
}

fn build_cross_ink_adjacency(
    roots: &[u32],
    root_to_region: &HashMap<u32, usize>,
    stats: &[RegionStats],
    line_mask: &GrayImage,
    width: u32,
    height: u32,
) -> Vec<Vec<(usize, f32)>> {
    const MAX_GAP: u32 = 4;
    let mut edges = HashMap::<(usize, usize), f32>::new();
    for y in 0..height {
        for x in 0..width {
            if line_mask.get_pixel(x, y)[0] > 90 {
                continue;
            }
            let index = (y * width + x) as usize;
            let Some(&origin) = root_to_region.get(&roots[index]) else {
                continue;
            };
            for (dx, dy) in [(1_u32, 0_u32), (0_u32, 1_u32)] {
                let mut crossed_ink = false;
                for distance in 1..=MAX_GAP {
                    let nx = x + dx * distance;
                    let ny = y + dy * distance;
                    if nx >= width || ny >= height {
                        break;
                    }
                    if line_mask.get_pixel(nx, ny)[0] > 90 {
                        crossed_ink = true;
                        continue;
                    }
                    if !crossed_ink {
                        break;
                    }
                    let neighbor_index = (ny * width + nx) as usize;
                    let Some(&neighbor) = root_to_region.get(&roots[neighbor_index]) else {
                        break;
                    };
                    if origin != neighbor {
                        let pair = if origin < neighbor {
                            (origin, neighbor)
                        } else {
                            (neighbor, origin)
                        };
                        *edges.entry(pair).or_insert(0.0) +=
                            (MAX_GAP + 1 - distance) as f32 / MAX_GAP as f32;
                    }
                    break;
                }
            }
        }
    }

    let mut adjacency = vec![Vec::new(); stats.len()];
    for ((left, right), boundary) in edges {
        let left_luma = stats[left].luma_sum / stats[left].count.max(1) as f64;
        let right_luma = stats[right].luma_sum / stats[right].count.max(1) as f64;
        let similarity = (-(left_luma - right_luma).abs() as f32 / 42.0).exp();
        let weight = (boundary.sqrt() * similarity).clamp(0.0, 7.0);
        if weight > 0.06 {
            adjacency[left].push((right, weight));
            adjacency[right].push((left, weight));
        }
    }
    adjacency
}

fn propagate_region_semantics(
    semantics: &mut Vec<RegionSemantic>,
    stats: &[RegionStats],
    adjacency: &[Vec<(usize, f32)>],
) {
    let mut next = semantics.clone();
    for _ in 0..8 {
        for index in 0..semantics.len() {
            if stats[index].count < 20 || semantics[index].skin_seed {
                next[index] = semantics[index];
                continue;
            }
            let mut strongest = 0.0_f32;
            let mut weighted_skin = 0.0_f32;
            let mut support = 0.0_f32;
            for &(neighbor, boundary) in &adjacency[index] {
                let evidence = semantics[neighbor].skin * boundary;
                strongest = strongest.max(evidence);
                weighted_skin += evidence;
                support += boundary;
            }
            let proposal = if support > 0.0 {
                (weighted_skin / support).max(strongest / (strongest + 1.8))
            } else {
                0.0
            };
            let fabric_gate = 1.0 - semantics[index].fabric * 0.96;
            next[index].skin = semantics[index].skin.max(proposal * 0.82 * fabric_gate);
        }
        std::mem::swap(semantics, &mut next);
    }
}

fn maybe_union(
    union_find: &mut UnionFind,
    left: usize,
    right: usize,
    left_gray: u8,
    right_gray: u8,
    right_ink: u8,
) {
    if right_ink > 118 {
        return;
    }
    let average = (left_gray as u16 + right_gray as u16) / 2;
    let tolerance = 9 + ((255 - average) / 30) as u8;
    if left_gray.abs_diff(right_gray) <= tolerance {
        union_find.union(left as u32, right as u32);
    }
}

fn accumulate_region(
    stats: &mut RegionStats,
    pixel: Rgb<u8>,
    original: u8,
    smooth: u8,
    x: u32,
    y: u32,
    dimensions: (u32, u32),
) {
    let (width, height) = dimensions;
    let luma = rgb_luma(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
    let red = pixel[0] as f32 - luma;
    let blue = pixel[2] as f32 - luma;
    let chroma = pixel[0].max(pixel[1]).max(pixel[2]) - pixel[0].min(pixel[1]).min(pixel[2]);
    let seed_weight = ((chroma as f32 - 7.0) / 28.0).clamp(0.0, 1.0) as f64;

    stats.count += 1;
    stats.min_x = stats.min_x.min(x);
    stats.min_y = stats.min_y.min(y);
    stats.max_x = stats.max_x.max(x);
    stats.max_y = stats.max_y.max(y);
    stats.luma_sum += luma as f64;
    stats.texture_sum += original.abs_diff(smooth) as f64;
    stats.existing_red_sum += red as f64;
    stats.existing_blue_sum += blue as f64;
    stats.seed_weight += seed_weight;
    stats.red_chroma_sum += red as f64 * seed_weight;
    stats.blue_chroma_sum += blue as f64 * seed_weight;
    stats.touches_border |= x == 0 || y == 0 || x + 1 == width || y + 1 == height;
}

fn infer_region_color(region: &RegionStats, palette: &[(f32, f32, f32)]) -> RegionColor {
    if region.count < 20 {
        return RegionColor::default();
    }
    let count = region.count as f64;
    let mean_luma = (region.luma_sum / count) as f32;
    let existing_red = (region.existing_red_sum / count) as f32;
    let existing_blue = (region.existing_blue_sum / count) as f32;
    let seed_ratio = (region.seed_weight / count) as f32;
    let existing_magnitude = existing_red.abs() + existing_blue.abs();
    if region.touches_border && mean_luma > 150.0 && existing_blue > 1.0 {
        if let Some((_, red, blue)) = palette
            .iter()
            .filter(|(_, red, blue)| *red >= 0.0 && *blue <= 0.0)
            .min_by(|left, right| {
                (left.0 - mean_luma)
                    .abs()
                    .total_cmp(&(right.0 - mean_luma).abs())
            })
            .copied()
        {
            return RegionColor {
                red: red * 0.78,
                blue: blue * 0.78,
                confidence: 0.36,
                seeded: false,
            };
        }
    }
    if region.seed_weight >= 1.8 && seed_ratio >= 0.008 {
        return RegionColor {
            red: (region.red_chroma_sum / region.seed_weight) as f32,
            blue: (region.blue_chroma_sum / region.seed_weight) as f32,
            confidence: (0.38 + seed_ratio * 5.5).clamp(0.38, 0.94),
            seeded: true,
        };
    }
    if mean_luma > 218.0 && existing_magnitude < 5.0 {
        return RegionColor::default();
    }
    if existing_magnitude >= 3.5 {
        return RegionColor {
            red: existing_red * 1.3,
            blue: existing_blue * 1.3,
            confidence: 0.28,
            seeded: false,
        };
    }

    if mean_luma > 244.0 || palette.is_empty() {
        return RegionColor::default();
    }

    let texture = (region.texture_sum / count) as f32;
    let target =
        palette.iter().min_by(|left, right| {
            palette_score(**left, mean_luma, texture, region.touches_border).total_cmp(
                &palette_score(**right, mean_luma, texture, region.touches_border),
            )
        });
    let Some((_, red, blue)) = target.copied() else {
        return RegionColor::default();
    };
    let area_confidence = ((region.count as f32).ln_1p() / 12.0).clamp(0.12, 0.5);
    RegionColor {
        red: red * 0.72,
        blue: blue * 0.72,
        confidence: area_confidence * if mean_luma > 224.0 { 0.45 } else { 0.72 },
        seeded: false,
    }
}

fn palette_score(color: (f32, f32, f32), luma: f32, texture: f32, touches_border: bool) -> f32 {
    let (palette_luma, red, blue) = color;
    let luma_cost = (palette_luma - luma).abs();
    let chroma = red.abs() + blue.abs();
    let bright_penalty = if luma > 220.0 { chroma * 0.42 } else { 0.0 };
    let border_bonus = if touches_border && texture < 9.0 {
        -chroma.min(36.0) * 0.12
    } else {
        0.0
    };
    luma_cost + bright_penalty + border_bonus
}

fn build_adjacency(
    roots: &[u32],
    root_to_region: &HashMap<u32, usize>,
    stats: &[RegionStats],
    line_mask: &GrayImage,
    width: u32,
    height: u32,
) -> Vec<Vec<(usize, f32)>> {
    let mut edges = HashMap::<(usize, usize), f32>::new();
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let Some(&left) = root_to_region.get(&roots[index]) else {
                continue;
            };
            for (nx, ny) in [(x + 1, y), (x, y + 1)] {
                if nx >= width || ny >= height {
                    continue;
                }
                let neighbor_index = (ny * width + nx) as usize;
                let Some(&right) = root_to_region.get(&roots[neighbor_index]) else {
                    continue;
                };
                if left == right {
                    continue;
                }
                let ink = line_mask.get_pixel(x, y)[0].max(line_mask.get_pixel(nx, ny)[0]);
                if ink > 90 {
                    continue;
                }
                let pair = if left < right {
                    (left, right)
                } else {
                    (right, left)
                };
                *edges.entry(pair).or_insert(0.0) += 1.0 - ink as f32 / 255.0;
            }
        }
    }

    let mut adjacency = vec![Vec::new(); stats.len()];
    for ((left, right), boundary) in edges {
        let left_luma = stats[left].luma_sum / stats[left].count.max(1) as f64;
        let right_luma = stats[right].luma_sum / stats[right].count.max(1) as f64;
        let similarity = (-(left_luma - right_luma).abs() as f32 / 24.0).exp();
        let weight = (boundary.sqrt() * similarity).clamp(0.0, 8.0);
        if weight > 0.08 {
            adjacency[left].push((right, weight));
            adjacency[right].push((left, weight));
        }
    }
    adjacency
}

fn propagate_graph_colors(
    colors: &mut Vec<RegionColor>,
    stats: &[RegionStats],
    adjacency: &[Vec<(usize, f32)>],
) {
    let mut next = colors.clone();
    for _ in 0..7 {
        for index in 0..colors.len() {
            if stats[index].count < 20 || colors[index].seeded {
                next[index] = colors[index];
                continue;
            }
            let mut red = 0.0;
            let mut blue = 0.0;
            let mut support = 0.0;
            for &(neighbor, boundary) in &adjacency[index] {
                let confidence = colors[neighbor].confidence;
                if confidence <= 0.04 {
                    continue;
                }
                let weight = boundary * confidence;
                red += colors[neighbor].red * weight;
                blue += colors[neighbor].blue * weight;
                support += weight;
            }
            if support > 0.08 {
                let proposal_red = red / support;
                let proposal_blue = blue / support;
                let incoming = (support / (support + 2.5)).clamp(0.0, 0.62);
                next[index].red += (proposal_red - next[index].red) * incoming;
                next[index].blue += (proposal_blue - next[index].blue) * incoming;
                next[index].confidence = next[index]
                    .confidence
                    .max((incoming * 0.72).clamp(0.0, 0.48));
            } else {
                next[index] = colors[index];
            }
        }
        std::mem::swap(colors, &mut next);
    }
}

#[allow(clippy::too_many_arguments)]
fn render_material_pixel(
    pixel: &mut Rgb<u8>,
    original: u8,
    illumination: u8,
    ink_distance: u16,
    ink: f32,
    x: u32,
    y: u32,
    region: RegionStats,
    color: RegionColor,
    semantic: RegionSemantic,
    face_cavity: f32,
    style: &StyleProfile,
) {
    let current_luma = rgb_luma(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32);
    let original_detail = original as f32 - illumination as f32;
    let base_luma = current_luma * 0.68
        + (illumination as f32 + original_detail.clamp(-24.0, 24.0) * 0.58) * 0.32;
    let region_width = (region.max_x - region.min_x + 1).max(1) as f32;
    let region_height = (region.max_y - region.min_y + 1).max(1) as f32;
    let local_x = (x - region.min_x) as f32 / region_width;
    let local_y = (y - region.min_y) as f32 / region_height;
    let directional_light =
        ((1.0 - local_x) * 0.55 + (1.0 - local_y) * 0.45 - 0.5) * style.highlight_strength * 10.0;

    let mut red = color.red;
    let mut blue = color.blue;
    let region_area =
        ((region.max_x - region.min_x + 1) * (region.max_y - region.min_y + 1)).max(1) as f32;
    let fill_ratio = region.count as f32 / region_area;
    let texture = (region.texture_sum / region.count.max(1) as f64) as f32;
    let region_red = (region.existing_red_sum / region.count.max(1) as f64) as f32;
    let region_blue = (region.existing_blue_sum / region.count.max(1) as f64) as f32;
    let observed_skin = region_red > 5.0 && region_blue < -3.0;
    let warm_candidate = ((red > 9.0 && blue < 1.0) || observed_skin) && base_luma > 72.0;
    let face_skin = face_cavity > 0.12;
    let semantic_skin = semantic.skin > 0.42 && semantic.fabric < 0.45;
    let skin = face_skin
        || ((warm_candidate || semantic_skin)
            && !region.touches_border
            && fill_ratio > 0.2
            && texture < 9.5);
    let dark_material = base_luma < 105.0;
    if skin && observed_skin {
        red = red.max(region_red * 1.35);
        blue = blue.min(region_blue * 1.3);
    }
    if skin && (semantic_skin || face_skin) {
        let strength = if face_skin {
            face_cavity.clamp(0.35, 1.0)
        } else {
            ((semantic.skin - 0.32) / 0.5).clamp(0.25, 1.0)
        };
        red = red.max(13.0 + strength * 12.0);
        blue = blue.min(-8.0 - strength * 10.0);
    }
    if warm_candidate && !skin && base_luma > 164.0 {
        red *= 0.18;
        blue *= 0.18;
    }
    let pale_material = base_luma > 178.0 && red.abs() + blue.abs() < 20.0;
    let shadow = ((150.0 - original as f32) / 120.0).clamp(0.0, 1.0);
    let highlight = ((original as f32 - 205.0) / 50.0).clamp(0.0, 1.0);
    let mut target_luma = base_luma + directional_light - shadow * style.shadow_strength * 4.5
        + highlight * style.highlight_strength * 2.0;
    let ambient_occlusion = (1.0 - ink_distance.min(12) as f32 / 12.0).powi(2);

    if skin {
        red += 19.0 * (1.0 - shadow * 0.26);
        blue -= 14.0 * (1.0 - shadow * 0.4);
        let cheek_band = (1.0 - ((local_y - 0.58).abs() / 0.22)).clamp(0.0, 1.0);
        let side_weight = ((local_x - 0.5).abs() * 2.0).clamp(0.0, 1.0);
        red += cheek_band * side_weight * 5.8;
        blue -= cheek_band * side_weight * 2.2;
        target_luma += highlight * 2.0 - 26.0 - shadow * 5.0 - ambient_occlusion * 8.5;
    } else if dark_material {
        red += style.shadow_strength * (1.3 + highlight);
        blue += style.shadow_strength * (2.3 + highlight * 1.4);
        target_luma -= ambient_occlusion * 3.0;
    } else if pale_material && shadow > 0.05 {
        red -= shadow * 1.2;
        blue += shadow * 3.4;
        target_luma -= ambient_occlusion * 4.5;
    }

    let confidence = color.confidence.clamp(0.0, 1.0);
    let existing_red = pixel[0] as f32 - current_luma;
    let existing_blue = pixel[2] as f32 - current_luma;
    let blend = (0.28 + confidence * 0.68) * (1.0 - ink).powi(2);
    red = existing_red + (red - existing_red) * blend;
    blue = existing_blue + (blue - existing_blue) * blend;
    *pixel = rgb_from_luma_chroma(target_luma.clamp(0.0, 255.0), red, blue);
}

fn rgb_from_luma_chroma(luma: f32, red: f32, blue: f32) -> Rgb<u8> {
    let green = (-0.2126 * red - 0.0722 * blue) / 0.7152;
    let mut scale = 1.0f32;
    for chroma in [red, green, blue] {
        if chroma > 0.0 {
            scale = scale.min((255.0 - luma) / chroma);
        } else if chroma < 0.0 {
            scale = scale.min(luma / -chroma);
        }
    }
    let scale = scale.clamp(0.0, 1.0);
    Rgb([
        (luma + red * scale).round().clamp(0.0, 255.0) as u8,
        (luma + green * scale).round().clamp(0.0, 255.0) as u8,
        (luma + blue * scale).round().clamp(0.0, 255.0) as u8,
    ])
}

fn rgb_luma(red: f32, green: f32, blue: f32) -> f32 {
    0.2126 * red + 0.7152 * green + 0.0722 * blue
}

fn box_blur_gray(image: &GrayImage, radius: u32) -> GrayImage {
    let (width, height) = image.dimensions();
    let stride = width as usize + 1;
    let mut integral = vec![0u64; stride * (height as usize + 1)];
    for y in 0..height {
        let mut row_sum = 0u64;
        for x in 0..width {
            row_sum += image.get_pixel(x, y)[0] as u64;
            let position = (y as usize + 1) * stride + x as usize + 1;
            integral[position] = integral[y as usize * stride + x as usize + 1] + row_sum;
        }
    }

    let mut output = GrayImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let x0 = x.saturating_sub(radius);
            let y0 = y.saturating_sub(radius);
            let x1 = (x + radius + 1).min(width);
            let y1 = (y + radius + 1).min(height);
            let sum = integral[y1 as usize * stride + x1 as usize]
                + integral[y0 as usize * stride + x0 as usize]
                - integral[y0 as usize * stride + x1 as usize]
                - integral[y1 as usize * stride + x0 as usize];
            let area = (x1 - x0) as u64 * (y1 - y0) as u64;
            output.put_pixel(x, y, image::Luma([(sum / area.max(1)) as u8]));
        }
    }
    output
}

fn distance_to_ink(line_mask: &GrayImage) -> Vec<u16> {
    let (width, height) = line_mask.dimensions();
    let mut distance = vec![u16::MAX / 4; width as usize * height as usize];
    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            if line_mask.get_pixel(x, y)[0] > 72 {
                distance[index] = 0;
                continue;
            }
            if x > 0 {
                distance[index] = distance[index].min(distance[index - 1].saturating_add(1));
            }
            if y > 0 {
                distance[index] =
                    distance[index].min(distance[index - width as usize].saturating_add(1));
            }
        }
    }
    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let index = (y * width + x) as usize;
            if x + 1 < width {
                distance[index] = distance[index].min(distance[index + 1].saturating_add(1));
            }
            if y + 1 < height {
                distance[index] =
                    distance[index].min(distance[index + width as usize].saturating_add(1));
            }
        }
    }
    distance
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    use mangahue_core::ColorRgb;

    #[test]
    fn region_graph_respects_ink_boundaries() {
        let mut image = RgbImage::from_pixel(24, 10, Rgb([160, 160, 160]));
        image.put_pixel(3, 5, Rgb([210, 126, 102]));
        image.put_pixel(4, 5, Rgb([210, 126, 102]));
        image.put_pixel(3, 6, Rgb([210, 126, 102]));
        let original = GrayImage::from_pixel(24, 10, Luma([160]));
        let mut lines = GrayImage::from_pixel(24, 10, Luma([0]));
        for y in 0..10 {
            lines.put_pixel(12, y, Luma([255]));
        }
        let style = StyleProfile {
            palette: Vec::new(),
            ..StyleProfile::default()
        };

        let report = render_region_graph(&mut image, &original, &lines, &style);

        let left = image.get_pixel(8, 5);
        let right = image.get_pixel(18, 5);
        assert!(left[0] > left[1] && left[1] > left[2]);
        assert!((right[0] as i16 - right[2] as i16).abs() < 5);
        assert!(report.seeded_regions >= 1);
    }

    #[test]
    fn palette_inference_colors_large_unseeded_materials() {
        let mut image = RgbImage::from_pixel(20, 20, Rgb([172, 172, 172]));
        let original = GrayImage::from_pixel(20, 20, Luma([172]));
        let mut lines = GrayImage::from_pixel(20, 20, Luma([0]));
        for coordinate in 0..20 {
            lines.put_pixel(coordinate, 0, Luma([255]));
            lines.put_pixel(coordinate, 19, Luma([255]));
            lines.put_pixel(0, coordinate, Luma([255]));
            lines.put_pixel(19, coordinate, Luma([255]));
        }
        let style = StyleProfile {
            palette: vec![ColorRgb::new(198, 146, 126)],
            ..StyleProfile::default()
        };

        let report = render_region_graph(&mut image, &original, &lines, &style);

        let pixel = image.get_pixel(10, 10);
        assert!(pixel[0] > pixel[2]);
        assert!(report.inferred_regions >= 1);
        assert_eq!(report.pixels_rendered, 324);
    }

    #[test]
    fn bright_neutral_fabric_is_not_forced_into_skin_palette() {
        let mut image = RgbImage::from_pixel(20, 20, Rgb([232, 232, 232]));
        let original = GrayImage::from_pixel(20, 20, Luma([232]));
        let lines = GrayImage::from_pixel(20, 20, Luma([0]));
        let style = StyleProfile {
            palette: vec![ColorRgb::new(239, 184, 160)],
            ..StyleProfile::default()
        };

        render_region_graph(&mut image, &original, &lines, &style);

        let pixel = image.get_pixel(10, 10);
        assert!((pixel[0] as i16 - pixel[2] as i16).abs() < 5);
    }

    #[test]
    fn ink_distance_field_expands_from_line_geometry() {
        let mut lines = GrayImage::from_pixel(7, 1, Luma([0]));
        lines.put_pixel(3, 0, Luma([255]));

        let distance = distance_to_ink(&lines);

        assert_eq!(distance, vec![3, 2, 1, 0, 1, 2, 3]);
    }

    #[test]
    fn semantic_skin_crosses_ink_while_fabric_gate_blocks_transfer() {
        let region = RegionStats {
            count: 400,
            min_x: 2,
            min_y: 2,
            max_x: 21,
            max_y: 21,
            luma_sum: 80_000.0,
            texture_sum: 800.0,
            seed_weight: 0.0,
            red_chroma_sum: 0.0,
            blue_chroma_sum: 0.0,
            existing_red_sum: 0.0,
            existing_blue_sum: 0.0,
            touches_border: false,
        };
        let stats = vec![region; 3];
        let mut semantics = vec![
            RegionSemantic {
                skin: 0.9,
                fabric: 0.0,
                skin_seed: true,
            },
            RegionSemantic::default(),
            RegionSemantic {
                skin: 0.0,
                fabric: 0.98,
                skin_seed: false,
            },
        ];
        let adjacency = vec![vec![(1, 4.0), (2, 4.0)], vec![(0, 4.0)], vec![(0, 4.0)]];

        propagate_region_semantics(&mut semantics, &stats, &adjacency);

        assert!(semantics[1].skin > 0.42);
        assert!(semantics[2].skin < 0.08);
    }

    #[test]
    fn bright_round_region_is_not_mistaken_for_large_fabric() {
        let round_region = RegionStats {
            count: 4_000,
            min_x: 10,
            min_y: 10,
            max_x: 109,
            max_y: 109,
            luma_sum: 920_000.0,
            texture_sum: 4_000.0,
            seed_weight: 0.0,
            red_chroma_sum: 0.0,
            blue_chroma_sum: 0.0,
            existing_red_sum: 0.0,
            existing_blue_sum: 0.0,
            touches_border: false,
        };

        let original = GrayImage::from_pixel(120, 120, Luma([230]));
        let lines = GrayImage::from_pixel(120, 120, Luma([0]));
        let semantics = infer_region_semantics(
            &[round_region],
            &[RegionColor::default()],
            &original,
            &lines,
        );

        assert_eq!(semantics[0].fabric, 0.0);
    }

    #[test]
    fn dark_cap_and_eye_structure_identify_bright_face_cavity() {
        let region = RegionStats {
            count: 1_800,
            min_x: 20,
            min_y: 20,
            max_x: 79,
            max_y: 79,
            luma_sum: 396_000.0,
            texture_sum: 3_600.0,
            seed_weight: 0.0,
            red_chroma_sum: 0.0,
            blue_chroma_sum: 0.0,
            existing_red_sum: 0.0,
            existing_blue_sum: 0.0,
            touches_border: false,
        };
        let original = GrayImage::from_pixel(100, 100, Luma([220]));
        let mut lines = GrayImage::from_pixel(100, 100, Luma([0]));
        for y in 20..=40 {
            for x in 20..=79 {
                lines.put_pixel(x, y, Luma([180]));
            }
        }

        let score = face_topology_score(&region, &original, &lines);

        assert!(score > 0.34);
    }

    #[test]
    fn hair_component_generates_a_bounded_face_cavity_field() {
        let mut original = GrayImage::from_pixel(120, 120, Luma([220]));
        let mut lines = GrayImage::from_pixel(120, 120, Luma([0]));
        for y in 20..=40 {
            for x in 20..=99 {
                original.put_pixel(x, y, Luma([38]));
                lines.put_pixel(x, y, Luma([220]));
            }
        }
        for y in 20..=90 {
            for x in 20..=29 {
                original.put_pixel(x, y, Luma([42]));
                lines.put_pixel(x, y, Luma([210]));
            }
            for x in 90..=99 {
                original.put_pixel(x, y, Luma([42]));
                lines.put_pixel(x, y, Luma([210]));
            }
        }

        let field = detect_face_cavity_field(&original, &lines);

        assert!(field[62 * 120 + 60] > 0.2);
        assert_eq!(field[110 * 120 + 60], 0.0);
    }
}
