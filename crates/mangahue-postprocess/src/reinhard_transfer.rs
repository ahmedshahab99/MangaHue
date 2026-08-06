use image::{GrayImage, Rgb, RgbImage};

/// Reinhard et al. 2001 color transfer in LAB space.
///
/// Shifts the output's L/a/b mean and standard deviation to match a
/// reference photo, so the overall grading (lightness, warmth, contrast)
/// matches the reference look while keeping the output's own structure.
pub(crate) fn apply_reinhard_transfer(
    image: &mut RgbImage,
    reference: &RgbImage,
    line_mask: &GrayImage,
    strength: f32,
) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 || line_mask.dimensions() != (width, height) {
        return;
    }

    // Resize the reference to the page size so the transfer always applies,
    // even when the page was resized by preprocessing (max_input_dimension).
    let reference = if reference.dimensions() != (width, height) {
        image::imageops::resize(
            reference,
            width,
            height,
            image::imageops::FilterType::Lanczos3,
        )
    } else {
        reference.clone()
    };

    // Compute LAB stats on non-ink pixels only (avoid the black lineart
    // dominating the statistics).
    let mut src_l = Vec::with_capacity((width * height) as usize);
    let mut src_a = Vec::with_capacity(src_l.capacity());
    let mut src_b = Vec::with_capacity(src_l.capacity());
    let mut ref_l = Vec::with_capacity(src_l.capacity());
    let mut ref_a = Vec::with_capacity(src_l.capacity());
    let mut ref_b = Vec::with_capacity(src_l.capacity());

    for y in 0..height {
        for x in 0..width {
            if line_mask.get_pixel(x, y)[0] > 100 {
                continue;
            }
            let src = image.get_pixel(x, y);
            let (sl, sa, sb) = rgb_to_lab(src[0], src[1], src[2]);
            src_l.push(sl);
            src_a.push(sa);
            src_b.push(sb);

            let rp = reference.get_pixel(x, y);
            let (rl, ra, rb) = rgb_to_lab(rp[0], rp[1], rp[2]);
            ref_l.push(rl);
            ref_a.push(ra);
            ref_b.push(rb);
        }
    }

    if src_l.len() < 64 || ref_l.len() < 64 {
        return;
    }

    let (sm_l, ss_l) = stats(&src_l);
    let (sm_a, ss_a) = stats(&src_a);
    let (sm_b, ss_b) = stats(&src_b);
    let (rm_l, rs_l) = stats(&ref_l);
    let (rm_a, rs_a) = stats(&ref_a);
    let (rm_b, rs_b) = stats(&ref_b);

    let t = strength.clamp(0.0, 1.0);
    let chroma_gain = 1.05_f32;
    // Cast-only transfer: shift each pixel by the difference in the channel
    // MEANS (a global warm/cool tint) while keeping the pixel's own hue
    // direction. Re-scaling per-pixel std dragged all hues toward the
    // reference mean and collapsed the palette to a single hue.
    let da = (rm_a - sm_a) * t;
    let db = (rm_b - sm_b) * t;
    let dl = (rm_l - sm_l) * t;
    for y in 0..height {
        for x in 0..width {
            if line_mask.get_pixel(x, y)[0] > 100 {
                continue;
            }
            let p = *image.get_pixel(x, y);
            let (l, a, b) = rgb_to_lab(p[0], p[1], p[2]);

            let new_l = l + dl;
            // Apply the reference cast as a tint; keep the pixel's chroma
            // magnitude (saturation) but let the cast rotate its hue gently.
            let chroma = (a * a + b * b).sqrt();
            let new_a = if chroma > 0.01 {
                (a + da * (1.0 + chroma * 0.06)) * chroma_gain
            } else {
                (a + da * 0.25) * chroma_gain
            };
            let new_b = if chroma > 0.01 {
                (b + db * (1.0 + chroma * 0.06)) * chroma_gain
            } else {
                (b + db * 0.25) * chroma_gain
            };
            // Clamp a/b to physically reachable LAB range for sRGB.
            let new_a = new_a.clamp(-120.0, 120.0);
            let new_b = new_b.clamp(-120.0, 120.0);

            let (r, g, b) = lab_to_rgb(new_l, new_a, new_b);
            image.put_pixel(
                x,
                y,
                Rgb([
                    (r.clamp(0.0, 255.0)) as u8,
                    (g.clamp(0.0, 255.0)) as u8,
                    (b.clamp(0.0, 255.0)) as u8,
                ]),
            );
        }
    }
}

fn stats(values: &[f32]) -> (f32, f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// sRGB -> CIELAB (D65).
fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let r = srgb_to_linear(r as f32 / 255.0);
    let g = srgb_to_linear(g as f32 / 255.0);
    let b = srgb_to_linear(b as f32 / 255.0);
    let x = 0.4124 * r + 0.3576 * g + 0.1805 * b;
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let z = 0.0193 * r + 0.1192 * g + 0.9505 * b;
    let fx = lab_f(x / 0.95047);
    let fy = lab_f(y);
    let fz = lab_f(z / 1.08883);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

/// CIELAB -> sRGB (D65).
fn lab_to_rgb(l: f32, a: f32, b: f32) -> (f32, f32, f32) {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;

    let x = lab_f_inv(fx) * 0.95047;
    let y = lab_f_inv(fy);
    let z = lab_f_inv(fz) * 1.08883;

    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;

    (
        linear_to_srgb(r) * 255.0,
        linear_to_srgb(g) * 255.0,
        linear_to_srgb(b) * 255.0,
    )
}

fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(c: f32) -> f32 {
    let c = c.max(0.0);
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn lab_f(t: f32) -> f32 {
    if t > 0.008856 {
        t.cbrt()
    } else {
        7.787 * t + 16.0 / 116.0
    }
}

fn lab_f_inv(t: f32) -> f32 {
    let t3 = t * t * t;
    if t3 > 0.008856 {
        t3
    } else {
        (t - 16.0 / 116.0) / 7.787
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn lab_round_trip() {
        for (r, g, b) in [
            (120u8, 80u8, 60u8),
            (200, 180, 170),
            (30, 30, 30),
            (255, 255, 255),
        ] {
            let (l, a, b_) = rgb_to_lab(r, g, b);
            let (nr, ng, nb) = lab_to_rgb(l, a, b_);
            assert!((nr - r as f32).abs() < 3.0, "r {r} -> {nr}");
            assert!((ng - g as f32).abs() < 3.0, "g {g} -> {ng}");
            assert!((nb - b as f32).abs() < 3.0, "b {b} -> {nb}");
        }
    }

    #[test]
    fn transfer_moves_stats_toward_reference() {
        // Warm reference, cool output: after transfer, output warms up.
        let mut image = RgbImage::from_pixel(64, 64, Rgb([100, 140, 180])); // cool
        let reference = RgbImage::from_pixel(64, 64, Rgb([220, 160, 110])); // warm
        let lines = GrayImage::from_pixel(64, 64, Luma([0]));
        let before = *image.get_pixel(32, 32);
        apply_reinhard_transfer(&mut image, &reference, &lines, 1.0);
        let after = *image.get_pixel(32, 32);
        // Output should become warmer (R up, B down).
        assert!(
            after[0] as i32 > before[0] as i32,
            "R {} -> {}",
            before[0],
            after[0]
        );
        assert!(
            (after[2] as i32) < (before[2] as i32),
            "B {} -> {}",
            before[2],
            after[2]
        );
    }
}
