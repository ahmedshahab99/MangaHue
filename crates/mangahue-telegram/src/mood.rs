//! Content-aware mood selection: choose a color style that fits the
//! manga's nature (dark/horror -> dark, romance -> pastel, fantasy ->
//! vibrant, ...) using source tags when available and page statistics
//! otherwise. All style ids below resolve via `styles_dir/{id}.yaml`.

use image::GrayImage;

/// Tag keyword -> style ids with weights. Lowercase substring match.
/// Higher weight wins; ties break alphabetically for determinism.
pub fn mood_table() -> &'static [(&'static str, &'static [(&'static str, i32)])] {
    &[
        // Dark / heavy
        (
            "dark",
            &[
                ("dark_seinen", 3),
                ("creative_cool_twilight", 2),
                ("creative_violet_dusk", 1),
            ],
        ),
        ("horror", &[("dark_seinen", 3), ("creative_violet_dusk", 2)]),
        ("gore", &[("dark_seinen", 3), ("creative_cool_twilight", 1)]),
        (
            "thriller",
            &[("dark_seinen", 3), ("creative_cool_twilight", 2)],
        ),
        (
            "psychological",
            &[("creative_cool_twilight", 3), ("dark_seinen", 2)],
        ),
        (
            "tragedy",
            &[("creative_violet_dusk", 3), ("dark_seinen", 2)],
        ),
        (
            "crime",
            &[("dark_seinen", 3), ("creative_cool_twilight", 2)],
        ),
        (
            "detective",
            &[("dark_seinen", 3), ("creative_cool_twilight", 2)],
        ),
        ("noir", &[("dark_seinen", 3), ("creative_cool_twilight", 2)]),
        ("mind break", &[("dark_seinen", 3)]),
        (
            "netorare",
            &[("creative_cool_twilight", 3), ("creative_violet_dusk", 2)],
        ),
        // Romance / soft
        (
            "romance",
            &[
                ("creative_soft_pastel", 3),
                ("creative_warm_sunset", 2),
                ("watercolor_soft", 1),
            ],
        ),
        (
            "vanilla",
            &[("creative_soft_pastel", 3), ("creative_warm_sunset", 2)],
        ),
        (
            "shoujo",
            &[("creative_soft_pastel", 3), ("watercolor_soft", 2)],
        ),
        (
            "wholesome",
            &[("creative_warm_cocoa", 3), ("creative_soft_pastel", 2)],
        ),
        (
            "love",
            &[("creative_warm_sunset", 3), ("creative_soft_pastel", 2)],
        ),
        (
            "harem",
            &[("creative_warm_sunset", 3), ("creative_soft_pastel", 1)],
        ),
        (
            "marriage",
            &[("creative_warm_sunset", 3), ("creative_warm_cocoa", 2)],
        ),
        // Fantasy / magic
        (
            "fantasy",
            &[
                ("creative_violet_dusk", 3),
                ("creative_ocean_teal", 2),
                ("fantasy_magic", 2),
            ],
        ),
        (
            "magic",
            &[("creative_violet_dusk", 3), ("fantasy_magic", 2)],
        ),
        (
            "elf",
            &[("creative_fresh_sage", 3), ("creative_violet_dusk", 2)],
        ),
        (
            "dragon",
            &[("creative_autumn_amber", 3), ("creative_ocean_teal", 2)],
        ),
        (
            "adventure",
            &[
                ("creative_ocean_teal", 2),
                ("creative_autumn_amber", 2),
                ("realistic_comic", 1),
            ],
        ),
        ("demon", &[("creative_violet_dusk", 3), ("dark_seinen", 2)]),
        // Sci-fi / tech
        (
            "sci-fi",
            &[
                ("creative_cool_twilight", 3),
                ("creative_ocean_teal", 2),
                ("realistic_comic", 1),
            ],
        ),
        (
            "science fiction",
            &[("creative_cool_twilight", 3), ("creative_ocean_teal", 2)],
        ),
        (
            "cyborg",
            &[("creative_cool_twilight", 3), ("realistic_comic", 2)],
        ),
        (
            "alien",
            &[("creative_cool_twilight", 3), ("creative_ocean_teal", 2)],
        ),
        (
            "mecha",
            &[("creative_cool_twilight", 3), ("realistic_comic", 2)],
        ),
        (
            "robot",
            &[("creative_cool_twilight", 3), ("realistic_comic", 2)],
        ),
        (
            "space",
            &[("creative_cool_twilight", 3), ("creative_ocean_teal", 2)],
        ),
        // Action / intensity
        (
            "action",
            &[
                ("realistic_comic", 3),
                ("creative_cool_twilight", 2),
                ("dark_seinen", 1),
            ],
        ),
        (
            "fighting",
            &[("realistic_comic", 3), ("creative_autumn_amber", 2)],
        ),
        (
            "martial",
            &[("realistic_comic", 3), ("creative_autumn_amber", 2)],
        ),
        ("military", &[("realistic_comic", 3), ("dark_seinen", 2)]),
        ("war", &[("dark_seinen", 3), ("realistic_comic", 2)]),
        (
            "superhero",
            &[("realistic_comic", 3), ("creative_cool_twilight", 2)],
        ),
        // Nature / calm
        (
            "nature",
            &[("creative_fresh_sage", 3), ("creative_warm_cocoa", 2)],
        ),
        (
            "animal",
            &[("creative_fresh_sage", 3), ("creative_warm_cocoa", 2)],
        ),
        (
            "farm",
            &[("creative_fresh_sage", 3), ("creative_warm_cocoa", 2)],
        ),
        (
            "fishing",
            &[("creative_ocean_teal", 3), ("creative_fresh_sage", 2)],
        ),
        (
            "slice of life",
            &[
                ("creative_warm_cocoa", 3),
                ("creative_fresh_sage", 2),
                ("watercolor_soft", 1),
            ],
        ),
        (
            "school",
            &[
                ("creative_soft_pastel", 2),
                ("creative_fresh_sage", 2),
                ("watercolor_soft", 1),
            ],
        ),
        (
            "daily life",
            &[("creative_warm_cocoa", 3), ("creative_fresh_sage", 2)],
        ),
        (
            "comedy",
            &[
                ("creative_warm_sunset", 2),
                ("creative_soft_pastel", 2),
                ("watercolor_soft", 1),
            ],
        ),
        // Seasons / weather
        (
            "autumn",
            &[("creative_autumn_amber", 3), ("creative_warm_cocoa", 2)],
        ),
        (
            "winter",
            &[("creative_cool_twilight", 3), ("creative_ocean_teal", 1)],
        ),
        ("snow", &[("creative_cool_twilight", 3)]),
        ("cold", &[("creative_cool_twilight", 3)]),
        ("ice", &[("creative_cool_twilight", 3)]),
        // Water
        (
            "beach",
            &[("creative_ocean_teal", 3), ("creative_warm_sunset", 2)],
        ),
        (
            "swimsuit",
            &[("creative_ocean_teal", 3), ("creative_warm_sunset", 2)],
        ),
        (
            "ocean",
            &[("creative_ocean_teal", 3), ("creative_fresh_sage", 1)],
        ),
        ("pool", &[("creative_ocean_teal", 3)]),
        (
            "boat",
            &[("creative_ocean_teal", 3), ("creative_cool_twilight", 1)],
        ),
        (
            "water",
            &[("creative_ocean_teal", 3), ("creative_fresh_sage", 1)],
        ),
        // Historical / food / misc
        (
            "historical",
            &[("creative_autumn_amber", 3), ("creative_warm_cocoa", 2)],
        ),
        (
            "samurai",
            &[("creative_autumn_amber", 3), ("creative_warm_cocoa", 2)],
        ),
        ("ninja", &[("creative_autumn_amber", 3), ("dark_seinen", 2)]),
        (
            "cooking",
            &[("creative_warm_cocoa", 3), ("creative_autumn_amber", 2)],
        ),
        ("food", &[("creative_warm_cocoa", 3)]),
        ("restaurant", &[("creative_warm_cocoa", 3)]),
        (
            "music",
            &[("creative_violet_dusk", 2), ("creative_soft_pastel", 2)],
        ),
        (
            "idol",
            &[("creative_soft_pastel", 3), ("creative_violet_dusk", 2)],
        ),
        (
            "performance",
            &[("creative_violet_dusk", 3), ("creative_soft_pastel", 1)],
        ),
        (
            "sports",
            &[("creative_fresh_sage", 2), ("creative_warm_sunset", 2)],
        ),
        (
            "ghost",
            &[("creative_cool_twilight", 3), ("creative_violet_dusk", 2)],
        ),
        (
            "paranormal",
            &[("creative_violet_dusk", 3), ("creative_cool_twilight", 2)],
        ),
    ]
}

/// Pick the best style for a set of source tags (lowercased). Returns the
/// style file id, or None when nothing matches (caller falls back).
pub fn select_style_for_tags(tags: &[String]) -> Option<&'static str> {
    let mut scores: std::collections::HashMap<&'static str, i32> = std::collections::HashMap::new();
    for tag in tags {
        let tag = tag.to_ascii_lowercase();
        for &(keyword, styles) in mood_table() {
            if tag.contains(keyword) {
                for &(style, weight) in styles {
                    *scores.entry(style).or_insert(0) += weight;
                }
            }
        }
    }
    scores
        .into_iter()
        .max_by_key(|&(style, score)| (score, std::cmp::Reverse(style)))
        .map(|(style, _)| style)
}

/// Pick a style from the page's own statistics (photos / tagless links).
/// Mirrors the scene lighting classifier used by the postprocess pipeline:
/// dark ink-heavy pages get dark styles, bright flat pages get pastels.
pub fn select_style_for_scene(gray: &GrayImage) -> Option<&'static str> {
    let (width, height) = gray.dimensions();
    let total = (width as u64).saturating_mul(height as u64).max(1);
    let mut sum = 0u64;
    let mut dark = 0u64;
    let mut bright = 0u64;
    for pixel in gray.pixels() {
        let v = pixel[0] as u32;
        sum += v as u64;
        if v < 70 {
            dark += 1;
        } else if v > 225 {
            bright += 1;
        }
    }
    let mean = (sum as f64 / total as f64) as f32;
    let dark_ratio = dark as f64 / total as f64;

    if mean < 105.0 && dark_ratio > 0.35 {
        Some("dark_seinen") // night scenes
    } else if mean < 120.0 {
        Some("creative_cool_twilight") // low light
    } else if mean > 170.0 && dark_ratio < 0.10 {
        Some("creative_soft_pastel") // airy, bright
    } else if mean > 170.0 && bright > dark * 3 {
        Some("creative_fresh_sage") // daylight
    } else if mean > 160.0 {
        Some("creative_warm_sunset") // warm daylight
    } else {
        None // neutral midtones: let the rotation decide
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn tag_list(tags: &[&str]) -> Vec<String> {
        tags.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn dark_tags_select_dark_style() {
        let style = select_style_for_tags(&tag_list(&["horror", "dark", "gore"])).unwrap();
        assert_eq!(style, "dark_seinen");
    }

    #[test]
    fn romance_tags_select_pastel() {
        let style = select_style_for_tags(&tag_list(&["romance", "shoujo"])).unwrap();
        assert_eq!(style, "creative_soft_pastel");
    }

    #[test]
    fn fantasy_tags_select_violet_or_teal() {
        let style = select_style_for_tags(&tag_list(&["fantasy", "magic"])).unwrap();
        assert!(style == "creative_violet_dusk" || style == "fantasy_magic");
    }

    #[test]
    fn mixed_tags_aggregate_weights() {
        // fantasy(3 violet) + magic(3 violet) + adventure(2 teal) -> violet
        let style = select_style_for_tags(&tag_list(&["fantasy", "magic", "adventure"])).unwrap();
        assert_eq!(style, "creative_violet_dusk");
    }

    #[test]
    fn no_matching_tags_returns_none() {
        assert!(select_style_for_tags(&tag_list(&["zzz_nonexistent_tag"])).is_none());
    }

    #[test]
    fn dark_page_selects_dark_style() {
        let dark = GrayImage::from_fn(64, 64, |_, _| Luma([40]));
        assert_eq!(select_style_for_scene(&dark), Some("dark_seinen"));
    }

    #[test]
    fn bright_airy_page_selects_pastel() {
        let bright = GrayImage::from_fn(64, 64, |_, _| Luma([240]));
        assert_eq!(
            select_style_for_scene(&bright),
            Some("creative_soft_pastel")
        );
    }

    #[test]
    fn neutral_page_falls_back_to_rotation() {
        let neutral = GrayImage::from_fn(64, 64, |_, _| Luma([140]));
        assert_eq!(select_style_for_scene(&neutral), None);
    }
}
