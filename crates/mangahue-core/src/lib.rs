use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};
use std::path::PathBuf;
use std::str::FromStr;
use thiserror::Error;

pub type EngineResult<T> = Result<T, EngineError>;

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("config error: {0}")]
    Config(String),
    #[error("image error: {0}")]
    Image(String),
    #[error("layout error: {0}")]
    Layout(String),
    #[error("hint error: {0}")]
    Hint(String),
    #[error("memory error: {0}")]
    Memory(String),
    #[error("inference error: {0}")]
    Inference(String),
    #[error("postprocess error: {0}")]
    Postprocess(String),
    #[error("unsupported format: {0}")]
    UnsupportedFormat(String),
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),
    #[error("invalid model contract: {0}")]
    InvalidModelContract(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ColorRgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl ColorRgb {
    pub const BLACK: Self = Self { r: 0, g: 0, b: 0 };
    pub const WHITE: Self = Self {
        r: 255,
        g: 255,
        b: 255,
    };

    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub fn to_hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.r, self.g, self.b)
    }

    pub fn distance(self, other: Self) -> f32 {
        let dr = self.r as f32 - other.r as f32;
        let dg = self.g as f32 - other.g as f32;
        let db = self.b as f32 - other.b as f32;
        (dr * dr + dg * dg + db * db).sqrt()
    }

    pub fn mix(self, other: Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        Self {
            r: lerp_u8(self.r, other.r, t),
            g: lerp_u8(self.g, other.g, t),
            b: lerp_u8(self.b, other.b, t),
        }
    }
}

impl Display for ColorRgb {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl Serialize for ColorRgb {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for ColorRgb {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum ColorRepr {
            Hex(String),
            Channels { r: u8, g: u8, b: u8 },
        }

        match ColorRepr::deserialize(deserializer)? {
            ColorRepr::Hex(value) => ColorRgb::from_str(&value).map_err(serde::de::Error::custom),
            ColorRepr::Channels { r, g, b } => Ok(ColorRgb::new(r, g, b)),
        }
    }
}

impl FromStr for ColorRgb {
    type Err = EngineError;

    fn from_str(input: &str) -> Result<Self, Self::Err> {
        let hex = input.trim().trim_start_matches('#');
        if hex.len() != 6 {
            return Err(EngineError::Config(format!(
                "expected #rrggbb color, got {input}"
            )));
        }
        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|_| EngineError::Config(format!("invalid red channel in {input}")))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|_| EngineError::Config(format!("invalid green channel in {input}")))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|_| EngineError::Config(format!("invalid blue channel in {input}")))?;
        Ok(Self { r, g, b })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ColorLinear {
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

impl ImageSize {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }

    pub fn aspect_ratio(self) -> f32 {
        if self.height == 0 {
            0.0
        } else {
            self.width as f32 / self.height as f32
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl BoundingBox {
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn right(self) -> u32 {
        self.x.saturating_add(self.width)
    }

    pub fn bottom(self) -> u32 {
        self.y.saturating_add(self.height)
    }

    pub fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }

    pub fn contains(self, x: u32, y: u32) -> bool {
        x >= self.x && y >= self.y && x < self.right() && y < self.bottom()
    }

    pub fn clamp_to(self, size: ImageSize) -> Self {
        let x = self.x.min(size.width);
        let y = self.y.min(size.height);
        let right = self.right().min(size.width);
        let bottom = self.bottom().min(size.height);
        Self {
            x,
            y,
            width: right.saturating_sub(x),
            height: bottom.saturating_sub(y),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Webp,
    Tiff,
    Bmp,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MangaPageInfo {
    pub id: String,
    pub path: PathBuf,
    pub size: ImageSize,
    pub aspect_ratio: f32,
    pub format: ImageFormat,
    pub is_webtoon_like: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MangaPanel {
    pub id: String,
    pub bbox: BoundingBox,
    pub reading_order: usize,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileInfo {
    pub id: String,
    pub bbox: BoundingBox,
    pub overlap: u32,
    pub panel_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaskKind {
    LineInk,
    SpeechBubble,
    Text,
    ProtectedWhite,
    UserHint,
    CharacterHint,
    Background,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HintSource {
    User,
    CharacterMemory,
    StyleProfile,
    PaletteRule,
    ScenePreset,
    ReferenceImage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleProfile {
    pub id: String,
    pub name: String,
    pub saturation: f32,
    pub contrast: f32,
    pub brightness: f32,
    pub shadow_strength: f32,
    pub highlight_strength: f32,
    pub line_strength: f32,
    pub palette: Vec<ColorRgb>,
    pub protected_colors: Vec<ColorRgb>,
    pub notes: Option<String>,
}

impl Default for StyleProfile {
    fn default() -> Self {
        Self {
            id: "anime_cel".to_string(),
            name: "Anime Cel".to_string(),
            saturation: 1.08,
            contrast: 1.04,
            brightness: 0.0,
            shadow_strength: 0.45,
            highlight_strength: 0.35,
            line_strength: 1.0,
            palette: vec![
                ColorRgb::new(242, 179, 143),
                ColorRgb::new(48, 55, 72),
                ColorRgb::new(66, 103, 214),
            ],
            protected_colors: vec![ColorRgb::WHITE],
            notes: Some("Default clean anime color profile.".to_string()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CharacterProfile {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub hair_color: Option<ColorRgb>,
    pub skin_color: Option<ColorRgb>,
    pub eye_color: Option<ColorRgb>,
    #[serde(default)]
    pub outfit_colors: Vec<ColorRgb>,
    #[serde(default)]
    pub accent_colors: Vec<ColorRgb>,
    pub confidence: f32,
    #[serde(default)]
    pub reference_images: Vec<PathBuf>,
    pub notes: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorHint {
    pub x: u32,
    pub y: u32,
    pub radius: u32,
    pub color: ColorRgb,
    pub label: Option<String>,
    pub strength: f32,
    #[serde(default = "default_hint_source")]
    pub source: HintSource,
}

fn default_hint_source() -> HintSource {
    HintSource::User
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UserHintFile {
    #[serde(default)]
    pub hints: Vec<ColorHint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionRecord {
    pub page: String,
    pub region: BoundingBox,
    pub label: Option<String>,
    pub old_color: ColorRgb,
    pub new_color: ColorRgb,
    pub character_id: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProcessingReport {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub model_path: Option<PathBuf>,
    pub style_id: Option<String>,
    #[serde(default)]
    pub character_ids: Vec<String>,
    pub preprocessing_ms: u128,
    pub layout_ms: u128,
    pub hint_ms: u128,
    pub inference_ms: u128,
    pub postprocess_ms: u128,
    pub total_ms: u128,
    pub tiles_count: usize,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub debug_files: Vec<PathBuf>,
    pub page_classification: Option<String>,
    pub director_verdict: Option<String>,
    #[serde(default)]
    pub stage_reports: Vec<StageReport>,
}

impl ProcessingReport {
    pub fn warn(&mut self, warning: impl Into<String>) {
        self.warnings.push(warning.into());
    }

    pub fn add_stage_report(&mut self, stage: StageReport) {
        self.stage_reports.push(stage);
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageReport {
    pub name: String,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub notes: Vec<String>,
}

impl StageReport {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            metrics: BTreeMap::new(),
            notes: Vec::new(),
        }
    }

    pub fn metric(mut self, name: impl Into<String>, value: impl Into<f64>) -> Self {
        self.metrics.insert(name.into(), value.into());
        self
    }

    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t)
        .round()
        .clamp(0.0, 255.0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_hex_colors() {
        let color = ColorRgb::from_str("#e8b89a").unwrap();
        assert_eq!(color, ColorRgb::new(232, 184, 154));
        assert_eq!(color.to_hex(), "#e8b89a");
    }

    #[test]
    fn clamps_boxes_to_image_bounds() {
        let bbox = BoundingBox::new(80, 90, 50, 40).clamp_to(ImageSize::new(100, 100));
        assert_eq!(bbox, BoundingBox::new(80, 90, 20, 10));
    }
}
