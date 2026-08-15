use anyhow::{Context, Result};
use chrono::Utc;
use mangahue_core::{
    BoundingBox, CharacterProfile, ColorHint, ColorRgb, CorrectionRecord, HintSource, UserHintFile,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectMemory {
    pub id: String,
    pub name: String,
    pub default_style: Option<String>,
    pub reading_direction: String,
    #[serde(default)]
    pub characters: Vec<String>,
    #[serde(default)]
    pub processed_pages: Vec<String>,
    #[serde(default)]
    pub corrections: Vec<CorrectionRecord>,
}

impl ProjectMemory {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            default_style: None,
            reading_direction: "right_to_left".to_string(),
            characters: Vec::new(),
            processed_pages: Vec::new(),
            corrections: Vec::new(),
        }
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .with_context(|| format!("failed to read project memory {}", path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("failed to parse project memory {}", path.display()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(path, serde_json::to_string_pretty(self)?)
            .with_context(|| format!("failed to write project memory {}", path.display()))
    }

    pub fn mark_processed(&mut self, page_id: impl Into<String>) {
        let page_id = page_id.into();
        if !self.processed_pages.contains(&page_id) {
            self.processed_pages.push(page_id);
        }
    }

    pub fn add_correction(&mut self, correction: CorrectionRecord) {
        self.corrections.push(correction);
    }
}

pub fn create_project_directory(base: impl AsRef<Path>, name: &str) -> Result<PathBuf> {
    let base = base.as_ref();
    fs::create_dir_all(base.join("characters"))?;
    fs::create_dir_all(base.join("history"))?;
    fs::create_dir_all(base.join("references"))?;
    fs::create_dir_all(base.join("corrections"))?;
    let project = ProjectMemory::new(name);
    let project_path = base.join("project.json");
    project.save(&project_path)?;
    Ok(project_path)
}

pub fn load_character(path: impl AsRef<Path>) -> Result<CharacterProfile> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read character {}", path.display()))?;
    serde_json::from_str(&text)
        .with_context(|| format!("failed to parse character {}", path.display()))
}

pub fn save_character(path: impl AsRef<Path>, character: &CharacterProfile) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, serde_json::to_string_pretty(character)?)
        .with_context(|| format!("failed to write character {}", path.display()))
}

pub fn create_character(
    project_dir: impl AsRef<Path>,
    name: &str,
    hair_color: Option<ColorRgb>,
    eye_color: Option<ColorRgb>,
    skin_color: Option<ColorRgb>,
) -> Result<PathBuf> {
    let project_dir = project_dir.as_ref();
    let id = slugify(name);
    anyhow::ensure!(
        !id.is_empty(),
        "character name must contain at least one letter or digit"
    );
    let path = project_dir.join("characters").join(format!("{id}.json"));
    if path.exists() {
        // "Hero" and "hero" slugify to the same id; silently overwriting
        // the existing profile loses its memory. Refuse with a clear error.
        let existing = load_character(&path)?;
        anyhow::bail!(
            "a character with id '{id}' already exists ('{}'); choose a different name",
            existing.name
        );
    }
    let character = CharacterProfile {
        id: id.clone(),
        name: name.to_string(),
        aliases: Vec::new(),
        hair_color,
        skin_color,
        eye_color,
        outfit_colors: Vec::new(),
        accent_colors: Vec::new(),
        confidence: 0.85,
        reference_images: Vec::new(),
        notes: Some(format!("Created by MangaHue CLI at {}", Utc::now())),
    };
    save_character(&path, &character)?;

    let project_path = project_dir.join("project.json");
    if project_path.exists() {
        let mut project = ProjectMemory::load(&project_path)?;
        if !project.characters.contains(&id) {
            project.characters.push(id);
        }
        project.save(project_path)?;
    }
    Ok(path)
}

#[derive(Debug, Clone)]
pub struct NewCorrection {
    pub page: String,
    pub region: BoundingBox,
    pub label: Option<String>,
    pub old_color: ColorRgb,
    pub new_color: ColorRgb,
    pub character_id: Option<String>,
}

pub fn add_correction(project_path: impl AsRef<Path>, correction: NewCorrection) -> Result<()> {
    let project_path = project_path.as_ref();
    let mut project = ProjectMemory::load(project_path)?;
    let record = CorrectionRecord {
        page: correction.page,
        region: correction.region,
        label: correction.label.clone(),
        old_color: correction.old_color,
        new_color: correction.new_color,
        character_id: correction.character_id.clone(),
        timestamp: Utc::now(),
    };
    update_character_from_correction(project_path, &record)?;
    project.add_correction(record);
    project.save(project_path)
}

pub fn export_corrections_jsonl(
    project_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<usize> {
    let project = ProjectMemory::load(project_path)?;
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut text = String::new();
    for correction in &project.corrections {
        text.push_str(&serde_json::to_string(correction)?);
        text.push('\n');
    }
    fs::write(output_path, text)
        .with_context(|| format!("failed to write {}", output_path.display()))?;
    Ok(project.corrections.len())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionTrainingRow {
    pub page: String,
    pub lineart: PathBuf,
    pub color: PathBuf,
    pub hints: PathBuf,
    pub correction_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionTrainingManifestSummary {
    pub output: PathBuf,
    pub rows: usize,
    pub pages_with_corrections: usize,
    pub missing_lineart: Vec<String>,
    pub missing_color: Vec<String>,
}

pub fn export_correction_hints(
    project_path: impl AsRef<Path>,
    page: &str,
    output_path: impl AsRef<Path>,
) -> Result<usize> {
    let project = ProjectMemory::load(project_path)?;
    let hints = UserHintFile {
        source_width: None,
        source_height: None,
        hints: project
            .corrections
            .iter()
            .filter(|correction| correction.page == page)
            .map(correction_to_hint)
            .collect(),
    };
    let output_path = output_path.as_ref();
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(output_path, serde_json::to_string_pretty(&hints)?)
        .with_context(|| format!("failed to write {}", output_path.display()))?;
    Ok(hints.hints.len())
}

pub fn export_correction_training_manifest(
    project_path: impl AsRef<Path>,
    lineart_dir: impl AsRef<Path>,
    color_dir: impl AsRef<Path>,
    hints_dir: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<CorrectionTrainingManifestSummary> {
    let project_path = project_path.as_ref().to_path_buf();
    let project = ProjectMemory::load(&project_path)?;
    let lineart_dir = lineart_dir.as_ref();
    let color_dir = color_dir.as_ref();
    let hints_dir = hints_dir.as_ref();
    let output_path = output_path.as_ref();
    fs::create_dir_all(hints_dir)
        .with_context(|| format!("failed to create {}", hints_dir.display()))?;
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let manifest_dir = output_path.parent().unwrap_or_else(|| Path::new("."));

    let mut pages = project
        .corrections
        .iter()
        .map(|correction| correction.page.clone())
        .collect::<Vec<_>>();
    pages.sort();
    pages.dedup();

    let mut rows = Vec::new();
    let mut missing_lineart = Vec::new();
    let mut missing_color = Vec::new();

    for page in &pages {
        let Some(lineart) = find_page_asset(lineart_dir, page) else {
            missing_lineart.push(page.clone());
            continue;
        };
        let Some(color) = find_page_asset(color_dir, page) else {
            missing_color.push(page.clone());
            continue;
        };
        let hints_path = hints_dir.join(format!("{}.hints.json", safe_page_id(page)));
        let correction_count = export_correction_hints(&project_path, page, &hints_path)?;
        rows.push(CorrectionTrainingRow {
            page: page.clone(),
            lineart: manifest_relative_path(manifest_dir, &lineart),
            color: manifest_relative_path(manifest_dir, &color),
            hints: manifest_relative_path(manifest_dir, &hints_path),
            correction_count,
        });
    }

    fs::write(output_path, serde_json::to_string_pretty(&rows)?)
        .with_context(|| format!("failed to write {}", output_path.display()))?;

    Ok(CorrectionTrainingManifestSummary {
        output: output_path.to_path_buf(),
        rows: rows.len(),
        pages_with_corrections: pages.len(),
        missing_lineart,
        missing_color,
    })
}

fn correction_to_hint(correction: &CorrectionRecord) -> ColorHint {
    let x = correction
        .region
        .x
        .saturating_add(correction.region.width / 2);
    let y = correction
        .region
        .y
        .saturating_add(correction.region.height / 2);
    let radius = (correction.region.width.max(correction.region.height) / 2).max(1);
    let label = match (&correction.character_id, &correction.label) {
        (Some(character), Some(label)) => Some(format!("{character}.{label}")),
        (_, label) => label.clone(),
    };
    ColorHint {
        x,
        y,
        radius,
        color: correction.new_color,
        label,
        strength: 1.0,
        source: HintSource::CharacterMemory,
    }
}

fn find_page_asset(dir: &Path, page: &str) -> Option<PathBuf> {
    let page_path = Path::new(page);
    if page_path.is_absolute() && page_path.exists() {
        return Some(page_path.to_path_buf());
    }
    let direct = dir.join(page);
    if direct.exists() {
        return Some(direct);
    }
    for extension in ["png", "jpg", "jpeg", "webp", "tif", "tiff", "bmp"] {
        let candidate = dir.join(format!("{page}.{extension}"));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn safe_page_id(page: &str) -> String {
    page.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn manifest_relative_path(base_dir: &Path, path: &Path) -> PathBuf {
    let base = std::fs::canonicalize(base_dir).unwrap_or_else(|e| {
        tracing::warn!("cannot canonicalize {}: {e}", base_dir.display());
        base_dir.to_path_buf()
    });
    let resolved = std::fs::canonicalize(path).unwrap_or_else(|e| {
        tracing::warn!("cannot canonicalize {}: {e}", path.display());
        path.to_path_buf()
    });
    resolved
        .strip_prefix(&base)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| path.to_path_buf())
}

fn update_character_from_correction(project_path: &Path, record: &CorrectionRecord) -> Result<()> {
    let Some(character_id) = record.character_id.as_deref() else {
        return Ok(());
    };
    let Some(label) = record.label.as_deref() else {
        return Ok(());
    };
    let project_dir = project_path.parent().unwrap_or_else(|| Path::new("."));
    let character_path = project_dir
        .join("characters")
        .join(format!("{character_id}.json"));
    if !character_path.exists() {
        return Ok(());
    }
    let mut character = load_character(&character_path)?;
    match label.to_ascii_lowercase().as_str() {
        "hair" => character.hair_color = Some(record.new_color),
        "skin" => character.skin_color = Some(record.new_color),
        "eye" | "eyes" => character.eye_color = Some(record.new_color),
        "outfit" => {
            // Keep up to four distinct outfits; once the list is full, the
            // nearest existing outfit is replaced. Previously every new
            // correction overwrote outfit[0], so a character with multiple
            // outfits could never learn a second one.
            if character.outfit_colors.len() < 4 {
                if !character.outfit_colors.contains(&record.new_color) {
                    character.outfit_colors.push(record.new_color);
                }
            } else {
                character.outfit_colors[0] = record.new_color;
            }
        }
        "accent" => {
            if character.accent_colors.len() < 4 {
                if !character.accent_colors.contains(&record.new_color) {
                    character.accent_colors.push(record.new_color);
                }
            } else {
                character.accent_colors[0] = record.new_color;
            }
        }
        _ => return Ok(()),
    }
    character.confidence = (character.confidence + 0.04).clamp(0.0, 1.0);
    save_character(character_path, &character)
}

pub fn load_project_characters(
    project_path: impl AsRef<Path>,
    ids: &[String],
) -> Result<Vec<CharacterProfile>> {
    let project_path = project_path.as_ref();
    let project_dir = project_path.parent().unwrap_or_else(|| Path::new("."));
    let project = ProjectMemory::load(project_path)?;
    let wanted: Vec<String> = if ids.is_empty() {
        project.characters
    } else {
        ids.to_vec()
    };
    let mut out = Vec::new();
    for id in wanted {
        let path = project_dir.join("characters").join(format!("{id}.json"));
        if path.exists() {
            out.push(load_character(path)?);
        }
    }
    Ok(out)
}

fn slugify(input: &str) -> String {
    let slug: String = input
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '_'
            }
        })
        .collect();
    slug.trim_matches('_').to_string()
}

pub const COLOR_BIBLE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionScope {
    Region,
    Character,
    Chapter,
    Series,
}

impl CorrectionScope {
    pub fn priority(self) -> u8 {
        match self {
            Self::Region => 4,
            Self::Character => 3,
            Self::Chapter => 2,
            Self::Series => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NormalizedRegion {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl NormalizedRegion {
    pub fn validate(self) -> Result<()> {
        anyhow::ensure!(
            [self.x, self.y, self.width, self.height]
                .into_iter()
                .all(f32::is_finite),
            "normalized region contains a non-finite value"
        );
        anyhow::ensure!(
            (0.0..=1.0).contains(&self.x) && (0.0..=1.0).contains(&self.y),
            "normalized region origin must be within 0..=1"
        );
        anyhow::ensure!(
            self.width > 0.0 && self.height > 0.0,
            "normalized region must have positive dimensions"
        );
        anyhow::ensure!(
            self.x + self.width <= 1.000_001 && self.y + self.height <= 1.000_001,
            "normalized region extends beyond the page"
        );
        Ok(())
    }

    pub fn contains(self, x: f32, y: f32) -> bool {
        x >= self.x && y >= self.y && x <= self.x + self.width && y <= self.y + self.height
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColorSelector {
    pub scope: CorrectionScope,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chapter_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub character_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub object_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semantic: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub region: Option<NormalizedRegion>,
}

impl ColorSelector {
    pub fn validate(&self) -> Result<()> {
        match self.scope {
            CorrectionScope::Region => {
                anyhow::ensure!(
                    self.page_id.as_deref().is_some_and(non_empty)
                        && (self.region_id.as_deref().is_some_and(non_empty)
                            || self.region.is_some()),
                    "region correction requires page_id and region_id or normalized region"
                );
            }
            CorrectionScope::Character => anyhow::ensure!(
                self.character_id.as_deref().is_some_and(non_empty),
                "character correction requires character_id"
            ),
            CorrectionScope::Chapter => anyhow::ensure!(
                self.chapter_id.as_deref().is_some_and(non_empty),
                "chapter correction requires chapter_id"
            ),
            CorrectionScope::Series => {}
        }
        anyhow::ensure!(
            self.semantic.as_deref().is_some_and(non_empty)
                || self.region_id.as_deref().is_some_and(non_empty)
                || self.object_id.as_deref().is_some_and(non_empty)
                || self.location_id.as_deref().is_some_and(non_empty),
            "color selector needs a semantic, region, object, or location identity"
        );
        if let Some(region) = self.region {
            region.validate()?;
        }
        Ok(())
    }

    pub fn stable_key(&self) -> String {
        format!(
            "{:?}|{}|{}|{}|{}|{}|{}|{}",
            self.scope,
            normalized(self.page_id.as_deref()),
            normalized(self.chapter_id.as_deref()),
            normalized(self.character_id.as_deref()),
            normalized(self.location_id.as_deref()),
            normalized(self.object_id.as_deref()),
            normalized(self.region_id.as_deref()),
            normalized(self.semantic.as_deref()),
        )
        .to_ascii_lowercase()
    }

    pub fn matches_context(&self, context: &ColorContext<'_>) -> bool {
        selector_match_score(self, context).is_some()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ColorContext<'a> {
    pub page_id: Option<&'a str>,
    pub chapter_id: Option<&'a str>,
    pub character_id: Option<&'a str>,
    pub location_id: Option<&'a str>,
    pub object_id: Option<&'a str>,
    pub region_id: Option<&'a str>,
    pub semantic: Option<&'a str>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalColorEntry {
    pub id: String,
    pub selector: ColorSelector,
    pub color: ColorRgb,
    pub confidence: f32,
    pub source: String,
    #[serde(default)]
    pub reference_ids: Vec<String>,
    pub updated_at: chrono::DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferenceKind {
    Style,
    Character,
    Location,
    Object,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorBibleReference {
    pub id: String,
    pub kind: ReferenceKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entity_id: Option<String>,
    pub path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    pub approved: bool,
    pub created_at: chrono::DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ColorBibleRevisionAction {
    Apply,
    Undo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorBibleRevision {
    pub revision: u64,
    pub id: String,
    pub action: ColorBibleRevisionAction,
    pub selector_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous: Option<CanonicalColorEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub current: Option<CanonicalColorEntry>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_revision: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub created_at: chrono::DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColorBible {
    pub schema_version: u32,
    pub id: String,
    pub series_id: String,
    pub name: String,
    pub revision: u64,
    pub created_at: chrono::DateTime<Utc>,
    pub updated_at: chrono::DateTime<Utc>,
    #[serde(default)]
    pub colors: BTreeMap<String, CanonicalColorEntry>,
    #[serde(default)]
    pub references: Vec<ColorBibleReference>,
    #[serde(default)]
    pub history: Vec<ColorBibleRevision>,
}

#[derive(Debug, Clone)]
pub struct ColorBibleCorrection {
    pub selector: ColorSelector,
    pub color: ColorRgb,
    pub confidence: f32,
    pub source: String,
    pub reference_ids: Vec<String>,
    pub note: Option<String>,
}

impl ColorBible {
    pub fn new(series_id: impl Into<String>, name: impl Into<String>) -> Result<Self> {
        let series_id = series_id.into();
        anyhow::ensure!(non_empty(&series_id), "series_id cannot be empty");
        let now = Utc::now();
        Ok(Self {
            schema_version: COLOR_BIBLE_SCHEMA_VERSION,
            id: Uuid::new_v4().to_string(),
            series_id,
            name: name.into(),
            revision: 0,
            created_at: now,
            updated_at: now,
            colors: BTreeMap::new(),
            references: Vec::new(),
            history: Vec::new(),
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bible: Self = serde_json::from_slice(
            &fs::read(path)
                .with_context(|| format!("failed to read color bible {}", path.display()))?,
        )
        .with_context(|| format!("failed to parse color bible {}", path.display()))?;
        anyhow::ensure!(
            bible.schema_version == COLOR_BIBLE_SCHEMA_VERSION,
            "unsupported color bible schema {}",
            bible.schema_version
        );
        Ok(bible)
    }

    pub fn load_or_create(
        path: impl AsRef<Path>,
        series_id: impl Into<String>,
        name: impl Into<String>,
    ) -> Result<Self> {
        let path = path.as_ref();
        if path.is_file() {
            Self::load(path)
        } else {
            let bible = Self::new(series_id, name)?;
            bible.save(path)?;
            Ok(bible)
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(self)?)
            .with_context(|| format!("failed to write {}", temporary.display()))?;
        fs::rename(&temporary, path)
            .with_context(|| format!("failed to commit color bible {}", path.display()))
    }

    pub fn apply_correction(&mut self, correction: ColorBibleCorrection) -> Result<u64> {
        correction.selector.validate()?;
        anyhow::ensure!(
            correction.confidence.is_finite(),
            "correction confidence must be finite"
        );
        let selector_key = correction.selector.stable_key();
        let previous = self.colors.get(&selector_key).cloned();
        let now = Utc::now();
        let current = CanonicalColorEntry {
            id: previous
                .as_ref()
                .map(|entry| entry.id.clone())
                .unwrap_or_else(|| Uuid::new_v4().to_string()),
            selector: correction.selector,
            color: correction.color,
            confidence: correction.confidence.clamp(0.0, 1.0),
            source: correction.source,
            reference_ids: correction.reference_ids,
            updated_at: now,
        };
        self.colors.insert(selector_key.clone(), current.clone());
        self.revision = self.revision.saturating_add(1);
        self.updated_at = now;
        self.history.push(ColorBibleRevision {
            revision: self.revision,
            id: Uuid::new_v4().to_string(),
            action: ColorBibleRevisionAction::Apply,
            selector_key,
            previous,
            current: Some(current),
            target_revision: None,
            note: correction.note,
            created_at: now,
        });
        Ok(self.revision)
    }

    pub fn undo_last(&mut self) -> Result<u64> {
        let undone = self
            .history
            .iter()
            .filter_map(|revision| {
                (revision.action == ColorBibleRevisionAction::Undo)
                    .then_some(revision.target_revision)
                    .flatten()
            })
            .collect::<BTreeSet<_>>();
        let target = self
            .history
            .iter()
            .rev()
            .find(|revision| {
                revision.action == ColorBibleRevisionAction::Apply
                    && !undone.contains(&revision.revision)
            })
            .cloned()
            .context("color bible has no active correction to undo")?;
        let before_undo = self.colors.get(&target.selector_key).cloned();
        if let Some(previous) = target.previous.clone() {
            self.colors.insert(target.selector_key.clone(), previous);
        } else {
            self.colors.remove(&target.selector_key);
        }
        let now = Utc::now();
        self.revision = self.revision.saturating_add(1);
        self.updated_at = now;
        self.history.push(ColorBibleRevision {
            revision: self.revision,
            id: Uuid::new_v4().to_string(),
            action: ColorBibleRevisionAction::Undo,
            selector_key: target.selector_key,
            previous: before_undo,
            current: target.previous,
            target_revision: Some(target.revision),
            note: Some(format!("undo revision {}", target.revision)),
            created_at: now,
        });
        Ok(self.revision)
    }

    pub fn resolve_color(&self, context: &ColorContext<'_>) -> Option<&CanonicalColorEntry> {
        self.colors
            .values()
            .filter_map(|entry| {
                selector_match_score(&entry.selector, context).map(|score| (entry, score))
            })
            .max_by(|(left, left_score), (right, right_score)| {
                left_score
                    .cmp(right_score)
                    .then_with(|| left.confidence.total_cmp(&right.confidence))
                    .then_with(|| left.updated_at.cmp(&right.updated_at))
            })
            .map(|(entry, _)| entry)
    }

    pub fn add_reference(&mut self, reference: ColorBibleReference) -> Result<()> {
        anyhow::ensure!(
            !reference.path.as_os_str().is_empty(),
            "reference path is empty"
        );
        if let Some(existing) = self
            .references
            .iter_mut()
            .find(|existing| existing.id == reference.id)
        {
            *existing = reference;
        } else {
            self.references.push(reference);
        }
        self.updated_at = Utc::now();
        Ok(())
    }
}

fn selector_match_score(selector: &ColorSelector, context: &ColorContext<'_>) -> Option<u16> {
    let matches = |expected: Option<&str>, actual: Option<&str>| {
        expected.is_none_or(|value| actual.is_some_and(|other| value.eq_ignore_ascii_case(other)))
    };
    if !matches(selector.page_id.as_deref(), context.page_id)
        || !matches(selector.chapter_id.as_deref(), context.chapter_id)
        || !matches(selector.character_id.as_deref(), context.character_id)
        || !matches(selector.location_id.as_deref(), context.location_id)
        || !matches(selector.object_id.as_deref(), context.object_id)
        || !matches(selector.region_id.as_deref(), context.region_id)
        || !matches(selector.semantic.as_deref(), context.semantic)
    {
        return None;
    }
    match selector.scope {
        CorrectionScope::Region
            if selector.page_id.as_deref().is_none_or(|value| {
                context
                    .page_id
                    .is_some_and(|other| value.eq_ignore_ascii_case(other))
            }) && selector.region_id.as_deref().is_none_or(|value| {
                context
                    .region_id
                    .is_some_and(|other| value.eq_ignore_ascii_case(other))
            }) => {}
        CorrectionScope::Character
            if selector.character_id.as_deref().is_some_and(|value| {
                context
                    .character_id
                    .is_some_and(|other| value.eq_ignore_ascii_case(other))
            }) => {}
        CorrectionScope::Chapter
            if selector.chapter_id.as_deref().is_some_and(|value| {
                context
                    .chapter_id
                    .is_some_and(|other| value.eq_ignore_ascii_case(other))
            }) => {}
        CorrectionScope::Series => {}
        _ => return None,
    }
    let qualifiers = [
        selector.page_id.is_some(),
        selector.chapter_id.is_some(),
        selector.character_id.is_some(),
        selector.location_id.is_some(),
        selector.object_id.is_some(),
        selector.region_id.is_some(),
        selector.semantic.is_some(),
    ]
    .into_iter()
    .filter(|present| *present)
    .count() as u16;
    Some(selector.scope.priority() as u16 * 100 + qualifiers)
}

fn normalized(value: Option<&str>) -> &str {
    value.map(str::trim).unwrap_or("")
}

fn non_empty(value: &str) -> bool {
    !value.trim().is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn correction_updates_character_and_exports_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let project_path = create_project_directory(dir.path(), "Memory Test").unwrap();
        create_character(
            dir.path(),
            "Hero",
            Some(ColorRgb::new(90, 70, 50)),
            None,
            None,
        )
        .unwrap();

        add_correction(
            &project_path,
            NewCorrection {
                page: "page_001".to_string(),
                region: BoundingBox::new(1, 2, 3, 4),
                label: Some("hair".to_string()),
                old_color: ColorRgb::new(90, 70, 50),
                new_color: ColorRgb::new(17, 17, 17),
                character_id: Some("hero".to_string()),
            },
        )
        .unwrap();

        let project = ProjectMemory::load(&project_path).unwrap();
        assert_eq!(project.corrections.len(), 1);

        let character = load_character(dir.path().join("characters").join("hero.json")).unwrap();
        assert_eq!(character.hair_color, Some(ColorRgb::new(17, 17, 17)));

        let export_path = dir.path().join("corrections.jsonl");
        let count = export_corrections_jsonl(&project_path, &export_path).unwrap();
        assert_eq!(count, 1);
        assert!(std::fs::read_to_string(export_path)
            .unwrap()
            .contains("page_001"));
    }

    #[test]
    fn corrections_export_as_hints_and_training_manifest() {
        let dir = tempfile::tempdir().unwrap();
        let project_path = create_project_directory(dir.path(), "Training Bridge").unwrap();
        add_correction(
            &project_path,
            NewCorrection {
                page: "page_001".to_string(),
                region: BoundingBox::new(10, 20, 30, 40),
                label: Some("hair".to_string()),
                old_color: ColorRgb::new(90, 70, 50),
                new_color: ColorRgb::new(17, 17, 17),
                character_id: Some("hero".to_string()),
            },
        )
        .unwrap();

        let hints_path = dir.path().join("page_001.hints.json");
        let count = export_correction_hints(&project_path, "page_001", &hints_path).unwrap();
        assert_eq!(count, 1);
        let hints: UserHintFile =
            serde_json::from_str(&std::fs::read_to_string(&hints_path).unwrap()).unwrap();
        assert_eq!(hints.hints[0].x, 25);
        assert_eq!(hints.hints[0].y, 40);
        assert_eq!(hints.hints[0].radius, 20);
        assert_eq!(hints.hints[0].label.as_deref(), Some("hero.hair"));
        assert_eq!(hints.hints[0].source, HintSource::CharacterMemory);

        let lineart_dir = dir.path().join("lineart");
        let color_dir = dir.path().join("color");
        std::fs::create_dir_all(&lineart_dir).unwrap();
        std::fs::create_dir_all(&color_dir).unwrap();
        std::fs::write(lineart_dir.join("page_001.png"), []).unwrap();
        std::fs::write(color_dir.join("page_001.png"), []).unwrap();

        let manifest_path = dir.path().join("manifest.json");
        let summary = export_correction_training_manifest(
            &project_path,
            &lineart_dir,
            &color_dir,
            dir.path().join("hints"),
            &manifest_path,
        )
        .unwrap();
        assert_eq!(summary.rows, 1);
        assert!(summary.missing_lineart.is_empty());
        assert!(summary.missing_color.is_empty());
        let rows: Vec<CorrectionTrainingRow> =
            serde_json::from_str(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
        assert_eq!(rows[0].page, "page_001");
        assert_eq!(
            rows[0].lineart,
            PathBuf::from("lineart").join("page_001.png")
        );
        assert_eq!(rows[0].color, PathBuf::from("color").join("page_001.png"));
        assert_eq!(
            rows[0].hints,
            PathBuf::from("hints").join("page_001.hints.json")
        );
        assert_eq!(rows[0].correction_count, 1);
    }

    fn correction(
        scope: CorrectionScope,
        page: Option<&str>,
        chapter: Option<&str>,
        character: Option<&str>,
        region: Option<&str>,
        semantic: &str,
        color: ColorRgb,
    ) -> ColorBibleCorrection {
        ColorBibleCorrection {
            selector: ColorSelector {
                scope,
                page_id: page.map(str::to_string),
                chapter_id: chapter.map(str::to_string),
                character_id: character.map(str::to_string),
                location_id: None,
                object_id: None,
                region_id: region.map(str::to_string),
                semantic: Some(semantic.to_string()),
                region: None,
            },
            color,
            confidence: 0.95,
            source: "test".to_string(),
            reference_ids: Vec::new(),
            note: None,
        }
    }

    #[test]
    fn color_bible_resolves_most_specific_scope_and_undo_restores_previous() {
        let mut bible = ColorBible::new("series-a", "Series A").unwrap();
        bible
            .apply_correction(correction(
                CorrectionScope::Series,
                None,
                None,
                None,
                None,
                "hair",
                ColorRgb::new(20, 20, 30),
            ))
            .unwrap();
        bible
            .apply_correction(correction(
                CorrectionScope::Character,
                None,
                None,
                Some("hero"),
                None,
                "hair",
                ColorRgb::new(180, 40, 30),
            ))
            .unwrap();
        let context = ColorContext {
            page_id: Some("page-1"),
            chapter_id: Some("chapter-1"),
            character_id: Some("hero"),
            semantic: Some("hair"),
            ..ColorContext::default()
        };
        assert_eq!(
            bible.resolve_color(&context).unwrap().color,
            ColorRgb::new(180, 40, 30)
        );

        bible.undo_last().unwrap();
        assert_eq!(
            bible.resolve_color(&context).unwrap().color,
            ColorRgb::new(20, 20, 30)
        );
        assert_eq!(bible.history.len(), 3);
    }

    #[test]
    fn color_bible_round_trips_atomically() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("series").join("color-bible.json");
        let mut bible = ColorBible::new("series-b", "Series B").unwrap();
        bible
            .apply_correction(correction(
                CorrectionScope::Region,
                Some("page-1"),
                Some("chapter-1"),
                None,
                Some("region-7"),
                "clothing",
                ColorRgb::new(20, 90, 180),
            ))
            .unwrap();
        bible.save(&path).unwrap();
        let loaded = ColorBible::load(&path).unwrap();
        assert_eq!(loaded.series_id, "series-b");
        assert_eq!(loaded.revision, 1);
        assert_eq!(loaded.colors.len(), 1);
        assert!(!path.with_extension("json.tmp").exists());
    }
}
