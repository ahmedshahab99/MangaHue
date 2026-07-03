use anyhow::{Context, Result};
use chrono::Utc;
use mangahue_core::{
    BoundingBox, CharacterProfile, ColorHint, ColorRgb, CorrectionRecord, HintSource, UserHintFile,
};
use serde::{Deserialize, Serialize};
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
    let path = project_dir.join("characters").join(format!("{id}.json"));
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
    let base = base_dir
        .canonicalize()
        .unwrap_or_else(|_| base_dir.to_path_buf());
    let resolved = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
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
    match label {
        "hair" => character.hair_color = Some(record.new_color),
        "skin" => character.skin_color = Some(record.new_color),
        "eye" | "eyes" => character.eye_color = Some(record.new_color),
        "outfit" => {
            if character.outfit_colors.is_empty() {
                character.outfit_colors.push(record.new_color);
            } else {
                character.outfit_colors[0] = record.new_color;
            }
        }
        "accent" => {
            if character.accent_colors.is_empty() {
                character.accent_colors.push(record.new_color);
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
}
