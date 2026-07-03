use anyhow::{Context, Result};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use mangahue_core::ImageSize;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    SourcePage,
    ImportedPage,
    PreprocessMask,
    HintMap,
    ModelTile,
    ColoredPage,
    QualityReport,
    Gallery,
    Archive,
    Manifest,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey(pub String);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedArtifact {
    pub key: CacheKey,
    pub kind: ArtifactKind,
    pub path: PathBuf,
    pub bytes: u64,
    pub media_type: Option<String>,
    pub image_size: Option<ImageSize>,
    pub source_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CacheManifest {
    pub artifacts: Vec<CachedArtifact>,
}

#[derive(Debug, Clone)]
pub struct ArtifactCache {
    root: PathBuf,
    manifest_path: PathBuf,
}

impl ArtifactCache {
    pub fn new(root: impl AsRef<Path>) -> Self {
        let root = root.as_ref().to_path_buf();
        Self {
            manifest_path: root.join("manifest.json"),
            root,
        }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn load_manifest(&self) -> Result<CacheManifest> {
        if !self.manifest_path.exists() {
            return Ok(CacheManifest::default());
        }
        let text = fs::read_to_string(&self.manifest_path)
            .with_context(|| format!("failed to read {}", self.manifest_path.display()))?;
        serde_json::from_str(&text)
            .with_context(|| format!("failed to parse {}", self.manifest_path.display()))
    }

    pub fn put_file(
        &self,
        source: impl AsRef<Path>,
        kind: ArtifactKind,
        media_type: Option<String>,
    ) -> Result<CachedArtifact> {
        let source = source.as_ref();
        let mut file = fs::File::open(source)
            .with_context(|| format!("failed to open {}", source.display()))?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .with_context(|| format!("failed to read {}", source.display()))?;
        let image_size = image_dimensions(source);
        let mut artifact = self.put_bytes(&data, kind, extension_for(source), media_type)?;
        artifact.source_path = Some(source.to_path_buf());
        artifact.image_size = image_size;
        self.upsert_manifest(artifact.clone())?;
        Ok(artifact)
    }

    pub fn put_bytes(
        &self,
        bytes: &[u8],
        kind: ArtifactKind,
        extension: Option<&str>,
        media_type: Option<String>,
    ) -> Result<CachedArtifact> {
        fs::create_dir_all(&self.root)
            .with_context(|| format!("failed to create {}", self.root.display()))?;
        let key = CacheKey::from_bytes(bytes);
        let ext = extension.unwrap_or("bin").trim_start_matches('.');
        let path = self.root.join(format!("{}.{}", key.0, ext));
        if !path.exists() {
            fs::write(&path, bytes)
                .with_context(|| format!("failed to write {}", path.display()))?;
        }
        let artifact = CachedArtifact {
            key,
            kind,
            path,
            bytes: bytes.len() as u64,
            media_type,
            image_size: None,
            source_path: None,
        };
        self.upsert_manifest(artifact.clone())?;
        Ok(artifact)
    }

    pub fn get(&self, key: &CacheKey) -> Result<Option<CachedArtifact>> {
        Ok(self
            .load_manifest()?
            .artifacts
            .into_iter()
            .find(|artifact| &artifact.key == key && artifact.path.exists()))
    }

    pub fn contains(&self, key: &CacheKey) -> Result<bool> {
        Ok(self.get(key)?.is_some())
    }

    fn upsert_manifest(&self, artifact: CachedArtifact) -> Result<()> {
        let mut manifest = self.load_manifest()?;
        manifest.artifacts.retain(|item| item.key != artifact.key);
        manifest.artifacts.push(artifact);
        manifest
            .artifacts
            .sort_by(|a, b| a.path.to_string_lossy().cmp(&b.path.to_string_lossy()));
        if let Some(parent) = self.manifest_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        fs::write(
            &self.manifest_path,
            serde_json::to_string_pretty(&manifest)?,
        )
        .with_context(|| format!("failed to write {}", self.manifest_path.display()))
    }
}

impl CacheKey {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let digest = Sha256::digest(bytes);
        Self(URL_SAFE_NO_PAD.encode(digest))
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        Ok(Self::from_bytes(&bytes))
    }
}

fn extension_for(path: &Path) -> Option<&str> {
    path.extension().and_then(|value| value.to_str())
}

fn image_dimensions(path: &Path) -> Option<ImageSize> {
    image::image_dimensions(path)
        .ok()
        .map(|(width, height)| ImageSize::new(width, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_artifacts_by_content_hash() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ArtifactCache::new(dir.path());
        let first = cache
            .put_bytes(b"same", ArtifactKind::Manifest, Some("json"), None)
            .unwrap();
        let second = cache
            .put_bytes(b"same", ArtifactKind::Manifest, Some("json"), None)
            .unwrap();
        assert_eq!(first.key, second.key);
        assert!(cache.contains(&first.key).unwrap());
        assert_eq!(cache.load_manifest().unwrap().artifacts.len(), 1);
    }

    #[test]
    fn stores_source_file_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("page.png");
        let image = image::RgbImage::from_pixel(4, 5, image::Rgb([1, 2, 3]));
        image.save(&source).unwrap();

        let cache = ArtifactCache::new(dir.path().join("cache"));
        let artifact = cache
            .put_file(
                &source,
                ArtifactKind::SourcePage,
                Some("image/png".to_string()),
            )
            .unwrap();
        assert_eq!(artifact.image_size, Some(ImageSize::new(4, 5)));
        assert_eq!(artifact.media_type.as_deref(), Some("image/png"));
    }
}
