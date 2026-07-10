use serde::{Deserialize, Serialize};
use sysinfo::System;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformProfile {
    pub cpu_cores: usize,
    pub total_ram_gb: f64,
    pub optimal_tile_size: u32,
    pub optimal_batch_size: usize,
    pub optimal_parallel_jobs: usize,
    pub fp16_supported: bool,
    pub ensemble_viable: bool,
    pub backend_priority: Vec<BackendKind>,
    pub gpu_detected: bool,
    pub gpu_name: Option<String>,
    pub gpu_vram_mb: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    NativeOnnxGpu,
    NativeOnnxCpu,
    PythonOnnx,
    Ensemble,
    Mock,
}

impl PlatformProfile {
    pub fn detect() -> Self {
        let mut system = System::new_all();
        system.refresh_all();

        let cpu_cores = system.cpus().len().max(1);
        let total_ram_bytes = system.total_memory();
        let total_ram_gb = total_ram_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

        let gpu_detected = false;
        let gpu_name = None;
        let gpu_vram_mb = None;

        let optimal_tile_size = if total_ram_gb >= 16.0 {
            1024
        } else if total_ram_gb >= 8.0 {
            896
        } else if total_ram_gb >= 4.0 {
            768
        } else {
            512
        };

        let optimal_batch_size = if total_ram_gb >= 32.0 {
            4
        } else if total_ram_gb >= 8.0 {
            2
        } else {
            1
        };

        let optimal_parallel_jobs = if cpu_cores >= 16 {
            8
        } else if cpu_cores >= 8 {
            4
        } else if cpu_cores >= 4 {
            2
        } else {
            1
        };

        let fp16_supported = gpu_detected;

        let ensemble_viable = cpu_cores >= 8 && total_ram_gb >= 8.0;

        let mut backend_priority = Vec::new();
        if gpu_detected {
            backend_priority.push(BackendKind::NativeOnnxGpu);
        }
        backend_priority.push(BackendKind::NativeOnnxCpu);
        backend_priority.push(BackendKind::PythonOnnx);
        if ensemble_viable {
            backend_priority.push(BackendKind::Ensemble);
        }
        backend_priority.push(BackendKind::Mock);

        let profile = Self {
            cpu_cores,
            total_ram_gb: (total_ram_gb * 100.0).round() / 100.0,
            optimal_tile_size,
            optimal_batch_size,
            optimal_parallel_jobs,
            fp16_supported,
            ensemble_viable,
            backend_priority,
            gpu_detected,
            gpu_name,
            gpu_vram_mb,
        };

        info!(
            cpu = cpu_cores,
            ram_gb = %profile.total_ram_gb,
            tile = optimal_tile_size,
            batch = optimal_batch_size,
            workers = optimal_parallel_jobs,
            "platform profile detected"
        );

        profile
    }

    /// Applies platform-optimal settings to an `EngineConfig` in place.
    ///
    /// Overrides tile size, batch size, parallelism (only when unset), fp16,
    /// and device selection. Intended for the `mangahue platform --apply-to`
    /// CLI flow so users can stamp a config file with hardware-tuned values
    /// without hand-editing YAML.
    pub fn apply_to_config(&self, config: &mut mangahue_config::EngineConfig) {
        config.layout.tile_size = self.optimal_tile_size;
        config.inference.batch_size = self.optimal_batch_size;
        if config.engine.parallel_jobs == 0 {
            config.engine.parallel_jobs = self.optimal_parallel_jobs;
        }
        config.inference.fp16_enabled = config.inference.fp16_enabled || self.fp16_supported;
        config.inference.device = if self.gpu_detected {
            "cuda".to_string()
        } else {
            "cpu".to_string()
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_platform_without_panic() {
        let profile = PlatformProfile::detect();
        assert!(profile.cpu_cores > 0);
        assert!(profile.total_ram_gb > 0.0);
        assert!(profile.optimal_tile_size >= 512);
        assert!(profile.optimal_batch_size >= 1);
        assert!(profile.optimal_parallel_jobs >= 1);
        assert!(!profile.backend_priority.is_empty());
    }
}
