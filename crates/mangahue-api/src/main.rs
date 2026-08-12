use anyhow::Result;
use clap::Parser;
use mangahue_api::{serve, ApiServerConfig};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "mangahue-api", about = "Run the MangaHue Engine HTTP API")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:7878")]
    addr: SocketAddr,
    #[arg(long, default_value = ".mangahue/jobs")]
    jobs_dir: PathBuf,
    #[arg(long, default_value = "configs/styles")]
    styles_dir: PathBuf,
    #[arg(long, default_value = ".mangahue/production.db")]
    production_db: PathBuf,
    #[arg(
        long,
        help = "Expose only authenticated v1 routes; disable local file, Studio, and operator APIs"
    )]
    public_api: bool,
    #[arg(
        long,
        help = "Root directory served by /api/file (default: current dir, or jobs dir in public mode)"
    )]
    file_root: Option<PathBuf>,
    #[arg(
        long,
        help = "Additional CORS origin allowed to call the API (repeatable)"
    )]
    cors_origin: Vec<String>,
    #[arg(long, default_value_t = 1, env = "MANGAHUE_MAX_CONCURRENT_JOBS")]
    max_concurrent_jobs: usize,
    #[arg(long, default_value_t = 168, env = "MANGAHUE_JOB_RETENTION_HOURS")]
    job_retention_hours: u64,
    #[arg(
        long,
        default_value = "models/ddcolor-tiny-fp16.onnx",
        env = "MANGAHUE_DEFAULT_MODEL"
    )]
    default_model: PathBuf,
    #[arg(
        long,
        default_value = "configs/engine.default.yaml",
        env = "MANGAHUE_DEFAULT_CONFIG"
    )]
    default_config: PathBuf,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("mangahue=info,tower_http=info")
        .init();
    let args = Args::parse();
    let file_root = args.file_root.unwrap_or_else(|| {
        if args.public_api {
            args.jobs_dir.clone()
        } else {
            PathBuf::from(".")
        }
    });
    serve(ApiServerConfig {
        addr: args.addr,
        jobs_dir: args.jobs_dir,
        styles_dir: args.styles_dir,
        production_db: args.production_db,
        public_api: args.public_api,
        file_root,
        allowed_origins: args.cors_origin,
        max_concurrent_jobs: args.max_concurrent_jobs,
        job_retention_hours: args.job_retention_hours,
        default_model_path: args.default_model,
        default_config_path: args.default_config,
    })
    .await
}
