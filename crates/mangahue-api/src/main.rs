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
    #[arg(long, default_value = "web/studio")]
    studio_dir: PathBuf,
    #[arg(long, default_value = ".mangahue/production.db")]
    production_db: PathBuf,
    #[arg(
        long,
        help = "Expose only authenticated v1 routes; disable local file, Studio, and operator APIs"
    )]
    public_api: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("mangahue=info,tower_http=info")
        .init();
    let args = Args::parse();
    serve(ApiServerConfig {
        addr: args.addr,
        jobs_dir: args.jobs_dir,
        styles_dir: args.styles_dir,
        studio_dir: args.studio_dir,
        production_db: args.production_db,
        public_api: args.public_api,
    })
    .await
}
