param(
  [string]$OutDir = "dist"
)

cargo fmt --all
cargo test
cargo clippy --all-targets -- -D warnings
cargo build --release -p mangahue-cli -p mangahue-api
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
Copy-Item target\release\mangahue.exe $OutDir -Force
Copy-Item target\release\mangahue-api.exe $OutDir -Force
Copy-Item configs $OutDir -Recurse -Force
Copy-Item docs $OutDir -Recurse -Force
Copy-Item models\README.md $OutDir -Force

