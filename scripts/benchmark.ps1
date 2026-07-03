param(
  [string]$InputDir = "examples\input",
  [string]$OutputDir = "examples\output\benchmark",
  [string]$Report = "examples\output\benchmark.json"
)

cargo run -p mangahue-cli -- benchmark $InputDir --output $OutputDir --mock --report $Report

