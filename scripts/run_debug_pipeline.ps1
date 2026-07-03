param(
  [string]$Input = "examples\input\smoke_page.png",
  [string]$Output = "examples\output\debug_pipeline.png",
  [string]$DebugDir = "examples\output\debug_pipeline"
)

cargo run -p mangahue-cli -- colorize $Input --output $Output --mock --debug $DebugDir

