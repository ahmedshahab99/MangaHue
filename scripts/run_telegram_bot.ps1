param(
    [string]$Token = $env:TELEGRAM_BOT_TOKEN,
    [string]$WorkDir = ".mangahue/telegram",
    [int]$MaxPages = 32,
    [int]$MaxConcurrentJobs = 2,
    [switch]$Mock,
    [switch]$Pyramid,
    [switch]$HintPropagate,
    [switch]$FailFast,
    [string]$Model = "",
    [string]$Config = "",
    [string]$Style = "",
    [string]$RegionLocks = ""
)

if (-not $Token) {
    throw "Set TELEGRAM_BOT_TOKEN or pass -Token."
}

$env:TELEGRAM_BOT_TOKEN = $Token

$argsList = @(
    "run", "-p", "mangahue-telegram", "--",
    "--work-dir", $WorkDir,
    "--max-pages", "$MaxPages",
    "--max-concurrent-jobs", "$MaxConcurrentJobs"
)

if ($Mock) { $argsList += "--mock" }
if ($Pyramid) { $argsList += "--pyramid" }
if ($HintPropagate) { $argsList += "--hint-propagate" }
if ($FailFast) { $argsList += "--fail-fast" }
if ($Model) { $argsList += @("--model", $Model) }
if ($Config) { $argsList += @("--config", $Config) }
if ($Style) { $argsList += @("--style", $Style) }
if ($RegionLocks) { $argsList += @("--region-locks", $RegionLocks) }

cargo @argsList
