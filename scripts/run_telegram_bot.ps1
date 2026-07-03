param(
    [string]$Token = $env:TELEGRAM_BOT_TOKEN,
    [string]$WorkDir = ".mangahue/telegram",
    [int]$MaxPages = 4,
    [switch]$Mock,
    [string]$Model = "",
    [string]$Config = "",
    [string]$Style = ""
)

if (-not $Token) {
    throw "Set TELEGRAM_BOT_TOKEN or pass -Token."
}

$env:TELEGRAM_BOT_TOKEN = $Token

$argsList = @(
    "run", "-p", "mangahue-telegram", "--",
    "--work-dir", $WorkDir,
    "--max-pages", "$MaxPages"
)

if ($Mock) { $argsList += "--mock" }
if ($Model) { $argsList += @("--model", $Model) }
if ($Config) { $argsList += @("--config", $Config) }
if ($Style) { $argsList += @("--style", $Style) }

cargo @argsList
