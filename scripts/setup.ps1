<# 
  Setup script for Personal RAG (Propositional) — PowerShell 5.1
  - Creates folders
  - Provisions .venv without requiring activation
  - Installs requirements (or a safe default set)
  - Generates cross-shell launch helpers (PS/CMD/Bash)

  Tested on Windows PowerShell 5.1
#>

[CmdletBinding()]
param(
  [string]$ProjectRoot = (Get-Location).Path,
  [string]$VenvDir     = ".venv",
  [string]$ReqFile     = "requirements.txt",
  [switch]$Force
)

$ErrorActionPreference = 'Stop'

function Write-Info($msg)  { Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Warn($msg)  { Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Err ($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# 0) mkdir structure
$dirs = @(
  (Join-Path $ProjectRoot "data"),
  (Join-Path $ProjectRoot "logs"),
  (Join-Path $ProjectRoot "artifacts"),
  (Join-Path $ProjectRoot "scripts")
)

foreach ($d in $dirs) {
  if (-not (Test-Path $d)) {
    New-Item -ItemType Directory -Path $d | Out-Null
    Write-Info "Created: $d"
  } else {
    Write-Info "Exists:  $d"
  }
}

# 1) locate Python (python or Windows launcher "py")
function Resolve-Python {
  $candidates = @("python", "py -3", "py")
  foreach ($c in $candidates) {
    try {
      $p = $c.Split()[0]
      if (Get-Command $p -ErrorAction SilentlyContinue) {
        # test it returns a version string
        $version = & $c - <<#%#>>c "import sys; print(sys.version)" 2>$null
        if ($LASTEXITCODE -eq 0 -or $version) { return $c }
      }
    } catch { continue }
  }
  return $null
}

$PythonCmd = Resolve-Python
if (-not $PythonCmd) {
  Write-Err "Python not found. Install Python 3.9+ and re-run."
  Write-Host "See official venv docs: https://docs.python.org/3/library/venv.html"
  exit 1
}
Write-Info "Using Python: $PythonCmd"

# 2) create virtual environment
$VenvPath   = Join-Path $ProjectRoot $VenvDir
$VenvPy     = Join-Path $VenvPath "Scripts\python.exe"   # use venv's python directly
$VenvPip    = @($VenvPy, "-m", "pip")

if ((Test-Path $VenvPy) -and -not $Force) {
  Write-Info "Virtual env already exists at $VenvPath (use -Force to rebuild)."
} else {
  if (Test-Path $VenvPath -and $Force) {
    Write-Info "Removing existing venv (Force)"
    Remove-Item -Recurse -Force $VenvPath
  }
  Write-Info "Creating venv: $VenvPath"
  & $PythonCmd -m venv $VenvPath
}

# 3) install requirements (prefer existing requirements.txt)
Push-Location $ProjectRoot
try {
  & $VenvPy -m pip install --upgrade pip
  if (Test-Path $ReqFile) {
    Write-Info "Installing from $ReqFile"
    & $VenvPy -m pip install -r $ReqFile
  } else {
    Write-Warn "No $ReqFile found; installing a minimal set for the scaffold."
    # Keep pinecone (new package), not pinecone-client (deprecated)
    & $VenvPy -m pip install `
      gradio `
      python-dotenv `
      requests `
      sentence-transformers `
      torch `
      pinecone `
      pypdf `
      tqdm `
      numpy `
      pandas
  }
} finally {
  Pop-Location
}

# 4) .env bootstrap
$EnvExample = Join-Path $ProjectRoot ".env.example"
$EnvFile    = Join-Path $ProjectRoot ".env"
if ((Test-Path $EnvExample) -and -not (Test-Path $EnvFile)) {
  Copy-Item $EnvExample $EnvFile
  Write-Info "Created .env from .env.example — add your keys before running the app."
} elseif (-not (Test-Path $EnvFile)) {
  # make a minimal .env if no template exists
  @"
OPENROUTER_API_KEY=
OPENROUTER_MODEL=openrouter/auto
OPENROUTER_REFERER=http://localhost:7860
OPENROUTER_TITLE=Personal RAG (Propositional)
PINECONE_API_KEY=
PINECONE_INDEX=personal-rag
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
EMBED_MODEL=BAAI/bge-small-en-v1.5
NAMESPACE=default
TOP_K=6
"@ | Out-File -FilePath $EnvFile -Encoding UTF8
  Write-Info "Created minimal .env — add your API keys."
}

# 5) generate cross-shell launch helpers
$ScriptsDir = Join-Path $ProjectRoot "scripts"

# PowerShell helper (activates and runs app.py)
$psHelper = @"
# PowerShell launch helper
# If your policy blocks activation, you can run: powershell -ExecutionPolicy Bypass -File .\scripts\activate.ps1
`$here = Split-Path -Parent `$(\$MyInvocation.MyCommand.Path)
Set-Location (Resolve-Path (Join-Path `$here ".."))
. .\$VenvDir\Scripts\Activate.ps1
python app.py
"@
$psHelper | Out-File (Join-Path $ScriptsDir "activate.ps1") -Encoding UTF8 -Force

# CMD helper
$batHelper = @"
@echo off
setlocal
cd /d "%~dp0\.."
call .\$VenvDir\Scripts\activate.bat
python app.py
"@
$batHelper | Out-File (Join-Path $ScriptsDir "activate.bat") -Encoding ASCII -Force

# Bash helper (for Git Bash / WSL)
$shHelper = @"
#!/usr/bin/env bash
set -euo pipefail
cd "`$(dirname "`$0`")/.."
if [ -f ".\$VenvDir/bin/activate" ]; then
  source ".\$VenvDir/bin/activate"
fi
python app.py
"@
$shPath = Join-Path $ScriptsDir "activate.sh"
$shHelper | Out-File $shPath -Encoding UTF8 -Force
# make it executable if git bash runs chmod
try { bash -lc "chmod +x `"$($shPath -replace '\\','/')`"" } catch {}

Write-Host ""
Write-Host "✅ Setup complete."
Write-Host ""
Write-Host "How to run (choose one):"
Write-Host "  PowerShell: powershell -ExecutionPolicy Bypass -File .\scripts\activate.ps1"
Write-Host "  CMD      : .\scripts\activate.bat"
Write-Host "  Bash     : bash ./scripts/activate.sh"
Write-Host ""
Write-Host "If PowerShell activation ever fails due to policy, you can:"
Write-Host "  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # current shell only"
Write-Host "  (or) run the app using the venv Python directly: .\.venv\Scripts\python.exe app.py"
