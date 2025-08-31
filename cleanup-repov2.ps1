#Requires -Version 5.1
<#
.SYNOPSIS
  Repository cleanup utility (PowerShell 5.1-safe)

.DESCRIPTION
  Removes common dev artifacts and updates .gitignore safely.
  Designed to be conservative and Windows PowerShell 5.1 compatible.

.PARAMETER RepoPath
  Target repository path. Defaults to current directory.

.PARAMETER WhatIf
  Preview actions without making changes.

.PARAMETER Force
  Skip confirmations.

.EXAMPLE
  PowerShell -ExecutionPolicy Bypass -File .\cleanup-repo.ps1 -WhatIf
#>

param(
  [Parameter(Mandatory = $false)]
  [string]$RepoPath = (Get-Location).Path,

  [switch]$WhatIf,
  [switch]$Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Color {
  param(
    [string]$Message,
    [ConsoleColor]$Color = [ConsoleColor]::Gray
  )
  $prev = $Host.UI.RawUI.ForegroundColor
  try {
    $Host.UI.RawUI.ForegroundColor = $Color
  } catch { }
  Write-Host $Message
  try {
    $Host.UI.RawUI.ForegroundColor = $prev
  } catch { }
}

function Confirm-Action {
  param(
    [string]$Prompt,
    [switch]$DefaultNo = $true
  )
  if ($Force) { return $true }
  $suffix = if ($DefaultNo) { " [y/N]" } else { " [Y/n]" }
  $answer = Read-Host ($Prompt + $suffix)
  if ([string]::IsNullOrWhiteSpace($answer)) { return -not $DefaultNo }
  return ($answer.Trim().ToLower() -in @('y','yes'))
}

function Safe-Remove {
  param([string]$PathToRemove)
  if ($WhatIf) {
    Write-Color "  (WhatIf) Would remove: $PathToRemove" DarkGray
    return
  }
  if (Test-Path -LiteralPath $PathToRemove) {
    Remove-Item -LiteralPath $PathToRemove -Recurse -Force -ErrorAction SilentlyContinue
    Write-Color "  Removed: $PathToRemove" DarkGray
  }
}

function Is-GitRepo {
  param([string]$Path)
  return (Test-Path -LiteralPath (Join-Path -Path $Path -ChildPath ".git"))
}

function Git-Status-Clean {
  try {
    $null = git rev-parse --is-inside-work-tree 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    $status = git status --porcelain 2>$null
    return [string]::IsNullOrWhiteSpace($status)
  } catch { return $false }
}

function Git-Create-Backup-Branch {
  param([string]$Name)
  try {
    git rev-parse --verify $Name 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
      Write-Color "  Backup branch '$Name' already exists" Yellow
      return
    }
    git branch $Name | Out-Null
    Write-Color "  Created backup branch: $Name" Gray
  } catch {
    Write-Color ("  Failed to create backup branch '{0}': {1}" -f $Name, $_.Exception.Message) Yellow
  }
}

# Prune sets (directories/files)
$DirectoriesToRemove = @(
  'logs','artifacts','memory-bank','venv','.venv','.pytest_cache',
  '.mypy_cache','.ruff_cache','.cache','node_modules','dist','build',
  '.coverage_html','.hypothesis','.ipynb_checkpoints'
)
$FilesToRemove = @(
  '*.log','*.tmp','*.bak','*.swp','.DS_Store','Thumbs.db',
  'qa_*.*','adversarial_*.*','verification_*.*'
)

# "Keep" (informational) globs
$DocsToKeep = @(
  'README.md','LICENSE','CONTRIBUTING.md','SECURITY.md',
  'docs/**','AGENTS.md','AGENTS_md_STANDARD.md'
)
$CoreToKeep = @(
  'src/**','app/**','backend/**','frontend/**','server/**',
  'client/**','scripts/**','tests/**','pyproject.toml',
  'requirements*.txt','package.json','package-lock.json','pnpm-lock.yaml'
)

# Pre-built .gitignore addition (avoid here-strings for PS 5.1 safety)
$nl = [Environment]::NewLine
$gitignoreAddition = @(
  '# === Added by repository cleanup script ' + (Get-Date -Format 'yyyy-MM-dd') + ' ===',
  '',
  '# Development artifacts',
  'qa_*.*',
  'logs/',
  'artifacts/',
  '*.log',
  'memory-bank/',
  'adversarial_*.*',
  'verification_*.*',
  '',
  '# Runtime data',
  'data/uploads/*',
  'data/tmp/*',
  '.cache/',
  '.ruff_cache/',
  '.pytest_cache/',
  '.mypy_cache/',
  '.hypothesis/',
  '.ipynb_checkpoints/',
  '',
  '# Python',
  '__pycache__/',
  '*.pyc',
  '*.pyo',
  '*.pyd',
  '*.egg-info/',
  '.eggs/',
  '',
  '# Node / frontend',
  'node_modules/',
  'npm-debug.log*',
  'yarn-debug.log*',
  'yarn-error.log*',
  '.pnpm-store/',
  'pnpm-lock.yaml',
  '',
  '# Build outputs',
  'dist/',
  'build/',
  '',
  '# OS generated files',
  '.DS_Store',
  '.DS_Store?',
  '._*',
  '.Spotlight-V100',
  '.Trashes',
  'ehthumbs.db',
  'Thumbs.db',
  ''
) -join $nl

# -------------------- MAIN --------------------
try {
  # Validate path and enter
  if (-not (Test-Path -LiteralPath $RepoPath -PathType Container)) {
    Write-Color ("Repository path not found: {0}" -f $RepoPath) Red
    exit 1
  }
  Push-Location -LiteralPath $RepoPath
  Write-Color ("Working in: {0}" -f (Get-Location)) Cyan

  $isGitRepo = Is-GitRepo -Path (Get-Location).Path
  if ($isGitRepo) {
    Write-Color "Git repository detected" Cyan
    if (-not (Git-Status-Clean)) {
      if (-not (Confirm-Action -Prompt "Uncommitted changes detected. Continue cleanup?")) {
        Write-Color "Aborting due to uncommitted changes." Yellow
        exit 1
      }
    }
    if (-not $WhatIf) {
      Git-Create-Backup-Branch -Name "backup-before-cleanup"
    }
  } else {
    Write-Color "No Git repository detected. Proceeding without VCS safeguards." Yellow
  }

  Write-Color "== Phase 1: Inventory (dry-run list) ==" Green
  $toRemove = New-Object System.Collections.Generic.List[string]

  foreach ($dir in $DirectoriesToRemove) {
    Get-ChildItem -Recurse -Force -Directory -Filter $dir -ErrorAction SilentlyContinue |
      ForEach-Object { $toRemove.Add($_.FullName) }
  }
  foreach ($pat in $FilesToRemove) {
    Get-ChildItem -Recurse -Force -File -Filter $pat -ErrorAction SilentlyContinue |
      ForEach-Object { $toRemove.Add($_.FullName) }
  }

  if ($toRemove.Count -eq 0) {
    Write-Color "Nothing to remove. Repository already looks clean." Gray
  } else {
    Write-Color "Candidates for removal:" Gray
    $toRemove | Sort-Object -Unique | ForEach-Object { Write-Color ("  - {0}" -f $_) DarkGray }
  }

  if (-not (Confirm-Action -Prompt "Proceed with cleanup?" -DefaultNo:$false)) {
    Write-Color "Aborted by user." Yellow
    exit 0
  }

  Write-Color "== Phase 2: Removing artifacts ==" Green
  foreach ($dir in $DirectoriesToRemove) {
    Get-ChildItem -Recurse -Force -Directory -Filter $dir -ErrorAction SilentlyContinue |
      ForEach-Object { Safe-Remove -PathToRemove $_.FullName }
  }
  foreach ($pat in $FilesToRemove) {
    Get-ChildItem -Recurse -Force -File -Filter $pat -ErrorAction SilentlyContinue |
      ForEach-Object { Safe-Remove -PathToRemove $_.FullName }
  }

  Write-Color "== Phase 3: .gitignore hygiene ==" Green
  if ($WhatIf) {
    Write-Color "(WhatIf) Would append standard ignores to .gitignore if not already present." DarkGray
  } else {
    $gitignorePath = ".gitignore"
    if (-not (Test-Path -LiteralPath $gitignorePath)) {
      New-Item -ItemType File -Path $gitignorePath | Out-Null
      Write-Color "Created new .gitignore file" Gray
    }
    $existing = Get-Content -LiteralPath $gitignorePath -Raw
    if ($existing -like "*Added by repository cleanup script*") {
      Write-Color ".gitignore already contains cleanup section; skipping append" Gray
    } else {
      Add-Content -LiteralPath $gitignorePath -Value $gitignoreAddition
      Write-Color "Appended cleanup section to .gitignore" Gray
    }
  }

  Write-Color "== Phase 4: Protection sweep (informational) ==" Green
  $keepGlobs = @($DocsToKeep + $CoreToKeep)
  $missing = @()
  foreach ($glob in $keepGlobs) {
    $hits = Get-ChildItem -Recurse -Force -ErrorAction SilentlyContinue -Path $glob
    if (-not $hits) { $missing += $glob }
  }
  if ($missing.Count -gt 0) {
    Write-Color "Missing expected keep paths (informational):" Yellow
    $missing | ForEach-Object { Write-Color ("  - {0}" -f $_) Yellow }
  } else {
    Write-Color "Keep set looks good." Gray
  }

  Write-Color "== Phase 5: Post-clean guidance ==" Green
  if ($isGitRepo) {
    Write-Color "Next steps:" Gray
    Write-Color "1. Review:  git status" White
    Write-Color "2. Commit:  git add -A && git commit -m 'chore: cleanup repository artifacts'" White
    Write-Color "3. Push:    git push origin HEAD" White
    Write-Color "4. Revert (if needed): git checkout backup-before-cleanup" White
  }

  Write-Color "Repository cleanup completed." Green
}
catch {
  Write-Color ("Error during cleanup: {0}" -f $_.Exception.Message) Red
  Write-Color "Cleanup may be incomplete. If using git, check the backup branch." Yellow
  exit 1
}
finally {
  try { Pop-Location } catch { }
}
