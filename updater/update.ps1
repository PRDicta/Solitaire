# Solitaire Universal Updater (PowerShell)
#
# Fallback for when Python is not available.
# Implements the same safety gates as update.py:
#   1. Target state discovery
#   2. Data-first backup (atomic SQLite via Python, file-copy fallback)
#   3. Circuit breaker (abort after 2 consecutive failures)

param(
    [Parameter(Position=0)]
    [string]$UpdateDir
)

$ErrorActionPreference = "Stop"

# --- Configuration ---

$Distributable = @(
    "solitaire",
    "pyproject.toml",
    "CLAUDE.md",
    "README.md",
    "skill",
    "mcp-server",
    "LICENSE",
    "COMMERCIAL_LICENSE.md"
)

$LegacyCleanup = @(
    "ai_writing_tells.md",
    "FIRST_INTERACTION.md",
    "starter"
)

$SkipDirs = @("__pycache__", ".git", ".pytest_cache")
$SkipExts = @(".pyc", ".pyo")

$Protected = @(
    "rolodex.db",
    "rolodex.db-wal",
    "rolodex.db-shm",
    "personas",
    "backups",
    "sessions",
    ".solitaire_session"
)

# --- Helpers ---

function Write-OK($msg) { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Err($msg) { Write-Host "  [!!] $msg" -ForegroundColor Red }
function Write-Info($msg) { Write-Host "  $msg" }

function Get-SolitaireVersion($solitaireDir) {
    $versionFile = Join-Path $solitaireDir "__version__.py"
    if (-not (Test-Path $versionFile)) { return $null }
    $content = Get-Content $versionFile -Raw -Encoding UTF8
    if ($content -match '__version__\s*=\s*["'']([^"'']+)["'']') {
        return $Matches[1]
    }
    return $null
}

function Find-Workspace($updateDir) {
    # Strategy 1: explicit workspace file
    $wsFile = Join-Path $updateDir "solitaire_workspace.txt"
    if (Test-Path $wsFile) {
        $wsPath = (Get-Content $wsFile -Raw -Encoding UTF8).Trim()
        if ((Test-Path $wsPath) -and (Test-Path (Join-Path $wsPath "rolodex.db"))) {
            return $wsPath
        }
    }

    # Strategy 2: walk upward looking for rolodex.db or pyproject.toml
    $current = Split-Path $updateDir -Parent
    for ($i = 0; $i -lt 5; $i++) {
        if (Test-Path (Join-Path $current "rolodex.db")) { return $current }
        if (Test-Path (Join-Path $current "pyproject.toml")) { return $current }
        $parent = Split-Path $current -Parent
        if ($parent -eq $current) { break }
        $current = $parent
    }

    # Strategy 3: ask the user
    Write-Host ""
    Write-Info "Could not auto-detect your Solitaire workspace."
    Write-Info "Please enter the full path to your Solitaire folder"
    Write-Info "(the folder that contains rolodex.db):"
    Write-Host ""
    while ($true) {
        $userPath = Read-Host "  Path"
        $userPath = $userPath.Trim().Trim('"').Trim("'")
        if ([string]::IsNullOrEmpty($userPath)) { return $null }
        if ((Test-Path $userPath) -and (Test-Path (Join-Path $userPath "rolodex.db"))) {
            return $userPath
        }
        Write-Err "No rolodex.db found at $userPath"
        Write-Info "Please try again, or press Enter to cancel."
    }
}

function Copy-Filtered($src, $dst) {
    if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
    $items = Get-ChildItem $src -Recurse | Where-Object {
        $skip = $false
        foreach ($d in $SkipDirs) {
            if ($_.FullName -like "*\$d\*" -or $_.Name -eq $d) { $skip = $true; break }
        }
        if (-not $skip) {
            foreach ($e in $SkipExts) {
                if ($_.Extension -eq $e) { $skip = $true; break }
            }
        }
        -not $skip
    }
    $items | Where-Object { $_.PSIsContainer } | ForEach-Object {
        $destPath = $_.FullName.Replace($src, $dst)
        if (-not (Test-Path $destPath)) {
            New-Item -ItemType Directory -Path $destPath -Force | Out-Null
        }
    }
    $items | Where-Object { -not $_.PSIsContainer } | ForEach-Object {
        $destPath = $_.FullName.Replace($src, $dst)
        $destDir = Split-Path $destPath -Parent
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        Copy-Item $_.FullName $destPath -Force
    }
}

# --- Target State Discovery ---

function Get-TargetState($workspace) {
    $state = @{
        version = $null
        layout = "unknown"
        has_rolodex = $false
        rolodex_size = 0
        has_personas = $false
        persona_count = 0
        has_sessions = $false
        has_session_marker = $false
        code_locations = @()
        legacy_files = @()
    }

    # Check unified layout (v1.1.0+)
    $vFile = Join-Path $workspace "solitaire\__version__.py"
    if (Test-Path $vFile) {
        $state.version = Get-SolitaireVersion (Join-Path $workspace "solitaire")
        $state.layout = "v1_unified"
        $state.code_locations += "solitaire/"
    }

    # Check dual-tree layout (v1.0.0)
    foreach ($oldPath in @("src", "starter\solitaire")) {
        $vFile = Join-Path $workspace "$oldPath\__version__.py"
        if (Test-Path $vFile) {
            if (-not $state.version) {
                $state.version = Get-SolitaireVersion (Join-Path $workspace $oldPath)
            }
            $state.layout = "v1_dual"
            $state.code_locations += "$oldPath/"
        }
    }

    # Data inventory
    $rdb = Join-Path $workspace "rolodex.db"
    if (Test-Path $rdb) {
        $state.has_rolodex = $true
        $state.rolodex_size = (Get-Item $rdb).Length
    }

    $personas = Join-Path $workspace "personas"
    if ((Test-Path $personas) -and (Get-ChildItem $personas -ErrorAction SilentlyContinue)) {
        $items = @(Get-ChildItem $personas)
        $state.has_personas = $items.Count -gt 0
        $state.persona_count = $items.Count
    }

    $sessions = Join-Path $workspace "sessions"
    $state.has_sessions = (Test-Path $sessions) -and @(Get-ChildItem $sessions -ErrorAction SilentlyContinue).Count -gt 0

    $state.has_session_marker = Test-Path (Join-Path $workspace ".solitaire_session")

    # Legacy scan
    foreach ($name in @("ai_writing_tells.md", "FIRST_INTERACTION.md", "starter", "src")) {
        if (Test-Path (Join-Path $workspace $name)) {
            $state.legacy_files += $name
        }
    }

    return $state
}

# --- Data Backup (Prerequisite Gate) ---

function Backup-UserData($workspace) {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $backupDir = Join-Path $workspace "backups\pre-update-data-$timestamp"
    New-Item -ItemType Directory -Path $backupDir -Force | Out-Null

    $report = @{
        backup_dir = $backupDir
        rolodex = "not_present"
        personas = "not_present"
        sessions = "not_present"
        session_marker = "not_present"
    }

    # 1. Backup rolodex.db
    $rdb = Join-Path $workspace "rolodex.db"
    if (Test-Path $rdb) {
        $backupDb = Join-Path $backupDir "rolodex.db"

        # Try atomic SQLite backup via Python
        $python = $null
        foreach ($cmd in @("python", "python3", "py")) {
            try {
                $null = & $cmd --version 2>&1
                $python = $cmd
                break
            } catch {}
        }

        $atomicOk = $false
        if ($python) {
            $pyScript = "import sqlite3; s=sqlite3.connect('$($rdb.Replace('\','\\'))'); d=sqlite3.connect('$($backupDb.Replace('\','\\'))'); s.backup(d); d.close(); s.close()"
            try {
                & $python -c $pyScript 2>&1
                if ($LASTEXITCODE -eq 0) { $atomicOk = $true }
            } catch {}
        }

        if (-not $atomicOk) {
            # Fallback: file copy (less safe but better than nothing)
            Write-Info "  (Using file-copy backup; atomic SQLite unavailable)"
            Copy-Item $rdb $backupDb -Force
            # Also copy WAL if present
            $wal = Join-Path $workspace "rolodex.db-wal"
            if (Test-Path $wal) {
                Copy-Item $wal (Join-Path $backupDir "rolodex.db-wal") -Force
            }
        }

        # Verify backup exists and has size
        if ((Test-Path $backupDb) -and (Get-Item $backupDb).Length -gt 0) {
            $sizeKB = [math]::Round((Get-Item $backupDb).Length / 1024)
            $report.rolodex = @{ path = $backupDb; size_kb = $sizeKB }
        } else {
            throw "Rolodex backup verification failed: backup is empty or missing"
        }
    }

    # 2. Backup personas/
    $personas = Join-Path $workspace "personas"
    if ((Test-Path $personas) -and @(Get-ChildItem $personas -ErrorAction SilentlyContinue).Count -gt 0) {
        $personasDst = Join-Path $backupDir "personas"
        Copy-Item $personas $personasDst -Recurse -Force
        $count = @(Get-ChildItem $personasDst -Recurse -File).Count
        $report.personas = @{ path = $personasDst; files = $count }
    }

    # 3. Backup sessions/
    $sessions = Join-Path $workspace "sessions"
    if ((Test-Path $sessions) -and @(Get-ChildItem $sessions -ErrorAction SilentlyContinue).Count -gt 0) {
        $sessionsDst = Join-Path $backupDir "sessions"
        Copy-Item $sessions $sessionsDst -Recurse -Force
        $count = @(Get-ChildItem $sessionsDst -Recurse -File).Count
        $report.sessions = @{ path = $sessionsDst; files = $count }
    }

    # 4. Backup .solitaire_session
    $marker = Join-Path $workspace ".solitaire_session"
    if (Test-Path $marker) {
        Copy-Item $marker (Join-Path $backupDir ".solitaire_session") -Force
        $report.session_marker = "backed_up"
    }

    return $report
}

# --- Main ---

if ([string]::IsNullOrEmpty($UpdateDir)) {
    Write-Err "No update directory specified."
    exit 1
}

$UpdateDir = (Resolve-Path $UpdateDir).Path

$updateSolitaire = Join-Path $UpdateDir "solitaire"
if (-not (Test-Path $updateSolitaire)) {
    Write-Err "No solitaire/ folder found in $UpdateDir"
    exit 1
}

$newVersion = Get-SolitaireVersion $updateSolitaire
if (-not $newVersion) {
    Write-Err "Could not read version from update package"
    exit 1
}

# Step 1: Find workspace
Write-Info "Looking for your Solitaire workspace..."
$workspace = Find-Workspace $UpdateDir
if (-not $workspace) {
    Write-Err "Could not find workspace. Update cancelled."
    exit 1
}

$workspace = (Resolve-Path $workspace).Path

# Step 2: Target state discovery
Write-Host ""
Write-Info "Examining current installation..."
$preState = Get-TargetState $workspace

$currentVersion = $preState.version
$versionLabel = if ($currentVersion) { $currentVersion } else { "pre-1.0" }
$layout = $preState.layout

Write-Host ""
Write-Info "Workspace:   $workspace"
Write-Info "Current:     v$versionLabel"
Write-Info "Layout:      $layout"
if ($preState.code_locations.Count -gt 0) {
    Write-Info "Code dirs:   $($preState.code_locations -join ', ')"
}
Write-Info "Updating to: v$newVersion"
if ($preState.has_rolodex) {
    $sizeKB = [math]::Round($preState.rolodex_size / 1024)
    Write-Info "Data:        rolodex.db ($sizeKB KB)"
}
if ($preState.has_personas) {
    Write-Info "             personas/ ($($preState.persona_count) items)"
}
Write-Host ""

# Step 3: Data-first backup (prerequisite gate)
Write-Info "Backing up user data (prerequisite gate)..."
try {
    $backupReport = Backup-UserData $workspace
    $dataBackupDir = $backupReport.backup_dir

    if ($backupReport.rolodex -ne "not_present") {
        Write-OK "Rolodex backed up ($($backupReport.rolodex.size_kb) KB)"
    }
    if ($backupReport.personas -ne "not_present") {
        Write-OK "Personas backed up ($($backupReport.personas.files) files)"
    }
    if ($backupReport.sessions -ne "not_present") {
        Write-OK "Sessions backed up ($($backupReport.sessions.files) files)"
    }
    Write-OK "Data backup saved to: $dataBackupDir"
}
catch {
    Write-Err "DATA BACKUP FAILED: $_"
    Write-Err "Update cancelled. Nothing was changed."
    Write-Err "Your data is safe. The backup gate prevented any operations."
    exit 1
}

# Step 4: Code backup (best-effort)
Write-Host ""
Write-Info "Backing up current code (best-effort)..."
$codeBackupDir = $null
try {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $codeBackupDir = Join-Path $workspace "backups\pre-update-code-v$versionLabel-$timestamp"
    New-Item -ItemType Directory -Path $codeBackupDir -Force | Out-Null

    $backedUp = 0
    foreach ($dirname in @("solitaire", "src", "starter", "skill")) {
        $src = Join-Path $workspace $dirname
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $codeBackupDir $dirname) -Recurse -Force
            $backedUp++
        }
    }
    foreach ($fname in @("pyproject.toml", "CLAUDE.md", "README.md")) {
        $src = Join-Path $workspace $fname
        if (Test-Path $src) {
            Copy-Item $src (Join-Path $codeBackupDir $fname) -Force
            $backedUp++
        }
    }
    Write-OK "Code backed up ($backedUp items)"
}
catch {
    Write-Info "Code backup note: $_ (non-fatal, continuing)"
    $codeBackupDir = $null
}

# Step 5: Execute migration with circuit breaker
Write-Host ""
Write-Info "Running migration for layout: $layout"

$consecutiveFailures = 0
$maxFailures = 2
$allFailures = @()

function Invoke-Step($stepName, $scriptBlock) {
    try {
        $result = & $scriptBlock
        $script:consecutiveFailures = 0
        Write-OK "${stepName}: $result"
        return $true
    }
    catch {
        $script:consecutiveFailures++
        $script:allFailures += @{ step = $stepName; error = $_.ToString() }
        Write-Err "${stepName}: $_"

        if ($script:consecutiveFailures -ge $maxFailures) {
            throw "Circuit breaker tripped after $maxFailures consecutive failures"
        }
        return $false
    }
}

try {
    # Remove old code based on layout
    switch ($layout) {
        "v1_dual" {
            Invoke-Step "remove_src" {
                $t = Join-Path $workspace "src"
                if (Test-Path $t) { Remove-Item $t -Recurse -Force; "Removed src/" } else { "src/ not present" }
            }
            Invoke-Step "remove_starter" {
                $t = Join-Path $workspace "starter"
                if (Test-Path $t) { Remove-Item $t -Recurse -Force; "Removed starter/" } else { "starter/ not present" }
            }
        }
        "v1_unified" {
            Invoke-Step "remove_solitaire" {
                $t = Join-Path $workspace "solitaire"
                if (Test-Path $t) { Remove-Item $t -Recurse -Force; "Removed solitaire/" } else { "solitaire/ not present" }
            }
        }
    }

    # Legacy cleanup
    Invoke-Step "remove_legacy_files" {
        $removed = @()
        foreach ($name in $LegacyCleanup) {
            $t = Join-Path $workspace $name
            if (Test-Path $t) { Remove-Item $t -Recurse -Force; $removed += $name }
        }
        if ($removed.Count -gt 0) { "Cleaned: $($removed -join ', ')" } else { "Nothing to remove" }
    }

    # Egg-info cleanup
    Invoke-Step "clean_egg_info" {
        $removed = @()
        Get-ChildItem $workspace -Directory -Filter "*.egg-info" | ForEach-Object {
            Remove-Item $_.FullName -Recurse -Force
            $removed += $_.Name
        }
        if ($removed.Count -gt 0) { "Cleaned: $($removed -join ', ')" } else { "No egg-info dirs" }
    }

    # Copy new code
    Invoke-Step "copy_new_code" {
        $copied = @()
        foreach ($name in $Distributable) {
            $src = Join-Path $UpdateDir $name
            $dst = Join-Path $workspace $name
            if (-not (Test-Path $src)) { continue }
            if (Test-Path $src -PathType Container) {
                Copy-Filtered $src $dst
            } else {
                Copy-Item $src $dst -Force
            }
            $copied += $name
        }
        "Copied: $($copied -join ', ')"
    }

    # Verify data intact
    Invoke-Step "verify_data_intact" {
        $issues = @()
        if ($preState.has_rolodex -and -not (Test-Path (Join-Path $workspace "rolodex.db"))) {
            $issues += "rolodex.db MISSING"
        }
        if ($preState.has_personas -and -not (Test-Path (Join-Path $workspace "personas"))) {
            $issues += "personas/ MISSING"
        }
        if ($preState.has_sessions -and -not (Test-Path (Join-Path $workspace "sessions"))) {
            $issues += "sessions/ MISSING"
        }
        if ($issues.Count -gt 0) { throw "Data integrity check failed: $($issues -join '; ')" }
        "All user data intact"
    }
}
catch {
    Write-Host ""
    Write-Err "UPDATE ABORTED: $_"
    Write-Info "Attempting rollback..."

    # Restore data if missing
    $restored = @()
    if ($dataBackupDir) {
        $backupRdb = Join-Path $dataBackupDir "rolodex.db"
        $liveRdb = Join-Path $workspace "rolodex.db"
        if ((Test-Path $backupRdb) -and -not (Test-Path $liveRdb)) {
            Copy-Item $backupRdb $liveRdb -Force
            $restored += "rolodex.db"
        }
        foreach ($dir in @("personas", "sessions")) {
            $backupD = Join-Path $dataBackupDir $dir
            $liveD = Join-Path $workspace $dir
            if ((Test-Path $backupD) -and -not (Test-Path $liveD)) {
                Copy-Item $backupD $liveD -Recurse -Force
                $restored += "$dir/"
            }
        }
    }

    # Restore code
    if ($codeBackupDir) {
        try {
            $backupSol = Join-Path $codeBackupDir "solitaire"
            if (Test-Path $backupSol) {
                $dst = Join-Path $workspace "solitaire"
                if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
                Copy-Item $backupSol $dst -Recurse -Force
                $restored += "solitaire/"
            }
        } catch {}
    }

    if ($restored.Count -gt 0) { Write-OK "Restored: $($restored -join ', ')" }
    Write-Info "Data backup: $dataBackupDir"
    if ($codeBackupDir) { Write-Info "Code backup: $codeBackupDir" }
    exit 1
}

# Step 6: Verify
Write-Host ""
$checks = @{}

$rdb = Join-Path $workspace "rolodex.db"
if (Test-Path $rdb) {
    $sizeKB = [math]::Round((Get-Item $rdb).Length / 1024)
    $status = "OK ($sizeKB KB)"
    if ($preState.has_rolodex) {
        $preKB = [math]::Round($preState.rolodex_size / 1024)
        $status += " [was $preKB KB]"
    }
    $checks["rolodex.db"] = $status
}
elseif ($preState.has_rolodex) {
    $checks["rolodex.db"] = "MISSING (was present before update!)"
}

$personas = Join-Path $workspace "personas"
if (Test-Path $personas) {
    $count = @(Get-ChildItem $personas).Count
    $status = "OK ($count items)"
    if ($preState.has_personas) { $status += " [was $($preState.persona_count)]" }
    $checks["personas/"] = $status
}
elseif ($preState.has_personas) {
    $checks["personas/"] = "MISSING (was present before update!)"
}

$installedVersion = Get-SolitaireVersion (Join-Path $workspace "solitaire")
if ($installedVersion -eq $newVersion) {
    $checks["version"] = "OK (v$installedVersion)"
}
else {
    $checks["version"] = "MISMATCH (expected $newVersion, got $installedVersion)"
}

# Report
Write-Host ""
$banner = "UPDATE COMPLETE: v$versionLabel  -->  v$newVersion"
$width = [Math]::Max($banner.Length + 4, 44)
Write-Host ("=" * $width) -ForegroundColor Cyan
Write-Host "  $banner" -ForegroundColor Cyan
Write-Host ("=" * $width) -ForegroundColor Cyan
Write-Host ""
Write-Info "Verification:"
foreach ($key in $checks.Keys) {
    $status = $checks[$key]
    Write-Info ("    {0,-20} {1}" -f $key, $status)
}
Write-Host ""
Write-Info "Backups:"
Write-Info "    Data: $dataBackupDir"
if ($codeBackupDir) { Write-Info "    Code: $codeBackupDir" }
Write-Host ""
Write-Info "Open Cowork and start a new session. You're all set."
Write-Host ""
