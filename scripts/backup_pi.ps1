# Pulls a nightly backup tarball from pi-1 over SSH (via Tailscale)
# and prunes local backups to keep the last 14 days.
#
# Run via Windows Task Scheduler nightly. No outbound creds needed
# on the Pi side; this script (Windows) pulls.
#
# Manual invocation:
#   powershell -ExecutionPolicy Bypass -File scripts\backup_pi.ps1
#
# Exits non-zero on any failure (Task Scheduler will surface it).

$ErrorActionPreference = 'Stop'

$LocalDir = 'C:\Users\gicha\GC-repos\Claude analyses\boris-backups'
$KeepDays = 14
$RemoteScript = '~/projects/tennis-pairings/scripts/backup_pi.sh'

if (-not (Test-Path $LocalDir)) {
    New-Item -ItemType Directory -Path $LocalDir -Force | Out-Null
}

# Path is deterministic from the Pi's date (the bash script uses
# `date +%Y-%m-%d`). Pi runs in BST/UTC+1 same as us; if midnight
# crosses while the script runs, worst case we name the file by
# yesterday's date -- not a problem.
$stamp = Get-Date -Format 'yyyy-MM-dd'
$remotePath = "/tmp/boris-backup-$stamp.tar.gz"
$localPath = Join-Path $LocalDir "boris-backup-$stamp.tar.gz"

# Invoke the Pi-side helper. Use cmd.exe to suppress ssh's stderr,
# because PowerShell 5.1 with $ErrorActionPreference='Stop' otherwise
# treats any stderr line from a native exe as a fatal error.
& cmd /c "ssh pi-1 ""bash $RemoteScript"" 2>nul" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Pi backup script exited $LASTEXITCODE" }
Write-Host "Pi tarball: $remotePath"

# SCP it back.
& cmd /c "scp pi-1:$remotePath ""$localPath"" 2>nul" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "scp failed (exit $LASTEXITCODE)" }

# Verify the local file is non-empty + can be listed by tar.
$size = (Get-Item $localPath).Length
if ($size -lt 1024) { throw "tarball too small ($size bytes); likely corrupt" }
$entries = (& tar -tzf $localPath | Measure-Object).Count
if ($entries -lt 5) { throw "tarball has only $entries entries; incomplete" }
Write-Host "Local copy: $localPath ($size bytes, $entries entries)"

# Delete the Pi-side tmp file (best-effort).
& cmd /c "ssh pi-1 ""rm -f $remotePath"" 2>nul" | Out-Null

# Prune local backups older than $KeepDays.
$cutoff = (Get-Date).AddDays(-$KeepDays)
Get-ChildItem -Path $LocalDir -Filter 'boris-backup-*.tar.gz' |
    Where-Object { $_.LastWriteTime -lt $cutoff } |
    ForEach-Object {
        Write-Host "Pruning $($_.Name) ($($_.LastWriteTime))"
        Remove-Item $_.FullName -Force
    }

Write-Host "Backup complete."
