# Pulls a nightly backup tarball from pi-1 over SSH (via Tailscale)
# and prunes local backups to keep the last 14 days.
#
# Run via Windows Task Scheduler nightly. No outbound creds needed on
# the Pi side; this script (Windows) pulls.
#
# Manual invocation:
#   powershell -ExecutionPolicy Bypass -File scripts\backup_pi.ps1
#
# Exits non-zero on any failure (Task Scheduler will surface it). On
# success, prints the local tarball path.

$ErrorActionPreference = 'Stop'

$LocalDir = 'C:\Users\gicha\GC-repos\Claude analyses\boris-backups'
$KeepDays = 14
$RemoteScript = '~/projects/tennis-pairings/scripts/backup_pi.sh'

if (-not (Test-Path $LocalDir)) {
    New-Item -ItemType Directory -Path $LocalDir -Force | Out-Null
}

# 1. Invoke the Pi-side helper. It builds the tarball at /tmp and
#    prints the path on stdout (size summary goes to stderr).
$remotePath = (ssh pi-1 "bash $RemoteScript" 2>$null).Trim()
if (-not $remotePath -or -not $remotePath.StartsWith('/')) {
    throw "Pi-side script returned unexpected output: '$remotePath'"
}
Write-Host "Pi tarball: $remotePath"

# 2. SCP it back.
$stamp = Get-Date -Format 'yyyy-MM-dd'
$localPath = Join-Path $LocalDir "boris-backup-$stamp.tar.gz"
& scp "pi-1:$remotePath" $localPath
if ($LASTEXITCODE -ne 0) { throw "scp failed (exit $LASTEXITCODE)" }

# 3. Verify the local file is non-empty + can be listed by tar (sanity
#    check that the transfer wasn't truncated). Windows 11 has bsdtar
#    via tar.exe.
$size = (Get-Item $localPath).Length
if ($size -lt 1024) { throw "tarball too small ($size bytes) — corrupt?" }
$entries = (& tar -tzf $localPath | Measure-Object).Count
if ($entries -lt 5) { throw "tarball has only $entries entries — likely incomplete" }
Write-Host "Local copy: $localPath ($size bytes, $entries entries)"

# 4. Delete the Pi-side tmp file (best-effort; not fatal if it fails).
ssh pi-1 "rm -f $remotePath" 2>$null

# 5. Prune local backups older than $KeepDays.
$cutoff = (Get-Date).AddDays(-$KeepDays)
Get-ChildItem -Path $LocalDir -Filter 'boris-backup-*.tar.gz' |
    Where-Object { $_.LastWriteTime -lt $cutoff } |
    ForEach-Object {
        Write-Host "Pruning $($_.Name) ($($_.LastWriteTime))"
        Remove-Item $_.FullName -Force
    }

Write-Host "Backup complete."
