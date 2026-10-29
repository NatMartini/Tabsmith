# Trains Tabsmith's three candidates on the three public datasets (CSV built by
# datasets-prep.py), one project per dataset, and keeps the JSON results.
# Writes only inside <data>\runs.
$t = '<repo>\target\release\tabsmith.exe'
$base = '<data>'
$runs = "$base\runs"
if (Test-Path $runs) { Remove-Item -Recurse -Force $runs -Confirm:$false }
New-Item -ItemType Directory -Force $runs | Out-Null
$sets = @(
    @{ name = 'adult';      args = @('train', "$base\adult.csv", '--target', 'income', '--positive', '>50K', '--set', 'data.null_values=[\"\",\"NA\",\"?\"]') },
    @{ name = 'california'; args = @('train', "$base\california.csv", '--target', 'median_house_value', '--task', 'regression') },
    @{ name = 'covertype';  args = @('train', "$base\covertype.csv", '--target', 'cover_type', '--task', 'multiclass') }
)
$summary = @()
foreach ($s in $sets) {
    $dir = "$runs\$($s.name)"
    New-Item -ItemType Directory -Force $dir | Out-Null
    Set-Location $dir
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $t -ArgumentList ($s.args + @('--set', 'execution.candidate_device="cpu"', '--json')) -NoNewWindow -PassThru -RedirectStandardOutput "$dir\train.json" -RedirectStandardError "$dir\train.err"
    $peak = 0
    while (-not $p.HasExited) {
        try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 } } catch {}
        Start-Sleep -Milliseconds 500
    }
    $line = "$($s.name): exit $($p.ExitCode), wall $([int]$sw.Elapsed.TotalSeconds) s, peak working set $([math]::Round($peak / 1MB)) MB"
    $summary += $line
    Write-Output $line
}
$summary | Set-Content -LiteralPath "$runs\summary.txt" -Encoding utf8
