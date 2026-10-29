# Scaling run: the gradient-boosted trees on 1,000,000 synthetic rows with the release
# binary, wall time and peak memory of the whole `tabsmith train` process.
# Usage: scale.ps1 [rows]   Writes only inside J:\Tabsmith\scratch\scale.
param([int]$Rows = 1000000)
$t = 'J:\Tabsmith\target\release\tabsmith.exe'
$work = "J:\Tabsmith\scratch\scale\rows-$Rows"
if (Test-Path $work) { Remove-Item -Recurse -Force $work -Confirm:$false }
New-Item -ItemType Directory -Force $work | Out-Null
Set-Location $work
$out = "J:\Tabsmith\scratch\scale\rows-$Rows.txt"
$lines = @()
$sw = [Diagnostics.Stopwatch]::StartNew()
& $t sample-data data.csv --rows $Rows --seed 7 2>&1 | Out-Null
$lines += "sample-data: $([int]$sw.Elapsed.TotalSeconds) s, $([math]::Round((Get-Item data.csv).Length / 1MB)) MB"
$sw.Restart()
$p = Start-Process -FilePath $t -ArgumentList @('train', 'data.csv', '--target', 'defect', '--group', 'device_id', '--id', 'unit_id', '--candidates', 'gbdt', '--set', 'execution.candidate_device="cpu"', '--json') -NoNewWindow -PassThru -RedirectStandardOutput train.json -RedirectStandardError train.err
$peak = 0
while (-not $p.HasExited) {
    try { $p.Refresh(); if ($p.PeakWorkingSet64 -gt $peak) { $peak = $p.PeakWorkingSet64 } } catch {}
    Start-Sleep -Milliseconds 500
}
$lines += "train (gbdt): exit $($p.ExitCode), wall $([int]$sw.Elapsed.TotalSeconds) s, peak working set $([math]::Round($peak / 1MB)) MB"
try {
    $j = Get-Content train.json -Raw | ConvertFrom-Json
    $run = $j.result.runs[0]
    $lines += "run: $($run.run_id) candidate=$($run.candidate) status=$($run.status) rounds=$($run.history.Count) steps_done=$($run.steps_done)"
    $m = $run.metrics.valid
    $lines += "valid: balanced_accuracy=$($m.balanced_accuracy) roc_auc=$($m.roc_auc)"
    $lines += "timings: " + (($run.PSObject.Properties | Where-Object { $_.Name -like '*_ms' -or $_.Name -like '*seconds*' } | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' ')
    $lines += "stages: " + (($j.result.PSObject.Properties | Where-Object { $_.Name -like '*_ms' } | ForEach-Object { "$($_.Name)=$($_.Value)" }) -join ' ')
} catch { $lines += "could not parse train.json: $_" }
$lines += "stderr tail:"
$lines += (Get-Content train.err -Tail 12)
$lines | Set-Content -LiteralPath $out -Encoding utf8
Get-Content $out
