# Trains the three candidates on the same synthetic data on Windows, for the
# cross-machine reproducibility experiment (the WSL side is scratch/linux/repro-gbdt.sh).
# Writes only inside J:\Tabsmith\scratch.
$env:CARGO_HOME = 'J:\Tabsmith\scratch\cargo-home'
$env:TEMP = 'J:\Tabsmith\scratch\tmp'
$env:TMP = 'J:\Tabsmith\scratch\tmp'
$env:CARGO_TERM_COLOR = 'never'
$env:VSLANG = '1033'
Set-Location J:\Tabsmith
$log = 'J:\Tabsmith\scratch\repro-gbdt.log'
Set-Content -LiteralPath $log -Value '' -Encoding utf8
cmd /c "cargo build --release --locked --offline -p tabsmith-cli >> $log 2>&1"
Write-Output "build: exit $LASTEXITCODE"
$work = 'J:\Tabsmith\scratch\repro\windows'
if (Test-Path $work) { Remove-Item -Recurse -Force $work -Confirm:$false }
New-Item -ItemType Directory -Force $work | Out-Null
Set-Location $work
$t = 'J:\Tabsmith\target\release\tabsmith.exe'
cmd /c "$t sample-data data.csv --rows 3000 --seed 7 >> $log 2>&1"
cmd /c "$t train data.csv --target defect --group device_id --id unit_id --set execution.candidate_device=""cpu"" >> $log 2>&1"
Write-Output "train: exit $LASTEXITCODE"
$out = 'J:\Tabsmith\scratch\repro\windows-hashes.txt'
$lines = @()
$lines += "data.csv  " + (Get-FileHash -Algorithm SHA256 data.csv).Hash.ToLower()
foreach ($f in Get-ChildItem -Recurse -Filter model.safetensors | Sort-Object FullName) {
    $rel = $f.FullName.Substring($work.Length + 1) -replace '\\', '/'
    $lines += "$rel  " + (Get-FileHash -Algorithm SHA256 $f.FullName).Hash.ToLower()
}
foreach ($f in Get-ChildItem -Recurse -Filter run.json | Sort-Object FullName) {
    $j = Get-Content $f.FullName -Raw | ConvertFrom-Json
    $rel = $f.FullName.Substring($work.Length + 1) -replace '\\', '/'
    $hist = ($j.history | ForEach-Object { "$($_.epoch):$($_.train_loss):$($_.valid_loss)" }) -join ' '
    $lines += "$rel  candidate=$($j.candidate) best_epoch=$($j.best_epoch) epochs=$($j.history.Count)"
    $lines += "  history $hist"
}
$lines | Set-Content -LiteralPath $out -Encoding utf8
Get-Content $out
