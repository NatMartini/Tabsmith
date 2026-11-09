# Compile time of each cost crate: one build for the dependencies, then three rebuilds of the
# crate alone (main.rs touched), release and check. Writes only inside J:\Tabsmith\scratch.
$env:CARGO_HOME='J:\Tabsmith\scratch\cargo-home'; $env:TEMP='J:\Tabsmith\scratch\tmp'; $env:TMP='J:\Tabsmith\scratch\tmp'; $env:VSLANG='1033'; $env:CARGO_INCREMENTAL='0'
$root = 'J:\Tabsmith\scratch\const-pred\cost'
foreach ($m in 'adult','california','covertype') {
  foreach ($v in 'head','const') {
    $dir = Join-Path $root "$m-$v"
    $env:CARGO_TARGET_DIR = Join-Path $root "target-$v"
    cargo build --release --offline --manifest-path "$dir\Cargo.toml" 2>&1 | Select-String -Pattern '^error|error\[|Finished' | Select-Object -First 3 | ForEach-Object { "  $m-$v first: $($_.Line)" }
    foreach ($mode in 'build','check') {
      $times = @()
      foreach ($r in 1..3) {
        (Get-Item "$dir\src\main.rs").LastWriteTime = Get-Date
        $t = Measure-Command { if ($mode -eq 'build') { cargo build --release --offline --manifest-path "$dir\Cargo.toml" 2>$null } else { cargo check --release --offline --manifest-path "$dir\Cargo.toml" 2>$null } }
        $times += [math]::Round($t.TotalSeconds, 1)
      }
      "$m $v $mode " + ($times -join ' ')
    }
  }
}
