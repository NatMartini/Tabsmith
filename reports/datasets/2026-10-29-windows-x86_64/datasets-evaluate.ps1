# Scores every candidate of the three dataset projects on the test split (a benchmark,
# not model selection: --allow-test-reuse, recorded in each evaluation).
# Writes only inside <data>\runs.
$t = '<repo>\target\release\tabsmith.exe'
$runs = '<data>\runs'
foreach ($name in 'adult', 'california', 'covertype') {
    $dir = "$runs\$name"
    Set-Location $dir
    $j = Get-Content "$dir\train.json" -Raw | ConvertFrom-Json
    foreach ($run in $j.result.group.runs) {
        $id = $run.run_id
        cmd /c "$t --json evaluate --run $id --split test --allow-test-reuse > $dir\eval-$id.json 2> $dir\eval-$id.err"
        Write-Output "$name $id ($($run.candidate)): exit $LASTEXITCODE"
    }
}
