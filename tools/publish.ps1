# Pushes this repository to GitHub and publishes its crates to crates.io. Double-click
# publish.cmd next to this file. It commits nothing: commit first, then double-click.
#
# Nothing changes before it has shown the plan and you have pressed Enter. The steps run
# from the easiest to undo to the hardest, because a version on crates.io is there forever:
#   1. Checks: on main, nothing uncommitted, not behind GitHub.
#   2. The plan, and one Enter.
#   3. For a new version: package and build every crate locally (cargo publish --dry-run).
#   4. git push.
#   5. For a new version: wait for CI on that commit, and stop if it fails.
#   6. Tag vX.Y.Z and push the tag, which starts the Release workflow.
#   7. cargo publish --workspace.
#
# If the version in Cargo.toml is already on crates.io, only step 4 runs. Run it again after
# any failure: whatever is already done is skipped. There is never a force push.
#
#   -DryRun   shows the plan and changes nothing
#   -Yes      does not ask before starting
param(
    [switch]$DryRun,
    [switch]$Yes,
    # Where it asks, and how often; the tests point these at a local server.
    [string]$IndexBase = 'https://index.crates.io',
    [string]$ApiBase = 'https://api.github.com',
    [int]$PollSeconds = 90
)

$Remote = 'tabsmith'
$Branch = 'main'
$GitHubRepo = 'NatMartini/Tabsmith'
$CiWorkflow = '.github/workflows/ci.yml'
# Sent with every request, so it names the project and nobody's address.
$UserAgent = "tabsmith-publish (https://github.com/$GitHubRepo)"
$Done = New-Object System.Collections.ArrayList

try { [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false } catch {}
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
# The linker's Chinese "creating library" message reads as a warning for proc-macro crates.
$env:VSLANG = '1033'
# A few builds at a time: rustc runs out of memory here when too many run at once.
if (-not $env:CARGO_BUILD_JOBS) { $env:CARGO_BUILD_JOBS = '4' }

function Write-Step([string]$Text) {
    Write-Host ''
    Write-Host "== $Text" -ForegroundColor Cyan
}

function Stop-Publish([string]$Why, [string]$Next = '') {
    Write-Host ''
    Write-Host "停止：$Why" -ForegroundColor Red
    if ($Next) { Write-Host $Next -ForegroundColor Yellow }
    if ($Done.Count -gt 0) {
        Write-Host "已经完成：$($Done -join '；')。"
    } else {
        Write-Host 'GitHub 和 crates.io 都没有变化。'
    }
    exit 1
}

function Invoke-Git {
    $output = & git @args
    if ($LASTEXITCODE -ne 0) { Stop-Publish "git $($args -join ' ') 失败。" }
    return $output
}

# The text at a URL, or $null when it does not exist. Any other failure throws.
function Get-Text([string]$Url) {
    $client = New-Object System.Net.WebClient
    $client.Encoding = [Text.Encoding]::UTF8
    $client.Headers['User-Agent'] = $UserAgent
    try {
        return $client.DownloadString($Url)
    } catch {
        $error0 = $_.Exception
        if ($error0.InnerException) { $error0 = $error0.InnerException }
        if ($error0.Response -and [int]$error0.Response.StatusCode -eq 404) { return $null }
        throw
    } finally {
        $client.Dispose()
    }
}

# Whether crates.io has this version of a crate, yanked or not: from the public index.
function Test-Published([string]$Name, [string]$Version) {
    $text = Get-Text ('{0}/{1}/{2}/{3}' -f $IndexBase, $Name.Substring(0, 2), $Name.Substring(2, 2), $Name)
    if ($null -eq $text) { return $false }
    foreach ($line in ($text -split "`n")) {
        if ($line.Trim() -and (ConvertFrom-Json $line).vers -eq $Version) { return $true }
    }
    return $false
}

# Waits for the CI run of this commit. Returns only when it passed.
function Wait-ForCi([string]$Sha) {
    $url = "$ApiBase/repos/$GitHubRepo/actions/runs?head_sha=$Sha&per_page=30"
    $started = Get-Date
    $failures = 0
    while ($true) {
        $minutes = [int][math]::Floor(((Get-Date) - $started).TotalMinutes)
        try {
            $text = Get-Text $url
            if ($null -eq $text) { throw "$url 不存在" }
            $runs = (ConvertFrom-Json $text).workflow_runs
            $failures = 0
        } catch {
            $failures++
            Write-Host "暂时读不到 CI 状态（第 $failures 次）：$($_.Exception.Message)"
            if ($failures -ge 40) {
                Stop-Publish '一直读不到 CI 状态。' '网络恢复后再双击一次，已完成的步骤会跳过。'
            }
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        $run = @($runs | Where-Object { $_.head_sha -eq $Sha -and $_.path -like "$CiWorkflow*" } |
                Sort-Object { [int64]$_.run_number } -Descending)[0]
        if ($run -and $run.status -eq 'completed') {
            if ($run.conclusion -eq 'success') {
                Write-Host 'CI 通过。' -ForegroundColor Green
                return
            }
            Stop-Publish "CI 没有通过（$($run.conclusion)）：$($run.html_url)" '修好并提交后再双击一次。这次没有打标签，也没有发布到 crates.io。'
        }
        if (-not $run -and $minutes -ge 10) {
            Stop-Publish "等了 10 分钟，CI 还没有开始：https://github.com/$GitHubRepo/actions" '确认 CI 跑起来之后再双击一次。'
        }
        if ($minutes -ge 180) {
            Stop-Publish "等了 3 小时，CI 还没有跑完：$($run.html_url)" '跑完之后再双击一次。'
        }
        if ($run) {
            Write-Host "CI 还在跑（$($run.status)），已等 $minutes 分钟：$($run.html_url)"
        } else {
            Write-Host "等 CI 开始，已等 $minutes 分钟。"
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

# ---------------------------------------------------------------------------------------
Write-Host 'Tabsmith：push 到 GitHub，发布到 crates.io' -ForegroundColor Cyan

foreach ($tool in 'git', 'cargo') {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { Stop-Publish "找不到 $tool。" }
}
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location -LiteralPath $root
$top = (Invoke-Git rev-parse --show-toplevel) -replace '/', '\'
if ($top -ine $root.TrimEnd('\')) { Stop-Publish "$root 不是仓库的根目录。" }

Write-Step '检查本地仓库'
$current = Invoke-Git rev-parse --abbrev-ref HEAD
if ($current -ne $Branch) { Stop-Publish "当前分支是 $current，不是 $Branch。" "切回 $Branch 再双击。" }
$uncommitted = @(Invoke-Git status --porcelain)
if ($uncommitted.Count -gt 0) {
    $uncommitted | Select-Object -First 30 | ForEach-Object { Write-Host "  $_" }
    if ($uncommitted.Count -gt 30) { Write-Host "  ……共 $($uncommitted.Count) 项" }
    Stop-Publish '还有没提交的改动。' '审查并提交之后再双击。这个程序不会替你提交。'
}
$url = Invoke-Git remote get-url $Remote
if ($url -notmatch "$([regex]::Escape($GitHubRepo))(\.git)?/?$") {
    Stop-Publish "远端 $Remote 指向 $url，不是 $GitHubRepo。"
}
& git fetch $Remote --tags
if ($LASTEXITCODE -ne 0) { Stop-Publish '读取 GitHub 失败。' '看上面 git 的错误信息；可能是网络或标签冲突。' }
$behind, $ahead = (Invoke-Git rev-list --left-right --count "$Remote/$Branch...$Branch") -split '\s+' | ForEach-Object { [int]$_ }
if ($behind -gt 0) {
    Stop-Publish "GitHub 上有 $behind 个提交是本地没有的。" '先把它们拉下来再双击。这个程序从不强制 push。'
}
$head = Invoke-Git rev-parse HEAD

Write-Step '读取版本和 crates.io'
$metadata = & cargo metadata --no-deps --format-version 1 --locked
if ($LASTEXITCODE -ne 0) { Stop-Publish 'cargo metadata 失败。' '如果刚改过版本号，先运行 cargo update --workspace 并提交 Cargo.lock。' }
$crates = @(($metadata -join "`n" | ConvertFrom-Json).packages | Where-Object { $null -eq $_.publish -or @($_.publish).Count -gt 0 })
$versions = @($crates | ForEach-Object { $_.version } | Sort-Object -Unique)
if ($versions.Count -ne 1) { Stop-Publish "各 crate 的版本不一致：$($versions -join '、')。" }
$version = $versions[0]
$missing = New-Object System.Collections.ArrayList
$published = New-Object System.Collections.ArrayList
try {
    foreach ($crate in $crates) {
        if (Test-Published $crate.name $version) { [void]$published.Add($crate.name) } else { [void]$missing.Add($crate.name) }
    }
} catch {
    Stop-Publish "读取 crates.io 失败：$($_.Exception.Message)" '网络恢复后再双击一次。'
}
$newVersion = $missing.Count -gt 0
$tag = "v$version"

if ($newVersion) {
    $changelog = Get-Content -LiteralPath 'CHANGELOG.md' -Raw -Encoding UTF8
    if ($changelog -notmatch "(?m)^## \[$([regex]::Escape($version))\]") {
        Stop-Publish "CHANGELOG.md 里没有 $version 这一节，Release 工作流要从那里读发布说明。" '把 Unreleased 的条目移到新版本下，提交后再双击。'
    }
    $cargoHome = if ($env:CARGO_HOME) { $env:CARGO_HOME } else { Join-Path $env:USERPROFILE '.cargo' }
    if (-not $env:CARGO_REGISTRY_TOKEN -and -not (Test-Path -LiteralPath (Join-Path $cargoHome 'credentials.toml'))) {
        Stop-Publish '还没有登录 crates.io。' '先在终端运行一次 cargo login，再双击。'
    }
    $tagCommit = & git rev-parse -q --verify "refs/tags/$tag^{commit}"
    $global:LASTEXITCODE = 0
    if ($tagCommit -and $tagCommit -ne $head) {
        Stop-Publish "标签 $tag 已经指向另一个提交（$tagCommit）。" '这个版本号已经打过标签。发布新的提交要换一个版本号。'
    }
    $tagOnRemote = @(Invoke-Git ls-remote --tags $Remote "refs/tags/$tag").Count -gt 0
}

# ---------------------------------------------------------------------------------------
if ($ahead -eq 0 -and -not $newVersion) {
    Write-Host ''
    Write-Host "GitHub 已是最新，crates.io 上也已有 $version。没有要做的事。" -ForegroundColor Green
    exit 0
}

Write-Step '这次会做的事'
Write-Host "GitHub（https://github.com/$GitHubRepo）"
if ($ahead -gt 0) {
    Write-Host "  push $Branch，$ahead 个提交："
    Invoke-Git log --max-count=20 '--format=    %h %s' "$Remote/$Branch..$Branch" | ForEach-Object { Write-Host $_ }
    if ($ahead -gt 20) { Write-Host '    ……' }
} else {
    Write-Host "  $Branch 已是最新。"
}
if ($newVersion) {
    if (-not $tagOnRemote) { Write-Host "  打标签 $tag 并 push，这会启动 Release 工作流，构建各平台的二进制。" }
    Write-Host 'crates.io'
    Write-Host "  发布 $version：$($missing -join '、')"
    if ($published.Count -gt 0) { Write-Host "  已经发布，跳过：$($published -join '、')" }
    Write-Host '  先在本地校验，再等这个提交的 CI 通过，都通过才发布。发布后不能撤销。' -ForegroundColor Yellow
} else {
    Write-Host 'crates.io'
    Write-Host "  上面已有 $version，这次不发布。要发布新版本，先改版本号，并把 CHANGELOG 的条目移到新版本下。"
}

if ($DryRun) {
    Write-Host ''
    Write-Host '-DryRun：到此为止，什么都没有改。'
    exit 0
}
if (-not $Yes) {
    Write-Host ''
    if ((Read-Host '按回车开始，输入 n 取消') -match '^\s*[nN]') {
        Write-Host '已取消，什么都没有改。'
        exit 0
    }
}

$exclude = @($published | ForEach-Object { '--exclude'; $_ })

if ($newVersion) {
    Write-Step '在本地校验打包（第一次可能要几分钟）'
    # Cargo unpacks the packages it verifies under a folder whose name starts with "-", and
    # does not replace them when the same version is packaged again with different code.
    Get-ChildItem -LiteralPath (Join-Path $cargoHome 'registry\src') -Directory -Filter '-*' -ErrorAction SilentlyContinue | ForEach-Object {
        foreach ($name in $missing) {
            $old = Join-Path $_.FullName "$name-$version"
            if (Test-Path -LiteralPath $old) { Remove-Item -LiteralPath $old -Recurse -Force }
        }
    }
    & cargo publish --workspace --locked --dry-run @exclude
    if ($LASTEXITCODE -ne 0) { Stop-Publish '本地校验没有通过。' '看上面 cargo 的错误信息，修好并提交后再双击。' }
    [void]$Done.Add('本地校验通过')
}

if ($ahead -gt 0) {
    Write-Step "push $Branch 到 GitHub"
    & git push $Remote $Branch
    if ($LASTEXITCODE -ne 0) { Stop-Publish 'git push 失败。' '看上面 git 的错误信息。' }
    [void]$Done.Add("已 push $Branch")
}

if ($newVersion) {
    Write-Step '等 CI 通过（通常几分钟，慢的时候要一小时）'
    Wait-ForCi $head
    [void]$Done.Add('CI 通过')

    Write-Step "标签 $tag"
    if (-not $tagCommit) {
        & git tag -a $tag -m "Tabsmith $version"
        if ($LASTEXITCODE -ne 0) { Stop-Publish "打标签 $tag 失败。" }
    }
    if (-not $tagOnRemote) {
        & git push $Remote "refs/tags/$tag"
        if ($LASTEXITCODE -ne 0) { Stop-Publish "push 标签 $tag 失败。" '再双击一次会接着做。' }
    }
    [void]$Done.Add("标签 $tag 已在 GitHub 上")

    Write-Step "发布到 crates.io：$($missing -join '、')"
    & cargo publish --workspace --locked @exclude
    if ($LASTEXITCODE -ne 0) {
        Stop-Publish '发布没有全部完成。' '等一会儿再双击一次，已发布的 crate 会跳过。如果是 429 限流，错误信息里有可以重试的时间。'
    }
    [void]$Done.Add('crates.io 发布完成')
}

Write-Host ''
Write-Host '全部完成。' -ForegroundColor Green
if ($ahead -gt 0) { Write-Host "  CI 和 Pages：https://github.com/$GitHubRepo/actions" }
if ($newVersion) {
    Write-Host "  GitHub Release：https://github.com/$GitHubRepo/releases/tag/$tag"
    Write-Host "  crates.io：https://crates.io/crates/tabsmith-cli/$version"
}
exit 0
