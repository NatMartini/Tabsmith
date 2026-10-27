# Install the prebuilt `rustpy` command-line tool from a GitHub release (Windows x86_64).
#
#   irm https://github.com/__OWNER__/__REPO__/releases/latest/download/install.ps1 | iex
#
# Environment variables:
#   RUSTPY_VERSION         release to install, e.g. 0.1.0 (default: the latest release)
#   RUSTPY_INSTALL_DIR     directory for rustpy.exe (default: %LOCALAPPDATA%\rustpy\bin)
#   RUSTPY_NO_MODIFY_PATH  set to 1 to leave the user PATH unchanged
#
# The archive's SHA-256 is checked against the .sha256 file published with it.
# On Linux and macOS use install.sh. Elsewhere: cargo install rustpy-cli --locked
#
# Everything runs inside one script block so that `irm ... | iex` leaves no variables
# or functions behind in your session.

& {
    Set-StrictMode -Version 3.0
    $ErrorActionPreference = 'Stop'
    $ProgressPreference = 'SilentlyContinue'   # the progress bar makes downloads very slow in Windows PowerShell

    # The GitHub repository that publishes the releases (filled in by tools/release/set-repo.sh).
    $RustpyRepo = if ($env:RUSTPY_REPO) { $env:RUSTPY_REPO } else { '__OWNER__/__REPO__' }

    $FromSource = 'Install from source instead: cargo install rustpy-cli --locked (needs Rust, https://rustup.rs)'

    function Write-Step([string] $Message) {
        Write-Host "rustpy-install: $Message"
    }

    function Get-RustpyTarget {
        $arch = $null
        try { $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture.ToString() } catch { }
        if (-not $arch) {
            $arch = if ($env:PROCESSOR_ARCHITEW6432) { $env:PROCESSOR_ARCHITEW6432 } else { $env:PROCESSOR_ARCHITECTURE }
        }
        switch ($arch) {
            { $_ -in @('X64', 'AMD64') } { return 'x86_64-pc-windows-msvc' }
            { $_ -in @('Arm64', 'ARM64') } {
                Write-Step 'no native ARM64 build yet; installing the x86_64 build, which Windows 11 runs under emulation'
                return 'x86_64-pc-windows-msvc'
            }
            default { throw "no prebuilt binary for this CPU architecture ($arch). $FromSource" }
        }
    }

    function Get-RustpyTag {
        if ($env:RUSTPY_VERSION) {
            return 'v' + ($env:RUSTPY_VERSION -replace '^v', '')
        }
        $api = "https://api.github.com/repos/$RustpyRepo/releases/latest"
        try {
            $release = Invoke-RestMethod -Uri $api -UseBasicParsing -UserAgent 'rustpy-install'
        } catch {
            throw "could not look up the latest release at $api ($($_.Exception.Message)). Set `$env:RUSTPY_VERSION to choose a version. $FromSource"
        }
        $tagName = $null
        if ($release -and $release.PSObject.Properties['tag_name']) { $tagName = [string] $release.tag_name }
        if (-not $tagName) { throw "no published release found for $RustpyRepo. $FromSource" }
        return $tagName
    }

    # Adds $Dir to the user PATH in the registry, keeping unexpanded entries such as
    # %USERPROFILE%\bin intact. Returns $false when it is already there.
    function Add-UserPath([string] $Dir) {
        $key = Get-Item -Path 'HKCU:\Environment'
        $current = [string] $key.GetValue('Path', '', 'DoNotExpandEnvironmentNames')
        $entries = @($current -split ';' | Where-Object { $_ -ne '' })
        foreach ($entry in $entries) {
            if ([Environment]::ExpandEnvironmentVariables($entry).TrimEnd('\') -ieq $Dir.TrimEnd('\')) { return $false }
        }
        Set-ItemProperty -Path 'HKCU:\Environment' -Name 'Path' -Value ((@($entries) + $Dir) -join ';') -Type ExpandString
        # Setting a variable through .NET broadcasts WM_SETTINGCHANGE, so new terminals see the new PATH.
        [Environment]::SetEnvironmentVariable('RUSTPY_INSTALL_REFRESH', '1', 'User')
        [Environment]::SetEnvironmentVariable('RUSTPY_INSTALL_REFRESH', $null, 'User')
        return $true
    }

    function Install-Rustpy {
        if ($RustpyRepo -match '__OWNER__|__REPO__') {
            throw "this copy of install.ps1 has no GitHub repository configured (RUSTPY_REPO=$RustpyRepo)"
        }
        if ($PSVersionTable.PSEdition -eq 'Core' -and -not $IsWindows) {
            throw "install.ps1 is for Windows. On Linux and macOS use: curl -fsSL https://github.com/$RustpyRepo/releases/latest/download/install.sh | sh"
        }
        try {
            [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
        } catch { }

        $installDir = if ($env:RUSTPY_INSTALL_DIR) {
            $env:RUSTPY_INSTALL_DIR
        } elseif ($env:LOCALAPPDATA) {
            Join-Path $env:LOCALAPPDATA 'rustpy\bin'
        } else {
            throw 'LOCALAPPDATA is not set; set RUSTPY_INSTALL_DIR to choose where to install'
        }

        $target = Get-RustpyTarget
        $tag = Get-RustpyTag
        $version = $tag -replace '^v', ''
        $name = "rustpy-$version-$target"
        $url = "https://github.com/$RustpyRepo/releases/download/$tag/$name.zip"

        $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('rustpy-install-' + [System.Guid]::NewGuid().ToString('N'))
        New-Item -ItemType Directory -Path $tmp | Out-Null
        try {
            $zip = Join-Path $tmp "$name.zip"
            Write-Step "downloading rustpy $version for $target"
            try {
                Invoke-WebRequest -Uri $url -OutFile $zip -UseBasicParsing
            } catch {
                throw "could not download $url ($($_.Exception.Message)). Does release $tag exist and include ${target}?"
            }
            try {
                Invoke-WebRequest -Uri "$url.sha256" -OutFile "$zip.sha256" -UseBasicParsing
            } catch {
                throw "could not download the checksum $url.sha256 ($($_.Exception.Message))"
            }

            $expected = ((Get-Content -LiteralPath "$zip.sha256" -Raw).Trim() -split '\s+')[0].ToLowerInvariant()
            if ($expected -notmatch '^[0-9a-f]{64}$') { throw "malformed checksum file $name.zip.sha256" }
            $actual = (Get-FileHash -LiteralPath $zip -Algorithm SHA256).Hash.ToLowerInvariant()
            if ($actual -ne $expected) {
                throw "checksum mismatch for $name.zip: expected $expected, got $actual. Nothing was installed."
            }
            Write-Step 'checksum verified (sha256)'

            Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
            $exe = Join-Path (Join-Path $tmp $name) 'rustpy.exe'
            if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) { throw "the archive does not contain $name\rustpy.exe" }

            # Native stderr must not turn into a terminating error here.
            $ErrorActionPreference = 'Continue'
            try {
                $probe = (& $exe --version 2>&1 | Out-String).Trim()
                $probeCode = $LASTEXITCODE
            } catch {
                $probe = $_.Exception.Message
                $probeCode = 1
            }
            $ErrorActionPreference = 'Stop'
            if ($probeCode -ne 0) { throw "the prebuilt binary does not run on this system: $probe. $FromSource" }

            New-Item -ItemType Directory -Force -Path $installDir | Out-Null
            $dest = Join-Path $installDir 'rustpy.exe'
            if (Test-Path -LiteralPath $dest) {
                # A running rustpy.exe cannot be overwritten, but it can be renamed.
                Move-Item -LiteralPath $dest -Destination "$dest.old" -Force
            }
            Copy-Item -LiteralPath $exe -Destination $dest -Force
            Remove-Item -LiteralPath "$dest.old" -Force -ErrorAction SilentlyContinue
        } finally {
            Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
        }

        Write-Step "installed $probe to $dest"
        if ($env:RUSTPY_NO_MODIFY_PATH -eq '1') {
            Write-Step "RUSTPY_NO_MODIFY_PATH=1: add $installDir to your PATH yourself"
        } elseif (Add-UserPath $installDir) {
            $env:Path = "$env:Path;$installDir"
            Write-Step "added $installDir to your user PATH; open a new terminal to use rustpy there"
        }
        Write-Step "get started: rustpy --help   (to uninstall, delete $dest)"
    }

    try {
        Install-Rustpy
    } catch {
        Write-Host "rustpy-install: error: $($_.Exception.Message)" -ForegroundColor Red
        # Run as a script file: report failure through the exit code. Run through
        # `irm | iex`: just return, because `exit` would close the user's PowerShell window.
        if ($PSCommandPath) { exit 1 }
    }
}
