#!/bin/sh
# Install the prebuilt `rustpy` command-line tool from a GitHub release.
#
#   curl -fsSL https://github.com/__OWNER__/__REPO__/releases/latest/download/install.sh | sh
#
# Environment variables:
#   RUSTPY_VERSION      release to install, e.g. 0.1.0 (default: the latest release)
#   RUSTPY_INSTALL_DIR  directory for the binary (default: ~/.local/bin)
#
# Prebuilt binaries exist for Linux (x86_64, aarch64; glibc) and macOS (x86_64,
# arm64). On other systems, build from source: cargo install rustpy-cli --locked
#
# The archive's SHA-256 is checked against the .sha256 file published with it.

set -eu

# The GitHub repository that publishes the releases (filled in by tools/release/set-repo.sh).
RUSTPY_REPO="${RUSTPY_REPO:-__OWNER__/__REPO__}"

FROM_SOURCE="Install from source instead: cargo install rustpy-cli --locked (needs Rust, https://rustup.rs)"

say() {
    printf 'rustpy-install: %s\n' "$1"
}

err() {
    printf 'rustpy-install: error: %s\n' "$1" >&2
    exit 1
}

usage() {
    cat <<EOF
Install the prebuilt rustpy binary from https://github.com/$RUSTPY_REPO/releases

Usage: install.sh [-h|--help]

Environment variables:
  RUSTPY_VERSION      release to install, e.g. 0.1.0 (default: latest)
  RUSTPY_INSTALL_DIR  directory for the binary (default: ~/.local/bin)
EOF
}

has() {
    command -v "$1" > /dev/null 2>&1
}

# Sets $target to the Rust target triple of a published archive, or exits.
detect_target() {
    os=$(uname -s)
    arch=$(uname -m)
    case "$os" in
        Linux)
            os_part=unknown-linux-gnu
            # musl's ldd prints "musl libc" in its version banner.
            if has ldd && ldd --version 2>&1 | grep -qi musl; then
                err "musl-based Linux (e.g. Alpine) has no prebuilt binary. $FROM_SOURCE"
            fi
            ;;
        Darwin)
            os_part=apple-darwin
            # A shell running under Rosetta 2 reports x86_64 on Apple Silicon; prefer the native build.
            if [ "$arch" = x86_64 ] && [ "$(sysctl -n sysctl.proc_translated 2> /dev/null || true)" = 1 ]; then
                arch=arm64
            fi
            ;;
        MINGW* | MSYS* | CYGWIN* | Windows_NT)
            err "on Windows, run install.ps1 in PowerShell: irm https://github.com/$RUSTPY_REPO/releases/latest/download/install.ps1 | iex"
            ;;
        *)
            err "no prebuilt binary for this operating system ($os). $FROM_SOURCE"
            ;;
    esac
    case "$arch" in
        x86_64 | amd64) arch_part=x86_64 ;;
        aarch64 | arm64) arch_part=aarch64 ;;
        *) err "no prebuilt binary for this CPU architecture ($arch on $os). $FROM_SOURCE" ;;
    esac
    target="$arch_part-$os_part"
}

# download URL FILE
download() {
    if has curl; then
        curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location --retry 3 --output "$2" "$1"
    else
        wget --quiet --output-document="$2" "$1"
    fi
}

# Sets $tag from RUSTPY_VERSION or the latest GitHub release, or exits.
resolve_tag() {
    if [ -n "${RUSTPY_VERSION:-}" ]; then
        tag="v${RUSTPY_VERSION#v}"
        return
    fi
    latest="https://github.com/$RUSTPY_REPO/releases/latest"
    if has curl; then
        # github.com/OWNER/REPO/releases/latest redirects to .../releases/tag/<tag>.
        final=$(curl --proto '=https' --tlsv1.2 --fail --silent --show-error --location --head \
            --output /dev/null --write-out '%{url_effective}' "$latest") ||
            err "could not look up the latest release at $latest. Set RUSTPY_VERSION=<version> to choose one."
        case "$final" in
            */releases/tag/*) tag=${final##*/releases/tag/} ;;
            *) tag= ;;
        esac
    else
        api="https://api.github.com/repos/$RUSTPY_REPO/releases/latest"
        json=$(wget --quiet --output-document=- "$api") ||
            err "could not look up the latest release at $api. Set RUSTPY_VERSION=<version> to choose one."
        tag=$(printf '%s\n' "$json" | sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -n 1)
    fi
    [ -n "$tag" ] || err "no published release found for $RUSTPY_REPO. $FROM_SOURCE"
}

# verify FILE SHA256FILE
verify() {
    expected=$(awk '{ print tolower($1); exit }' "$2")
    if has sha256sum; then
        actual=$(sha256sum "$1" | awk '{ print $1 }')
    elif has shasum; then
        actual=$(shasum -a 256 "$1" | awk '{ print $1 }')
    elif has openssl; then
        actual=$(openssl dgst -sha256 "$1" | awk '{ print $NF }')
    else
        err "need sha256sum, shasum or openssl to verify the download"
    fi
    case "$expected" in
        *[!0-9a-f]* | "") err "malformed checksum file $(basename "$2")" ;;
    esac
    [ "${#expected}" -eq 64 ] || err "malformed checksum file $(basename "$2")"
    [ "$expected" = "$actual" ] ||
        err "checksum mismatch for $(basename "$1"): expected $expected, got $actual. Nothing was installed."
}

main() {
    case "${1:-}" in
        "") ;;
        -h | --help)
            usage
            exit 0
            ;;
        *) err "unknown argument: $1 (configuration is read from environment variables; see --help)" ;;
    esac

    case "$RUSTPY_REPO" in
        *__OWNER__* | *__REPO__*) err "this copy of install.sh has no GitHub repository configured (RUSTPY_REPO=$RUSTPY_REPO)" ;;
    esac

    for tool in uname tar awk sed mktemp chmod mkdir cp mv; do
        has "$tool" || err "required command not found: $tool"
    done
    has curl || has wget || err "need curl or wget to download rustpy"

    if [ -n "${RUSTPY_INSTALL_DIR:-}" ]; then
        install_dir=$RUSTPY_INSTALL_DIR
    else
        [ -n "${HOME:-}" ] || err "HOME is not set; set RUSTPY_INSTALL_DIR to choose where to install"
        install_dir="$HOME/.local/bin"
    fi

    detect_target
    resolve_tag
    version=${tag#v}
    name="rustpy-$version-$target"
    url="https://github.com/$RUSTPY_REPO/releases/download/$tag/$name.tar.gz"

    tmp=$(mktemp -d 2> /dev/null || mktemp -d -t rustpy-install)
    trap 'rm -rf "$tmp"' EXIT
    trap 'exit 130' INT TERM

    say "downloading rustpy $version for $target"
    download "$url" "$tmp/$name.tar.gz" ||
        err "could not download $url (does release $tag exist and include $target?)"
    download "$url.sha256" "$tmp/$name.tar.gz.sha256" ||
        err "could not download the checksum $url.sha256"
    verify "$tmp/$name.tar.gz" "$tmp/$name.tar.gz.sha256"
    say "checksum verified (sha256)"

    tar -xzf "$tmp/$name.tar.gz" -C "$tmp" || err "could not extract $name.tar.gz (needs tar with gzip support)"
    bin="$tmp/$name/rustpy"
    [ -f "$bin" ] || err "the archive does not contain $name/rustpy"
    chmod 755 "$bin"
    if ! "$bin" --version > /dev/null 2>&1; then
        detail=$("$bin" --version 2>&1 | head -n 1 || true)
        err "the prebuilt binary does not run on this system: $detail. On Linux this usually means the system glibc is older than the one it was built against. $FROM_SOURCE"
    fi

    mkdir -p "$install_dir" || err "cannot create $install_dir; set RUSTPY_INSTALL_DIR to a writable directory"
    # Copy next to the destination, then rename, so a running rustpy is never half-overwritten.
    cp "$bin" "$install_dir/.rustpy.tmp.$$" || err "cannot write to $install_dir; set RUSTPY_INSTALL_DIR to a writable directory"
    mv -f "$install_dir/.rustpy.tmp.$$" "$install_dir/rustpy"

    say "installed $("$install_dir/rustpy" --version) to $install_dir/rustpy"
    case ":$PATH:" in
        *":$install_dir:"*) ;;
        *)
            say "$install_dir is not on your PATH. Add it in your shell profile (~/.profile, ~/.bashrc or ~/.zshrc):"
            # shellcheck disable=SC2016 # $PATH is meant literally: the user's shell expands it.
            printf '\n    export PATH="%s:$PATH"\n\n' "$install_dir"
            ;;
    esac
    say "get started: rustpy --help   (to uninstall, delete $install_dir/rustpy)"
}

main "$@"
