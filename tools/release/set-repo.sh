#!/bin/sh
# Point the project at its GitHub repository, once the owner and name are decided.
#
#   tools/release/set-repo.sh OWNER/REPO [HOMEPAGE_URL]
#
# - Replaces the placeholder OWNER/REPO token (written __OWNER__ / __REPO__ joined by a
#   slash) in every tracked or new text file: install.sh, install.ps1, the
#   cargo-binstall metadata, Markdown badges and links, issue templates, ...
# - Adds `repository` and `homepage` to [workspace.package] in Cargo.toml (homepage
#   defaults to the repository URL) and makes every crate inherit them.
#
# Safe to run again: a second run with the same arguments changes nothing. It refuses
# to switch an already configured project to a different repository.

set -eu

usage() {
    echo "usage: $0 OWNER/REPO [HOMEPAGE_URL]" >&2
    exit 2
}

fail() {
    echo "set-repo: error: $1" >&2
    exit 1
}

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    usage
fi
repo=$1
if ! printf '%s\n' "$repo" | grep -Eq '^[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?/[A-Za-z0-9._-]+$'; then
    fail "'$repo' is not a GitHub OWNER/REPO name"
fi
case "$repo" in
    */. | */..) fail "'$repo' is not a GitHub OWNER/REPO name" ;;
esac
url="https://github.com/$repo"
homepage=${2:-$url}
case "$homepage" in
    https://*) ;;
    *) fail "HOMEPAGE_URL must start with https://" ;;
esac
case "$homepage" in
    *[\"\\\|\&[:space:]]*) fail "HOMEPAGE_URL contains characters that are not allowed here" ;;
esac

root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
if ! grep -q '^\[workspace\.package\]' Cargo.toml 2> /dev/null; then
    fail "no [workspace.package] in $root/Cargo.toml"
fi

# Spelled in pieces so that this script does not contain (and rewrite) the token itself.
placeholder="__OWNER__""/""__REPO__"
self="tools/release/set-repo.sh"
changes=0

# Value of KEY in [workspace.package] of Cargo.toml (empty when absent).
workspace_field() {
    awk -v key="$1" '
        /^\[/ { in_ws = ($0 == "[workspace.package]"); next }
        in_ws && $0 ~ ("^" key "[ \t]*=") {
            sub(/^[^=]*=[ \t]*"/, ""); sub(/"[ \t]*$/, ""); print; exit
        }
    ' Cargo.toml
}

# Check everything before changing anything.
current_repo=$(workspace_field repository)
current_home=$(workspace_field homepage)
if [ -n "$current_repo" ] && [ "$current_repo" != "$url" ]; then
    fail "Cargo.toml already has repository = \"$current_repo\". Revert the earlier run (e.g. git checkout) before switching to $url."
fi
if [ -n "$current_home" ] && [ "$current_home" != "$homepage" ]; then
    fail "Cargo.toml already has homepage = \"$current_home\"; edit it by hand to change it."
fi
grep -q '^license[[:blank:]]*=' Cargo.toml || fail "no license line in [workspace.package] to insert after"

# Rewrites FILE with the output of the remaining arguments (a filter reading FILE).
rewrite() {
    file=$1
    shift
    "$@" < "$file" > "$file.set-repo.tmp"
    cat "$file.set-repo.tmp" > "$file"
    rm -f "$file.set-repo.tmp"
}

# 1. The placeholder in text files.
list_files() {
    if command -v git > /dev/null 2>&1 && git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        git -c core.quotePath=false ls-files --cached --others --exclude-standard
    else
        find . \( -name .git -o -name target -o -name .claude -o -name node_modules \) -prune -o -type f -print |
            sed 's|^\./||'
    fi
}
files=$(list_files | sort -u)
old_ifs=$IFS
IFS='
'
set -f # no globbing of file names
for f in $files; do
    if [ ! -f "$f" ] || [ "$f" = "$self" ]; then
        continue
    fi
    grep -qIF -e "$placeholder" "$f" 2> /dev/null || continue
    n=$(awk -v p="$placeholder" '{ n += gsub(p, "") } END { print n + 0 }' "$f")
    rewrite "$f" sed "s|$placeholder|$repo|g"
    echo "updated $f: $n placeholder(s) -> $repo"
    changes=$((changes + 1))
done
set +f
IFS=$old_ifs

# 2. repository / homepage in [workspace.package].
insert_workspace_field() {
    # shellcheck disable=SC2016 # $0 belongs to awk
    rewrite Cargo.toml awk -v line="$1 = \"$2\"" '
        { print }
        /^\[/ { in_ws = ($0 == "[workspace.package]") }
        in_ws && !done && /^license[ \t]*=/ { print line; done = 1 }
    '
}
# Each is inserted right after `license`, so homepage goes first to end up second.
if [ -z "$current_home" ]; then
    insert_workspace_field homepage "$homepage"
    echo "updated Cargo.toml: [workspace.package] homepage = \"$homepage\""
    changes=$((changes + 1))
fi
if [ -z "$current_repo" ]; then
    insert_workspace_field repository "$url"
    echo "updated Cargo.toml: [workspace.package] repository = \"$url\""
    changes=$((changes + 1))
fi
if [ "$(workspace_field repository)" != "$url" ] || [ "$(workspace_field homepage)" != "$homepage" ]; then
    fail "could not set repository/homepage in Cargo.toml"
fi

# 3. Every crate inherits them.
for manifest in crates/*/Cargo.toml; do
    [ -f "$manifest" ] || continue
    grep -q '^license\.workspace[[:blank:]]*=[[:blank:]]*true' "$manifest" || {
        echo "warning: $manifest does not inherit license from the workspace; add repository/homepage by hand" >&2
        continue
    }
    for key in homepage repository; do
        grep -q "^$key\.workspace[[:blank:]]*=" "$manifest" && continue
        rewrite "$manifest" awk -v line="$key.workspace = true" '
            { print }
            !done && /^license\.workspace[ \t]*=/ { print line; done = 1 }
        '
        echo "updated $manifest: $key.workspace = true"
        changes=$((changes + 1))
    done
done

left=$(list_files | while IFS= read -r f; do
    [ -f "$f" ] && [ "$f" != "$self" ] && grep -lIF -e "$placeholder" "$f" 2> /dev/null
done || true)
[ -z "$left" ] || fail "placeholder still present in: $left"

if [ "$changes" -eq 0 ]; then
    echo "nothing to change: already set to $repo"
else
    echo "done: $changes change(s). Review with 'git diff' and commit."
fi
