#!/bin/bash
# Pre-release version check for SMG workspace crates.
#
# For each workspace crate, verifies:
#   1. Whether there are code changes since the latest git tag
#   2. Whether the crate version was bumped in its own Cargo.toml
#   3. Whether the workspace root Cargo.toml reflects the new version
#
# Detects bump level from conventional commits:
#   - feat!: or BREAKING CHANGE → major
#   - feat: → minor
#   - fix:, refactor:, perf:, etc. → patch
#
# After the check, offers to auto-bump any unbumped crates.
#
# Usage: ./check_release_versions.sh [tag]
#        If no tag is given, the latest tag is used.
#
# Exit code 0 = all good, 1 = issues found (user declined fix).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Portable sed in-place: macOS uses `sed -i ''`, GNU uses `sed -i`
sed_inplace() {
    if [[ "$(uname)" == "Darwin" ]]; then
        sed -i '' "$@"
    else
        sed -i "$@"
    fi
}

# Escape dots in a version string for use in sed regex
escape_version() {
    echo "$1" | sed 's/\./\\./g'
}

# ---------------------------------------------------------------------------
# Workspace crate registry
# Format: "crate_name|directory|workspace_dep_key"
# workspace_dep_key is the key in root Cargo.toml [workspace.dependencies].
# Use "-" for the main gateway crate (no workspace dep entry).
# ---------------------------------------------------------------------------
CRATES=(
    "openai-protocol|protocols|openai-protocol"
    "reasoning-parser|reasoning_parser|reasoning-parser"
    "tool-parser|tool_parser|tool-parser"
    "wfaas|workflow|wfaas"
    "llm-tokenizer|tokenizer|llm-tokenizer"
    "smg-auth|auth|smg-auth"
    "smg-mcp|mcp|smg-mcp"
    "kv-index|kv_index|kv-index"
    "data-connector|data_connector|smg-data-connector"
    "llm-multimodal|multimodal|llm-multimodal"
    "smg-wasm|wasm|smg-wasm"
    "smg-mesh|mesh|smg-mesh"
    "smg-grpc-client|grpc_client|smg-grpc-client"
    "smg|model_gateway|-"
)

# ---------------------------------------------------------------------------
# Resolve tag
# ---------------------------------------------------------------------------
if [[ $# -ge 1 ]]; then
    TAG="$1"
else
    TAG=$(git tag --sort=-creatordate 2>/dev/null | head -1)
    if [[ -z "$TAG" ]]; then
        echo -e "${RED}No tags found in repository. Pass a tag explicitly.${NC}"
        echo "Usage: $0 [tag]"
        exit 1
    fi
fi

if ! git rev-parse "$TAG" >/dev/null 2>&1; then
    echo -e "${RED}Tag '$TAG' does not exist.${NC}"
    exit 1
fi

echo -e "${BOLD}Checking workspace versions against tag: ${BLUE}$TAG${NC}"
echo ""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Extract version from a Cargo.toml (first version = line in [package])
get_crate_version() {
    local file="$1"
    if grep -qE 'version\.workspace\s*=\s*true|version\s*=\s*\{\s*workspace\s*=\s*true' "$file"; then
        echo -e "${RED}ERROR: $file uses workspace versioning; this script expects explicit version strings.${NC}" >&2
        exit 1
    fi
    grep -m1 '^version' "$file" | sed 's/.*"\(.*\)".*/\1/'
}

# Extract version at a specific git ref (returns empty string if crate missing)
get_crate_version_at_ref() {
    local path="$1"
    local ref="$2"
    local content
    content=$(git show "$ref:$path/Cargo.toml" 2>/dev/null) || return 0
    if echo "$content" | grep -qE 'version\.workspace\s*=\s*true|version\s*=\s*\{\s*workspace\s*=\s*true'; then
        echo -e "${RED}ERROR: $path/Cargo.toml at $ref uses workspace versioning; this script expects explicit version strings.${NC}" >&2
        exit 1
    fi
    echo "$content" | grep -m1 '^version' | sed 's/.*"\(.*\)".*/\1/'
}

# Extract workspace dep version from root Cargo.toml
get_workspace_dep_version() {
    local dep_key="$1"
    local root_toml="$REPO_ROOT/Cargo.toml"
    grep "^${dep_key} " "$root_toml" 2>/dev/null \
        | grep -o 'version = "[^"]*"' \
        | sed 's/version = "\(.*\)"/\1/'
}

# Detect semver bump level from conventional commits touching a directory.
# Scans commit subjects and bodies for:
#   - "feat!:", "fix!:", "<type>!:" or "BREAKING CHANGE" → major
#   - "feat:" or "feat(<scope>):" → minor
#   - everything else → patch
# Returns: "major", "minor", or "patch"
detect_bump_level() {
    local path="$1"
    local level="patch"

    # Get commit hashes that touch this path
    local commits
    commits=$(git log "$TAG"..HEAD --format='%H' --no-merges -- "$path/")
    if [[ -z "$commits" ]]; then
        echo "patch"
        return
    fi

    while IFS= read -r hash; do
        local subject body
        subject=$(git log -1 --format='%s' "$hash")
        body=$(git log -1 --format='%b' "$hash")

        # Check for breaking change indicators
        # 1. Type with ! suffix: feat!:, fix!:, refactor!:, etc.
        if echo "$subject" | grep -qE '^[a-z]+(\([^)]*\))?!:'; then
            echo "major"
            return
        fi
        # 2. BREAKING CHANGE in commit body or footer
        if echo "$body" | grep -q 'BREAKING CHANGE'; then
            echo "major"
            return
        fi

        # Check for feat → minor
        if echo "$subject" | grep -qE '^feat(\([^)]*\))?:'; then
            level="minor"
        fi
    done <<< "$commits"

    echo "$level"
}

# Bump version by level: major, minor, or patch
bump_version() {
    local version="$1"
    local level="$2"
    local major minor patch
    IFS='.' read -r major minor patch <<< "$version"
    case "$level" in
        major) echo "$((major + 1)).0.0" ;;
        minor) echo "${major}.$((minor + 1)).0" ;;
        patch) echo "${major}.${minor}.$((patch + 1))" ;;
    esac
}

# Pretty label for bump level
bump_label() {
    case "$1" in
        major) echo -e "${RED}major${NC}" ;;
        minor) echo -e "${YELLOW}minor${NC}" ;;
        patch) echo -e "${CYAN}patch${NC}" ;;
    esac
}

# Update version in a Cargo.toml (first version = line only)
set_crate_version() {
    local file="$1"
    local new_version="$2"
    sed_inplace "0,/^version = \".*\"/s//version = \"${new_version}\"/" "$file"
    if ! grep -q "^version = \"${new_version}\"" "$file"; then
        echo -e "    ${RED}FAILED to update $file${NC}" >&2
        return 1
    fi
}

# Update workspace dep version in root Cargo.toml
set_workspace_dep_version() {
    local dep_key="$1"
    local old_version="$2"
    local new_version="$3"
    local root_toml="$REPO_ROOT/Cargo.toml"
    local escaped_old
    escaped_old=$(escape_version "$old_version")
    sed_inplace "s/^${dep_key} = { version = \"${escaped_old}\"/${dep_key} = { version = \"${new_version}\"/" "$root_toml"
    if ! grep -q "^${dep_key} .* version = \"${new_version}\"" "$root_toml"; then
        echo -e "    ${RED}FAILED to update $dep_key in workspace Cargo.toml to v${new_version}${NC}" >&2
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Phase 1: Check all crates, collect unbumped ones
# ---------------------------------------------------------------------------
issues=0
changed=0
clean=0

# Collect crates that need bumping: "name|path|dep_key|current_version|bump_level"
NEEDS_BUMP=()
# Collect crates with workspace Cargo.toml mismatch: "name|dep_key|crate_version|ws_version|path"
NEEDS_WS_SYNC=()

for entry in "${CRATES[@]}"; do
    IFS='|' read -r name path dep_key <<< "$entry"

    # 1. Check for code changes since tag (exclude Cargo.toml itself)
    diff_count=$(git diff --name-only "$TAG"..HEAD -- "$path/" | grep -cv 'Cargo\.toml$' || true)
    if [[ "$diff_count" -eq 0 ]]; then
        clean=$((clean + 1))
        continue
    fi

    changed=$((changed + 1))
    current_version=$(get_crate_version "$path/Cargo.toml")
    tag_version=$(get_crate_version_at_ref "$path" "$TAG")

    # Handle crate not existing at the tag (new crate)
    if [[ -z "$tag_version" ]]; then
        echo -e "  ${GREEN}✓${NC} ${BOLD}$name${NC} ($path/) — new crate (v$current_version), $diff_count file(s) changed"
        if [[ "$dep_key" != "-" ]]; then
            ws_version=$(get_workspace_dep_version "$dep_key")
            if [[ "$ws_version" != "$current_version" ]]; then
                echo -e "    ${RED}✗ workspace Cargo.toml has $dep_key v$ws_version, expected v$current_version${NC}"
                NEEDS_WS_SYNC+=("$name|$dep_key|$current_version|$ws_version|$path")
                issues=$((issues + 1))
            fi
        fi
        continue
    fi

    # 2. Check if version was bumped
    if [[ "$current_version" == "$tag_version" ]]; then
        level=$(detect_bump_level "$path")
        echo -e "  ${YELLOW}!${NC} ${BOLD}$name${NC} ($path/) — $diff_count file(s) changed but version not bumped (v$current_version) [$(bump_label "$level")]"
        NEEDS_BUMP+=("$name|$path|$dep_key|$current_version|$level")
        issues=$((issues + 1))
        continue
    fi

    echo -e "  ${GREEN}✓${NC} ${BOLD}$name${NC} ($path/) — v$tag_version → v$current_version ($diff_count file(s) changed)"

    # 3. Check workspace root Cargo.toml
    if [[ "$dep_key" != "-" ]]; then
        ws_version=$(get_workspace_dep_version "$dep_key")
        if [[ "$ws_version" != "$current_version" ]]; then
            echo -e "    ${RED}✗ workspace Cargo.toml has $dep_key v$ws_version, expected v$current_version${NC}"
            NEEDS_WS_SYNC+=("$name|$dep_key|$current_version|$ws_version|$path")
            issues=$((issues + 1))
        fi
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}Summary:${NC} $changed crate(s) with changes, $clean unchanged"

if [[ "$issues" -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}All versions consistent.${NC}"
    exit 0
fi

echo -e "${RED}${BOLD}$issues issue(s) found.${NC}"

# ---------------------------------------------------------------------------
# Phase 2: Offer to fix
# ---------------------------------------------------------------------------
total_fixes=$(( ${#NEEDS_BUMP[@]} + ${#NEEDS_WS_SYNC[@]} ))
if [[ "$total_fixes" -eq 0 ]]; then
    exit 1
fi

echo ""
echo -e "${BOLD}Proposed fixes:${NC}"

for entry in "${NEEDS_BUMP[@]}"; do
    IFS='|' read -r name path dep_key current_version level <<< "$entry"
    new_version=$(bump_version "$current_version" "$level")
    echo -e "  $(bump_label "$level") $name v$current_version → v$new_version ($path/Cargo.toml)"
    if [[ "$dep_key" != "-" ]]; then
        echo -e "       sync workspace Cargo.toml $dep_key → v$new_version"
    fi
done

if [[ ${#NEEDS_WS_SYNC[@]} -gt 0 ]]; then
    for entry in "${NEEDS_WS_SYNC[@]}"; do
        IFS='|' read -r name dep_key crate_version ws_version path <<< "$entry"
        echo -e "  ${BLUE}sync${NC} workspace Cargo.toml $dep_key v$ws_version → v$crate_version"
    done
fi

echo ""
read -rp "Apply fixes? [y/N] " answer
if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
    echo "No changes made."
    exit 1
fi

# ---------------------------------------------------------------------------
# Phase 3: Apply fixes
# ---------------------------------------------------------------------------
echo ""
fix_failed=0

for entry in "${NEEDS_BUMP[@]}"; do
    IFS='|' read -r name path dep_key current_version level <<< "$entry"
    new_version=$(bump_version "$current_version" "$level")

    # Bump crate Cargo.toml
    if set_crate_version "$path/Cargo.toml" "$new_version"; then
        echo -e "  ${GREEN}✓${NC} $path/Cargo.toml → v$new_version"
    else
        fix_failed=$((fix_failed + 1))
    fi

    # Sync workspace root Cargo.toml (read actual ws version in case it drifted)
    if [[ "$dep_key" != "-" ]]; then
        local ws_old
        ws_old=$(get_workspace_dep_version "$dep_key")
        if set_workspace_dep_version "$dep_key" "$ws_old" "$new_version"; then
            echo -e "  ${GREEN}✓${NC} Cargo.toml $dep_key → v$new_version"
        else
            fix_failed=$((fix_failed + 1))
        fi
    fi
done

if [[ ${#NEEDS_WS_SYNC[@]} -gt 0 ]]; then
    for entry in "${NEEDS_WS_SYNC[@]}"; do
        IFS='|' read -r name dep_key crate_version ws_version path <<< "$entry"
        if set_workspace_dep_version "$dep_key" "$ws_version" "$crate_version"; then
            echo -e "  ${GREEN}✓${NC} Cargo.toml $dep_key → v$crate_version"
        else
            fix_failed=$((fix_failed + 1))
        fi
    done
fi

echo ""
if [[ "$fix_failed" -gt 0 ]]; then
    echo -e "${RED}${BOLD}$fix_failed fix(es) failed. Check output above.${NC}"
    exit 1
fi
echo -e "${GREEN}${BOLD}Done. Re-run to verify.${NC}"
