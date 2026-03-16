#!/usr/bin/env bash
# DCO sign-off check for pre-commit (commit-msg stage).
# Verifies the commit message contains a Signed-off-by line
# that matches the committer's git user.name and user.email.
set -euo pipefail

commit_msg_file="$1"

# 1. Check that a Signed-off-by line exists
if ! grep -q "^Signed-off-by: .* <.*>$" "$commit_msg_file"; then
  echo "ERROR: Commit message must contain a Signed-off-by line."
  echo "Tip: use \"git commit -s\" to auto-add sign-off."
  exit 1
fi

# 2. Ensure git identity is configured
expected_name="$(git config --get user.name 2>/dev/null || true)"
expected_email="$(git config --get user.email 2>/dev/null || true)"

if [ -z "$expected_name" ] || [ -z "$expected_email" ]; then
  echo "ERROR: git user.name and user.email must be configured."
  echo "Run: git config user.name \"Your Name\" && git config user.email \"you@example.com\""
  exit 1
fi

# 3. Guard against newline characters in git config values
if [[ "$expected_name" == *$'\n'* ]] || [[ "$expected_email" == *$'\n'* ]]; then
  echo "ERROR: git user.name and user.email must not contain newline characters."
  exit 1
fi

# 4. Verify the sign-off matches git config (exact line match)
expected="Signed-off-by: $expected_name <$expected_email>"
if ! grep -qxF "$expected" "$commit_msg_file"; then
  echo "ERROR: DCO sign-off does not match your git config."
  echo "Expected: $expected"
  echo ""
  echo "Tip: use \"git commit -s\" to auto-add sign-off, or update your git config:"
  echo "  git config user.name \"Your Name\""
  echo "  git config user.email \"you@example.com\""
  exit 1
fi
