#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CI_AGENTIC_SVC_DEPS_LIB_ONLY=1 source "$REPO_ROOT/scripts/ci_agentic_svc_deps.sh"

assert_eq() {
    local expected="$1" actual="$2"
    if [ "$expected" != "$actual" ]; then
        echo "expected: $expected"
        echo "actual:   $actual"
        exit 1
    fi
}

assert_matches() {
    local value="$1" pattern="$2"
    if [[ ! "$value" =~ $pattern ]]; then
        echo "expected '$value' to match /$pattern/"
        exit 1
    fi
}

assert_max_len() {
    local value="$1" max="$2"
    if [ "${#value}" -gt "$max" ]; then
        echo "expected '$value' to be at most $max chars, got ${#value}"
        exit 1
    fi
}

test_run_scoped_oracle_usernames_are_unique_and_oracle_safe() {
    local test_user_a
    test_user_a=$(
        HOSTNAME=2-gpu-h100 \
            GITHUB_RUN_ID=27463234653 \
            GITHUB_RUN_ATTEMPT=4 \
            CI_ORACLE_NAME_RANDOM=A1B2C3 \
            ci_oracle_username TEST
    )
    assert_eq "TEST_7463234653_4_A1B2C3" "$test_user_a"

    local test_user_b
    test_user_b=$(
        HOSTNAME=2-gpu-h100 \
            GITHUB_RUN_ID=27463234653 \
            GITHUB_RUN_ATTEMPT=4 \
            CI_ORACLE_NAME_RANDOM=D4E5F6 \
            ci_oracle_username TEST
    )
    assert_eq "TEST_7463234653_4_D4E5F6" "$test_user_b"

    local flyway_user
    flyway_user=$(
        HOSTNAME=2-gpu-h100 \
            GITHUB_RUN_ID=27463234653 \
            GITHUB_RUN_ATTEMPT=4 \
            CI_ORACLE_NAME_RANDOM=A1B2C3 \
            ci_oracle_username FLYWAY
    )
    assert_eq "FLYWAY_7463234653_4_A1B2C3" "$flyway_user"
    assert_max_len "$flyway_user" 30
}

test_hostname_fallback_keeps_entropy_and_stays_under_oracle_limit() {
    local username
    username=$(
        HOSTNAME=arc-runner-gpu-h100-jfvzm-m2lm8 \
            GITHUB_RUN_ID= \
            GITHUB_RUN_ATTEMPT= \
            CI_ORACLE_NAME_RANDOM=ABCDEF \
            ci_oracle_username TEST
    )

    assert_matches "$username" '^TEST_[A-Z0-9_]+_ABCDEF$'
    assert_max_len "$username" 30
}

test_run_scoped_oracle_usernames_are_unique_and_oracle_safe
test_hostname_fallback_keeps_entropy_and_stays_under_oracle_limit

echo "ci_agentic_svc_deps tests passed"
