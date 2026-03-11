# Upstream Sync SOP

Use this flow for every upstream sync.

## Remotes

- `origin`: internal downstream repo
- `upstream`: open-source SMG repo

## Branches and commits

- Base every sync on `origin/main`
- Work on `sync/upstream-YYYYMMDD`
- Keep the sync as:
  1. `Sync/upstream YYYYMMDD`
  2. `Reapply downstream changes` (only if needed)

## Source of truth for sync baseline

Do not use local reflog as the durable source of truth.

The source of truth is the latest `Sync/upstream ...` commit message on `origin/main`.

That commit message must contain:

```text
Sync/upstream YYYYMMDD

Upstream branch: upstream/main
Upstream old: <OLD_UPSTREAM>
Upstream new: <NEW_UPSTREAM>
Upstream range: <OLD_UPSTREAM>..<NEW_UPSTREAM>
```

For the next sync:

- `OLD_UPSTREAM` = previous `Upstream new`
- `NEW_UPSTREAM` = current `upstream/main`

## Workflow

0) Ensure a clean working tree

```bash
git status -sb
```

1) Fetch remotes

```bash
git fetch origin --prune --tags
git fetch upstream --prune --tags
```

2) Read the previous sync baseline from commit message

```bash
git log origin/main --grep='^Sync/upstream ' --format=%B -n 1
```

Set:

```bash
OLD_UPSTREAM=<previous Upstream new>
NEW_UPSTREAM=$(git rev-parse upstream/main)
```

If there is no prior sync commit, use the previous sync PR description.

3) Create the sync branch from `origin/main`

```bash
git checkout -b sync/upstream-YYYYMMDD origin/main
```

4) Classify upstream-changed files vs downstream-overlap files

```bash
git diff --name-only "$OLD_UPSTREAM".."$NEW_UPSTREAM" | sort > /tmp/up_files
git diff --name-only "$OLD_UPSTREAM"..origin/main | sort > /tmp/down_files

comm -23 /tmp/up_files /tmp/down_files > /tmp/safe_accept_upstream
comm -12 /tmp/up_files /tmp/down_files > /tmp/has_downstream_changes

wc -l /tmp/safe_accept_upstream /tmp/has_downstream_changes
```

Interpretation:

- `/tmp/safe_accept_upstream`: take upstream directly
- `/tmp/has_downstream_changes`: manual review required; start from upstream, then reapply only the downstream behavior still needed

5) Apply upstream file states for `OLD_UPSTREAM..NEW_UPSTREAM`

Do not rely on a plain merge for this. Handle adds/modifies, deletes, and renames so old paths do not linger.

Rules:

- add/modify: restore from `upstream/main`
- delete: remove the path
- rename: remove old path, restore new path from `upstream/main`

6) Create the upstream sync commit

```bash
git commit -m "Sync/upstream YYYYMMDD" \
  -m "Upstream branch: upstream/main" \
  -m "Upstream old: $OLD_UPSTREAM
Upstream new: $NEW_UPSTREAM
Upstream range: $OLD_UPSTREAM..$NEW_UPSTREAM"
```

7) Review files in `/tmp/has_downstream_changes`

Rules:

- Keep upstream as the baseline
- Reapply only the minimal downstream behavior that is still required
- Do not bulk-copy `origin/main` back over the file
- Review diff hunk by hunk if needed

8) Validate before any downstream reapply commit

```bash
cargo check
```

9) If needed, create the downstream reapply commit

```bash
git commit -m "Reapply downstream changes"
```

10) Push the branch and open the PR

```bash
git push -u origin sync/upstream-YYYYMMDD
```

PR description should include:

- `Upstream branch`
- `Upstream old`
- `Upstream new`
- `Upstream range`
- counts for `/tmp/safe_accept_upstream` and `/tmp/has_downstream_changes`
- any downstream files re-applied
- validation results

## Useful commands

```bash
git log --oneline origin/main..sync/upstream-YYYYMMDD
git diff --stat origin/main...sync/upstream-YYYYMMDD
```

## Abort

```bash
git checkout main
git branch -D sync/upstream-YYYYMMDD
```
