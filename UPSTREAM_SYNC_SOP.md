# Upstream Sync SOP

This repository is an internal fork for downstream customization.
We periodically sync changes from the open-source upstream SMG repo into this
internal Bitbucket repo in a controlled and auditable way.

## Remotes

- **origin**: internal Bitbucket repository (read/write)
- **upstream**: open-source SMG repository (read-only)
  - https://github.com/lightseekorg/smg.git

## Principles

- Never commit to or push to `upstream/*` remote-tracking branches.
- Do not force-push to `main`.
- Every upstream sync must be done on a dedicated branch: `sync/upstream-YYYYMMDD`.
- Upstream syncs must use **merge** (not rebase) to preserve a clear audit trail.

## Branch Naming

- `main`: internal stable branch
- `sync/upstream-YYYYMMDD`: one branch per upstream sync (via PR)

Example: `sync/upstream-20260127`

## One-time Setup (per local clone)

Run once after cloning the internal repo:

```bash
git remote add upstream https://github.com/lightseekorg/smg.git
git fetch upstream --prune --tags
git remote -v
```

## Every-time Upstream Sync Workflow

Replace `YYYYMMDD` with today's date.

1) Update internal main

```bash
git checkout main
git pull origin main
```

2) Fetch upstream

```bash
git fetch upstream --prune --tags
```

3) Create a sync branch

```bash
git checkout -b sync/upstream-YYYYMMDD
```

4) Merge upstream into the sync branch

```bash
git merge --no-ff upstream/main
```

5) Resolve conflicts and validate

```bash
git status
```

Resolve conflicts if any, then run required build/tests

1) Push the sync branch to origin

```bash
git push -u origin sync/upstream-YYYYMMDD
```

7) Create a Pull Request (Bitbucket)

From: `sync/upstream-YYYYMMDD`

To: `main`

PR description must include:

- Upstream branch synced (e.g., `upstream/main`)
- Upstream commit/tag range
- Conflict summary (if any)
- Test results

## Useful Commands

List commits introduced by the sync branch:

```bash
git log --oneline main..sync/upstream-YYYYMMDD
```

Show diff summary:

```bash
git diff --stat main...sync/upstream-YYYYMMDD
```

## Abort / Rollback

Abort merge (before completing the merge commit):

```bash
git merge --abort
```

Delete local sync branch:

```bash
git checkout main
git branch -D sync/upstream-YYYYMMDD
```
