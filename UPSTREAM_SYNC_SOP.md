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
- Upstream syncs may be **squashed into a single commit** for review simplicity.
- Always base sync branches on `origin/main`, not a local `main`.

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

0) Ensure a clean working tree

```bash
git status -sb
```

If the working tree is not clean, stop and resolve it before continuing.

1) Fetch origin and upstream

```bash
git fetch origin --prune --tags
git fetch upstream --prune --tags
```

2) Create a sync branch (from origin/main)

```bash
git checkout -b sync/upstream-YYYYMMDD origin/main
```

3) Merge upstream into the sync branch

```bash
git merge --no-ff upstream/main
```

4) Resolve conflicts and validate

```bash
git status
```

Resolve conflicts if any, then run required build/tests.

### Conflict Handling Guidance

When conflicts happen:

1) Identify whether the conflicting file was last changed by a prior upstream sync.
   - Use `git log -1 origin/main -- <file>` to confirm.
   - If yes, accept **upstream** for that file.
2) If the file contains downstream customizations, keep **upstream as the baseline** and
   reapply the downstream changes on top (do not overwrite with the old file).
3) Do not use bulk overwrite commands (`git checkout origin/main -- <file>`, `git apply`, etc.)
   to reapply downstream changes. Read the context and edit by hand.

#### Special Case: `data_connector/src/oracle.rs`

- Do **not** accept upstream wholesale.
- Keep the downstream `oracle.rs` as the baseline.
- Manually integrate upstream changes into the downstream version, step by step, after
  reading context.
- Overwrite `data_connector/src/oracle_old.rs` with the latest upstream `oracle.rs` as a
  reference when doing the manual merge:

```bash
git show upstream/main:data_connector/src/oracle.rs > data_connector/src/oracle_old.rs
```

Recommended workflow for downstream customizations:

- Keep downstream changes as a **separate commit** (after the upstream squash commit).
- Reapply only the specific diff for the affected files; do not cherry-pick an entire
  downstream PR if it touches unrelated files (to avoid extra conflicts).

This keeps upstream history clean and makes future syncs predictable.

### Squash Mode Steps

When using squash mode for upstream sync:

1) Merge `upstream/main` into `sync/upstream-YYYYMMDD` and resolve conflicts.
2) Create a temporary merge commit to exit the merge state.
3) Squash all upstream changes into one commit:

```bash
git reset --soft origin/main
git commit -m "Sync/upstream YYYYMMDD"
```

4) Reapply downstream changes as a separate commit.

5) Push the sync branch to origin

```bash
git push -u origin sync/upstream-YYYYMMDD
```

6) Create a Pull Request (Bitbucket)

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
