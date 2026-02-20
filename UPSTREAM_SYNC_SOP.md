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
- Because sync commits are typically squashed, `origin/main` and `upstream/main` may not share recent history.
  Do not rely on `git merge-base origin/main upstream/main` for "what changed since last sync".

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

2) Record the upstream baseline for this sync (required for file classification)

Because we squash upstream sync commits, the reliable baseline is the previous
`upstream/main` value from your local reflog.

```bash
git reflog -n 5 upstream/main
OLD_UPSTREAM=$(git rev-parse upstream/main@{1})
NEW_UPSTREAM=$(git rev-parse upstream/main)
echo "OLD_UPSTREAM=$OLD_UPSTREAM"
echo "NEW_UPSTREAM=$NEW_UPSTREAM"
```

If `upstream/main@{1}` does not exist (fresh clone), fall back to using the last
tag/commit recorded in the previous sync PR description.

3) Create a sync branch (from origin/main)

```bash
git checkout -b sync/upstream-YYYYMMDD origin/main
```

4) Classify files impacted by upstream vs downstream (do not limit to conflicts)

This step prevents missing non-conflicting but required downstream reapply work
(e.g., signature/callsite changes that only show up in `cargo check`).

Generate the file lists:

```bash
git diff --name-only "$OLD_UPSTREAM".."$NEW_UPSTREAM" | sort > /tmp/up_files
git diff --name-only "$OLD_UPSTREAM"..origin/main | sort > /tmp/down_files

# Files changed in upstream, but not customized downstream since OLD_UPSTREAM.
comm -23 /tmp/up_files /tmp/down_files > /tmp/safe_accept_upstream

# Files changed in upstream AND customized downstream since OLD_UPSTREAM.
comm -12 /tmp/up_files /tmp/down_files > /tmp/has_downstream_changes

wc -l /tmp/safe_accept_upstream /tmp/has_downstream_changes
```

Interpretation:

- `/tmp/safe_accept_upstream`: usually safe to accept upstream for these files.
- `/tmp/has_downstream_changes`: requires human review. These files may or may not
  need downstream changes re-applied; do not blindly copy the full `origin/main`
  diff back.

Always treat `data_connector/src/oracle.rs` as a manual-review file even if it
does not appear in `/tmp/has_downstream_changes`.

Optional sanity checks per file:

```bash
git diff upstream/main..origin/main -- <file>
git blame -n origin/main -- <file> | head
```

5) Merge upstream into the sync branch

```bash
git merge --no-ff upstream/main
```

6) Resolve conflicts and validate

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

Important: conflict files are only a subset. You must also review files in
`/tmp/has_downstream_changes` even if they did not conflict.

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
- Reapply only the *downstream semantics that are still required*; do not copy the entire
  `origin/main` vs `upstream/main` diff back.
  - A file being in `/tmp/has_downstream_changes` means "manual review required", not
    "always reapply everything".
  - Use `git blame origin/main -- <file>` to validate whether a hunk is actually a
    downstream customization or just historical evolution.
  - Do not resurrect upstream-removed targets/flags by accident (common pitfall with
    Makefile changes).

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

4) Run a compile check *before* downstream reapply.

This catches non-conflicting callsite/signature issues.

```bash
cargo check
```

5) Reapply downstream changes as a separate commit.

Guidelines:

- Prefer upstream as the baseline for manual-review files, then add the minimal
  downstream semantics required.
- If task-local context is used, be careful with background tasks:
  - `tokio::spawn` does not automatically inherit task-locals.
  - Capture needed values before spawning and pass them into the spawned task.

6) Push the sync branch to origin

```bash
git push -u origin sync/upstream-YYYYMMDD
```

7) Create a Pull Request (Bitbucket)

From: `sync/upstream-YYYYMMDD`

To: `main`

PR description must include:

- Upstream branch synced (e.g., `upstream/main`)
- Upstream commit range (include `OLD_UPSTREAM..NEW_UPSTREAM` from reflog)
- File classification notes:
  - count of `/tmp/safe_accept_upstream`
  - count of `/tmp/has_downstream_changes`
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
