# Archived Kubernetes runner manifests

The YAML files in this directory are not part of the active Moirai runner
configuration. Do not apply this directory recursively or include it in GitOps,
Kustomize, Helm, or deployment automation.

## Contents

- `arc-runner-cpu.yaml`, `arc-runner-gpu.yaml`,
  `arc-runner-autoscaler.yaml`, and `arc-runner-rbac.yaml` use the deprecated
  `actions.summerwind.dev` API. Their runner deployments and autoscalers are pinned
  to zero.
- `runner-values-1-gpu.yaml` and `runner-values-4-gpu-general.yaml` describe
  official ARC scale sets that are not installed in the current cluster.
- `arc-secret-template.yaml` is a placeholder-only schema reference. Never store
  real GitHub App credentials or private keys in this file.

If one of these configurations is needed for recovery, review it against the
current ARC version and cluster topology, then move or copy that single file back
to the parent directory in a dedicated change.
