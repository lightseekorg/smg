# ARC (Actions Runner Controller) Deployment Guide

This directory configures the `smg-project/smg` runner scale sets. The shared ARC
controller and listener pods run in `org-actions-runner`; runner scale-set objects,
credentials, and runner pods run in `actions-runner-system`.

The controller and runner scale-set charts are pinned to `0.14.2` so their CRDs and
controller behavior stay compatible.

## Prerequisites

- Kubernetes cluster v1.28 or newer
- `kubectl` configured with cluster access
- Helm v3
- Organization-owner access to install and authorize the GitHub App

## 1. Create and install a GitHub App

Create a GitHub App under the `smg-project` organization with webhooks disabled.
Grant only the permissions required by ARC:

- Repository permissions:
  - **Administration**: Read and write
  - **Metadata**: Read-only
- Organization permissions:
  - **Self-hosted runners**: Read and write

Install the App on `smg-project` and grant it access to the `smg` repository. Record
the App ID and the installation ID, then generate and securely store a private key.
The installation ID is the final number in the installation settings URL:

```text
https://github.com/organizations/smg-project/settings/installations/<installation-id>
```

See GitHub's [ARC authentication documentation](https://docs.github.com/en/actions/how-tos/manage-runners/use-actions-runner-controller/authenticate-to-the-api)
for the authoritative permission list.

## 2. Create or update the Kubernetes secret

Never paste credentials into `archived/arc-secret-template.yaml` or commit a private key.
Create the secret directly from the values and PEM file instead:

```bash
kubectl create namespace actions-runner-system --dry-run=client -o yaml \
  | kubectl apply -f -

kubectl create secret generic github-arc-secret \
  --namespace actions-runner-system \
  --from-literal=github_app_id='<app-id>' \
  --from-literal=github_app_installation_id='<installation-id>' \
  --from-file=github_app_private_key='<path-to-private-key.pem>' \
  --dry-run=client -o yaml \
  | kubectl apply -f -
```

The archived `arc-secret-template.yaml` contains placeholders only and is retained
for schema/reference purposes.

## 3. Install or upgrade the shared controller

```bash
helm upgrade --install arc \
  --namespace org-actions-runner \
  --create-namespace \
  --version 0.14.2 \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller
```

Verify the controller:

```bash
kubectl get pods \
  --namespace org-actions-runner \
  -l app.kubernetes.io/part-of=gha-rs-controller
```

## 4. Install or upgrade runner scale sets

Each values file explicitly references the shared controller service account in
`org-actions-runner`. Install the required scale sets into `actions-runner-system`:

```bash
helm upgrade --install k8s-runner-cpu \
  --namespace actions-runner-system \
  --create-namespace \
  --version 0.14.2 \
  -f scripts/k8s-runner-resources/runner-values-cpu.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

helm upgrade --install 1-gpu-h100 \
  --namespace actions-runner-system \
  --create-namespace \
  --version 0.14.2 \
  -f scripts/k8s-runner-resources/runner-values-1-gpu-h100.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

helm upgrade --install 2-gpu-h100 \
  --namespace actions-runner-system \
  --create-namespace \
  --version 0.14.2 \
  -f scripts/k8s-runner-resources/runner-values-2-gpu-h100.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

helm upgrade --install 4-gpu-h100 \
  --namespace actions-runner-system \
  --create-namespace \
  --version 0.14.2 \
  -f scripts/k8s-runner-resources/runner-values-4-gpu-h100.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set
```

## 5. Verify

```bash
# Scale-set objects and runner pods
kubectl get autoscalingrunnersets,pods --namespace actions-runner-system

# Listener pods managed by the shared controller
kubectl get pods \
  --namespace org-actions-runner \
  -l actions.github.com/scale-set-name
```

Each scale set should have a listener pod in `Running` state. Runner pods are created
on demand when a workflow uses the corresponding `runnerScaleSetName` as its
`runs-on` label.

## Uninstalling

Remove scale sets before removing the shared controller:

```bash
helm uninstall <runner-set-name> --namespace actions-runner-system
helm uninstall arc --namespace org-actions-runner
```

Do not uninstall the shared controller until every runner scale set that uses it has
been removed.

## Archived manifests

Inactive scale-set values and deprecated `actions.summerwind.dev` resources are in
[`archived/`](archived/README.md). They are retained only for historical recovery
and must not be applied alongside the active runner scale sets.
