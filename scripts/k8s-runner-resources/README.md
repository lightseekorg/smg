# ARC (Actions Runner Controller) Deployment Guide

## Prerequisites

- Kubernetes cluster (v1.28+)
- `kubectl` configured with cluster access
- `helm` v3 installed

## 1. Create a GitHub App

1. Go to **GitHub Organization Settings > Developer settings > GitHub Apps > New GitHub App**
2. Set the following:
   - **GitHub App name**: e.g. `smg-k8s-arc-app`
   - **Homepage URL**: any valid URL
   - **Webhook**: uncheck "Active" (not needed)
3. Grant **Repository permissions**:
   - **Actions**: Read and Write
   - **Administration**: Read and Write
   - **Checks**: Read and Write
   - **Metadata**: Read-only
4. Click **Create GitHub App**
5. Note the **App ID** from the app's settings page
6. Generate a **Private Key** (PEM file) and save it

## 2. Install the GitHub App

1. From the GitHub App settings, click **Install App**
2. Select the organization and choose the target repository (e.g. `lightseekorg/smg`)
3. Click **Install**
4. Note the **Installation ID** from the URL: `https://github.com/organizations/<org>/settings/installations/<installation-id>`

## 3. Create the Kubernetes Secret

Edit `arc-secret-template.yaml` with the values collected above:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: github-arc-secret
  namespace: actions-runner-system
type: Opaque
stringData:
  github_app_id: "<your-app-id>"
  github_app_installation_id: "<your-installation-id>"
  github_app_private_key: "<contents-of-your-pem-file>"
```

Create the namespace and secret:

```bash
kubectl create namespace actions-runner-system
kubectl apply -f scripts/k8s-runner-resources/arc-secret-template.yaml
```

## 4. Install the ARC Controller

```bash
helm install arc \
  --namespace actions-runner-system \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set-controller
```

Verify the controller is running:

```bash
kubectl get pods -n actions-runner-system -l app.kubernetes.io/name=gha-runner-scale-set-controller
```

## 5. Install Runner Scale Sets

Install each runner set using its values file:

```bash
# CPU runners
helm install k8s-runner-cpu \
  --namespace actions-runner-system \
  -f scripts/k8s-runner-resources/runner-values-cpu.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

# 4x GPU H100/A10 runners
helm install k8s-runner-gpu \
  --namespace actions-runner-system \
  -f scripts/k8s-runner-resources/runner-values-4-gpu-general.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

# 4x GPU H100 runners
helm install 4-gpu-h100 \
  --namespace actions-runner-system \
  -f scripts/k8s-runner-resources/runner-values-4-gpu-h100.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

# 2x GPU H100 runners
helm install 2-gpu-h100 \
  --namespace actions-runner-system \
  -f scripts/k8s-runner-resources/runner-values-2-gpu-h100.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set

# 1x GPU H100 runners
helm install 1-gpu-h100 \
  --namespace actions-runner-system \
  -f scripts/k8s-runner-resources/runner-values-1-gpu-h100.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set
```

## 6. Verify

Check that listener pods are running for each runner set:

```bash
kubectl get pods -n actions-runner-system
```

Each runner set should have a listener pod in `Running` state. Runner pods will be created on-demand when workflows target the corresponding `runs-on` label (matching `runnerScaleSetName` in each values file).

## Upgrading

To update a runner set (e.g. after changing values):

```bash
helm upgrade <runner-set-name> \
  --namespace actions-runner-system \
  -f scripts/k8s-runner-resources/<runner-set-values-file>.yaml \
  oci://ghcr.io/actions/actions-runner-controller-charts/gha-runner-scale-set
```

## Uninstalling

```bash
# Remove a runner set
helm uninstall <runner-set-name> -n actions-runner-system

# Remove the controller (after all runner sets are removed)
helm uninstall arc -n actions-runner-system
```
