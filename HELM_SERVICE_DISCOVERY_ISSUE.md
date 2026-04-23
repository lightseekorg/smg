# Helm Service Discovery Issue

## Summary

The `deploy/helm/smg` chart had a few small service-discovery rendering bugs that prevented SMG from discovering externally managed predictor pods in Kubernetes.

This showed up when using SMG to discover KServe predictor pods created outside the chart, where pod metadata already included:

- `component=predictor`
- `smg.lightseek.io/model=<cluster-id>`

## Observed Behavior

With service discovery enabled, SMG started but failed to discover tenant predictor pods correctly.

Typical symptoms:

- `/v1/models` returned `503 No models available`
- router logs showed `Router ready | workers: []`
- rendered args still included `--service-discovery-namespace default`
- chart defaulted to `--service-discovery-port 80`

## Root Cause

There were three chart issues:

1. Empty `router.serviceDiscovery.namespace` did not mean "all namespaces".
   - The template always rendered:
   - `--service-discovery-namespace {{ .Release.Namespace }}`
   - This prevented cluster-wide discovery.

2. The chart only created namespaced RBAC.
   - `Role` / `RoleBinding` were fine for single-namespace discovery.
   - All-namespace discovery needs `ClusterRole` / `ClusterRoleBinding`.

3. Service discovery values were too restrictive for external predictor pods.
   - Default port was `80`, but KServe predictor pods were serving on `8080`.
   - `modelIdFrom` was unset, so SMG did not automatically use the model ID label.

## Fix Applied

### `values.yaml`

Updated defaults to better support external predictor discovery:

- `router.serviceDiscovery.selector: "component=predictor"`
- `router.serviceDiscovery.port: 8080`
- `router.serviceDiscovery.modelIdFrom: "label:smg.lightseek.io/model"`

### `templates/_helpers.tpl`

Updated router arg rendering so that:

- `--service-discovery-namespace` is omitted when the namespace value is empty
- selector values can be rendered from either a string or a list

This allows SMG's documented behavior:

- no namespace flag => watch all namespaces

### `templates/role.yaml`

Updated RBAC generation so that:

- cluster-wide discovery renders a `ClusterRole`
- permissions include:
  - `pods`
  - `services`
  - `endpoints`
  - `endpointslices`
- verbs include:
  - `get`
  - `list`
  - `watch`

### `templates/rolebinding.yaml`

Updated binding generation so that:

- cluster-wide discovery renders a `ClusterRoleBinding`
- namespaced discovery continues to use `RoleBinding`

## Result

After redeploying the chart:

- SMG discovered predictor pods across tenant namespaces
- router logs showed:
  - `Adding pod: ... | url: http://<pod-ip>:8080`
- `/v1/models` returned discovered cluster IDs successfully

Example discovered models:

- `b861100d-7739-4c69-90e2-9487e68e95d9`
- `4da88683-768c-4306-9979-fc666eedc528`

## Notes

- This is a small Helm/chart fix, not a router-core change.
- There is still a separate tokenizer-related warning for workers exposing `/mnt/models`, but it does not block service discovery or worker registration.
