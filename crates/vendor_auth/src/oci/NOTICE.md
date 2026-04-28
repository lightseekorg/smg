# NOTICE — `crates/vendor_auth/src/oci/`

This subtree contains source files copied verbatim from
[`oci-rust-sdk`](https://github.com/oracle/oci-rust-sdk) and used under the
**Universal Permissive License (UPL), Version 1.0**.

## Origin

| | |
|---|---|
| Upstream repo | `https://github.com/oracle/oci-rust-sdk` |
| Origin commit | `0590d5dcebabc68d9115520e2be5e42f9dbf1ffb` |
| Local mirror at copy time | `/Users/simolin/opensource/oci-rust-sdk` |
| Source path | `crates/common/src/*.rs` |
| Copy date | 2026-04-25 |
| Reason | Per SMG design doc `.claude/plans/container-backend-design.md` D8: avoid external path-dep, vendor minimum subset under UPL-1.0. |

## Files copied (v1 — instance principals only, per D10)

| File (here) | Upstream file | Modifications |
|---|---|---|
| `auth_utils.rs` | `auth_utils.rs` | None |
| `authentication_provider.rs` | `authentication_provider.rs` | None |
| `certificate_retriever.rs` | `certficate_retreiver.rs` | **Renamed** to fix typo (per D8). Internal type names like `UrlBasedCertificateRetriever` kept verbatim. |
| `constants.rs` | `constants.rs` | None |
| `federation_client.rs` | `federation_client.rs` | None |
| `file_utils.rs` | `file_utils.rs` | None — added as transitive compile dep of `private_key_supplier.rs`. (Design doc §5 lists `file_utils.rs` as RP-v2-only; this is a documented deviation — `private_key_supplier.rs` also requires it.) |
| `http_signature.rs` | `http_signature.rs` | None |
| `instance_principals_provider.rs` | `instance_principals_provider.rs` | `crate::*` paths rewritten to `crate::oci::*`. |
| `jwt_claim_set.rs` | `jwt_claim_set.rs` | `crate::auth_utils` rewritten to `crate::oci::auth_utils`. |
| `private_key_supplier.rs` | `private_key_supplier.rs` | `crate::file_utils` rewritten to `crate::oci::file_utils`. |
| `region_definitions.rs` | `region_definitions.rs` | None |
| `security_token_container.rs` | `security_token_container.rs` | `crate::jwt_claim_set` rewritten to `crate::oci::jwt_claim_set`. |
| `session_key_supplier.rs` | `session_key_supplier.rs` | None |
| `signer.rs` | `signer.rs` | `crate::authentication_provider` rewritten to `crate::oci::authentication_provider`; `crate::http_signature` (mod-path attr) rewritten to use the sibling `super::http_signature` module. |
| `x509_federation_client.rs` | `x509_federation_client.rs` | All `crate::*` paths rewritten to `crate::oci::*`. References to `certficate_retreiver` rewritten to `certificate_retriever` (file rename). |

Files explicitly NOT copied (v1 deferral, per design doc §5 + D10):

- `resource_principals_provider_v2.rs` (RP v2 — deferred to v2)
- `utils.rs` (RP v2 dep)
- `session_token_authentication_provider.rs` (local-dev path, unused in v1)
- `simple_authentication_provider.rs` (API-key on disk path, unused in v1)
- `config_file_authentication_provider.rs`, `config_file_reader.rs` (`~/.oci/config` reader, unused in v1)
- `endpoint_builder.rs`, `oci_error.rs`, `request_helper.rs`, `response_helper.rs`, `sdk_client.rs`, `serialization.rs` (unrelated SDK runtime)

## HeaderMap refactor decision

The upstream `signer.rs` is built around `reqwest::header::HeaderMap` and
`reqwest::Method`. SMG's HTTP stack at the public boundary is `axum`/`http`.

**Decision**: keep the `reqwest` types **internal** to `signer.rs` (verbatim
copy) and translate at the adapter boundary in
`crates/vendor_auth/src/signer_adapter.rs`. Rationale:

- Smaller diff vs. upstream (R8 drift management is easier).
- The conversion cost is one-shot per request — performance impact is
  negligible.
- The reqwest types are an implementation detail that does not leak past
  `signer_adapter.rs`.

## License (UPL-1.0)

```
Copyright (c) 2023, Oracle and/or its affiliates.

The Universal Permissive License (UPL), Version 1.0

Subject to the condition set forth below, permission is hereby granted to any
person obtaining a copy of this software, associated documentation and/or data
(collectively the "Software"), free of charge and under any and all copyright
rights in the Software, and any and all patent rights owned or freely
licensable by each licensor hereunder covering either (i) the unmodified
Software as contributed to or provided by such licensor, or (ii) the Larger
Works (as defined below), to deal in both

(a) the Software, and
(b) any piece of software and/or hardware listed in the lrgrwrks.txt file if
one is included with the Software (each a "Larger Work" to which the Software
is contributed by such licensors),

without restriction, including without limitation the rights to copy, create
derivative works of, display, perform, and distribute the Software and make,
use, sell, offer for sale, import, export, have made, and have sold the
Software and the Larger Work(s), and to sublicense the foregoing rights on
either these or other terms.

This license is subject to the following condition:

The above copyright notice and either this complete permission notice or at a
minimum a reference to the UPL must be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Drift management (R8)

This subtree is treated as **forked-and-frozen** at commit
`0590d5dcebabc68d9115520e2be5e42f9dbf1ffb`. Upstream changes do not flow in
automatically.

- **Quarterly job (CB-0.A)**: diff this subtree against upstream HEAD and
  triage delta.
- **CVE response**: re-pull only the affected file(s); update the per-file
  `Origin commit` header line; record in this NOTICE.
- **No upstream contribution unless asked**: changes here do not flow back to
  Oracle's repo as PRs unless we explicitly upstream them.

## License-header check

`crates/vendor_auth/scripts/check_oracle_headers.sh` validates that every
`*.rs` file in this subtree carries the Oracle copyright header. The check is
wired into `cargo test` via `tests/license_header.rs`.
