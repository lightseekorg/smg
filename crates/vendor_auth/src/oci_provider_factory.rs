//! Factory for the runtime-active OCI [`AuthenticationProvider`].
//!
//! Per design doc §5 + D10, v1 ships only the instance-principals branch
//! live; resource-principal v2 (workload identity) is deferred to v2.

use std::sync::Arc;

use crate::AuthError;
use crate::oci::authentication_provider::AuthenticationProvider;
use crate::oci::instance_principals_provider::InstancePrincipalAuthProvider;

/// Detect and construct the active OCI auth provider.
///
/// **v1 behavior**: always returns an [`InstancePrincipalAuthProvider`]
/// constructed from IMDS. Per D9, SMG's CI cluster has node-level
/// instance-principal access, so this branch is exercised on every CI run.
///
/// **v2 follow-up (deferred per D10)**: when SMG adopts resource principals
/// v2, the factory will inspect `OCI_RESOURCE_PRINCIPAL_VERSION` and prefer
/// the RP v2 provider. The placeholder branch lives below as a comment.
pub async fn detect_oci_provider(
) -> Result<Arc<Box<dyn AuthenticationProvider + Send + Sync>>, AuthError> {
    // v2 follow-up per D10:
    //
    // ```rust,ignore
    // if std::env::var("OCI_RESOURCE_PRINCIPAL_VERSION").is_ok() {
    //     let rp = ResourcePrincipalAuthProviderV2::new()
    //         .map_err(|e| AuthError::Signer(anyhow::anyhow!("rp v2: {e}")))?;
    //     return Ok(Arc::new(Box::new(rp)));
    // }
    // ```

    let ip = InstancePrincipalAuthProvider::new()
        .await
        .map_err(|e| AuthError::Signer(anyhow::anyhow!("instance principals: {e}")))?;
    Ok(Arc::new(Box::new(ip)))
}
