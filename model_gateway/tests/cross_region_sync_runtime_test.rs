//! Boot-path smoke test for the cross-region sync runtime bundle.
//!
//! Exercises the construction the gateway boot block in `server::startup`
//! goes through: build a `CrossRegionContext` from a fully-populated
//! `CrossRegionConfig`, register the `cross_region:` broadcast stream
//! namespace on a shared `MeshKV`, call
//! `CrossRegionSyncRuntime::start_with_mesh_kv(...)`, and verify the
//! resulting bundle exposes a live sync handle whose publishes stage
//! envelopes in the outbox ready for the next gossip round.
//!
//! Real mesh transport / mTLS allowlist is out of scope — `MeshKV::new`
//! gives us an in-process namespace that exercises the same publish path
//! adapters use in production.

use std::{sync::Arc, time::Duration};

use smg::{
    config::{
        CrossRegionConfig, CrossRegionMtlsConfig, CrossRegionPeerConfig,
        CrossRegionRequestPlaneConfig, CrossRegionSyncPlaneConfig,
    },
    cross_region::{
        CrossRegionContext, CrossRegionSyncRuntime, CrossRegionSyncService, SignalEnvelope,
        SignalKey, SignalKind, SmgReadinessSignal,
    },
    worker::WorkerRegistry,
};
use smg_mesh::MeshKV;

fn valid_cross_region_config() -> CrossRegionConfig {
    CrossRegionConfig {
        enabled: true,
        region_id: Some("us-ashburn-1".to_string()),
        server_name: Some("smg-router-a".to_string()),
        realm: Some("oc1".to_string()),
        environment: Some("prod".to_string()),
        local_only_on_degraded_sync: true,
        request_plane: CrossRegionRequestPlaneConfig::default(),
        sync_plane: CrossRegionSyncPlaneConfig::default(),
        peers: vec![CrossRegionPeerConfig {
            region_id: Some("us-chicago-1".to_string()),
            request_url: Some("https://smg-region-agent.us-chicago-1.internal:8443".to_string()),
            sync_url: Some("https://smg-region-agent.us-chicago-1.internal:9443".to_string()),
            realm: Some("oc1".to_string()),
            environment: Some("prod".to_string()),
            ..CrossRegionPeerConfig::default()
        }],
        mtls: CrossRegionMtlsConfig {
            ca_cert_path: Some("/etc/smg/certs/ca.crt".to_string()),
            server_cert_path: Some("/etc/smg/certs/tls.crt".to_string()),
            server_key_path: Some("/etc/smg/certs/tls.key".to_string()),
            client_cert_path: Some("/etc/smg/certs/client.crt".to_string()),
            client_key_path: Some("/etc/smg/certs/client.key".to_string()),
        },
    }
}

#[expect(
    clippy::expect_used,
    reason = "test helper — the fixture is known-valid"
)]
fn build_context() -> CrossRegionContext {
    CrossRegionContext::from_router_config(&valid_cross_region_config())
        .expect("valid cross-region config should convert")
        .expect("enabled cross-region config should produce a runtime context")
}

fn make_mesh_kv() -> Arc<MeshKV> {
    Arc::new(MeshKV::new("smg-router-a".to_string()))
}

fn fetch_envelope(
    sync: &CrossRegionSyncService,
    key: &SignalKey,
) -> Option<SignalEnvelope<SignalKind>> {
    sync.outbox_snapshot()
        .into_iter()
        .find(|env| env.key == *key)
}

#[tokio::test]
async fn boot_starts_runtime_with_live_producers_and_publishable_sync_handle() {
    let context = build_context();
    let registry = Arc::new(WorkerRegistry::new());
    let mesh_kv = make_mesh_kv();

    let runtime = CrossRegionSyncRuntime::start_with_mesh_kv(&context, &mesh_kv, registry)
        .expect("sync runtime should start");

    // sync() exposes a live handle stamped with the resolved identity.
    assert_eq!(runtime.sync().region_id(), "us-ashburn-1");
    assert_eq!(runtime.sync().server_name(), "smg-router-a");

    // peers() round-trips the configured peer registry so request-plane
    // forwarding code can resolve targets against the same source.
    assert_eq!(runtime.peers().regions(), vec!["us-chicago-1".to_string()]);

    // Producer adapters publish through the same sync handle exposed on the
    // bundle: publishing a readiness signal directly mirrors what
    // `RegionReadinessAdapter::publish_ready` does internally.
    let key = SignalKey::SmgReadiness {
        region_id: "us-ashburn-1".to_string(),
        server_name: "smg-router-a".to_string(),
    };
    runtime
        .sync()
        .publish_signal(
            key.clone(),
            SignalKind::SmgReadiness(SmgReadinessSignal {
                region_id: "us-ashburn-1".to_string(),
                server_name: "smg-router-a".to_string(),
                ready: true,
            }),
            30_000,
        )
        .expect("manual readiness publish should succeed");
    let envelope = fetch_envelope(&runtime.sync(), &key)
        .expect("manual readiness publish should stage in the outbox");
    assert!(matches!(
        envelope.signal,
        Some(SignalKind::SmgReadiness(s)) if s.ready
    ));
}

#[tokio::test]
async fn boot_reconcile_loop_publishes_readiness_without_manual_intervention() {
    let context = build_context();
    let registry = Arc::new(WorkerRegistry::new());
    let mesh_kv = make_mesh_kv();

    let runtime = CrossRegionSyncRuntime::start_with_mesh_kv(&context, &mesh_kv, registry)
        .expect("sync runtime should start");
    let sync = runtime.sync();

    let readiness_key = SignalKey::SmgReadiness {
        region_id: "us-ashburn-1".to_string(),
        server_name: "smg-router-a".to_string(),
    };

    // The periodic readiness reconcile loop publishes immediately on its
    // first tick. Give the spawned task time to run and verify the outbox
    // carries a readiness envelope without the test having published one.
    let mut readiness_envelope = None;
    for _ in 0..20 {
        if let Some(env) = fetch_envelope(&sync, &readiness_key) {
            readiness_envelope = Some(env);
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let envelope = readiness_envelope
        .expect("periodic readiness reconcile loop should publish a readiness envelope at boot");

    assert_eq!(envelope.actor, "smg-router-a");
    assert!(matches!(
        envelope.key,
        SignalKey::SmgReadiness { ref region_id, ref server_name }
        if region_id == "us-ashburn-1" && server_name == "smg-router-a"
    ));
}

#[test]
fn disabled_cross_region_config_returns_no_context() {
    let disabled = CrossRegionConfig::default();
    let context = CrossRegionContext::from_router_config(&disabled)
        .expect("disabled config should convert without error");
    assert!(
        context.is_none(),
        "the boot block's `Ok(None)` branch should fire when cross_region is disabled",
    );
}
