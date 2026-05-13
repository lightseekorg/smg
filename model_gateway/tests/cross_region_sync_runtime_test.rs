//! Boot-path smoke test for the cross-region sync runtime bundle.
//!
//! Exercises the construction the gateway boot block in `server::startup`
//! goes through: build a `CrossRegionContext` from a fully-populated
//! `CrossRegionConfig`, call `CrossRegionSyncRuntime::start(...)`, attach a
//! `PullClientOrchestrator` via `with_pull_client(...)`, and verify the
//! resulting bundle is live (producers publishing into the log, pull-server
//! state can be built from the runtime's accessors, the pull-client
//! orchestrator wires onto a reqwest client).
//!
//! The mTLS listeners and the real reqwest client over mTLS are out of scope
//! — they need real certs and bound ports — but every other piece of the
//! boot path is exercised end-to-end through the runtime bundle.
//!
//! Test cert fixtures aren't required here because we never load
//! `MTLSManager::load_*_config()`. The pull client uses a default reqwest
//! client; its background tasks will fail their first call (no real peer) and
//! back off, which is invisible to assertions on the bundle.

use std::{sync::Arc, time::Duration};

use smg::{
    config::{
        CrossRegionConfig, CrossRegionMtlsConfig, CrossRegionPeerConfig,
        CrossRegionRequestPlaneConfig, CrossRegionSyncPlaneConfig,
    },
    cross_region::{
        CrossRegionContext, CrossRegionSyncRuntime, PullClientConfig, PullClientOrchestrator,
        PullServerState, SignalKey, SignalKind, SmgReadinessSignal,
    },
    worker::WorkerRegistry,
};

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

#[tokio::test]
async fn boot_starts_runtime_with_live_producers_and_publishable_sync_handle() {
    let context = build_context();
    let registry = Arc::new(WorkerRegistry::new());

    let runtime =
        CrossRegionSyncRuntime::start(&context, registry).expect("sync runtime should start");

    // sync() exposes a live handle stamped with the resolved identity.
    assert_eq!(runtime.sync().region_id(), "us-ashburn-1");
    assert_eq!(runtime.sync().server_name(), "smg-router-a");

    // peers() round-trips the configured peer registry so the pull-server
    // listener and pull-client orchestrator can be built off the same source.
    assert_eq!(runtime.peers().regions(), vec!["us-chicago-1".to_string()]);

    // Producer adapters publish through the same sync handle exposed on the
    // bundle: publishing a readiness signal directly mirrors what
    // `RegionReadinessAdapter::publish_ready` does internally.
    runtime
        .sync()
        .publish_signal(
            SignalKey::SmgReadiness {
                region_id: "us-ashburn-1".to_string(),
                server_name: "smg-router-a".to_string(),
            },
            SignalKind::SmgReadiness(SmgReadinessSignal {
                region_id: "us-ashburn-1".to_string(),
                server_name: "smg-router-a".to_string(),
                ready: true,
            }),
            30_000,
        )
        .expect("manual readiness publish should succeed");
    let (entries, _) = runtime.sync().local_log_snapshot();
    assert!(
        entries.iter().any(|env| matches!(
            &env.signal,
            Some(SignalKind::SmgReadiness(s)) if s.ready
        )),
        "manual readiness publish should be visible in the producer log",
    );
}

#[tokio::test]
async fn boot_reconcile_loop_publishes_readiness_without_manual_intervention() {
    let context = build_context();
    let registry = Arc::new(WorkerRegistry::new());

    let runtime =
        CrossRegionSyncRuntime::start(&context, registry).expect("sync runtime should start");
    let sync = runtime.sync();

    // The periodic readiness reconcile loop publishes immediately on its
    // first tick. Give the spawned task time to run and verify the log
    // contains a readiness envelope without the test having published one.
    let mut readiness_envelope = None;
    for _ in 0..20 {
        let (entries, _) = sync.local_log_snapshot();
        if let Some(env) = entries
            .iter()
            .find(|env| matches!(&env.signal, Some(SignalKind::SmgReadiness(_))))
        {
            readiness_envelope = Some(env.clone());
            break;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    let envelope = readiness_envelope
        .expect("periodic readiness reconcile loop should publish a readiness envelope at boot");

    // The envelope's `actor` matches the resolved server_name (the same
    // value the boot path feeds in from runtime config).
    assert_eq!(envelope.actor, "smg-router-a");
    assert!(matches!(
        envelope.key,
        SignalKey::SmgReadiness { ref region_id, ref server_name }
        if region_id == "us-ashburn-1" && server_name == "smg-router-a"
    ));
}

#[tokio::test]
async fn boot_wires_pull_server_state_off_the_runtime_handles() {
    // The pull-server listener is mounted in server.rs over an axum stack
    // with mTLS termination — we don't bind that here, but we *do* verify
    // that PullServerState can be constructed off the bundle's sync handle
    // and peer registry. That's the exact wiring the listener block uses.
    let context = build_context();
    let registry = Arc::new(WorkerRegistry::new());
    let runtime =
        CrossRegionSyncRuntime::start(&context, registry).expect("sync runtime should start");

    let state = PullServerState::new(runtime.sync(), runtime.peers().clone());
    // The state itself is opaque; the smoke check is that the typed wiring
    // accepts the same handle types the boot path uses without runtime
    // surgery. (Construction would fail to compile if the bundle exposed
    // mismatched types.)
    let _ = state;
}

#[tokio::test]
async fn boot_attaches_pull_client_orchestrator_through_with_pull_client() {
    // Mirrors the boot path's chain:
    //   let runtime = CrossRegionSyncRuntime::start(...)?;
    //   let orchestrator = PullClientOrchestrator::start(
    //       runtime.sync(), runtime.peers(), http_client, config)?;
    //   let runtime = runtime.with_pull_client(orchestrator);
    //
    // The HTTP client is built without mTLS because we never make a real
    // outbound request — the orchestrator spawns one task per peer that will
    // fail its first connect and back off. The point is that
    // `with_pull_client` accepts the orchestrator and the runtime continues
    // to expose the same sync/peers/producer handles afterward.
    let context = build_context();
    let registry = Arc::new(WorkerRegistry::new());
    let runtime =
        CrossRegionSyncRuntime::start(&context, registry).expect("sync runtime should start");

    let http_client = reqwest::Client::builder()
        .build()
        .expect("default reqwest client should build");
    let orchestrator = PullClientOrchestrator::start(
        runtime.sync(),
        runtime.peers(),
        http_client,
        PullClientConfig::default(),
    )
    .expect("pull client orchestrator should start");

    let runtime = runtime.with_pull_client(orchestrator);

    // After attachment the bundle still exposes a live sync handle stamped
    // with the configured identity.
    assert_eq!(runtime.sync().region_id(), "us-ashburn-1");
    assert_eq!(runtime.sync().server_name(), "smg-router-a");
    assert_eq!(runtime.peers().regions(), vec!["us-chicago-1".to_string()]);
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
