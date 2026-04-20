fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=src/proto/gossip.proto");

    // Compile gossip protobuf files. Emit StreamEntry.data as
    // bytes::Bytes instead of Vec<u8> so the sender can split a value
    // into zero-copy chunks (Bytes::slice is refcount-shared) and the
    // receiver keeps decoded chunks as Bytes without an extra copy.
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .bytes(".mesh.gossip.StreamEntry.data")
        .compile_protos(&["src/proto/gossip.proto"], &["src/proto"])?;

    Ok(())
}
