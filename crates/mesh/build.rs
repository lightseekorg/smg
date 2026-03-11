fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=src/proto/gossip.proto");

    // Compile gossip protobuf files
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .compile_protos(&["src/proto/gossip.proto"], &["src/proto"])?;

    Ok(())
}
