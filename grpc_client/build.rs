fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=proto/sglang_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/vllm_engine.proto");
    println!("cargo:rerun-if-changed=proto/trtllm_service.proto");

    // Compile protobuf files
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .type_attribute("GetModelInfoResponse", "#[derive(serde::Serialize)]")
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "proto/sglang_scheduler.proto",
                "proto/vllm_engine.proto",
                "proto/trtllm_service.proto",
            ],
            &["proto"],
        )?;

    Ok(())
}
