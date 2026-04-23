fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Rebuild triggers
    println!("cargo:rerun-if-changed=proto/common.proto");
    println!("cargo:rerun-if-changed=proto/sglang_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/tokenspeed_scheduler.proto");
    println!("cargo:rerun-if-changed=proto/vllm_engine.proto");
    println!("cargo:rerun-if-changed=proto/trtllm_service.proto");
    println!("cargo:rerun-if-changed=proto/mlx_engine.proto");

    // Pass 1: compile shared message types (no gRPC service generation)
    tonic_prost_build::configure()
        .build_server(false)
        .build_client(false)
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/common.proto"], &["proto"])?;

    // Pass 2: compile engine protos, referencing common types via extern_path
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .extern_path(".smg.grpc.common", "crate::common_proto")
        .type_attribute("GetModelInfoResponse", "#[derive(serde::Serialize)]")
        // vllm + trtllm ServerInfo have only primitive fields.
        // sglang's contains prost_types::{Struct,Timestamp} so it's handled separately.
        .type_attribute(
            "vllm.grpc.engine.GetServerInfoResponse",
            "#[derive(serde::Serialize)]",
        )
        .type_attribute(
            "trtllm.GetServerInfoResponse",
            "#[derive(serde::Serialize)]",
        )
        .type_attribute(
            "mlx.grpc.engine.GetServerInfoResponse",
            "#[derive(serde::Serialize)]",
        )
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                "proto/sglang_scheduler.proto",
                "proto/vllm_engine.proto",
                "proto/trtllm_service.proto",
                "proto/mlx_engine.proto",
            ],
            &["proto"],
        )?;

    // Pass 3: compile the TokenSpeed scheduler proto. It imports SGLang
    // message types (see proto/tokenspeed_scheduler.proto), so we point
    // tonic at the already-generated ``crate::sglang_scheduler::proto``
    // module via ``extern_path`` — otherwise tonic would try to generate a
    // parallel ``sglang.grpc.scheduler`` module under tokenspeed's output
    // file and produce ``too many leading super keywords`` errors.
    //
    // Write into a dedicated sub-directory so this pass's ``extern_path``
    // on the SGLang package doesn't clobber Pass 2's ``sglang.grpc.scheduler.rs``
    // — tonic still emits a stub file for every compiled proto even when
    // the message types inside are extern-pathed out.
    let ts_out_dir = std::path::PathBuf::from(std::env::var("OUT_DIR")?).join("tokenspeed");
    std::fs::create_dir_all(&ts_out_dir)?;
    tonic_prost_build::configure()
        .build_server(true)
        .build_client(true)
        .out_dir(&ts_out_dir)
        .extern_path(".smg.grpc.common", "crate::common_proto")
        .extern_path(".sglang.grpc.scheduler", "crate::sglang_scheduler::proto")
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(&["proto/tokenspeed_scheduler.proto"], &["proto"])?;

    Ok(())
}
