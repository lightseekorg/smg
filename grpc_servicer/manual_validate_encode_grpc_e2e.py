"""Manual GPU e2e: gateway Encode RPC -> engine encode worker -> Mooncake -> prefill.

Two processes on one box (GPUs free, Mooncake auto-falls-back to TCP):
- ``server``: serve_grpc(disaggregation_mode=encode) -- launches the engine's
  run_encode_loop (vision-only Qwen3.5 + Mooncake encode manager + bootstrap
  server) behind the gRPC TokenSpeedEncoder service.
- main: builds a proto EncodeRequest (random pixels + grid_thw), sends it over
  gRPC, then receives the image embedding back over Mooncake via the real
  receive_encoded_embeddings and asserts it arrived with the right shape and is
  finite + non-trivial (the tower actually ran). Exact-byte correctness of the
  transfer is covered separately by manual_validate_encode_e2e.

Run: <env> python manual_validate_encode_grpc_e2e.py  (auto-spawns the server).
Skips cleanly without CUDA / model / mooncake.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

MODEL = os.environ.get("ENCODE_VIT_MODEL", "/scratch/models/Qwen/Qwen3.5-2B")
HOST = "127.0.0.1"
GRPC_PORT = 19100
BOOT_PORT = 19101
ROOM = 778899
GRID_T, GRID_H, GRID_W = 1, 4, 6  # -> 24 patches, post_merge = 1*(4//2)*(6//2) = 6
PATCH_DIM = 1536  # Qwen3.5 vision patch features (3*2*16*16)
POST_MERGE = GRID_T * (GRID_H // 2) * (GRID_W // 2)


def _server_args():
    from tokenspeed.runtime.utils.server_args import ServerArgs

    return ServerArgs(
        model=MODEL,
        trust_remote_code=True,
        disaggregation_mode="encode",
        disaggregation_bootstrap_port=BOOT_PORT,
        host=HOST,
        port=GRPC_PORT,
    )


def run_server() -> int:
    from smg_grpc_servicer.tokenspeed.server import serve_grpc

    asyncio.run(serve_grpc(_server_args()))
    return 0


def _tensor_data(torch_tensor, dtype_str):
    import smg_grpc_proto.generated.tokenspeed_scheduler_pb2 as sch
    import torch

    t = torch_tensor.cpu().contiguous()
    if t.dtype == torch.bfloat16:
        raw = t.view(torch.int16).numpy().tobytes()
    else:
        raw = t.numpy().tobytes()
    return sch.TensorData(data=raw, shape=list(t.shape), dtype=dtype_str)


def run_client() -> int:
    import grpc
    import torch

    import smg_grpc_proto.generated.tokenspeed_encoder_pb2 as enc
    import smg_grpc_proto.generated.tokenspeed_encoder_pb2_grpc as enc_grpc
    import smg_grpc_proto.generated.tokenspeed_scheduler_pb2 as sch
    from tokenspeed.runtime.multimodal.inputs import Modality, MultimodalDataItem
    from tokenspeed.runtime.pd.encode_receiver import receive_encoded_embeddings
    from tokenspeed.runtime.pd.mooncake.embedding_transfer import (
        EmbeddingArgs,
        MooncakeEmbeddingManagerPrefill,
    )
    from tokenspeed.runtime.pd.mooncake.entities import ManagerArgs
    from tokenspeed.runtime.utils.network import get_local_ip_by_remote

    # --- Build the proto EncodeRequest: random pixels + grid_thw, one image. ---
    pixels = torch.randn(GRID_T * GRID_H * GRID_W, PATCH_DIM, dtype=torch.bfloat16)
    grid = torch.tensor([[GRID_T, GRID_H, GRID_W]], dtype=torch.int64)
    mm = sch.MultimodalInputs(
        pixel_values=_tensor_data(pixels, "bfloat16"),
        model_specific_tensors={"image_grid_thw": _tensor_data(grid, "int64")},
    )
    request = enc.EncodeRequest(
        request_id="grpc-e2e-0",
        mm_inputs=mm,
        items=[enc.EncodeItemAssignment(item_index=0, bootstrap_room=ROOM)],
    )

    # --- Send the Encode RPC (retry until the server is up). ---
    channel = grpc.insecure_channel(f"{HOST}:{GRPC_PORT}")
    stub = enc_grpc.TokenSpeedEncoderStub(channel)
    deadline = time.monotonic() + 180
    while True:
        try:
            resp = stub.Encode(request, timeout=30)
            break
        except grpc.RpcError as e:
            if time.monotonic() > deadline:
                print(f"FAIL: Encode RPC never succeeded: {e}")
                return 1
            time.sleep(1.0)
    print(f"Encode RPC accepted={resp.accepted}")

    # --- Receive the embedding over Mooncake via the real receive glue. ---
    margs = ManagerArgs(
        bootstrap_port=BOOT_PORT + 1000,
        dist_init_addr=None,
        world_size=1,
        dp_size=1,
        attn_tp_rank=0,
        attn_dp_rank=0,
        is_mla_backend=False,
        draft_is_mla_backend=False,
        enable_metrics=False,
        enable_mla_l1_5_cache=False,
        served_model_name="x",
        app_key="",
        metrics_reporters=None,
        enable_dp_attention=False,
    )
    mgr = MooncakeEmbeddingManagerPrefill(margs, EmbeddingArgs(0, 0, None, 0, 0))

    hidden = 2048  # Qwen3.5-2B text hidden = tower output width (no deepstack)
    item = MultimodalDataItem(modality=Modality.IMAGE, offsets=[(0, POST_MERGE - 1)])
    ip = get_local_ip_by_remote()
    handshakes = [
        {"item_index": 0, "bootstrap_room": ROOM, "bootstrap_host": ip, "bootstrap_port": BOOT_PORT}
    ]
    receive_encoded_embeddings(
        items=[item],
        handshakes=handshakes,
        manager=mgr,
        hidden=hidden,
        num_deepstack=0,
        dtype=torch.bfloat16,
        device="cuda:0",
        timeout=120,
    )
    torch.cuda.synchronize()

    enc_t = item.encoded
    ok = (
        enc_t is not None
        and tuple(enc_t.shape) == (POST_MERGE, hidden)
        and torch.isfinite(enc_t.float()).all().item()
        and enc_t.float().abs().sum().item() > 0
    )
    print(
        f"PREFILL received shape={tuple(enc_t.shape) if enc_t is not None else None} "
        f"finite_nonzero={ok}"
    )
    print("GRPC ENCODE E2E OK" if ok else "GRPC ENCODE E2E FAILED")
    return 0 if ok else 1


def main() -> int:
    import torch

    if not torch.cuda.is_available():
        print("SKIP: no CUDA")
        return 0
    if not os.path.isdir(MODEL):
        print(f"SKIP: model not found at {MODEL}")
        return 0
    try:
        import mooncake  # noqa: F401
    except Exception:
        print("SKIP: mooncake not importable")
        return 0

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        return run_server()

    server = subprocess.Popen([sys.executable, __file__, "server"])
    try:
        rc = run_client()
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
