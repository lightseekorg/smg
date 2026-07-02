//! No-op multimodal RDMA transport used when the `mm-rdma` feature is disabled.

pub(crate) fn rdma_enabled() -> bool {
    false
}

pub(crate) fn export_pixel_buffer(_room: i64, bytes: Vec<u8>) -> Result<Vec<u8>, Vec<u8>> {
    Err(bytes)
}
