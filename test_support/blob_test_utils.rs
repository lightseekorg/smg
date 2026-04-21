use std::io::Cursor;

use anyhow::Result;
use smg_blob_storage::{GetBlobResponse, PutBlobRequest};
use tokio::io::AsyncReadExt;

pub fn put_request(bytes: &[u8]) -> PutBlobRequest {
    PutBlobRequest {
        reader: Box::pin(Cursor::new(bytes.to_vec())),
        content_length: bytes.len() as u64,
        content_type: None,
    }
}

pub async fn read_all(response: GetBlobResponse) -> Result<Vec<u8>> {
    let mut reader = response.reader;
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer).await?;
    Ok(buffer)
}
