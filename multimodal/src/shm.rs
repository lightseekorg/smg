use std::{
    fs::{self, OpenOptions},
    path::PathBuf,
    sync::Arc,
};

use memmap2::MmapMut;
use thiserror::Error;
use uuid::Uuid;

#[derive(Error, Debug)]
pub enum ShmError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Failed to create shared memory file: {0}")]
    Creation(String),
}

/// Handle to a shared memory region.
/// 
/// This struct manages the lifecycle of the shared memory file.
/// When it is dropped, the underlying file is deleted (unlinked).
#[derive(Debug)]
pub struct ShmHandle {
    /// Unique identifier for this shared memory region
    pub uuid: String,
    /// Size of the region in bytes
    pub size: usize,
    /// Path to the backing file
    pub path: PathBuf,
    /// Memory mapped mutable slice
    mmap: Option<MmapMut>,
    /// Whether to delete the file on drop
    should_delete: bool,
}

impl ShmHandle {
    /// Get a mutable slice to the shared memory
    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        if let Some(mmap) = &mut self.mmap {
            mmap
        } else {
            &mut []
        }
    }

    /// Persist the shared memory file (do not delete on drop).
    /// This is useful when passing ownership to another process.
    pub fn persist(&mut self) {
        self.should_delete = false;
    }
}

impl Drop for ShmHandle {
    fn drop(&mut self) {
        self.mmap.take();

        if self.should_delete && self.path.exists() {
            let _ = fs::remove_file(&self.path);
        }
    }
}

/// Manager for creating shared memory regions.
#[derive(Debug, Clone)]
pub struct SharedMemoryManager {
    /// Directory where shared memory files are stored.
    /// On Linux used /dev/shm, on Windows uses temp dir.
    base_path: PathBuf,
}

impl SharedMemoryManager {
    pub fn new() -> Self {
        let base_path = if cfg!(target_os = "linux") {
            PathBuf::from("/dev/shm")
        } else {
            std::env::temp_dir()
        };

        Self { base_path }
    }

    /// Allocate a new shared memory region of `size` bytes.
    pub fn alloc(&self, size: usize) -> Result<ShmHandle, ShmError> {
        let uuid = Uuid::new_v4().to_string();
        let filename = format!("smg-{}", uuid);
        let path = self.base_path.join(filename);

        // Create the file
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        // Set file size
        file.set_len(size as u64)?;

        // Map into memory
        let mmap = unsafe { MmapMut::map_mut(&file)? };

        Ok(ShmHandle {
            uuid,
            size,
            path,
            mmap: Some(mmap),
            should_delete: true,
        })
    }
}

impl Default for SharedMemoryManager {
    fn default() -> Self {
        Self::new()
    }
}
