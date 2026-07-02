//! Optional multimodal pixel RDMA transport.
//!
//! The default build uses a no-op shim so ordinary gateway builds do not need
//! NIXL headers/bindgen. Enable the `mm-rdma` Cargo feature to compile the NIXL
//! implementation.

#[cfg(feature = "mm-rdma")]
mod nixl;

#[cfg(not(feature = "mm-rdma"))]
mod stub;

#[cfg(feature = "mm-rdma")]
pub(crate) use nixl::*;
#[cfg(not(feature = "mm-rdma"))]
pub(crate) use stub::*;
