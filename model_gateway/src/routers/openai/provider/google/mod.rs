#[expect(
    clippy::module_inception,
    reason = "keep module path explicit as provider/google/google.rs for parity-oriented reviewability"
)]
mod google;
mod request;

pub(crate) use google::GoogleProvider;
