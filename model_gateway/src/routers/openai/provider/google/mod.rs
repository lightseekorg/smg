#[expect(
    clippy::module_inception,
    reason = "keep module path explicit as provider/google/google.rs for parity-oriented reviewability"
)]
mod google;
mod request;
mod response;
mod streaming;

pub(crate) use google::GoogleProvider;

#[cfg(test)]
mod tests;
