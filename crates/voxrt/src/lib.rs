//! voxrt: inference engine for `moondream/parakeet-redux`.

pub mod audio;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod format;
pub mod frontend;
pub mod kernels;
pub mod longform;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub mod metal;
pub mod packed;
pub mod subsampling;
pub mod tensor;
pub mod ternary_linear;
pub mod tokenizer;
pub mod transcriber;
pub mod trits;
pub mod vad;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;
