//! prx-core: inference engine for `moondream/parakeet-redux`.
//!
//! Phase 2 (scalar fp32, correctness first). See `PLAN.md` and `docs/` at the repo root.

pub mod audio;
pub mod decoder;
pub mod encoder;
pub mod error;
pub mod format;
pub mod frontend;
pub mod subsampling;
pub mod tensor;
pub mod tokenizer;
pub mod transcriber;
pub mod trits;

pub use error::Error;
pub type Result<T> = std::result::Result<T, Error>;
