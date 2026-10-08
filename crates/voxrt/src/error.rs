#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("safetensors: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("model format: {0}")]
    Format(String),
    #[error("audio: {0}")]
    Audio(String),
}
