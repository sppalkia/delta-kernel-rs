pub type DeltaResult<T, E = Error> = std::result::Result<T, E>;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow_schema::ArrowError),

    #[error("Generic delta kernel error: {0}")]
    Generic(String),

    #[error("Generic error: {source}")]
    GenericError {
        /// Source error
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    #[error("Arrow error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),

    #[error("Error interacting with object store: {0}")]
    ObjectStore(object_store::Error),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("{0}")]
    MissingColumn(String),

    #[error("Expected column type: {0}")]
    UnexpectedColumnType(String),

    #[error("Expected is missing: {0}")]
    MissingData(String),

    #[error("No table version found.")]
    MissingVersion,

    #[error("Deleteion Vecor error: {0}")]
    DeletionVector(String),

    #[error("Invalid url: {0}")]
    InvalidUrl(#[from] url::ParseError),

    #[error("Invalid url: {0}")]
    MalformedJson(#[from] serde_json::Error),

    #[error("No table metadata found in delta log.")]
    MissingMetadata,
}

impl From<object_store::Error> for Error {
    fn from(value: object_store::Error) -> Self {
        match value {
            object_store::Error::NotFound { path, .. } => Self::FileNotFound(path),
            err => Self::ObjectStore(err),
        }
    }
}
