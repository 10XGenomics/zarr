use serde::{Deserialize, Serialize};

use crate::DataType;

/// Filters to transform data prior to compression and after decompression.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "id")]
pub enum Filter {
    /// Convert the data between two types.
    #[serde(rename = "astype")]
    AsType(AsType),
}

/// Filter to convert data between different types.
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct AsType {
    /// Data type to use for encoded data.
    encode_dtype: DataType,
    /// Data type to use for decoded data.
    decode_dtype: DataType,
}

impl AsType {
    /// Create a new [`AsType`] filter.
    #[must_use]
    pub fn new(encode: DataType, decode: DataType) -> Self {
        Self {
            encode_dtype: encode,
            decode_dtype: decode,
        }
    }
}
