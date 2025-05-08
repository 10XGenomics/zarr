use anyhow::{Context, Result};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::Filter;

use super::{compression, ChunkCoord, DataType, Error, GridCoord, ReflectedType};

/// Memory layout of a stored Zarr array.
#[derive(Default, Serialize, Deserialize, PartialEq, Debug, Clone, Copy)]
pub enum Order {
    /// Row-major (c-style) memory layout.
    #[serde(rename = "C")]
    RowMajor,
    /// Column-major (fortan-style) memory layout.
    #[serde(rename = "F")]
    #[default]
    ColumnMajor,
}

/// A list of codec configurations of filters to apply to data.
#[derive(Default, Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(untagged)]
pub enum Filters {
    /// List of filters.
    Array(Vec<Filter>),
    /// No filter to be applied.
    #[default]
    Null,
}

/// Attributes of a tensor array for the Zarr v2 spec.
#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
pub struct ArrayMetadata {
    /// An integer defining the version of the storage specification to which the array store
    /// adheres.
    zarr_format: usize,
    /// A list of integers defining the length of each dimension of the array.
    shape: GridCoord,
    /// A list of integers defining the length of each dimension of a chunk of the array.
    ///
    /// Note that all chunks within a Zarr array have the same shape.
    chunks: ChunkCoord,
    /// A string or list defining a valid data type for the array.
    ///
    /// See also the subsection below on data type encoding.
    dtype: DataType,
    /// A JSON object identifying the primary compression codec and providing configuration
    /// parameters, or null if no compressor is to be used.
    ///
    /// The object MUST contain an "id" key identifying the codec to be used.
    #[serde(default)]
    #[serde(skip_serializing_if = "compression::CompressionType::is_default")]
    compressor: compression::CompressionType,
    /// A scalar value providing the default value to use for uninitialized portions of the array,
    /// or null if no fill_value is to be used.
    fill_value: Option<Value>,
    /// Either “C” or “F”, defining the layout of bytes within each chunk of the array.
    ///
    /// “C” means row-major order, i.e., the last dimension varies fastest; “F” means column-major
    /// order, i.e., the first dimension varies fastest.
    order: Order,
    /// A list of JSON objects providing codec configurations, or null if no filters are to be
    /// applied.
    ///
    /// Each codec configuration object MUST contain a "id" key identifying the codec to be used.
    #[serde(default)]
    filters: Filters,
    /// If present, either the string "." or "/" defining the separator placed between the
    /// dimensions of a chunk.
    ///
    /// If the value is not set, then the default MUST be assumed to be ".", leading to chunk keys
    /// of the form “0.0”. Arrays defined with "/" as the dimension separator can be considered to
    /// have nested, or hierarchical, keys of the form “0/0” that SHOULD where possible produce a
    /// directory-like structure.
    #[serde(default = "default_dimension_separator")]
    dimension_separator: String,
}

fn default_dimension_separator() -> String {
    ".".to_string()
}

impl ArrayMetadata {
    /// Create a new [`ArrayMetadata`].
    ///
    /// # Note
    ///
    /// For more flexiblity, use the [`ArrayMetadataBuilder`].
    ///
    /// # Panics
    ///
    /// This function will panic if the shape and chunk dimensions don't match.
    pub fn new<D: Into<DataType>>(
        shape: GridCoord,
        chunk_shape: ChunkCoord,
        data_type: D,
        compressor: compression::CompressionType,
    ) -> ArrayMetadata {
        assert_eq!(
            shape.len(),
            chunk_shape.len(),
            "Number of array dimensions must match number of chunk size dimensions."
        );
        ArrayMetadataBuilder::default()
            .shape(shape)
            .dtype(data_type.into())
            .chunks(chunk_shape)
            .order(Order::ColumnMajor)
            .compressor(compressor)
            .build()
            .unwrap()
    }

    /// Returns a reference to the array shape.
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        self.shape.as_slice()
    }
    /// Returns a reference to the array shape.
    pub fn shape_mut(&mut self) -> &mut GridCoord {
        &mut self.shape
    }

    /// Returns a reference to the array chunk shape.
    #[must_use]
    pub fn chunk_shape(&self) -> &[u32] {
        &self.chunks
    }

    /// Returns a mutable reference to the array chunk shape.
    pub fn chunk_shape_mut(&mut self) -> &mut ChunkCoord {
        &mut self.chunks
    }

    /// Returns the array memory layout.
    #[must_use]
    pub fn order(&self) -> Order {
        self.order
    }

    /// Returns a mutable reference to the array memory layout.
    pub fn order_mut(&mut self) -> &mut Order {
        &mut self.order
    }

    /// Returns the array fill value.
    #[must_use]
    pub fn fill_value(&self) -> Option<&Value> {
        self.fill_value.as_ref()
    }

    /// Returns the [`DataType`] of the array.
    #[must_use]
    pub fn dtype(&self) -> DataType {
        self.dtype
    }

    /// Returns a reference to the array [compression type].
    ///
    /// [compression type]: crate::compression::CompressionType
    #[must_use]
    pub fn compressor(&self) -> &compression::CompressionType {
        &self.compressor
    }

    /// Returns a mutable reference to the array [compression type].
    ///
    /// [compression type]: crate::compression::CompressionType
    pub fn compressor_mut(&mut self) -> &mut compression::CompressionType {
        &mut self.compressor
    }

    /// Returns the dimension separator for the array chunks.
    #[must_use]
    pub fn dimension_separator(&self) -> &str {
        &self.dimension_separator
    }

    /// Returns the fill value as a JSON type.
    ///
    /// # Errors
    ///
    /// An error is returned if serialization of the data type fails.
    pub fn get_effective_fill_value<T: ReflectedType>(&self) -> Result<T, Error> {
        Ok(self
            .fill_value()
            .map(|v| serde_json::from_value(v.clone()))
            .transpose()?
            .unwrap_or_else(T::default))
    }

    /// Returns the number of array dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Returns the total number of elements possible given the shape.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Returns the total number of elements possible in a chunk.
    #[must_use]
    pub fn chunk_num_elements(&self) -> usize {
        self.chunks.iter().map(|&d| d as usize).product()
    }

    /// Returns the upper bound extent of grid coordinates.
    #[must_use]
    pub fn grid_extent(&self) -> GridCoord {
        self.shape
            .iter()
            .zip(self.chunks.iter())
            .map(|(&d, &s)| (d + u64::from(s) - 1) / u64::from(s))
            .collect()
    }

    /// Get the total number of chunks.
    ///
    /// ```
    /// use zarr::prelude::*;
    /// use zarr::smallvec::smallvec;
    /// let attrs = ArrayMetadata::new(
    ///     smallvec![50, 40, 30],
    ///     smallvec![11, 10, 10],
    ///     i8::ZARR_TYPE,
    ///     zarr::compression::CompressionType::default(),
    /// );
    /// assert_eq!(attrs.num_chunks(), 60);
    /// ```
    #[must_use]
    pub fn num_chunks(&self) -> u64 {
        self.grid_extent().iter().product()
    }

    /// Check whether a chunk grid position is in the bounds of this array.
    /// ```
    /// use zarr::prelude::*;
    /// use zarr::smallvec::smallvec;
    /// let attrs = ArrayMetadata::new(
    ///     smallvec![50, 40, 30],
    ///     smallvec![11, 10, 10],
    ///     i8::ZARR_TYPE,
    ///     zarr::compression::CompressionType::default(),
    /// );
    /// assert!(attrs.in_bounds(&smallvec![4, 3, 2]));
    /// assert!(!attrs.in_bounds(&smallvec![5, 3, 2]));
    /// ```
    #[must_use]
    pub fn in_bounds(&self, grid_position: &GridCoord) -> bool {
        self.shape.len() == grid_position.len()
            && self
                .grid_extent()
                .iter()
                .zip(grid_position.iter())
                .all(|(&bound, &coord)| coord < bound)
    }
}

impl ArrayMetadata {
    /// The filename for array metadata.
    #[must_use]
    pub fn metadata_filename() -> &'static str {
        ".zarray"
    }
}

/// A builder for creating [`ArrayMetadata`].
#[derive(Debug, Clone, Default)]
pub struct ArrayMetadataBuilder {
    /// A list of integers defining the length of each dimension of the array.
    shape: Option<GridCoord>,
    /// A list of integers defining the length of each dimension of a chunk of the array.
    ///
    /// Note that all chunks within a Zarr array have the same shape.
    chunks: Option<ChunkCoord>,
    /// A string or list defining a valid data type for the array.
    ///
    /// See also the subsection below on data type encoding.
    dtype: Option<DataType>,
    /// A JSON object identifying the primary compression codec and providing configuration
    /// parameters, or null if no compressor is to be used.
    ///
    /// The object MUST contain an "id" key identifying the codec to be used.
    compressor: Option<compression::CompressionType>,
    /// A scalar value providing the default value to use for uninitialized portions of the array,
    /// or null if no fill_value is to be used.
    fill_value: Option<Value>,
    /// Either “C” or “F”, defining the layout of bytes within each chunk of the array.
    ///
    /// “C” means row-major order, i.e., the last dimension varies fastest; “F” means column-major
    /// order, i.e., the first dimension varies fastest.
    order: Option<Order>,
    /// A list of JSON objects providing codec configurations, or null if no filters are to be
    /// applied.
    ///
    /// Each codec configuration object MUST contain a "id" key identifying the codec to be used.
    filters: Option<Filters>,
    /// If present, either the string "." or "/" defining the separator placed between the
    /// dimensions of a chunk.
    ///
    /// If the value is not set, then the default MUST be assumed to be ".", leading to chunk keys
    /// of the form “0.0”. Arrays defined with "/" as the dimension separator can be considered to
    /// have nested, or hierarchical, keys of the form “0/0” that SHOULD where possible produce a
    /// directory-like structure.
    dimension_separator: Option<String>,
}

impl ArrayMetadataBuilder {
    /// Set the shape of the array.
    #[must_use]
    pub fn shape(mut self, shape: GridCoord) -> Self {
        self.shape = Some(shape);
        self
    }
    /// Set the shape of the chunks.
    ///
    /// Note that all chunks within a Zarr array have the same shape.
    #[must_use]
    pub fn chunks(mut self, chunks: ChunkCoord) -> Self {
        self.chunks = Some(chunks);
        self
    }
    /// Set the data type of the array.
    ///
    /// See also the subsection below on data type encoding.
    #[must_use]
    pub fn dtype(mut self, dtype: DataType) -> Self {
        self.dtype = Some(dtype);
        self
    }
    /// Set the compressor for array storage.
    #[must_use]
    pub fn compressor(mut self, compressor: compression::CompressionType) -> Self {
        self.compressor = Some(compressor);
        self
    }
    /// Set the default value for missing data.
    #[must_use]
    pub fn fill_value(mut self, fill_value: Value) -> Self {
        self.fill_value = Some(fill_value);
        self
    }
    /// Set the memory layout of the array.
    #[must_use]
    pub fn order(mut self, order: Order) -> Self {
        self.order = Some(order);
        self
    }
    /// Set the filters applied before encoding and after decoding.
    #[must_use]
    pub fn filters(mut self, filters: Filters) -> Self {
        self.filters = Some(filters);
        self
    }
    /// Set the separator for chunk keys.
    #[must_use]
    pub fn dimension_separator(mut self, dimension_separator: String) -> Self {
        self.dimension_separator = Some(dimension_separator);
        self
    }

    /// Build the array metadata.
    ///
    /// # Errors
    ///
    /// An error is returned if any of the following are not specified:
    /// - `shape`
    /// - `chunks`
    /// - `dtype`
    pub fn build(self) -> Result<ArrayMetadata> {
        let ArrayMetadataBuilder {
            shape,
            chunks,
            dtype,
            compressor,
            fill_value,
            order,
            filters,
            dimension_separator,
        } = self;
        Ok(ArrayMetadata {
            zarr_format: 2,
            shape: shape.context("`shape` is required.")?,
            chunks: chunks.context("`chunks` is required.")?,
            dtype: dtype.context("`dtype` is required.")?,
            order: order.unwrap_or_default(),
            compressor: compressor.unwrap_or_default(),
            filters: filters.unwrap_or_default(),
            dimension_separator: dimension_separator.unwrap_or_else(default_dimension_separator),
            fill_value,
        })
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use serde_json::Value;

    use crate::{
        compression::blosc::{BloscCompressionBuilder, Compressor},
        data_type::{DataType, ReflectedType},
        filter::{AsType, Filter},
        Filters, Order,
    };

    use super::{ArrayMetadata, ArrayMetadataBuilder};

    #[cfg(feature = "gzip")]
    #[test]
    fn array_metadata_deserialization() {
        let example_json = r#"
        {
            "chunks": [10017, 1],
            "compressor": {
                "blocksize": 0,
                "clevel": 5,
                "cname": "zstd",
                "id": "blosc",
                "shuffle": 1
            },
            "dtype": "|b1",
            "fill_value": false,
            "filters": [
                {
                    "decode_dtype": "|b1",
                    "encode_dtype": "|b1",
                    "id": "astype"
                },
                {
                    "decode_dtype": "|b1",
                    "encode_dtype": "|b1",
                    "id": "astype"
                }
            ],
            "order": "C",
            "shape": [10017, 8],
            "zarr_format": 2
        }
        "#;
        let deserialized: ArrayMetadata = serde_json::from_str(example_json).unwrap();

        let mut expected = ArrayMetadataBuilder::default()
            .shape(smallvec![10017, 8])
            .dtype(DataType::bool())
            .chunks(smallvec![10017, 1])
            .dimension_separator(".".into())
            .order(Order::RowMajor)
            .compressor(
                BloscCompressionBuilder::new(Compressor::Zstd)
                    .blocksize(0)
                    .clevel(5)
                    .shuffle(1)
                    .build()
                    .into(),
            )
            .fill_value(serde_json::Value::Bool(false))
            .filters(Filters::Array(vec![
                Filter::AsType(AsType::new(DataType::bool(), DataType::bool())),
                Filter::AsType(AsType::new(DataType::bool(), DataType::bool())),
            ]))
            .build()
            .unwrap();

        assert_eq!(deserialized, expected);

        let example_json = r#"
        {
            "chunks": [10017, 1],
            "dtype": "|b1",
            "fill_value": false,
            "filters": [
                {
                    "decode_dtype": "|b1",
                    "encode_dtype": "|b1",
                    "id": "astype"
                },
                {
                    "decode_dtype": "|b1",
                    "encode_dtype": "|b1",
                    "id": "astype"
                }
            ],
            "order": "C",
            "shape": [10017, 8],
            "zarr_format": 2
        }
        "#;
        let deserialized: ArrayMetadata = serde_json::from_str(example_json).unwrap();
        *expected.compressor_mut() = crate::compression::raw::RawCompression.into();

        assert_eq!(deserialized, expected);
    }

    #[test]
    fn test_array_metadata_1d() -> Result<()> {
        let raw_metadata = r#"{
        "chunks": [
            2000000
        ],
        "compressor": {
            "blocksize": 0,
            "clevel": 5,
            "cname": "zstd",
            "id": "blosc",
            "shuffle": 1
        },
        "dtype": "<u2",
        "fill_value": 0,
        "filters": null,
        "order": "C",
        "shape": [
            4173030
        ],
        "zarr_format": 2
    }"#;

        let expected = ArrayMetadataBuilder::default()
            .chunks(smallvec![2_000_000])
            .dtype(DataType::u16_le())
            .fill_value(Value::Number(0.into()))
            .filters(Filters::Null)
            .order(Order::RowMajor)
            .shape(smallvec![4_173_030])
            .compressor(
                BloscCompressionBuilder::new(Compressor::Zstd)
                    .blocksize(0)
                    .clevel(5)
                    .shuffle(1)
                    .build()
                    .into(),
            )
            .build()?;

        let deserialized: ArrayMetadata = serde_json::from_str(raw_metadata).unwrap();
        assert_eq!(deserialized, expected);

        println!("{:?}", expected.grid_extent());

        Ok(())
    }

    #[test]
    fn test_array_metadata_props() -> Result<()> {
        let expected = ArrayMetadataBuilder::default()
            .shape(smallvec![30, 10])
            .chunks(smallvec![3, 5])
            .dtype(u16::ZARR_TYPE)
            .build()?;

        println!("{:?}", expected.grid_extent());

        Ok(())
    }
}
