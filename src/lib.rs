//! Zarr Version 2.0 implementation.

#![deny(missing_debug_implementations, unused_imports)]

#[macro_use]
pub extern crate smallvec;

use std::io::Error;
use std::time::SystemTime;

use anyhow::Result;
use camino::Utf8PathBuf;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use smallvec::SmallVec;
use thiserror::Error;

pub mod chunk;
use crate::chunk::{
    DataChunk, ReadableDataChunk, ReinitDataChunk, SliceDataChunk, VecDataChunk, WriteableDataChunk,
};

pub mod compression;

#[macro_use]
pub mod data_type;
pub use data_type::*;

#[cfg(feature = "use_ndarray")]
pub mod ndarray;

pub mod storage;

pub mod store;

pub mod prelude;

#[cfg(test)]
#[macro_use]
pub(crate) mod tests;

mod array;
pub use array::{ArrayMetadata, ArrayMetadataBuilder, Filters, Order};

mod filter;
pub use filter::{AsType, Filter};

pub use semver::{BuildMetadata, Prerelease, Version, VersionReq};

const COORD_SMALLVEC_SIZE: usize = 6;

/// Type for array shapes.
pub type CoordVec<T> = SmallVec<[T; COORD_SMALLVEC_SIZE]>;

/// Type specifying the shape of array chunking.
pub type ChunkCoord = CoordVec<u32>;

/// Type specifying the shape of an array.
pub type GridCoord = CoordVec<u64>;

/// Version of the Zarr spec supported by this library.
pub const VERSION: Version = Version {
    major: 2,
    minor: 0,
    patch: 0,
    pre: Prerelease::EMPTY,
    build: BuildMetadata::EMPTY,
};

const ARRAY_METADATA_KEY_EXT: &str = ".zarray";
const ATTRS_METADATA_KEY_EXT: &str = ".zattrs";
const GROUP_METADATA_KEY_EXT: &str = ".zgroup";

// Work around lack of Rust enum variant types (#2593) for `Value::Object(..)`
// to still provide guarantee of correct type for attributes.
type JsonObject = serde_json::Map<String, Value>;

/// An error raised from reading or writing metadata.
#[derive(Error, Debug)]
pub enum MetadataError {
    /// Error for an invalid/unexpected type in group/array metadata.
    #[error("value was not of the expected type: {0}")]
    UnexpectedType(Value),
}

impl From<MetadataError> for Error {
    fn from(e: MetadataError) -> Self {
        match e {
            MetadataError::UnexpectedType(..) => Error::new(std::io::ErrorKind::InvalidData, e),
        }
    }
}

fn zarr_format_version() -> u64 {
    VERSION.major
}

/// Store metadata about a node.
///
/// This is metadata from the persistence layer of the hierarchy, such as
/// filesystem access times and on-disk sizes, and is not to be confused with
/// semantic metadata stored as attributes in the hierarchy.
#[derive(Clone, Debug)]
pub struct StoreNodeMetadata {
    pub created: Option<SystemTime>,
    pub accessed: Option<SystemTime>,
    pub modified: Option<SystemTime>,
    pub size: Option<u64>,
}

/// Canonicalize path for concatenation into keys by stripping leading or trailing
/// slashes. This does not attempt to remove roots or relative paths as the
/// store may do.
fn canonicalize_path(path: &str) -> &str {
    path.trim_start_matches('/').trim_end_matches('/')
}

/// Core trait of a Zarr store.
pub trait Hierarchy {
    /// Returns the key to the metadata of the array at `path_name`.
    fn array_metadata_key(&self, path_name: &str) -> Utf8PathBuf {
        Utf8PathBuf::from(canonicalize_path(path_name)).join(ARRAY_METADATA_KEY_EXT)
    }

    /// Returns the key to the metadata of the attributes at `path_name`.
    fn attrs_metadata_key(&self, path_name: &str) -> Utf8PathBuf {
        Utf8PathBuf::from(canonicalize_path(path_name)).join(ATTRS_METADATA_KEY_EXT)
    }

    /// Returns the key to the metadata of the group at `path_name`.
    fn group_metadata_key(&self, path_name: &str) -> Utf8PathBuf {
        Utf8PathBuf::from(canonicalize_path(path_name)).join(GROUP_METADATA_KEY_EXT)
    }

    fn data_path_key(&self, path_name: &str) -> Utf8PathBuf {
        Utf8PathBuf::from(canonicalize_path(path_name))
    }
}

/// Non-mutating operations on Zarr hierarchys.
pub trait HierarchyReader: Hierarchy {
    /// Get metadata for an array.
    ///
    /// # Errors
    ///
    /// This function should return an error if trying to read from `path_name` results in an IO
    /// error or if `path_name` doesn't point to an existing array.
    fn get_array_metadata(&self, path_name: &str) -> Result<ArrayMetadata, Error>;

    /// Returns that attributes at the given path, if there are any.
    ///
    /// # Errors
    ///
    /// This function should return an error if trying to read from `path_name` results in an IO
    /// error or if `path_name` doesn't point to an existing array or group.
    fn get_attributes(&self, path_name: &str) -> Result<Option<JsonObject>, Error>;

    /// Returns `true` if a group or array exists.
    ///
    /// # Errors
    ///
    /// This function should return an error if trying to read from `path_name` results in an IO
    /// error.
    fn exists(&self, path_name: &str) -> Result<bool, Error>;

    /// Returns `true` if an array exists.
    ///
    /// # Errors
    ///
    /// This function should return an error if trying to read from `path_name` results in an IO
    /// error.
    fn array_exists(&self, path_name: &str) -> Result<bool, Error> {
        Ok(self.exists(path_name)? && self.get_array_metadata(path_name).is_ok())
    }

    /// Get a URI string for a data chunk.
    ///
    /// Whether this requires that the array and chunk exist is currently
    /// implementation dependent. Whether this URI is a URL is implementation
    /// dependent.
    ///
    /// # Errors
    ///
    /// A call to `get_chunk_uri` may return an I/O error indicating the operation could not be
    /// completed.
    fn get_chunk_uri(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<String, Error>;

    /// Read a single array chunk into a linear vec.
    ///
    /// # Errors
    ///
    /// A call to `read_chunk` may return an I/O error indicating the operation could not be
    /// completed.
    fn read_chunk<T>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataChunk<T>>, Error>
    where
        VecDataChunk<T>: DataChunk<T> + ReadableDataChunk,
        T: ReflectedType;

    /// Read a single array chunk into an existing buffer.
    ///
    /// # Errors
    ///
    /// A call to `read_chunk_into` may return an I/O error indicating the operation could not be
    /// completed.
    fn read_chunk_into<T: ReflectedType, B: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
        chunk: &mut B,
    ) -> Result<Option<()>, Error>;

    /// Read store metadata about a chunk.
    ///
    /// # Errors
    ///
    /// A call to `store_chunk_metadata` may return an I/O error indicating the operation could not
    /// be completed.
    fn store_chunk_metadata(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<Option<StoreNodeMetadata>, Error>;
}

/// Non-mutating operations on Zarr hierarchys that support group discoverability.
pub trait HierarchyLister {
    /// List all groups (including arrays) in a group.
    ///
    /// # Errors
    ///
    /// A call to `list_nodes` may return an I/O error indicating the operation could not be
    /// completed.
    fn list_nodes(&self, prefix_path: &str) -> Result<Vec<String>, Error>;
}

/// Mutating operations on Zarr hierarchys.
pub trait HierarchyWriter: HierarchyReader {
    /// Set a single attribute.
    ///
    /// # Errors
    ///
    /// A call to `set_attribute` may return an I/O error indicating the operation could not be
    /// completed.
    fn set_attribute<T: Serialize>(
        &self,
        path_name: &str,
        key: String,
        attribute: T,
    ) -> Result<(), Error> {
        self.set_attributes(
            path_name,
            vec![(key, serde_json::to_value(attribute)?)]
                .into_iter()
                .collect(),
        )
    }

    /// Set a map of attributes for a group or array.
    ///
    /// # Errors
    ///
    /// A call to `set_attributes` may return an I/O error indicating the operation could not be
    /// completed.
    // TODO: determine/fix behavior for implicit groups
    fn set_attributes(&self, path_name: &str, attributes: JsonObject) -> Result<(), Error>;

    /// Create a group (directory).
    ///
    /// # Errors
    ///
    /// A call to `create_group` may return an I/O error indicating the operation could not be
    /// completed.
    fn create_group(&self, path_name: &str) -> Result<(), Error>;

    /// Create an array. This will create the array group and attributes,
    /// but not populate any chunk data.
    ///
    /// # Errors
    ///
    /// A call to `create_array` may return an I/O error indicating the operation could not be
    /// completed.
    fn create_array(&self, path_name: &str, array_meta: &ArrayMetadata) -> Result<(), Error>;

    /// Remove the Zarr hierarchy.
    ///
    /// # Errors
    ///
    /// A call to `remove_all` may return an I/O error indicating the operation could not be
    /// completed.
    fn remove_all(&self) -> Result<(), Error> {
        self.remove("")
    }

    /// Remove a group or array (directory and all contained files).
    ///
    /// This will wait on locks acquired by other writers or readers.
    ///
    /// # Errors
    ///
    /// A call to `remove` may return an I/O error indicating the operation could not be
    /// completed or if the a file lock cannot be obtained.
    fn remove(&self, path_name: &str) -> Result<(), Error>;

    /// Write a chunk of an array to the store with key `path_name`.
    ///
    /// # Errors
    ///
    /// A call to `write_chunk` may return an I/O error indicating the operation could not be
    /// completed.
    fn write_chunk<T: ReflectedType, B: DataChunk<T> + WriteableDataChunk>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        chunk: &B,
    ) -> Result<(), Error>;

    /// Delete a chunk from an array.
    ///
    /// Returns `true` if the chunk does not exist on the backend at the
    /// completion of the call.
    ///
    /// # Errors
    ///
    /// A call to `delete_chunk` may return an I/O error indicating the operation could not be
    /// completed.
    fn delete_chunk(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<bool, Error>;
}

/// Metadata for groups.
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone, Copy)]
pub struct GroupMetadata {
    #[serde(default = "zarr_format_version")]
    zarr_format: u64,
}

impl Default for GroupMetadata {
    fn default() -> Self {
        Self {
            zarr_format: zarr_format_version(),
        }
    }
}
