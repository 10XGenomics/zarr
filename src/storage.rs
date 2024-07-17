use std::io::{Error, ErrorKind, Read, Write};

use camino::Utf8PathBuf;

use crate::{
    canonicalize_path,
    chunk::{DataChunk, ReadableDataChunk, ReinitDataChunk, VecDataChunk, WriteableDataChunk},
    ArrayMetadata, GridCoord, GroupMetadata, Hierarchy, HierarchyLister, HierarchyReader,
    HierarchyWriter, JsonObject, ReflectedType, StoreNodeMetadata,
};

/// A trait for a Zarr store that can be read.
pub trait ReadableStore {
    /// The store reader type.
    type GetReader: Read;

    /// Returns `true` if an item with the given `key` is present in the store.
    ///
    /// # Errors
    ///
    /// A call to `exists` may return an I/O error indicating the operation could not be completed.
    fn exists(&self, key: &str) -> Result<bool, Error>;

    /// Retreive the item in the store with the given `key`, if it exists.
    ///
    /// # Errors
    ///
    /// A call to `get` may return an I/O error indicating the operation could not be completed.
    fn get(&self, key: &str) -> Result<Option<Self::GetReader>, Error>;

    /// TODO: not in zarr spec
    fn uri(&self, key: &str) -> Result<String, Error>;
}

/// A trait for a Zarr store that can list each node.
pub trait ListableStore {
    /// Retrieve all keys in the store.
    ///
    /// # Errors
    ///
    /// A call to `list` may return an I/O error indicating the operation could not be completed.
    fn list(&self) -> Result<Vec<String>, Error> {
        self.list_prefix("/")
    }

    /// Retrieve all keys with a given prefix.
    ///
    /// # Errors
    ///
    /// A call to `list_prefix` may return an I/O error indicating the operation could not be
    /// completed.
    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>, Error> {
        let mut to_visit = vec![prefix.to_owned()];
        let mut result = vec![];

        while let Some(next) = to_visit.pop() {
            let dir = self.list_dir(&next)?;
            result.extend(dir.0);
            to_visit.extend(dir.1);
        }

        Ok(result)
    }

    /// Retrieve all keys and prefixes with a given prefix and which do not
    /// contain the character “/” after the given prefix.
    ///
    /// # Errors
    ///
    /// A call to `list_dir` may return an I/O error indicating the operation could not be
    /// completed.
    fn list_dir(&self, prefix: &str) -> Result<(Vec<String>, Vec<String>), Error>;
}

/// A trait for a Zarr store that can write new data.
pub trait WriteableStore {
    /// The store writer.
    type SetWriter: Write;

    /// Write a value to the Zarr store with the given `key` using a function `value` that yields
    /// the item.
    ///
    /// # Errors
    ///
    /// A call to `set` may return an I/O error indicating the operation could not be completed.
    fn set<F: FnOnce(Self::SetWriter) -> Result<(), Error>>(
        &self,
        key: &str,
        value: F,
    ) -> Result<(), Error>;

    /// Erase an item from the Zarr store.
    ///
    /// # Errors
    ///
    /// A call to `erase` may return an I/O error indicating the operation could not be completed.
    fn erase(&self, key: &str) -> Result<bool, Error>;

    /// Erase all items from the array store with a `key` containing the given `key_prefix`.
    ///
    /// # Errors
    ///
    /// A call to `erase_prefix` may return an I/O error indicating the operation could not be
    /// completed.
    fn erase_prefix(&self, key_prefix: &str) -> Result<bool, Error>;
}

/// Returns the chunk key of an array chunk at the given grid position.
///
/// # Examples
///
/// ```
/// use zarr::prelude::*;
/// use zarr::smallvec::smallvec;
/// use zarr::storage::get_chunk_key;
/// // Metadata for an array of shape (100, 4, 4) and chunk sizes of (100, 2, 2)
/// let meta = ArrayMetadataBuilder::default()
///     .shape(smallvec![100, 4, 4])
///     .chunks(smallvec![100, 2, 2])
///     .dtype(i8::ZARR_TYPE)
///     .compressor(zarr::compression::CompressionType::default())
///     .build()
///     .unwrap();
/// assert_eq!(get_chunk_key("/foo/baz", &meta, &[0, 0, 0]), "foo/baz/0.0.0");
/// assert_eq!(get_chunk_key("/foo/baz", &meta, &[0, 1, 0]), "foo/baz/0.1.0");
/// assert_eq!(get_chunk_key("/foo/baz", &meta, &[0, 0, 1]), "foo/baz/0.0.1");
/// assert_eq!(get_chunk_key("/foo/baz", &meta, &[0, 1, 1]), "foo/baz/0.1.1");
/// ```
#[must_use]
pub fn get_chunk_key(key: &str, array_meta: &ArrayMetadata, grid_position: &[u64]) -> String {
    use std::fmt::Write;
    // TODO: normalize relative or absolute paths
    let mut chunk_key = format!("{}/", canonicalize_path(key));

    for (i, coord) in grid_position.iter().enumerate() {
        write!(chunk_key, "{coord}").unwrap();
        if i < grid_position.len() - 1 {
            chunk_key.push_str(array_meta.dimension_separator());
        }
    }

    chunk_key
}

fn merge_top_level(a: &mut JsonObject, b: JsonObject) {
    for (k, v) in b {
        a.insert(k, v);
    }
}

impl<S: ReadableStore + Hierarchy> HierarchyReader for S {
    fn get_array_metadata(&self, path_name: &str) -> Result<ArrayMetadata, Error> {
        let array_path = self.array_metadata_key(path_name);
        let value_reader = ReadableStore::get(self, array_path.as_str())?
            .ok_or_else(|| Error::from(ErrorKind::NotFound))?;
        let metadata: ArrayMetadata = serde_json::from_reader(value_reader)?;
        Ok(metadata)
    }

    fn get_attributes(&self, path_name: &str) -> Result<Option<JsonObject>, Error> {
        let key = self.attrs_metadata_key(path_name);

        if let Some(rdr) = self.get(key.as_str())? {
            let attrs: JsonObject = serde_json::from_reader(rdr)?;
            Ok(Some(attrs))
        } else {
            Ok(None)
        }
    }

    fn exists(&self, path_name: &str) -> Result<bool, Error> {
        // TODO: needless path allocs
        // TODO: should follow spec more closely by using `list_dir` for implicit groups.
        Ok(self.exists(self.array_metadata_key(path_name).as_str())?
            || self.exists(self.group_metadata_key(path_name).as_str())?
            || self.exists(
                self.group_metadata_key(path_name)
                    .with_extension("")
                    .with_extension("")
                    .as_str(),
            )?)
    }

    fn get_chunk_uri(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<String, Error> {
        let chunk_key = get_chunk_key(path_name, array_meta, grid_position);
        self.uri(&chunk_key)
    }

    fn read_chunk<T>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
    ) -> Result<Option<VecDataChunk<T>>, Error>
    where
        VecDataChunk<T>: DataChunk<T> + ReadableDataChunk,
        T: ReflectedType,
    {
        // TODO convert asserts to errors
        assert!(array_meta.in_bounds(&grid_position));

        // Construct chunk path string
        let chunk_key = get_chunk_key(path_name, array_meta, &grid_position);

        // Get key from store
        let value_reader = ReadableStore::get(self, &chunk_key)?;

        // Read value into container
        value_reader
            .map(|reader| {
                <crate::chunk::DefaultChunk as crate::chunk::DefaultChunkReader<T, _>>::read_chunk(
                    reader,
                    array_meta,
                    grid_position,
                )
            })
            .transpose()
    }

    fn read_chunk_into<
        T: ReflectedType,
        B: DataChunk<T> + ReinitDataChunk<T> + ReadableDataChunk,
    >(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: GridCoord,
        chunk: &mut B,
    ) -> Result<Option<()>, Error> {
        // TODO convert asserts to errors
        assert!(array_meta.in_bounds(&grid_position));

        // Construct chunk path string
        let chunk_key = get_chunk_key(path_name, array_meta, &grid_position);

        // Get key from store
        let value_reader = ReadableStore::get(self, &chunk_key)?;

        // Read value into container
        value_reader
            .map(|reader| {
                <crate::chunk::DefaultChunk as crate::chunk::DefaultChunkReader<T, _>>::read_chunk_into(
                    reader,
                    array_meta,
                    grid_position,
                    chunk,
                )
            })
            .transpose()
    }

    fn store_chunk_metadata(
        &self,
        _path_name: &str,
        _array_meta: &ArrayMetadata,
        _grid_position: &[u64],
    ) -> Result<Option<StoreNodeMetadata>, Error> {
        todo!()
    }
}

impl<S: ListableStore + Hierarchy> HierarchyLister for S {
    fn list_nodes(&self, prefix_path: &str) -> Result<Vec<String>, Error> {
        // TODO: Inelegant.

        let key_prefix = canonicalize_path(prefix_path);
        let (mut keys, mut prefixes) = self.list_dir(key_prefix)?;

        // Find array and group metadata keys.
        keys.retain(|k| {
            k.ends_with(&crate::ARRAY_METADATA_KEY_EXT)
                || k.ends_with(&crate::GROUP_METADATA_KEY_EXT)
        });
        assert_eq!(
            crate::ARRAY_METADATA_KEY_EXT.len(),
            crate::GROUP_METADATA_KEY_EXT.len()
        );

        // Add potential implicit groups.
        prefixes
            .iter_mut()
            .for_each(|p| p.truncate(p.trim_end_matches('/').len()));
        keys.append(&mut prefixes);

        // Remove duplicates.
        keys.sort();
        keys.dedup();

        // Remove key prefix to convert to node path.
        for k in &mut keys {
            k.drain(..key_prefix.len());
        }

        Ok(keys)
    }
}

impl<S: ReadableStore + WriteableStore + Hierarchy> HierarchyWriter for S {
    fn set_attributes(
        &self, // TODO: should this be mut for semantics?
        path_name: &str,
        attributes: JsonObject,
    ) -> Result<(), Error> {
        // TODO: wasteful path recomputation
        let attrs_metadata_key = self.attrs_metadata_key(path_name);
        let attrs_metadata_key_str = attrs_metadata_key.as_str();

        let existing = if self.exists(attrs_metadata_key_str)? {
            let value_reader = ReadableStore::get(self, attrs_metadata_key_str)?
                .ok_or_else(|| Error::from(ErrorKind::NotFound))?;
            let existing: JsonObject = serde_json::from_reader(value_reader)?;
            existing
        } else {
            let value = serde_json::Map::new();
            self.set(attrs_metadata_key_str, |writer| {
                Ok(serde_json::to_writer(writer, &value)?)
            })?;
            value
        };

        // TODO: determine whether attribute merging is still necessary for zarr
        let mut merged = existing.clone();
        merge_top_level(&mut merged, attributes);

        if merged != existing {
            self.set(attrs_metadata_key_str, |writer| {
                Ok(serde_json::to_writer(writer, &merged)?)
            })?;
        }
        Ok(())
    }

    fn create_group(&self, path_name: &str) -> Result<(), Error> {
        // Walk through the parents of paths of the group and ensure that all ancestors have
        // `.zgroup` files. If they don't, create a group.
        let path = Utf8PathBuf::from(path_name);
        let mut ancestors = path.ancestors();
        // The first ancestor is always equal to the original path. We don't want to create the
        // group at this path yet.
        ancestors.next();

        for parent in ancestors {
            let parent_group_key = self.group_metadata_key(parent.as_str());
            if !self.exists(parent_group_key.as_str())? {
                self.create_group(parent.as_str())?;
            }
        }

        let metadata_key = self.group_metadata_key(path_name);
        if self.exists(self.array_metadata_key(path_name).as_str())? {
            Err(Error::new(
                ErrorKind::AlreadyExists,
                "Array already exists at group path",
            ))
        } else if self.exists(metadata_key.as_str())? {
            Ok(())
        } else {
            self.set(metadata_key.as_str(), |writer| {
                Ok(serde_json::to_writer(writer, &GroupMetadata::default())?)
            })
        }
    }

    fn create_array(&self, path_name: &str, array_meta: &ArrayMetadata) -> Result<(), Error> {
        // Walk through the parents of paths of the array and ensure that all ancestors have
        // `.zgroup` files. If they don't, create a group.
        let path = Utf8PathBuf::from(path_name);
        let mut ancestors = path.ancestors();
        // The first ancestor is always equal to the original path. We don't want to create the
        // group at this path.
        ancestors.next();

        for parent in ancestors {
            let parent_group_key = self.group_metadata_key(parent.as_str());
            if !self.exists(parent_group_key.as_str())? {
                self.create_group(parent.as_str())?;
            }
        }

        let metadata_key = self.array_metadata_key(path_name);
        if self.exists(self.group_metadata_key(path_name).as_str())?
            || self.exists(metadata_key.as_str())?
        {
            Err(Error::new(
                ErrorKind::AlreadyExists,
                "Node already exists at array path",
            ))
        } else {
            self.set(metadata_key.as_str(), |writer| {
                Ok(serde_json::to_writer(writer, array_meta)?)
            })
        }
    }

    fn remove(&self, path_name: &str) -> Result<(), Error> {
        // TODO: needless allocs
        let metadata_key = self.group_metadata_key(path_name);
        self.erase(metadata_key.as_str())?;
        let mut metadata_key = self.array_metadata_key(path_name);
        self.erase(metadata_key.as_str())?;
        metadata_key.set_extension("");
        metadata_key.set_extension("");
        self.erase_prefix(self.data_path_key(path_name).as_str())?;
        Ok(())
    }

    fn write_chunk<T: ReflectedType, B: DataChunk<T> + WriteableDataChunk>(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        chunk: &B,
    ) -> Result<(), Error> {
        // TODO convert assert
        // assert!(array_meta.in_bounds(chunk.get_grid_position()));
        let chunk_key = get_chunk_key(path_name, array_meta, chunk.grid_position());
        self.set(&chunk_key, |writer| {
            <crate::chunk::DefaultChunk as crate::chunk::DefaultChunkWriter<T, _, _>>::write_chunk(
                writer, array_meta, chunk,
            )
        })
    }

    fn delete_chunk(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<bool, Error> {
        let chunk_key = get_chunk_key(path_name, array_meta, grid_position);
        self.erase(&chunk_key)
    }
}
