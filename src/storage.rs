use std::io::{Error, ErrorKind, Read, Write};

use semver::VersionReq;
use serde_json::Value;

use crate::{
    canonicalize_path,
    chunk::{DataChunk, ReadableDataChunk, ReinitDataChunk, VecDataChunk, WriteableDataChunk},
    ArrayMetadata, GridCoord, GroupMetadata, Hierarchy, HierarchyLister, HierarchyReader,
    HierarchyWriter, JsonObject, MetadataError, ReflectedType, StoreNodeMetadata,
};

pub trait ReadableStore {
    type GetReader: Read;

    /// TODO: not in zarr spec
    fn exists(&self, key: &str) -> Result<bool, Error>;

    fn get(&self, key: &str) -> Result<Option<Self::GetReader>, Error>;

    /// TODO: not in zarr spec
    fn uri(&self, key: &str) -> Result<String, Error>;
}

pub trait ListableStore {
    /// Retrieve all keys in the store.
    fn list(&self) -> Result<Vec<String>, Error> {
        self.list_prefix("/")
    }

    /// Retrieve all keys with a given prefix.
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
    fn list_dir(&self, prefix: &str) -> Result<(Vec<String>, Vec<String>), Error>;
}

pub trait WriteableStore {
    type SetWriter: Write;

    fn set<F: FnOnce(Self::SetWriter) -> Result<(), Error>>(
        &self,
        key: &str,
        value: F,
    ) -> Result<(), Error>;

    // TODO differs from spec in that it returns a bool indicating existence of the key at the end of the operation.
    fn erase(&self, key: &str) -> Result<bool, Error>;

    // TODO
    fn erase_prefix(&self, key_prefix: &str) -> Result<bool, Error>;
}

/// TODO
///
/// ```
/// use zarr::prelude::*;
/// use zarr::storage::get_chunk_key;
/// use zarr::smallvec::smallvec;
/// let meta = ArrayMetadata::new(
///     smallvec![50, 40, 30],
///     smallvec![11, 10, 10],
///     i8::ZARR_TYPE,
///     zarr::compression::CompressionType::default(),
/// );
/// assert_eq!(get_chunk_key("/foo/baz", &meta, &[0, 0, 0]), "/data/root/foo/baz/c0/0/0");
/// assert_eq!(get_chunk_key("/foo/baz", &meta, &[1, 2, 3]), "/data/root/foo/baz/c1/2/3");
///
/// let meta = ArrayMetadata::new(
///     smallvec![],
///     smallvec![],
///     i8::ZARR_TYPE,
///     zarr::compression::CompressionType::default(),
/// );
/// assert_eq!(get_chunk_key("/foo/baz", &meta, &[]), "/data/root/foo/baz/c");
/// ```
pub fn get_chunk_key(base_path: &str, array_meta: &ArrayMetadata, grid_position: &[u64]) -> String {
    use std::fmt::Write;
    // TODO: normalize relative or absolute paths
    let canon_path = canonicalize_path(base_path);
    let mut chunk_key = if canon_path.is_empty() {
        format!("{}/c", crate::DATA_ROOT_PATH,)
    } else {
        format!("{}/{}/c", crate::DATA_ROOT_PATH, canon_path)
    };

    for (i, coord) in grid_position.iter().enumerate() {
        write!(chunk_key, "{}", coord).unwrap();
        if i < grid_position.len() - 1 {
            chunk_key.push_str(&array_meta.chunk_grid.separator)
        }
    }

    chunk_key
}

const ATTRIBUTES_NAME: &str = "attributes";

fn merge_top_level(a: &mut Value, b: JsonObject) {
    match a {
        &mut Value::Object(ref mut a) => {
            for (k, v) in b {
                a.insert(k, v);
            }
        }
        a => {
            *a = b.into();
        }
    }
}

impl<S: ReadableStore + Hierarchy> HierarchyReader for S {
    fn get_version(&self) -> Result<VersionReq, Error> {
        let vers_str = self
            .get_entry_point_metadata()
            .zarr_format
            .rsplit('/')
            .next()
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    "Entry point metadata zarr format URI does not have version",
                )
            })?;
        VersionReq::parse(vers_str).map_err(|_| {
            Error::new(
                ErrorKind::InvalidData,
                "Entry point metadata zarr format URI does not have version",
            )
        })
    }

    fn get_array_metadata(&self, path_name: &str) -> Result<ArrayMetadata, Error> {
        let array_path = self.array_metadata_key(path_name);
        let value_reader = ReadableStore::get(self, &array_path.to_str().expect("TODO"))?
            .ok_or_else(|| Error::from(ErrorKind::NotFound))?;
        let metadata: ArrayMetadata = serde_json::from_reader(value_reader)?;
        // TODO: erring immediately when encountering unknown extensions, while
        // it may be more appropriate to do so only when doing chunk IO.
        if let Some(ext) = metadata.extensions.iter().find(|e| e.must_understand) {
            // TODO: returning an io::Error wrapped custom error, rather than other
            // way around.
            return Err(MetadataError::UnknownRequiredExtension(ext.clone()).into());
        }
        Ok(metadata)
    }

    fn exists(&self, path_name: &str) -> Result<bool, Error> {
        // TODO: needless path allocs
        // TODO: should follow spec more closely by using `list_dir` for implicit groups.
        Ok(
            self.exists(self.array_metadata_key(path_name).to_str().expect("TODO"))?
                || self.exists(self.group_metadata_key(path_name).to_str().expect("TODO"))?
                || self.exists(
                    self.group_metadata_key(path_name)
                        .with_extension("")
                        .with_extension("")
                        .to_str()
                        .expect("TODO"),
                )?,
        )
    }

    fn get_chunk_uri(
        &self,
        path_name: &str,
        array_meta: &ArrayMetadata,
        grid_position: &[u64],
    ) -> Result<String, Error> {
        let chunk_key = get_chunk_key(path_name, array_meta, &grid_position);
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

    fn list_attributes(&self, path_name: &str) -> Result<JsonObject, Error> {
        // TODO: wasteful path recomputation
        let metadata_key =
            if self.exists(self.array_metadata_key(path_name).to_str().expect("TODO"))? {
                self.array_metadata_key(path_name)
            } else if self.exists(self.group_metadata_key(path_name).to_str().expect("TODO"))? {
                self.group_metadata_key(path_name)
            } else {
                return Err(Error::new(
                    ErrorKind::NotFound,
                    "Node does not exist at path",
                ));
            };

        // TODO: determine proper missing behavior for implicit groups.
        // For now return an error.
        let value_reader = ReadableStore::get(self, &metadata_key.to_str().expect("TODO"))?
            .ok_or_else(|| Error::from(ErrorKind::NotFound))?;
        let mut value: Value = serde_json::from_reader(value_reader)?;
        let attrs = match value
            .as_object_mut()
            .and_then(|o| o.remove(ATTRIBUTES_NAME))
        {
            Some(Value::Object(map)) => map,
            Some(v) => return Err(MetadataError::UnexpectedType(v).into()),
            _ => JsonObject::new(),
        };
        Ok(attrs)
    }
}

impl<S: ListableStore + Hierarchy> HierarchyLister for S {
    fn list_nodes(&self, prefix_path: &str) -> Result<Vec<String>, Error> {
        // TODO: Inelegant.

        let key_prefix = format!(
            "{}/{}",
            crate::META_ROOT_PATH,
            canonicalize_path(prefix_path)
        );
        let (mut keys, mut prefixes) = self.list_dir(&key_prefix)?;

        // Find array and group metadata keys.
        keys.retain(|k| k.ends_with(&self.get_entry_point_metadata().metadata_key_suffix));
        keys.iter_mut().for_each(|k| {
            k.truncate(k.len() - self.get_entry_point_metadata().metadata_key_suffix.len())
        });
        keys.retain(|k| {
            k.ends_with(&crate::ARRAY_METADATA_KEY_EXT)
                || k.ends_with(&crate::GROUP_METADATA_KEY_EXT)
        });
        assert_eq!(
            crate::ARRAY_METADATA_KEY_EXT.len(),
            crate::GROUP_METADATA_KEY_EXT.len()
        );

        // Remove metadata extensions from keys to partially convert to node paths.
        let suffix_len = self.get_entry_point_metadata().metadata_key_suffix.len()
            + crate::ARRAY_METADATA_KEY_EXT.len()
            + 1;
        keys.iter_mut()
            .for_each(|k| k.truncate(k.len() - suffix_len));

        // Add potential implicit groups.
        prefixes
            .iter_mut()
            .for_each(|p| p.truncate(p.trim_end_matches('/').len()));
        keys.extend(prefixes.drain(..));

        // Remove duplicates.
        keys.sort();
        keys.dedup();

        // Remove key prefix to convert to node path.
        keys.iter_mut().for_each(|k| {
            k.drain(..key_prefix.len());
        });

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
        let metadata_key =
            if self.exists(self.array_metadata_key(path_name).to_str().expect("TODO"))? {
                self.array_metadata_key(path_name)
            } else if self.exists(self.group_metadata_key(path_name).to_str().expect("TODO"))? {
                self.group_metadata_key(path_name)
            } else {
                return Err(Error::new(
                    ErrorKind::NotFound,
                    "Node does not exist at path",
                ));
            };

        // TODO: race condition
        let value_reader = ReadableStore::get(self, &metadata_key.to_str().expect("TODO"))?
            .ok_or_else(|| Error::from(ErrorKind::NotFound))?;
        let existing: JsonObject = serde_json::from_reader(value_reader)?;

        // TODO: determine whether attribute merging is still necessary for zarr
        let mut merged = existing.clone();
        match merged.get_mut(ATTRIBUTES_NAME) {
            Some(merged_attr) => merge_top_level(merged_attr, attributes),
            None => {
                merged.insert(ATTRIBUTES_NAME.into(), Value::Object(attributes));
            }
        }
        if merged != existing {
            self.set(metadata_key.to_str().expect("TODO"), |writer| {
                Ok(serde_json::to_writer(writer, &merged)?)
            })?;
        }
        Ok(())
    }

    fn create_group(&self, path_name: &str) -> Result<(), Error> {
        // Because of implicit hierarchy rules, it is not necessary to create
        // the parent group.
        // let path_buf = PathBuf::from(path_name);
        // if let Some(parent) = path_buf.parent() {
        //     self.create_group(parent.to_str().expect("TODO"))?;
        // }
        let metadata_key = self.group_metadata_key(path_name);
        if self.exists(self.array_metadata_key(path_name).to_str().expect("TODO"))? {
            Err(Error::new(
                ErrorKind::AlreadyExists,
                "Array already exists at group path",
            ))
        } else if self.exists(metadata_key.to_str().expect("TODO"))? {
            Ok(())
        } else {
            self.set(metadata_key.to_str().expect("TODO"), |writer| {
                Ok(serde_json::to_writer(writer, &GroupMetadata::default())?)
            })
        }
    }

    fn create_array(&self, path_name: &str, array_meta: &ArrayMetadata) -> Result<(), Error> {
        // Because of implicit hierarchy rules, it is not necessary to create
        // the parent group.
        // let path_buf = PathBuf::from(path_name);
        // if let Some(parent) = path_buf.parent() {
        //     self.create_group(parent.to_str().expect("TODO"))?;
        // }
        let metadata_key = self.array_metadata_key(path_name);
        if self.exists(self.group_metadata_key(path_name).to_str().expect("TODO"))?
            || self.exists(metadata_key.to_str().expect("TODO"))?
        {
            Err(Error::new(
                ErrorKind::AlreadyExists,
                "Node already exists at array path",
            ))
        } else {
            self.set(metadata_key.to_str().expect("TODO"), |writer| {
                Ok(serde_json::to_writer(writer, array_meta)?)
            })
        }
    }

    fn remove(&self, path_name: &str) -> Result<(), Error> {
        // TODO: needless allocs
        let metadata_key = self.group_metadata_key(path_name);
        self.erase(metadata_key.to_str().expect("TODO"))?;
        let mut metadata_key = self.array_metadata_key(path_name);
        self.erase(metadata_key.to_str().expect("TODO"))?;
        metadata_key.set_extension("");
        metadata_key.set_extension("");
        self.erase_prefix(self.data_path_key(path_name).to_str().expect("TODO"))?;
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
        let chunk_key = get_chunk_key(path_name, array_meta, chunk.get_grid_position());
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
