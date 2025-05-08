use std::io::Cursor;

use super::chunk::{DefaultChunk, DefaultChunkReader, DefaultChunkWriter};
use super::*;

const DOC_SPEC_CHUNK_DATA: [i16; 6] = [1, 2, 3, 4, 5, 6];

/// Wrapper type for holding a context from dropping during the lifetime of an
/// Zarr backend. This is useful for things like tempdirs.
pub struct ContextWrapper<C, N: HierarchyReader + HierarchyWriter> {
    pub context: C,
    pub zarr: N,
}

impl<C, N: HierarchyReader + HierarchyWriter> AsRef<N> for ContextWrapper<C, N> {
    fn as_ref(&self) -> &N {
        &self.zarr
    }
}

fn doc_spec_array_metadata(compression: compression::CompressionType) -> ArrayMetadata {
    ArrayMetadata::new(
        smallvec![5, 6, 7],
        smallvec![1, 2, 3],
        DataType::Int {
            size: IntSize::B2,
            endian: Endian::Big,
        },
        compression,
    )
}

pub(crate) fn test_read_doc_spec_chunk(chunk: &[u8], compression: compression::CompressionType) {
    let buff = Cursor::new(chunk);
    let array_meta = doc_spec_array_metadata(compression);

    let chunk = <DefaultChunk as DefaultChunkReader<i16, std::io::Cursor<&[u8]>>>::read_chunk(
        buff,
        &array_meta,
        smallvec![0, 0, 0],
    )
    .expect("read_chunk failed");

    assert_eq!(chunk.grid_position(), &[0, 0, 0]);
    assert_eq!(chunk.data(), &DOC_SPEC_CHUNK_DATA);
}

pub(crate) fn test_write_doc_spec_chunk(
    expected_chunk: &[u8],
    compression: compression::CompressionType,
) {
    let array_meta = doc_spec_array_metadata(compression);
    let chunk_in = SliceDataChunk::new(smallvec![0, 0, 0], DOC_SPEC_CHUNK_DATA);
    let mut buff: Vec<u8> = Vec::new();

    <DefaultChunk as DefaultChunkWriter<i16, _, _>>::write_chunk(&mut buff, &array_meta, &chunk_in)
        .expect("read_chunk failed");

    assert_eq!(buff, expected_chunk);
}

pub(crate) fn test_chunk_compression_rw(compression: compression::CompressionType) {
    let array_meta = ArrayMetadata::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        i32::ZARR_TYPE,
        compression,
    );
    let chunk_data: Vec<i32> = (0..125_i32).collect();
    let chunk_in = SliceDataChunk::new(smallvec![0, 0, 0], &chunk_data);

    let mut inner: Vec<u8> = Vec::new();

    <DefaultChunk as DefaultChunkWriter<i32, _, _>>::write_chunk(
        &mut inner,
        &array_meta,
        &chunk_in,
    )
    .expect("write_chunk failed");

    let chunk_out = <DefaultChunk as DefaultChunkReader<i32, _>>::read_chunk(
        &inner[..],
        &array_meta,
        smallvec![0, 0, 0],
    )
    .expect("read_chunk failed");

    assert_eq!(chunk_out.grid_position(), &[0, 0, 0]);
    assert_eq!(chunk_out.data(), &chunk_data[..]);
}

pub(crate) fn test_varlength_chunk_rw(compression: compression::CompressionType) {
    let array_meta = ArrayMetadata::new(
        smallvec![10, 10, 10],
        smallvec![5, 5, 5],
        i32::ZARR_TYPE,
        compression,
    );
    let chunk_data: Vec<i32> = (0..100_i32).collect();
    let chunk_in = SliceDataChunk::new(smallvec![0, 0, 0], &chunk_data);

    let mut inner: Vec<u8> = Vec::new();

    // In Zarr chunks must fill the chunk shape, so this should err.
    assert!(
        <DefaultChunk as DefaultChunkWriter<i32, _, _>>::write_chunk(
            &mut inner,
            &array_meta,
            &chunk_in,
        )
        .is_err()
    );

    assert!(<DefaultChunk as DefaultChunkReader<i32, _>>::read_chunk(
        &inner[..],
        &array_meta,
        smallvec![0, 0, 0],
    )
    .is_err());
}
