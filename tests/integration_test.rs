use std::path::Path;

use anyhow::{Context, Result};
use half::f16;
use rand::{distributions::Standard, Rng};
use smallvec::smallvec;
use walkdir::WalkDir;

use zarr::{prelude::*, ArrayMetadataBuilder, Order};

fn test_read_write<T, RT: Into<T>, Zarr: HierarchyReader + HierarchyWriter>(
    n: &Zarr,
    compression: &CompressionType,
    dim: usize,
    path: &Path,
) -> Result<()>
where
    T: 'static + std::fmt::Debug + ReflectedType + PartialEq + Default,
    rand::distributions::Standard: rand::distributions::Distribution<RT>,
    VecDataChunk<T>: zarr::chunk::ReadableDataChunk + zarr::chunk::WriteableDataChunk,
{
    println!("{compression}");
    let chunk_shape: ChunkCoord = (1..=dim as u32).rev().map(|d| d * 5).collect();

    let array_meta = ArrayMetadataBuilder::default()
        .shape((1..=dim as u64).map(|d| d * 100).collect())
        .chunks(chunk_shape)
        .dtype(T::ZARR_TYPE)
        .compressor(compression.clone())
        .order(Order::RowMajor)
        .build()?;

    let numel = array_meta.chunk_num_elements();
    let rng = rand::thread_rng();
    let chunk_data: Vec<T> = rng
        .sample_iter(&Standard)
        .take(numel)
        .map(Into::into)
        .collect();

    let chunk_in = SliceDataChunk::new(smallvec![0; dim], chunk_data);

    n.create_group("dataset")?;
    n.set_attributes("dataset", serde_json::from_str(r#"{"key": "value"}"#)?)
        .context("Failed to set attributes")?;

    n.create_group("dataset")?;

    let path_name = "dataset/array";
    n.create_array(path_name, &array_meta)
        .context("Failed to create array")?;
    n.set_attributes(path_name, serde_json::from_str(r#"{"key": "value"}"#)?)
        .context("Failed to set attributes")?;
    n.write_chunk(path_name, &array_meta, &chunk_in)
        .context("Failed to write chunk")?;

    let chunk_data = chunk_in.into_data();

    for item in WalkDir::new(path)
        .into_iter()
        .filter(Result::is_ok)
        .flatten()
    {
        println!("{:?}", item.path());
    }

    let chunk_out = n
        .read_chunk::<T>(path_name, &array_meta, smallvec![0; dim])
        .expect("Failed to read chunk")
        .expect("Chunk is empty");
    assert_eq!(chunk_out.data(), &chunk_data[..]);

    let mut into_chunk = VecDataChunk::new(smallvec![0; dim], vec![]);
    n.read_chunk_into(path_name, &array_meta, smallvec![0; dim], &mut into_chunk)
        .expect("Failed to read chunk")
        .expect("Chunk is empty");
    assert_eq!(into_chunk.data(), &chunk_data[..]);

    let attrs = n
        .get_attributes("dataset")?
        .context("Expected group to have attrs.")?;
    println!("{}", serde_json::to_string_pretty(&attrs)?);

    n.remove(path_name).unwrap();

    Ok(())
}

fn test_all_types<Zarr: HierarchyReader + HierarchyWriter>(
    n: &Zarr,
    compression: &CompressionType,
    dim: usize,
    path: &Path,
) -> Result<()> {
    test_read_write::<bool, bool, _>(n, compression, dim, path)?;
    test_read_write::<u8, u8, _>(n, compression, dim, path)?;
    test_read_write::<u16, u16, _>(n, compression, dim, path)?;
    test_read_write::<u32, u32, _>(n, compression, dim, path)?;
    test_read_write::<u64, u64, _>(n, compression, dim, path)?;
    test_read_write::<i8, i8, _>(n, compression, dim, path)?;
    test_read_write::<i16, i16, _>(n, compression, dim, path)?;
    test_read_write::<i32, i32, _>(n, compression, dim, path)?;
    test_read_write::<i64, i64, _>(n, compression, dim, path)?;
    test_read_write::<f16, u8, _>(n, compression, dim, path)?;
    test_read_write::<f32, f32, _>(n, compression, dim, path)?;
    test_read_write::<f64, f64, _>(n, compression, dim, path)?;

    Ok(())
}

fn test_zarr_filesystem_dim(dim: usize) -> Result<()> {
    let dir = tempdir::TempDir::new("rust_zarr_integration_tests")?;

    let n = FilesystemHierarchy::open_or_create(dir.path())
        .context("Failed to create Zarr filesystem")?;
    test_all_types(
        &n,
        &CompressionType::Raw(compression::raw::RawCompression),
        dim,
        dir.path(),
    )
}

#[test]
fn test_zarr_filesystem_dims() -> Result<()> {
    for dim in 1..=5 {
        test_zarr_filesystem_dim(dim)?;
    }
    Ok(())
}

fn test_all_compressions<Zarr: HierarchyReader + HierarchyWriter>(
    n: &Zarr,
    path: &Path,
) -> Result<()> {
    test_all_types(
        n,
        &CompressionType::Raw(compression::raw::RawCompression),
        3,
        path,
    )?;
    #[cfg(feature = "bzip")]
    test_all_types(
        n,
        &CompressionType::Bzip2(compression::bzip::Bzip2Compression::default()),
        3,
        path,
    )?;
    #[cfg(feature = "gzip")]
    test_all_types(
        n,
        &CompressionType::Gzip(compression::gzip::GzipCompression::default()),
        3,
        path,
    )?;
    #[cfg(feature = "xz")]
    test_all_types(
        n,
        &CompressionType::Xz(compression::xz::XzCompression::default()),
        3,
        path,
    )?;
    // #[cfg(feature = "lz")]
    // test_all_types(
    //     n,
    //     &CompressionType::Lz4(compression::lz::Lz4Compression::default()),
    //     3,
    //     path,
    // );
    #[cfg(feature = "blosc")]
    test_all_types(
        n,
        &CompressionType::Blosc(
            compression::blosc::BloscCompressionBuilder::new(compression::blosc::Compressor::Zstd)
                .clevel(5)
                .shuffle(1)
                .blocksize(0)
                .build(),
        ),
        3,
        path,
    )?;
    Ok(())
}

#[test]
fn test_zarr_filesystem_compressions() -> Result<()> {
    let dir = tempdir::TempDir::new("rust_zarr_integration_tests").unwrap();

    let n = FilesystemHierarchy::open_or_create(dir.path())
        .context("Failed to create Zarr filesystem")?;
    test_all_compressions(&n, dir.path())
}
