use blosc_src;

/*
Portions of the blosc decompression code originate from blosc-rs
(github.com/asomers/blosc-rs).  The code is inlined here in order
to use the source bindings from blosc-src in order to avoid
linkages issues in upstream/dependent modules and crates.

The license for blosc-rs is as follows:

Copyright (c) 2018 Alan Somers

Permission is hereby granted, free of charge, to any
person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the
Software without restriction, including without
limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice
shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF
ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT
SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
*/

use std::io::{Cursor, Read, Result as IoResult, Write};
use std::{
    error, fmt, mem,
    os::raw::{c_char, c_int, c_void},
};

use serde::{Deserialize, Serialize};

use blosc_src::*;

use super::Compression;

pub const BLOSC_BLOSCLZ_COMPNAME: &[u8; 8usize] = b"blosclz\0";
pub const BLOSC_LZ4_COMPNAME: &[u8; 4usize] = b"lz4\0";
pub const BLOSC_ZLIB_COMPNAME: &[u8; 5usize] = b"zlib\0";
pub const BLOSC_ZSTD_COMPNAME: &[u8; 5usize] = b"zstd\0";

/// compressor types. https://github.com/asomers/blosc-rs/blob/master/blosc/src/lib.rs#L73
#[derive(Clone, Debug, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Compressor {
    /// The default compressor, based on FastLZ.  It's very fast, but the
    /// compression isn't as good as the other compressors.
    BloscLZ,
    /// Another fast compressor.  See [lz4.org](http://www.lz4.org).
    LZ4,
    /// The venerable Zlib.  Slower, but better compression than most other
    /// algorithms.  See [zlib.net](https://www.zlib.net)
    Zlib,
    /// A high compression algorithm from Facebook.
    /// See [zstd](https://facebook.github.io/zstd).
    Zstd,
}

impl From<Compressor> for *const c_char {
    fn from(compressor: Compressor) -> Self {
        let compref = match compressor {
            Compressor::BloscLZ => BLOSC_BLOSCLZ_COMPNAME.as_ptr(),
            Compressor::LZ4 => BLOSC_LZ4_COMPNAME.as_ptr(),
            Compressor::Zlib => BLOSC_ZLIB_COMPNAME.as_ptr(),
            Compressor::Zstd => BLOSC_ZSTD_COMPNAME.as_ptr(),
        };
        compref as *const c_char
    }
}

/// An unspecified error from C-Blosc
/// Same BloscError as github.com/asomers/blosc-rs (blosc v0.1.3)
#[derive(Clone, Copy, Debug)]
pub struct BloscError;

impl fmt::Display for BloscError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "unspecified error from c-Blosc")
    }
}

impl error::Error for BloscError {}

#[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
#[serde(rename_all = "lowercase")]
pub struct BloscCompression {
    #[serde(default = "default_blosc_blocksize")]
    blocksize: usize,
    #[serde(default = "default_blosc_clevel")]
    clevel: u8,
    cname: Compressor,
    #[serde(default = "default_blosc_shufflemode")]
    shuffle: u8, // serialize shuffle mode into enum by index
    #[serde(default = "default_blosc_id")]
    id: String,
}

fn default_blosc_id() -> String {
    "blosc".to_string()
}

fn default_blosc_blocksize() -> usize {
    0
}

fn default_blosc_clevel() -> u8 {
    5
}

fn default_blosc_shufflemode() -> u8 {
    1
}

impl Default for BloscCompression {
    fn default() -> BloscCompression {
        BloscCompression {
            blocksize: default_blosc_blocksize(),
            clevel: 5,
            cname: Compressor::BloscLZ,
            shuffle: default_blosc_shufflemode(),
            id: default_blosc_id(),
        }
    }
}

impl BloscCompression {
    fn decompress<T>(src: &[u8]) -> Result<Vec<T>, BloscError> {
        unsafe { BloscCompression::decompress_bytes(src) }
    }

    // Adapted from https://github.com/asomers/blosc-rs
    //
    // same as decompress_bytes from blosc-0.1.3, but use the
    // blosc-src direct lib to allow easier builds without
    // linkage
    unsafe fn decompress_bytes<T>(src: &[u8]) -> Result<Vec<T>, BloscError> {
        let typesize = mem::size_of::<T>();
        let mut nbytes: usize = 0;
        let mut _cbytes: usize = 0;
        let mut _blocksize: usize = 0;

        // unsafe
        #[allow(trivial_casts)]
        blosc_cbuffer_sizes(
            src.as_ptr() as *const c_void,
            &mut nbytes as *mut usize,
            &mut _cbytes as *mut usize,
            &mut _blocksize as *mut usize,
        );
        let dest_size = nbytes / typesize;
        let mut dest: Vec<T> = Vec::with_capacity(dest_size);

        // unsafe
        let rsize = blosc_decompress_ctx(
            src.as_ptr() as *const c_void,
            dest.as_mut_ptr() as *mut c_void,
            nbytes,
            1,
        );
        if rsize > 0 {
            // unsafe
            dest.set_len(rsize as usize / typesize);
            dest.shrink_to_fit();
            Ok(dest)
        } else {
            Err(BloscError)
        }
    }

    // Adapted from https://github.com/asomers/blosc-rs
    /// Compress an array and return a newly allocated compressed buffer.
    fn compress<T>(&self, src: &[T]) -> Vec<u8> {
        let typesize = mem::size_of::<T>();
        let src_size = mem::size_of_val(src);
        let dest_size = src_size + BLOSC_MAX_OVERHEAD as usize;
        let mut dest: Vec<u8> = Vec::with_capacity(dest_size);
        let rsize = unsafe {
            blosc_compress_ctx(
                self.clevel as c_int,
                self.shuffle as c_int,
                typesize,
                src_size,
                src.as_ptr() as *const c_void,
                dest.as_mut_ptr() as *mut c_void,
                dest_size,
                self.cname.clone().into(),
                self.blocksize,
                1,
            )
        };
        // Blosc's docs claim that blosc_compress_ctx should never return an
        // error
        // LCOV_EXCL_START
        assert!(
            rsize >= 0,
            "C-Blosc internal error with Context={:?}, typesize={:?} nbytes={:?} and destsize={:?}",
            self,
            typesize,
            src_size,
            dest_size
        );
        // LCOV_EXCL_STOP
        unsafe {
            dest.set_len(rsize as usize);
        }
        dest.shrink_to_fit();
        dest
    }
}

struct Wrapper<W: Write> {
    writer: W,
    compressor: BloscCompression,
    inner_buffer: Vec<u8>,
    finished: bool,
}

impl<W: Write> Write for Wrapper<W> {
    fn write(&mut self, buffer: &[u8]) -> IoResult<usize> {
        self.inner_buffer.extend(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> IoResult<()> {
        Ok(())
    }
}

impl<W: Write> Wrapper<W> {
    fn finish(&mut self) -> IoResult<()> {
        if self.finished {
            return Ok(());
        }
        self.finished = true;
        let compressed_bytes = self.compressor.compress(&self.inner_buffer);
        self.writer.write_all(&compressed_bytes)
    }
}

impl<W: Write> Drop for Wrapper<W> {
    fn drop(&mut self) {
        self.finish().unwrap();
    }
}

impl Compression for BloscCompression {
    fn decoder<'a, R: Read + 'a>(&self, mut r: R) -> Box<dyn Read + 'a> {
        // blosc is all at the same time...
        let mut bytes: Vec<u8> = Vec::new();
        r.read_to_end(&mut bytes).unwrap();
        let decompressed = BloscCompression::decompress(&bytes).unwrap();
        Box::new(Cursor::new(decompressed))
    }

    fn encoder<'a, W: Write + 'a>(&self, w: W) -> Box<dyn Write + 'a> {
        Box::new(Wrapper {
            writer: w,
            compressor: self.clone(),
            inner_buffer: Vec::new(),
            finished: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compression::CompressionType;

    #[rustfmt::skip]
  const TEST_CHUNK_I16_BLOSC: [u8; 28] = [
      0x02, 0x01, 0x33, 0x01,
      0x0c, 0x00, 0x00, 0x00,
      0x0c, 0x00, 0x00, 0x00,
      0x1c, 0x00, 0x00, 0x00,
      0x00, 0x01, 0x00, 0x02, // target payload is big endian
      0x00, 0x03, 0x00, 0x04,
      0x00, 0x05, 0x00, 0x06, // not very compressed now is it
  ];

    #[test]
    fn test_read_doc_spec_chunk() {
        let blosc_lz4: BloscCompression = BloscCompression {
            blocksize: 0,
            clevel: 5,
            cname: Compressor::LZ4,
            shuffle: 1,
            id: "blosc".to_string(),
        };
        crate::tests::test_read_doc_spec_chunk(
            TEST_CHUNK_I16_BLOSC.as_ref(),
            CompressionType::Blosc(blosc_lz4),
        );
    }

    #[test]
    fn test_write_doc_spec_chunk() {
        let blosc_lz4: BloscCompression = BloscCompression {
            blocksize: 0,
            clevel: 5,
            cname: Compressor::LZ4,
            shuffle: 1,
            id: "blosc".to_string(),
        };
        crate::tests::test_write_doc_spec_chunk(
            TEST_CHUNK_I16_BLOSC.as_ref(),
            CompressionType::Blosc(blosc_lz4),
        )
    }
}
