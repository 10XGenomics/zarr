use std::io::Write;

use half::f16;
use serde::{Deserialize, Serialize};

use crate::{GridCoord, VecDataChunk};

/// Sizes for boolean types.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum BoolSize {
    /// A single byte.
    B1,
}

impl BoolSize {
    fn deserial_char(c: char) -> Option<Self> {
        match c {
            '1' => Some(BoolSize::B1),
            _ => None,
        }
    }

    fn serial_char(&self) -> char {
        match self {
            BoolSize::B1 => '1',
        }
    }
}

/// Sizes for integer types.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum IntSize {
    /// Single byte integer (`i8`, `u8`)
    B1,
    /// Two byte integer (`i16`, `u16`)
    B2,
    /// Four byte integer (`i32`, `u32`)
    B4,
    /// Eight byte integer (`i64`, `u64`)
    B8,
}

impl IntSize {
    fn deserial_char(c: char) -> Option<Self> {
        match c {
            '1' => Some(IntSize::B1),
            '2' => Some(IntSize::B2),
            '4' => Some(IntSize::B4),
            '8' => Some(IntSize::B8),
            _ => None,
        }
    }

    fn serial_char(&self) -> char {
        match self {
            IntSize::B1 => '1',
            IntSize::B2 => '2',
            IntSize::B4 => '4',
            IntSize::B8 => '8',
        }
    }
}

/// Sizes for floating-point types.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum FloatSize {
    /// Two byte float (`f16`)
    B2,
    /// Four byte float (`f32`)
    B4,
    /// Eight byte float (`f64`)
    B8,
}

impl FloatSize {
    fn deserial_char(c: char) -> Option<Self> {
        match c {
            '2' => Some(FloatSize::B2),
            '4' => Some(FloatSize::B4),
            '8' => Some(FloatSize::B8),
            _ => None,
        }
    }

    fn serial_char(&self) -> char {
        match self {
            FloatSize::B2 => '2',
            FloatSize::B4 => '4',
            FloatSize::B8 => '8',
        }
    }
}

/// Endianness of a data type.
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum Endian {
    /// Big endian (most significant bytes first)
    Big,
    /// Big endian (least significant bytes first)
    Little,
    /// Endianness doesn't matter (e.g., a `bool`)
    NotRelevant,
}

#[cfg(target_endian = "big")]
/// Default `Endian` for the system
pub const NATIVE_ENDIAN: Endian = Endian::Big;
#[cfg(target_endian = "little")]
/// Default `Endian` for the system
pub const NATIVE_ENDIAN: Endian = Endian::Little;
/// Default `Endian` for network protocols
pub const NETWORK_ENDIAN: Endian = Endian::Big;

impl Endian {
    fn deserial_char(c: char) -> Option<Self> {
        match c {
            '>' => Some(Endian::Big),
            '<' => Some(Endian::Little),
            '|' => Some(Endian::NotRelevant),
            _ => None,
        }
    }

    fn serial_char(&self) -> char {
        match self {
            Endian::Big => '>',
            Endian::Little => '<',
            Endian::NotRelevant => '|',
        }
    }
}

/// Data types representable in Zarr.
///
/// ```
/// use zarr::data_type::{Endian, IntSize, FloatSize, DataType};
/// use serde_json;
///
/// let d: DataType = serde_json::from_str("\"<f8\"").unwrap();
/// assert_eq!(d, DataType::Float {size: FloatSize::B8, endian: Endian::Little});
/// let d: DataType = serde_json::from_str("\">u4\"").unwrap();
/// assert_eq!(d, DataType::UInt {size: IntSize::B4, endian: Endian::Big});
/// let d: DataType = serde_json::from_str("\"r24\"").unwrap();
/// assert_eq!(d, DataType::Raw {size: 24});
/// ```
#[derive(PartialEq, Debug, Clone, Copy)]
pub enum DataType {
    /// A boolean data type
    Bool {
        /// The size of the boolean
        size: BoolSize,
        /// Endianness of the boolean
        endian: Endian,
    },
    /// A signed integer data type
    Int {
        /// The size of the signed integer type
        size: IntSize,
        /// The endianness of the signed integer type
        endian: Endian,
    },
    /// An unsigned integer data type
    UInt {
        /// The size of the unsigned integer type
        size: IntSize,
        /// The endianness of the unsigned integer type
        endian: Endian,
    },
    /// A floating point number data type
    Float {
        /// The size of the float type
        size: FloatSize,
        /// The endianness of the float type
        endian: Endian,
    },
    /// Raw bytes
    Raw {
        /// The number of bytes for each item
        size: usize,
    },
}

impl DataType {
    /// Create a boolean [`DataType`].
    #[must_use]
    pub fn bool() -> Self {
        Self::Bool {
            size: BoolSize::B1,
            endian: Endian::NotRelevant,
        }
    }
}

impl DataType {
    /// Create a 8-bit signed integer [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn i8_le() -> Self {
        DataType::Int {
            size: IntSize::B1,
            endian: Endian::Little,
        }
    }
    /// Create a 8-bit signed integer [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn i8_be() -> Self {
        DataType::Int {
            size: IntSize::B1,
            endian: Endian::Big,
        }
    }

    /// Create a 16-bit signed integer [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn i16_le() -> Self {
        DataType::Int {
            size: IntSize::B2,
            endian: Endian::Little,
        }
    }
    /// Create a 16-bit signed integer [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn i16_be() -> Self {
        DataType::Int {
            size: IntSize::B2,
            endian: Endian::Big,
        }
    }

    /// Create a 32-bit signed integer [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn i32_le() -> Self {
        DataType::Int {
            size: IntSize::B4,
            endian: Endian::Little,
        }
    }
    /// Create a 32-bit signed integer [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn i32_be() -> Self {
        DataType::Int {
            size: IntSize::B4,
            endian: Endian::Big,
        }
    }

    /// Create a 64-bit signed integer [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn i64_le() -> Self {
        DataType::Int {
            size: IntSize::B8,
            endian: Endian::Little,
        }
    }
    /// Create a 64-bit signed integer [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn i64_be() -> Self {
        DataType::Int {
            size: IntSize::B8,
            endian: Endian::Big,
        }
    }
}

impl DataType {
    /// Create a 8-bit unsigned integer [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn u8_le() -> Self {
        DataType::UInt {
            size: IntSize::B1,
            endian: Endian::Little,
        }
    }
    /// Create a 8-bit unsigned integer [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn u8_be() -> Self {
        DataType::UInt {
            size: IntSize::B1,
            endian: Endian::Big,
        }
    }

    /// Create a 16-bit unsigned integer [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn u16_le() -> Self {
        DataType::UInt {
            size: IntSize::B2,
            endian: Endian::Little,
        }
    }
    /// Create a 16-bit unsigned integer [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn u16_be() -> Self {
        DataType::UInt {
            size: IntSize::B2,
            endian: Endian::Big,
        }
    }

    /// Create a 32-bit unsigned integer [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn u32_le() -> Self {
        DataType::UInt {
            size: IntSize::B4,
            endian: Endian::Little,
        }
    }
    /// Create a 32-bit unsigned integer [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn u32_be() -> Self {
        DataType::UInt {
            size: IntSize::B4,
            endian: Endian::Big,
        }
    }

    /// Create a 64-bit unsigned integer [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn u64_le() -> Self {
        DataType::UInt {
            size: IntSize::B8,
            endian: Endian::Little,
        }
    }
    /// Create a 64-bit unsigned integer [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn u64_be() -> Self {
        DataType::UInt {
            size: IntSize::B8,
            endian: Endian::Big,
        }
    }
}

impl DataType {
    /// Create a 16-bit floating point [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn f16_le() -> Self {
        DataType::Float {
            size: FloatSize::B2,
            endian: Endian::Little,
        }
    }
    /// Create a 16-bit floating point [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn f16_be() -> Self {
        DataType::Float {
            size: FloatSize::B2,
            endian: Endian::Big,
        }
    }

    /// Create a 32-bit floating point [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn f32_le() -> Self {
        DataType::Float {
            size: FloatSize::B4,
            endian: Endian::Little,
        }
    }
    /// Create a 32-bit floating point [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn f32_be() -> Self {
        DataType::Float {
            size: FloatSize::B4,
            endian: Endian::Big,
        }
    }

    /// Create a 64-bit floating point [`DataType`] with little-endian ordering.
    #[must_use]
    pub fn f64_le() -> Self {
        DataType::Float {
            size: FloatSize::B8,
            endian: Endian::Little,
        }
    }
    /// Create a 64-bit floating point [`DataType`] with big-endian ordering.
    #[must_use]
    pub fn f64_be() -> Self {
        DataType::Float {
            size: FloatSize::B8,
            endian: Endian::Big,
        }
    }
}

impl DataType {
    /// Create a raw [`DataType`] with `size` bytes.
    #[must_use]
    pub fn raw(size: usize) -> Self {
        DataType::Raw { size }
    }
}

impl Serialize for DataType {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut buf = [0u8; 32];
        let s = match self {
            DataType::Bool { size, endian } => {
                endian.serial_char().encode_utf8(&mut buf[0..1]);
                'b'.encode_utf8(&mut buf[1..2]);
                size.serial_char().encode_utf8(&mut buf[2..3]);
                std::str::from_utf8(&buf[..3]).unwrap()
            }
            DataType::Int {
                size: IntSize::B1, ..
            } => "i1",
            DataType::UInt {
                size: IntSize::B1, ..
            } => "u1",
            DataType::Int { size, endian } => {
                endian.serial_char().encode_utf8(&mut buf[0..1]);
                'i'.encode_utf8(&mut buf[1..2]);
                size.serial_char().encode_utf8(&mut buf[2..3]);
                std::str::from_utf8(&buf[..3]).unwrap()
            }
            DataType::UInt { size, endian } => {
                endian.serial_char().encode_utf8(&mut buf[0..1]);
                'u'.encode_utf8(&mut buf[1..2]);
                size.serial_char().encode_utf8(&mut buf[2..3]);
                std::str::from_utf8(&buf[..3]).unwrap()
            }
            DataType::Float { size, endian } => {
                endian.serial_char().encode_utf8(&mut buf[0..1]);
                'f'.encode_utf8(&mut buf[1..2]);
                size.serial_char().encode_utf8(&mut buf[2..3]);
                std::str::from_utf8(&buf[..3]).unwrap()
            }
            DataType::Raw { size } => {
                write!(&mut buf[..], "r{size}").expect("TODO");
                std::str::from_utf8(&buf[..]).unwrap()
            }
        };
        serializer.serialize_str(s)
    }
}

struct DataTypeVisitor;

impl<'de> serde::de::Visitor<'de> for DataTypeVisitor {
    type Value = DataType;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a string of the format `bool|[<>|]?[iuf][1248]`")
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        let dtype = match value {
            "b1" => DataType::Bool {
                size: BoolSize::B1,
                endian: Endian::NotRelevant,
            },
            "i1" => DataType::Int {
                size: IntSize::B1,
                endian: Endian::Little,
            },
            "u1" => DataType::UInt {
                size: IntSize::B1,
                endian: Endian::Little,
            },
            dtype if dtype.starts_with('r') => {
                if let Ok(size) = dtype[1..].parse::<usize>() {
                    if size % 8 == 0 {
                        DataType::Raw { size }
                    } else {
                        // TODO: more specific error?
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(value),
                            &self,
                        ));
                    }
                } else {
                    return Err(serde::de::Error::invalid_value(
                        serde::de::Unexpected::Str(value),
                        &self,
                    ));
                }
            }
            dtype if dtype.len() == 3 => {
                let mut chars = dtype.chars();
                let endian = Endian::deserial_char(chars.next().unwrap()).expect("TODO");
                match chars.next().unwrap() {
                    'i' => {
                        let size = IntSize::deserial_char(chars.next().unwrap()).unwrap();
                        DataType::Int { size, endian }
                    }
                    'u' => {
                        let size = IntSize::deserial_char(chars.next().unwrap()).unwrap();
                        DataType::UInt { size, endian }
                    }
                    'f' => {
                        let size = FloatSize::deserial_char(chars.next().unwrap()).unwrap();
                        DataType::Float { size, endian }
                    }
                    'b' => {
                        let size = BoolSize::deserial_char(chars.next().unwrap()).unwrap();
                        DataType::Bool { size, endian }
                    }
                    _ => {
                        return Err(serde::de::Error::invalid_value(
                            serde::de::Unexpected::Str(value),
                            &self,
                        ))
                    }
                }
            }
            _ => {
                return Err(serde::de::Error::invalid_value(
                    serde::de::Unexpected::Str(value),
                    &self,
                ))
            }
        };

        Ok(dtype)
    }
}

impl<'de> Deserialize<'de> for DataType {
    fn deserialize<D>(deserializer: D) -> Result<DataType, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(DataTypeVisitor)
    }
}

/// Replace all RsType tokens with the provide type.
#[macro_export]
macro_rules! data_type_rstype_replace {
    // Open parenthesis.
    ($rstype:ty, @($($stack:tt)*) ($($first:tt)*) $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(() $($stack)*) $($first)* __paren $($rest)*)
    };

    // Open square bracket.
    ($rstype:ty, @($($stack:tt)*) [$($first:tt)*] $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(() $($stack)*) $($first)* __bracket $($rest)*)
    };

    // Open curly brace.
    ($rstype:ty, @($($stack:tt)*) {$($first:tt)*} $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(() $($stack)*) $($first)* __brace $($rest)*)
    };

    // Close parenthesis.
    ($rstype:ty, @(($($close:tt)*) ($($top:tt)*) $($stack:tt)*) __paren $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(($($top)* ($($close)*)) $($stack)*) $($rest)*)
    };

    // Close square bracket.
    ($rstype:ty, @(($($close:tt)*) ($($top:tt)*) $($stack:tt)*) __bracket $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(($($top)* [$($close)*]) $($stack)*) $($rest)*)
    };

    // Close curly brace.
    ($rstype:ty, @(($($close:tt)*) ($($top:tt)*) $($stack:tt)*) __brace $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(($($top)* {$($close)*}) $($stack)*) $($rest)*)
    };

    // Replace `RsType` token with $rstype.
    ($rstype:ty, @(($($top:tt)*) $($stack:tt)*) RsType $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(($($top)* $rstype) $($stack)*) $($rest)*)
    };

    // Munch a token that is not `RsType`.
    ($rstype:ty, @(($($top:tt)*) $($stack:tt)*) $first:tt $($rest:tt)*) => {
        data_type_rstype_replace!($rstype, @(($($top)* $first) $($stack)*) $($rest)*)
    };

    // Terminal case.
    ($rstype:ty, @(($($top:tt)+))) => {
        $($top)+
    };

    // Initial case.
    ($rstype:ty, $($input:tt)+) => {
        data_type_rstype_replace!($rstype, @(()) $($input)*)
    };
}

/// Match a DataType-valued expression, and in each arm repeat the provided
/// code chunk with the token `RsType` replaced with the primitive type
/// appropriate for that arm.
#[macro_export]
macro_rules! data_type_match {
    ($match_expr:expr, $raw_match:pat => $raw_expr:expr, $($expr:tt)*) => {
        {
            match $match_expr {
                $crate::DataType::Bool { .. } => $crate::data_type_rstype_replace!(bool, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B1, ..} => $crate::data_type_rstype_replace!(u8, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B2, ..}=> $crate::data_type_rstype_replace!(u16, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B4, ..} => $crate::data_type_rstype_replace!(u32, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B8, ..} => $crate::data_type_rstype_replace!(u64, $($expr)*),
                $crate::DataType::Int {size: IntSize::B1, ..} => $crate::data_type_rstype_replace!(i8, $($expr)*),
                $crate::DataType::Int {size: IntSize::B2, ..}=> $crate::data_type_rstype_replace!(i16, $($expr)*),
                $crate::DataType::Int {size: IntSize::B4, ..} => $crate::data_type_rstype_replace!(i32, $($expr)*),
                $crate::DataType::Int {size: IntSize::B8, ..} => $crate::data_type_rstype_replace!(i64, $($expr)*),
                $crate::DataType::Float {size: FloatSize::B2, ..}=> $crate::data_type_rstype_replace!(f16, $($expr)*),
                $crate::DataType::Float {size: FloatSize::B4, ..} => $crate::data_type_rstype_replace!(f32, $($expr)*),
                $crate::DataType::Float {size: FloatSize::B8, ..} => $crate::data_type_rstype_replace!(f64, $($expr)*),
                $raw_match => $raw_expr,
            }
        }
    };
    ($match_expr:expr, $($expr:tt)*) => {
        {
            match $match_expr {
                $crate::DataType::Bool { .. } => $crate::data_type_rstype_replace!(bool, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B1, ..} => $crate::data_type_rstype_replace!(u8, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B2, ..}=> $crate::data_type_rstype_replace!(u16, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B4, ..} => $crate::data_type_rstype_replace!(u32, $($expr)*),
                $crate::DataType::UInt {size: IntSize::B8, ..} => $crate::data_type_rstype_replace!(u64, $($expr)*),
                $crate::DataType::Int {size: IntSize::B1, ..} => $crate::data_type_rstype_replace!(i8, $($expr)*),
                $crate::DataType::Int {size: IntSize::B2, ..}=> $crate::data_type_rstype_replace!(i16, $($expr)*),
                $crate::DataType::Int {size: IntSize::B4, ..} => $crate::data_type_rstype_replace!(i32, $($expr)*),
                $crate::DataType::Int {size: IntSize::B8, ..} => $crate::data_type_rstype_replace!(i64, $($expr)*),
                $crate::DataType::Float {size: FloatSize::B2, ..}=> $crate::data_type_rstype_replace!(f16, $($expr)*),
                $crate::DataType::Float {size: FloatSize::B4, ..} => $crate::data_type_rstype_replace!(f32, $($expr)*),
                $crate::DataType::Float {size: FloatSize::B8, ..} => $crate::data_type_rstype_replace!(f64, $($expr)*),
                $crate::DataType::Raw { .. } => $crate::data_type_rstype_replace!([u8], $($expr)*),
            }
        }
    };
}

impl DataType {
    /// Boilerplate method for reflection of primitive type sizes.
    #[must_use]
    pub fn size_of(self) -> usize {
        data_type_match!(self, DataType::Raw { size } => { std::mem::size_of::<u8>() * size / 8 }, {
            std::mem::size_of::<RsType>()
        })
    }

    /// Returns the [`Endian`] of the data type.
    #[must_use]
    pub fn endian(self) -> Endian {
        match self {
            DataType::Int { endian, .. }
            | DataType::UInt { endian, .. }
            | DataType::Float { endian, .. } => endian,
            // These are single-byte types.
            _ => NATIVE_ENDIAN,
        }
    }

    pub(crate) fn eq_modulo_endian(&self, other: &Self) -> bool {
        match (self, other) {
            (DataType::Bool { size: s1, .. }, DataType::Bool { size: s2, .. }) => s1 == s2,
            (DataType::Int { size: s1, .. }, DataType::Int { size: s2, .. }) => s1 == s2,
            (DataType::UInt { size: s1, .. }, DataType::UInt { size: s2, .. }) => s1 == s2,
            (DataType::Float { size: s1, .. }, DataType::Float { size: s2, .. }) => s1 == s2,
            (DataType::Raw { size: s1 }, DataType::Raw { size: s2 }) => s1 == s2,
            _ => false,
        }
    }
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

/// Trait implemented by primitive types that are reflected in Zarr.
///
/// The supertraits are not necessary for this trait, but are used to
/// remove redundant bounds elsewhere when operating generically over
/// data types.
// `DeserializedOwned` is necessary for deserialization of metadata `fill_value`.
pub trait ReflectedType:
    Send + Sync + Clone + Default + serde::de::DeserializeOwned + 'static
{
    /// The corresponding Zarr data type for this rust type.
    const ZARR_TYPE: DataType;

    /// Create a [`VecDataChunk`] for this type, filled with the zeros/default values.
    #[must_use]
    fn create_data_chunk(grid_position: &GridCoord, num_el: u32) -> VecDataChunk<Self> {
        VecDataChunk::<Self>::new(
            grid_position.clone(),
            vec![Self::default(); num_el as usize],
        )
    }
}

macro_rules! reflected_type {
    ($d_name:expr, $d_type:ty) => {
        impl ReflectedType for $d_type {
            const ZARR_TYPE: DataType = $d_name;
        }
    };
}

#[rustfmt::skip] reflected_type!(DataType::Bool {size: BoolSize::B1, endian: Endian::NotRelevant}, bool);
#[rustfmt::skip] reflected_type!(DataType::UInt {size: IntSize::B1, endian: NATIVE_ENDIAN}, u8);
#[rustfmt::skip] reflected_type!(DataType::UInt {size: IntSize::B2, endian: NATIVE_ENDIAN}, u16);
#[rustfmt::skip] reflected_type!(DataType::UInt {size: IntSize::B4, endian: NATIVE_ENDIAN}, u32);
#[rustfmt::skip] reflected_type!(DataType::UInt {size: IntSize::B8, endian: NATIVE_ENDIAN}, u64);
#[rustfmt::skip] reflected_type!(DataType::Int {size: IntSize::B1, endian: NATIVE_ENDIAN}, i8);
#[rustfmt::skip] reflected_type!(DataType::Int {size: IntSize::B2, endian: NATIVE_ENDIAN}, i16);
#[rustfmt::skip] reflected_type!(DataType::Int {size: IntSize::B4, endian: NATIVE_ENDIAN}, i32);
#[rustfmt::skip] reflected_type!(DataType::Int {size: IntSize::B8, endian: NATIVE_ENDIAN}, i64);
#[rustfmt::skip] reflected_type!(DataType::Float {size: FloatSize::B2, endian: NATIVE_ENDIAN}, f16);
#[rustfmt::skip] reflected_type!(DataType::Float {size: FloatSize::B4, endian: NATIVE_ENDIAN}, f32);
#[rustfmt::skip] reflected_type!(DataType::Float {size: FloatSize::B8, endian: NATIVE_ENDIAN}, f64);

// TODO: As example
#[rustfmt::skip] reflected_type!(DataType::Raw {size: 8}, [u8; 1]);
#[rustfmt::skip] reflected_type!(DataType::Raw {size: 16}, [u8; 2]);
#[rustfmt::skip] reflected_type!(DataType::Raw {size: 24}, [u8; 3]);
#[rustfmt::skip] reflected_type!(DataType::Raw {size: 32}, [u8; 4]);

#[cfg(test)]
mod tests {
    use super::*;

    fn test_data_type_reflection<T: ReflectedType>() {
        assert_eq!(std::mem::size_of::<T>(), T::ZARR_TYPE.size_of());
    }

    #[test]
    fn test_all_data_type_reflections() {
        test_data_type_reflection::<bool>();
        test_data_type_reflection::<u8>();
        test_data_type_reflection::<u16>();
        test_data_type_reflection::<u32>();
        test_data_type_reflection::<u64>();
        test_data_type_reflection::<i8>();
        test_data_type_reflection::<i16>();
        test_data_type_reflection::<i32>();
        test_data_type_reflection::<i64>();
        test_data_type_reflection::<f16>();
        test_data_type_reflection::<f32>();
        test_data_type_reflection::<f64>();
        test_data_type_reflection::<[u8; 1]>();
        test_data_type_reflection::<[u8; 2]>();
        test_data_type_reflection::<[u8; 3]>();
        test_data_type_reflection::<[u8; 4]>();
    }
}
