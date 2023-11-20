#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]

extern crate alloc;

#[cfg(feature = "std")]
pub mod sync;

#[cfg(feature = "std")]
pub use sync::{LazyMap, OnceMap};

pub mod unsync;

mod map;

#[cfg(test)]
mod tests;

use core::hash::{BuildHasher, Hash, Hasher};

#[cfg(feature = "equivalent")]
pub use equivalent::Equivalent;

/// Generalization of `Borrow` that works with more types.
#[cfg(not(feature = "equivalent"))]
pub trait Equivalent<K: ?Sized> {
    fn equivalent(&self, key: &K) -> bool;
}

#[cfg(not(feature = "equivalent"))]
impl<Q, K> Equivalent<K> for Q
where
    Q: Eq + ?Sized,
    K: core::borrow::Borrow<Q> + ?Sized,
{
    fn equivalent(&self, key: &K) -> bool {
        self == key.borrow()
    }
}

/// Generalization of `ToOwned` that works with more types.
pub trait ToOwnedEquivalent<K>: Equivalent<K> {
    fn to_owned_equivalent(&self) -> K;
}

impl<Q> ToOwnedEquivalent<Q::Owned> for Q
where
    Q: alloc::borrow::ToOwned + Eq + ?Sized,
{
    fn to_owned_equivalent(&self) -> Q::Owned {
        self.to_owned()
    }
}

fn hash_one<S: BuildHasher, Q: Hash + ?Sized>(hash_builder: &S, key: &Q) -> u64 {
    let mut hasher = hash_builder.build_hasher();
    key.hash(&mut hasher);
    hasher.finish()
}

trait InfallibleResult {
    type Ok;

    fn unwrap_infallible(self) -> Self::Ok;
}

impl<T> InfallibleResult for Result<T, core::convert::Infallible> {
    type Ok = T;

    #[inline]
    fn unwrap_infallible(self) -> T {
        match self {
            Ok(v) => v,
            Err(void) => match void {},
        }
    }
}

#[cfg(feature = "ahash")]
use ahash::{AHasher as HasherInner, RandomState as RandomStateInner};

#[cfg(all(not(feature = "ahash"), feature = "std"))]
use std::collections::hash_map::{DefaultHasher as HasherInner, RandomState as RandomStateInner};

#[cfg(all(not(feature = "ahash"), not(feature = "std")))]
compile_error!("Either feature `ahash` or `std` must be enabled");

/// The default hasher used by this crate.
#[derive(Debug, Clone)]
pub struct RandomState(RandomStateInner);

#[derive(Debug, Clone, Default)]
pub struct DefaultHasher(HasherInner);

impl RandomState {
    #[inline]
    pub fn new() -> Self {
        Self(RandomStateInner::new())
    }
}

impl Default for RandomState {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl core::hash::BuildHasher for RandomState {
    type Hasher = DefaultHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        DefaultHasher(self.0.build_hasher())
    }
}

impl core::hash::Hasher for DefaultHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0.finish()
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        self.0.write(bytes)
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.0.write_u8(i)
    }

    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.0.write_u16(i)
    }

    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.0.write_u32(i)
    }

    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0.write_u64(i)
    }

    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.0.write_u128(i)
    }

    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.0.write_usize(i)
    }
}

/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<once_map::sync::ReadOnlyView<(), (), once_map::RandomState>>();
/// ```
struct PhantomUnsend(core::marker::PhantomData<*const ()>);
unsafe impl Sync for PhantomUnsend {}
