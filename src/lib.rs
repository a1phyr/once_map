#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
pub mod sync;

#[cfg(feature = "std")]
pub use sync::{OnceMap, LazyMap};

pub mod unsync;

#[cfg(test)]
mod tests;

use core::{
    borrow::Borrow,
    hash::{BuildHasher, Hash, Hasher},
};

fn hash_one<S: BuildHasher, Q: Hash + ?Sized>(hash_builder: &S, key: &Q) -> u64 {
    let mut hasher = hash_builder.build_hasher();
    key.hash(&mut hasher);
    hasher.finish()
}

trait HashMapExt {
    type Key;
    type Value;
    type Hasher;

    fn get_raw_entry<Q>(&self, hash: u64, key: &Q) -> Option<(&Self::Key, &Self::Value)>
    where
        Q: Eq + Hash + ?Sized,
        Self::Key: Borrow<Q>;

    fn get_raw_entry_mut<Q>(
        &mut self,
        hash: u64,
        key: &Q,
    ) -> hashbrown::hash_map::RawEntryMut<Self::Key, Self::Value, Self::Hasher>
    where
        Q: Eq + Hash + ?Sized,
        Self::Key: Borrow<Q>;
}

impl<K, V, S> HashMapExt for hashbrown::HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    type Key = K;
    type Value = V;
    type Hasher = S;

    fn get_raw_entry<Q>(&self, hash: u64, key: &Q) -> Option<(&Self::Key, &Self::Value)>
    where
        Q: Eq + Hash + ?Sized,
        Self::Key: Borrow<Q>,
    {
        self.raw_entry().from_key_hashed_nocheck(hash, key)
    }

    fn get_raw_entry_mut<Q>(
        &mut self,
        hash: u64,
        key: &Q,
    ) -> hashbrown::hash_map::RawEntryMut<Self::Key, Self::Value, Self::Hasher>
    where
        Q: Eq + Hash + ?Sized,
        Self::Key: Borrow<Q>,
    {
        self.raw_entry_mut().from_key_hashed_nocheck(hash, key)
    }
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
