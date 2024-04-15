use crate::Equivalent;
use core::{
    fmt,
    hash::{BuildHasher, Hash},
};
use hashbrown::hash_table;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[inline]
fn equivalent<Q, K, V>(key: &Q) -> impl Fn(&(K, V)) -> bool + '_
where
    Q: Hash + Equivalent<K> + ?Sized,
{
    |(k, _)| key.equivalent(k)
}

/// Unfortunately `hashbrown` can drop elements if the hash function panics.
///
/// We prevent this by:
/// - Catching unwinds when `std` is enabled and returning a dummy hash
/// - Panicking in panic when `std` is not enabled, which result in an abort
///
/// See https://github.com/a1phyr/once_map/issues/3
#[inline]
fn hash_one<S: BuildHasher, K: Hash, V>(hasher: &S) -> impl Fn(&(K, V)) -> u64 + '_ {
    #[cfg(feature = "std")]
    let hash_one = |(k, _): &(K, V)| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| crate::hash_one(hasher, k)))
            .unwrap_or(0)
    };

    #[cfg(not(feature = "std"))]
    let hash_one = |(k, _): &(K, V)| {
        struct Guard;
        impl Drop for Guard {
            #[inline]
            fn drop(&mut self) {
                panic!("Hash implementation panicked");
            }
        }

        let guard = Guard;
        let h = crate::hash_one(hasher, k);
        core::mem::forget(guard);
        h
    };

    hash_one
}

/// This is just like std's `HashMap`, but it does not store its `BuildHasher`,
/// so it has to be provided (or a hash) for each operation.
pub struct HashMap<K, V>(hash_table::HashTable<(K, V)>);

impl<K, V> HashMap<K, V> {
    #[inline]
    pub const fn new() -> Self {
        Self(hash_table::HashTable::new())
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.0.iter().map(|(k, v)| (k, v))
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.0.iter_mut().map(|(k, v)| (&*k, v))
    }

    #[inline]
    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.0.iter().map(|(k, _)| k)
    }

    #[inline]
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.0.iter().map(|(_, v)| v)
    }

    #[inline]
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.0.iter_mut().map(|(_, v)| v)
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

#[cfg(feature = "rayon")]
impl<K, V> HashMap<K, V>
where
    K: Send,
    V: Send,
{
    #[inline]
    pub fn into_par_iter(self) -> impl rayon::iter::ParallelIterator<Item = (K, V)> {
        self.0.into_par_iter()
    }
}

#[cfg(feature = "rayon")]
impl<K, V> HashMap<K, V>
where
    K: Sync,
    V: Sync,
{
    #[inline]
    pub fn par_iter(&self) -> impl rayon::iter::ParallelIterator<Item = (&K, &V)> + '_ {
        self.0.par_iter().map(|(k, v)| (k, v))
    }

    #[inline]
    pub fn par_keys(&self) -> impl rayon::iter::ParallelIterator<Item = &K> + '_ {
        self.0.par_iter().map(|(k, _)| k)
    }

    #[inline]
    pub fn par_values(&self) -> impl rayon::iter::ParallelIterator<Item = &V> + '_ {
        self.0.par_iter().map(|(_, v)| v)
    }
}

impl<K, V> HashMap<K, V>
where
    K: Eq + Hash,
{
    #[inline]
    #[allow(clippy::manual_map)]
    pub fn get<Q>(&self, hash: u64, k: &Q) -> Option<&V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        match self.0.find(hash, equivalent(k)) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    #[inline]
    #[allow(clippy::manual_map)]
    pub fn get_key_value<Q>(&self, hash: u64, k: &Q) -> Option<(&K, &V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        match self.0.find(hash, equivalent(k)) {
            Some((k, v)) => Some((k, v)),
            None => None,
        }
    }

    #[inline]
    pub fn contains_key<Q>(&self, hash: u64, k: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.0.find(hash, equivalent(k)).is_some()
    }

    #[inline]
    pub fn entry<Q, S>(&mut self, hash: u64, k: &Q, hasher: &S) -> Entry<K, V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
        S: core::hash::BuildHasher,
    {
        match self.0.entry(hash, equivalent(k), hash_one(hasher)) {
            hash_table::Entry::Occupied(e) => Entry::Occupied(OccupiedEntry(e)),
            hash_table::Entry::Vacant(e) => Entry::Vacant(VacantEntry(e)),
        }
    }

    #[inline]
    pub fn remove<Q>(&mut self, hash: u64, k: &Q) -> Option<V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        match self.0.find_entry(hash, equivalent(k)) {
            Ok(e) => Some(e.remove().0 .1),
            Err(_) => None,
        }
    }

    #[inline]
    pub fn remove_entry<Q>(&mut self, hash: u64, k: &Q) -> Option<(K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        match self.0.find_entry(hash, equivalent(k)) {
            Ok(e) => Some(e.remove().0),
            Err(_) => None,
        }
    }
}

impl<K, V> IntoIterator for HashMap<K, V> {
    type Item = (K, V);
    type IntoIter = hash_table::IntoIter<(K, V)>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<K, V> fmt::Debug for HashMap<K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

pub enum Entry<'a, K, V> {
    Vacant(VacantEntry<'a, K, V>),
    Occupied(OccupiedEntry<'a, K, V>),
}

pub struct OccupiedEntry<'a, K, V>(hash_table::OccupiedEntry<'a, (K, V)>);

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    #[inline]
    pub fn get(&self) -> &V {
        &self.0.get().1
    }
}

pub struct VacantEntry<'a, K, V>(hash_table::VacantEntry<'a, (K, V)>);

impl<'a, K, V> VacantEntry<'a, K, V>
where
    K: Hash,
{
    #[inline]
    pub fn insert(self, key: K, value: V) -> &'a mut (K, V) {
        self.0.insert((key, value)).into_mut()
    }
}
