use crate::{map, map::HashMap, Equivalent, InfallibleResult, ToOwnedEquivalent};
use alloc::boxed::Box;
use core::{
    borrow::Borrow,
    fmt,
    hash::{BuildHasher, Hash, Hasher},
    ptr::NonNull,
};
use parking_lot::{Condvar, Mutex, MutexGuard, RwLock};
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use stable_deref_trait::StableDeref;

#[cfg(feature = "std")]
#[inline]
fn default_shards_amount() -> usize {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        thread::available_parallelism,
    };

    // We want to cache the result to compute it only once
    static N_SHARDS: AtomicUsize = AtomicUsize::new(0);

    #[cold]
    fn load_slow() -> usize {
        // This is racy but that's fine
        let n = available_parallelism()
            .ok()
            .and_then(|n| n.get().checked_next_power_of_two())
            .and_then(|n| n.checked_mul(4))
            .unwrap_or(16);
        N_SHARDS.store(n, Ordering::Relaxed);
        n
    }

    let n = N_SHARDS.load(Ordering::Relaxed);
    if n != 0 {
        return n;
    }

    load_slow()
}

#[cfg(not(feature = "std"))]
#[inline]
fn default_shards_amount() -> usize {
    32
}

unsafe fn extend_lifetime<'a, T: StableDeref>(ptr: &T) -> &'a T::Target {
    &*(&**ptr as *const T::Target)
}

struct ValidPtr<T>(NonNull<T>);

unsafe impl<T: Send> Send for ValidPtr<T> {}
unsafe impl<T: Sync> Sync for ValidPtr<T> {}

impl<T> core::ops::Deref for ValidPtr<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { self.0.as_ref() }
    }
}

impl<T> Clone for ValidPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for ValidPtr<T> {}

impl<T> Borrow<T> for ValidPtr<T> {
    fn borrow(&self) -> &T {
        self
    }
}

impl<T: Hash> Hash for ValidPtr<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state);
    }
}

impl<T: PartialEq> PartialEq for ValidPtr<T> {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}

impl<T: Eq> Eq for ValidPtr<T> {}

/// Looks like a Condvar, but wait for all notified threads to wake up when
/// calling `notify_all`.

struct WaitingBarrier {
    condvar: Condvar,
    n_waiters: Mutex<usize>,
}

struct Waiter<'a> {
    guard: MutexGuard<'a, usize>,
    condvar: &'a Condvar,
}

impl WaitingBarrier {
    fn new() -> Self {
        Self {
            condvar: Condvar::new(),
            n_waiters: Mutex::new(0),
        }
    }

    /// Registers ourselves as willing to wait
    fn prepare_waiting(&self) -> Waiter {
        let mut guard = self.n_waiters.lock();
        *guard += 1;
        Waiter {
            guard,
            condvar: &self.condvar,
        }
    }

    /// Notifies all waiters and wait for them to wake up
    fn notify_all(&self) {
        let mut n = self.n_waiters.lock();
        self.condvar.notify_all();
        while *n != 0 {
            self.condvar.wait(&mut n);
        }
    }
}

impl Waiter<'_> {
    fn wait(mut self) {
        self.condvar.wait(&mut self.guard);
    }
}

impl Drop for Waiter<'_> {
    fn drop(&mut self) {
        *self.guard -= 1;
        if *self.guard == 0 {
            self.condvar.notify_one();
        }
    }
}

struct BarrierGuard<'a>(&'a WaitingBarrier);

impl Drop for BarrierGuard<'_> {
    fn drop(&mut self) {
        self.0.notify_all();
    }
}

type Waiters<K> = Mutex<HashMap<ValidPtr<K>, ValidPtr<WaitingBarrier>>>;

struct WaitersGuard<'a, K: Eq + Hash> {
    waiters: &'a Waiters<K>,
    key: &'a K,
    hash: u64,
}

impl<'a, K: Eq + Hash> Drop for WaitersGuard<'a, K> {
    fn drop(&mut self) {
        let mut writing = self.waiters.lock();
        writing.remove(self.hash, self.key);
    }
}

#[repr(align(64))]
struct Shard<K, V> {
    map: RwLock<HashMap<K, V>>,

    // This lock should always be taken after `map`
    waiters: Waiters<K>,
}

impl<K, V> Shard<K, V> {
    fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
            waiters: Mutex::new(HashMap::new()),
        }
    }
}

impl<K, V> Shard<K, V>
where
    K: Hash + Eq,
{
    fn get<Q, T>(&self, hash: u64, key: &Q, with_result: impl FnOnce(&K, &V) -> T) -> Option<T>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let this = self.map.read();
        let (k, v) = this.get_key_value(hash, key)?;
        Some(with_result(k, v))
    }

    fn try_get<Q, G, T, U>(&self, hash: u64, key: &Q, data: T, with_result: G) -> Result<U, (T, G)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
        G: FnOnce(T, &K, &V) -> U,
    {
        let this = self.map.read();
        match this.get_key_value(hash, key) {
            Some((k, v)) => Ok(with_result(data, k, v)),
            None => Err((data, with_result)),
        }
    }

    #[cold]
    fn get_or_try_insert<E, T, U>(
        &self,
        hash: u64,
        key: K,
        data: T,
        hasher: &impl BuildHasher,
        on_vacant: impl FnOnce(T, &K) -> Result<(V, U), E>,
        on_occupied: impl FnOnce(T, &K, &V) -> U,
    ) -> Result<U, E> {
        let barrier = WaitingBarrier::new();

        loop {
            // If a value already exists, we're done
            let map = self.map.read();
            if let Some((key, value)) = map.get_key_value(hash, &key) {
                return Ok(on_occupied(data, key, value));
            }

            // Else try to register ourselves as willing to write
            let mut writing = self.waiters.lock();

            drop(map);

            match writing.entry(hash, &key, hasher) {
                map::Entry::Occupied(entry) => {
                    // Somebody is already writing this value ! Wait until it
                    // is done, then start again.

                    // Safety: We call `prepare_wait` before dropping the mutex
                    // guard, so the barrier is guaranteed to be valid for the
                    // wait even if it was removed from the map.
                    let barrier = unsafe { entry.get().0.as_ref() };
                    let waiter = barrier.prepare_waiting();

                    // Ensure that other threads will be able to use the mutex
                    // while we wait for the value's writing to complete
                    drop(writing);
                    waiter.wait();
                    continue;
                }
                map::Entry::Vacant(entry) => {
                    // We're the first ! Register our barrier so other can wait
                    // on it.
                    let key_ref = ValidPtr(NonNull::from(&key));
                    let barrier_ref = ValidPtr(NonNull::from(&barrier));
                    entry.insert(key_ref, barrier_ref);
                    break;
                }
            }
        }

        // We know that are we know that we are the only one reaching this point
        // for this key

        // Now that our barrier is shared, some other thread might wait on it
        // even if it is removed from `self.waiters.tokens`, so we make sure
        // that we don't leave this function while someone still thinks the
        // barrier is alive.
        let _barrier_guard = BarrierGuard(&barrier);
        let guard = WaitersGuard {
            waiters: &self.waiters,
            key: &key,
            hash,
        };

        // It is important not to hold any lock here
        let (value, ret) = on_vacant(data, &key)?;

        // Take this lock first to avoid deadlocks
        let mut map = self.map.write();

        // We'll have to move the key to insert it in the map, which will
        // invalidate the pointer we put in `waiters`, so we remove it now.
        //
        // Note that the mutex guard will stay alive until the end of the
        // function, which is intentional.
        let mut writing = self.waiters.lock();

        match writing.remove(hash, &key) {
            Some(b) => debug_assert!(core::ptr::eq(b.0.as_ptr(), &barrier)),
            None => debug_assert!(false),
        }

        // We have just done the cleanup manually
        core::mem::forget(guard);

        // We can finally insert the value in the map.
        match map.entry(hash, &key, hasher) {
            map::Entry::Vacant(entry) => {
                entry.insert(key, value);
            }
            map::Entry::Occupied(_) => panic!("re-entrant init"),
        }
        Ok(ret)

        // Leaving the function will wake up waiting threads.
    }

    pub fn contains_key<Q>(&self, hash: u64, key: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.map.read().contains_key(hash, key)
    }

    pub fn remove_entry<Q>(&mut self, hash: u64, key: &Q) -> Option<(K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.map.get_mut().remove_entry(hash, key)
    }
}

pub struct OnceMap<K, V, S = crate::RandomState> {
    shards: Box<[Shard<K, V>]>,
    hash_builder: S,
}

impl<K, V> OnceMap<K, V> {
    /// Creates an empty `OnceMap`.
    pub fn new() -> Self {
        Self::with_hasher(crate::RandomState::new())
    }

    #[cfg(test)]
    pub(crate) fn with_single_shard() -> Self {
        let hash_builder = crate::RandomState::new();
        let shards = Box::new([Shard::new()]);
        Self {
            shards,
            hash_builder,
        }
    }
}

impl<K, V, S> OnceMap<K, V, S> {
    /// Creates an empty `OnceMap` which will use the given hash builder to hash keys.
    pub fn with_hasher(hash_builder: S) -> Self {
        let shards = (0..default_shards_amount()).map(|_| Shard::new()).collect();
        Self {
            shards,
            hash_builder,
        }
    }
}

impl<K, V, S> OnceMap<K, V, S> {
    /// Removes all key-value pairs from the map, but keeps the allocated memory.
    pub fn clear(&mut self) {
        self.shards.iter_mut().for_each(|s| s.map.get_mut().clear());
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.shards
            .iter_mut()
            .flat_map(|s| s.map.get_mut().values_mut())
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.shards
            .iter_mut()
            .flat_map(|s| s.map.get_mut().iter_mut())
    }

    #[allow(clippy::should_implement_trait)]
    pub fn into_iter(self) -> impl Iterator<Item = (K, V)> {
        self.shards
            .into_vec()
            .into_iter()
            .flat_map(|s| s.map.into_inner().into_iter())
    }

    /// Returns a reference to the map's [`BuildHasher`].
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Locks the whole map for reading.
    ///
    /// This enables more methods, such as iterating on the maps, but will cause
    /// a deadlock if trying to insert values in the map from the same thread.
    pub fn read_only_view(&self) -> ReadOnlyView<K, V, S> {
        ReadOnlyView::new(self)
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> OnceMap<K, V, S>
where
    K: Send,
    V: Send,
{
    pub fn into_par_iter(self) -> impl rayon::iter::ParallelIterator<Item = (K, V)> {
        self.shards
            .into_vec()
            .into_par_iter()
            .flat_map(|s| s.map.into_inner().into_par_iter())
    }
}

impl<K, V, S> OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn hash_one<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        crate::hash_one(&self.hash_builder, key)
    }

    fn get_shard(&self, hash: u64) -> &Shard<K, V> {
        let len = self.shards.len();
        &self.shards[(len - 1) & (hash as usize)]
    }

    fn get_shard_mut(&mut self, hash: u64) -> &mut Shard<K, V> {
        let len = self.shards.len();
        &mut self.shards[(len - 1) & (hash as usize)]
    }

    /// Returns `true` if the map contains a value for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.get_shard(hash).contains_key(hash, key)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        let (_, v) = self.get_shard_mut(hash).remove_entry(hash, key)?;
        Some(v)
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.get_shard_mut(hash).remove_entry(hash, key)
    }
}

impl<K, V, S> OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
    V: StableDeref,
{
    /// Returns a reference to the value corresponding to the key.
    pub fn get<Q>(&self, key: &Q) -> Option<&V::Target>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.map_get(key, |_, v| unsafe { extend_lifetime(v) })
    }

    /// Returns a reference to the value corresponding to the key or insert one
    /// with the given closure.
    pub fn insert(&self, key: K, make_val: impl FnOnce(&K) -> V) -> &V::Target {
        self.map_insert(key, make_val, |_, v| unsafe { extend_lifetime(v) })
    }

    /// Same as `insert` but the closure is allowed to fail.
    ///
    /// If the closure is called and an error is returned, no value is stored in
    /// the map.
    pub fn try_insert<E>(
        &self,
        key: K,
        make_val: impl FnOnce(&K) -> Result<V, E>,
    ) -> Result<&V::Target, E> {
        self.map_try_insert(key, make_val, |_, v| unsafe { extend_lifetime(v) })
    }
}

impl<K, V, S> OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
    V: Clone,
{
    pub fn get_cloned<Q>(&self, key: &Q) -> Option<V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.map_get(key, |_, v| v.clone())
    }

    pub fn insert_cloned(&self, key: K, make_val: impl FnOnce(&K) -> V) -> V {
        self.map_insert(key, make_val, |_, v| v.clone())
    }

    pub fn try_insert_cloned<E>(
        &self,
        key: K,
        make_val: impl FnOnce(&K) -> Result<V, E>,
    ) -> Result<V, E> {
        self.map_try_insert(key, make_val, |_, v| v.clone())
    }
}

impl<K, V, S> OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn map_get<Q, T>(&self, key: &Q, with_result: impl FnOnce(&K, &V) -> T) -> Option<T>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.get_shard(hash).get(hash, key, with_result)
    }

    pub fn map_insert<T>(
        &self,
        key: K,
        make_val: impl FnOnce(&K) -> V,
        with_result: impl FnOnce(&K, &V) -> T,
    ) -> T {
        self.map_try_insert(key, |k| Ok(make_val(k)), with_result)
            .unwrap_infallible()
    }

    pub fn map_insert_ref<Q, T>(
        &self,
        key: &Q,
        make_key: impl FnOnce(&Q) -> K,
        make_val: impl FnOnce(&K) -> V,
        with_result: impl FnOnce(&K, &V) -> T,
    ) -> T
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.map_try_insert_ref(key, make_key, |k| Ok(make_val(k)), with_result)
            .unwrap_infallible()
    }

    pub fn map_try_insert<T, E>(
        &self,
        key: K,
        make_val: impl FnOnce(&K) -> Result<V, E>,
        with_result: impl FnOnce(&K, &V) -> T,
    ) -> Result<T, E> {
        self.get_or_try_insert(
            key,
            with_result,
            |with_result, k| {
                let v = make_val(k)?;
                let ret = with_result(k, &v);
                Ok((v, ret))
            },
            |with_result, k, v| with_result(k, v),
        )
    }

    pub fn map_try_insert_ref<Q, T, E>(
        &self,
        key: &Q,
        make_key: impl FnOnce(&Q) -> K,
        make_val: impl FnOnce(&K) -> Result<V, E>,
        with_result: impl FnOnce(&K, &V) -> T,
    ) -> Result<T, E>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.get_or_try_insert_ref(
            key,
            with_result,
            make_key,
            |with_result, k| {
                let v = make_val(k)?;
                let ret = with_result(k, &v);
                Ok((v, ret))
            },
            |with_result, k, v| with_result(k, v),
        )
    }

    pub fn get_or_try_insert<T, U, E>(
        &self,
        key: K,
        data: T,
        on_vacant: impl FnOnce(T, &K) -> Result<(V, U), E>,
        on_occupied: impl FnOnce(T, &K, &V) -> U,
    ) -> Result<U, E> {
        let hash = self.hash_one(&key);
        let shard = self.get_shard(hash);

        match shard.try_get(hash, &key, data, on_occupied) {
            Ok(result) => Ok(result),
            Err((data, on_occupied)) => {
                shard.get_or_try_insert(hash, key, data, &self.hash_builder, on_vacant, on_occupied)
            }
        }
    }

    pub fn get_or_try_insert_ref<Q, T, U, E>(
        &self,
        key: &Q,
        data: T,
        make_key: impl FnOnce(&Q) -> K,
        on_vacant: impl FnOnce(T, &K) -> Result<(V, U), E>,
        on_occupied: impl FnOnce(T, &K, &V) -> U,
    ) -> Result<U, E>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        let shard = self.get_shard(hash);

        // Try to get the value from the map as a fast-path, to avoid having to
        // compute an owned version of the key.
        shard.try_get(hash, key, data, on_occupied).or_else(
            #[cold]
            |(data, on_occupied)| {
                let owned_key = make_key(key);
                debug_assert_eq!(self.hash_one::<K>(&owned_key), hash);
                debug_assert!(key.equivalent(&owned_key));
                shard.get_or_try_insert(
                    hash,
                    owned_key,
                    data,
                    &self.hash_builder,
                    on_vacant,
                    on_occupied,
                )
            },
        )
    }
}

impl<K, V, S: Default> Default for OnceMap<K, V, S> {
    fn default() -> Self {
        Self::with_hasher(S::default())
    }
}

impl<K, V, S> Extend<(K, V)> for OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        iter.into_iter()
            .for_each(|(k, v)| self.map_insert(k, |_| v, |_, _| ()))
    }
}

impl<K, V, S> Extend<(K, V)> for &'_ OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        iter.into_iter()
            .for_each(|(k, v)| self.map_insert(k, |_| v, |_, _| ()))
    }
}

impl<K, V, S> FromIterator<(K, V)> for OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut map = OnceMap::default();
        map.extend(iter);
        map
    }
}

impl<K, V, S, const N: usize> From<[(K, V); N]> for OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher + Default,
{
    fn from(array: [(K, V); N]) -> Self {
        Self::from_iter(array)
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> ParallelExtend<(K, V)> for OnceMap<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Sync,
{
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: IntoParallelIterator<Item = (K, V)>,
    {
        par_iter
            .into_par_iter()
            .for_each(|(k, v)| self.map_insert(k, |_| v, |_, _| ()));
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> ParallelExtend<(K, V)> for &'_ OnceMap<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Sync,
{
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: IntoParallelIterator<Item = (K, V)>,
    {
        par_iter
            .into_par_iter()
            .for_each(|(k, v)| self.map_insert(k, |_| v, |_, _| ()));
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<K, V, S> FromParallelIterator<(K, V)> for OnceMap<K, V, S>
where
    K: Eq + Hash + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Default + Sync,
{
    fn from_par_iter<I>(par_iter: I) -> Self
    where
        I: IntoParallelIterator<Item = (K, V)>,
    {
        let mut map = Self::default();
        map.par_extend(par_iter);
        map
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<K, V, S> serde::Serialize for OnceMap<K, V, S>
where
    K: serde::Serialize,
    V: serde::Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        serializer.collect_map(self.read_only_view().iter())
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<'de, K, V, S> serde::Deserialize<'de> for OnceMap<K, V, S>
where
    K: Eq + Hash + serde::Deserialize<'de>,
    V: serde::Deserialize<'de>,
    S: BuildHasher + Default,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct OnceMapVisitor<K, V, S>(OnceMap<K, V, S>);

        impl<'de, K, V, S> serde::de::Visitor<'de> for OnceMapVisitor<K, V, S>
        where
            K: Eq + Hash + serde::Deserialize<'de>,
            V: serde::Deserialize<'de>,
            S: BuildHasher,
        {
            type Value = OnceMap<K, V, S>;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a map")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::MapAccess<'de>,
            {
                while let Some((key, value)) = map.next_entry()? {
                    self.0.map_insert(key, |_| value, |_, _| ())
                }

                Ok(self.0)
            }
        }

        deserializer.deserialize_map(OnceMapVisitor(OnceMap::default()))
    }
}

impl<K, V, S> fmt::Debug for OnceMap<K, V, S>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.read_only_view().iter()).finish()
    }
}

#[repr(transparent)]
struct LockedShard<K, V>(Shard<K, V>);

unsafe impl<K: Sync, V: Sync> Sync for LockedShard<K, V> {}

impl<K, V> LockedShard<K, V> {
    fn get(&self) -> &HashMap<K, V> {
        unsafe { &*self.0.map.data_ptr() }
    }
}

pub struct ReadOnlyView<'a, K, V, S = crate::RandomState> {
    shards: &'a [LockedShard<K, V>],
    hasher: &'a S,
    _marker: crate::PhantomUnsend,
}

impl<'a, K, V, S> ReadOnlyView<'a, K, V, S> {
    fn new(map: &'a OnceMap<K, V, S>) -> Self {
        use parking_lot::lock_api::RawRwLockRecursive;

        for shard in map.shards.iter() {
            unsafe {
                shard.map.raw().lock_shared_recursive();
            }
        }

        let shards = unsafe { &*(&*map.shards as *const [_] as *const [_]) };

        Self {
            shards,
            hasher: &map.hash_builder,
            _marker: crate::PhantomUnsend(std::marker::PhantomData),
        }
    }

    #[inline]
    fn iter_shards(&self) -> impl ExactSizeIterator<Item = &HashMap<K, V>> {
        self.shards.iter().map(|shard| shard.get())
    }

    pub fn len(&self) -> usize {
        self.iter_shards().map(|s| s.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.iter_shards().all(|s| s.is_empty())
    }

    pub fn hasher(&self) -> &S {
        self.hasher
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.iter_shards().flat_map(|shard| shard.iter())
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.iter_shards().flat_map(|shard| shard.keys())
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.iter_shards().flat_map(|shard| shard.values())
    }
}

impl<'a, K, V, S> ReadOnlyView<'a, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    #[inline]
    fn hash_one<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        crate::hash_one(self.hasher, key)
    }

    fn get_shard(&self, hash: u64) -> &HashMap<K, V> {
        let len = self.shards.len();
        let shard = &self.shards[(len - 1) & (hash as usize)];
        shard.get()
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.get_shard(hash).get(hash, key)
    }

    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.get_shard(hash).get_key_value(hash, key)
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.get_shard(hash).contains_key(hash, key)
    }
}

impl<K, V, S> Drop for ReadOnlyView<'_, K, V, S> {
    fn drop(&mut self) {
        for shard in self.shards.iter() {
            unsafe { shard.0.map.force_unlock_read() }
        }
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, K, V, S> ReadOnlyView<'a, K, V, S>
where
    K: Sync,
    V: Sync,
{
    #[inline]
    fn par_iter_shards(&self) -> impl rayon::iter::IndexedParallelIterator<Item = &HashMap<K, V>> {
        self.shards.par_iter().map(|shard| shard.get())
    }

    pub fn par_iter(&self) -> impl rayon::iter::ParallelIterator<Item = (&K, &V)> {
        self.par_iter_shards().flat_map(|shard| shard.par_iter())
    }

    pub fn par_keys(&self) -> impl rayon::iter::ParallelIterator<Item = &K> {
        self.par_iter_shards().flat_map(|shard| shard.par_keys())
    }

    pub fn par_values(&self) -> impl rayon::iter::ParallelIterator<Item = &V> {
        self.par_iter_shards().flat_map(|shard| shard.par_values())
    }
}

#[cfg(feature = "serde")]
#[cfg_attr(docsrs, doc(cfg(feature = "serde")))]
impl<K, V, S> serde::Serialize for ReadOnlyView<'_, K, V, S>
where
    K: serde::Serialize,
    V: serde::Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        serializer.collect_map(self.iter())
    }
}

impl<K, V, S> fmt::Debug for ReadOnlyView<'_, K, V, S>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.iter()).finish()
    }
}

/// A map where values are automatically filled at access.
///
/// This type can be shared across threads.
///
/// ```
/// let map = once_map::LazyMap::new(|x: &i32| x.to_string());
///
/// assert_eq!(&map[&3], "3");
/// assert_eq!(map.get(&-67), "-67");
/// ```
pub struct LazyMap<K, V, S = crate::RandomState, F = fn(&K) -> V> {
    map: OnceMap<K, V, S>,
    init: F,
}

impl<K, V, F> LazyMap<K, V, crate::RandomState, F> {
    pub fn new(f: F) -> Self {
        Self::with_hasher(crate::RandomState::new(), f)
    }
}

impl<K, V, S, F> LazyMap<K, V, S, F> {
    pub fn with_hasher(hash_builder: S, f: F) -> Self {
        Self {
            map: OnceMap::with_hasher(hash_builder),
            init: f,
        }
    }

    /// Removes all entries from the map.
    pub fn clear(&mut self) {
        self.map.clear();
    }
}

impl<K, V, S, F> LazyMap<K, V, S, F>
where
    K: Eq + Hash,
    S: BuildHasher,
    F: Fn(&K) -> V,
    V: StableDeref,
{
    pub fn get<Q>(&self, key: &Q) -> &V::Target
    where
        Q: Hash + ToOwnedEquivalent<K> + ?Sized,
    {
        self.map_get(key, |_, v| unsafe { extend_lifetime(v) })
    }
}

impl<K, V, S, F> LazyMap<K, V, S, F>
where
    K: Eq + Hash,
    S: BuildHasher,
    F: Fn(&K) -> V,
    V: Clone,
{
    pub fn get_cloned<Q>(&self, key: &Q) -> V
    where
        Q: Hash + ToOwnedEquivalent<K> + ?Sized,
    {
        self.map_get(key, |_, v| v.clone())
    }
}

impl<K, V, S, F> LazyMap<K, V, S, F>
where
    K: Eq + Hash,
    S: BuildHasher,
    F: Fn(&K) -> V,
{
    pub fn map_get<Q, T>(&self, key: &Q, with_result: impl FnOnce(&K, &V) -> T) -> T
    where
        Q: Hash + ToOwnedEquivalent<K> + ?Sized,
    {
        self.map
            .map_insert_ref(key, Q::to_owned_equivalent, &self.init, with_result)
    }
}

impl<K, V, S, F> LazyMap<K, V, S, F>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.map.remove(key)
    }
}

impl<K, V, S, F, Q> core::ops::Index<&Q> for LazyMap<K, V, S, F>
where
    K: Eq + Hash,
    S: BuildHasher,
    F: Fn(&K) -> V,
    V: StableDeref,
    Q: Hash + ToOwnedEquivalent<K> + ?Sized,
{
    type Output = V::Target;

    fn index(&self, key: &Q) -> &V::Target {
        self.get(key)
    }
}

/// Creates a `LazyMap` that fills all values with `V::default()`.
impl<K, V: Default, S: Default> Default for LazyMap<K, V, S> {
    fn default() -> Self {
        Self::with_hasher(S::default(), |_| V::default())
    }
}

impl<K, V, S> fmt::Debug for LazyMap<K, V, S>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyMap")
            .field("values", &self.map)
            .finish_non_exhaustive()
    }
}
