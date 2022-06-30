use crate::{HashMapExt, InfallibleResult};
use alloc::boxed::Box;
use core::{
    borrow::Borrow,
    fmt,
    hash::{BuildHasher, Hash, Hasher},
    ptr::NonNull,
};
use hashbrown::{hash_map, HashMap};
use parking_lot::{Condvar, Mutex, MutexGuard, RwLock};
use stable_deref_trait::StableDeref;

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
        &**self
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

type Waiters<K, S> = Mutex<HashMap<ValidPtr<K>, ValidPtr<WaitingBarrier>, S>>;

struct WaitersGuard<'a, K: Eq + Hash, S: BuildHasher> {
    waiters: &'a Waiters<K, S>,
    key: &'a K,
    hash: u64,
}

impl<'a, K: Eq + Hash, S: BuildHasher> Drop for WaitersGuard<'a, K, S> {
    fn drop(&mut self) {
        let mut writing = self.waiters.lock();
        match writing.get_raw_entry_mut(self.hash, self.key) {
            hash_map::RawEntryMut::Occupied(e) => {
                e.remove();
            }
            hash_map::RawEntryMut::Vacant(_) => (),
        }
    }
}

struct Shard<K, V, S> {
    map: RwLock<HashMap<K, V, S>>,

    // This lock should always be taken after `map`
    waiters: Waiters<K, S>,
}

impl<K, V, S> Shard<K, V, S>
where
    S: Clone,
{
    fn new(hash_builder: S) -> Self {
        Self {
            map: RwLock::new(HashMap::with_hasher(hash_builder.clone())),
            waiters: Mutex::new(HashMap::with_hasher(hash_builder)),
        }
    }
}

impl<K, V, S> Shard<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    fn get<Q, T>(&self, hash: u64, key: &Q, with_result: impl FnOnce(&K, &V) -> T) -> Option<T>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        let this = self.map.read();
        let (k, v) = this.get_raw_entry(hash, key)?;
        Some(with_result(k, v))
    }

    fn try_get<Q, G, T, U>(&self, hash: u64, key: &Q, data: T, with_result: G) -> Result<U, (T, G)>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
        G: FnOnce(T, &K, &V) -> U,
    {
        let this = self.map.read();
        match this.get_raw_entry(hash, key) {
            Some((k, v)) => Ok(with_result(data, k, v)),
            None => Err((data, with_result)),
        }
    }

    fn get_or_try_insert_with<E, T, U>(
        &self,
        hash: u64,
        key: K,
        data: T,
        on_vacant: impl FnOnce(T, &K) -> Result<(V, U), E>,
        on_occupied: impl FnOnce(T, &K, &V) -> U,
    ) -> Result<U, E> {
        let barrier = WaitingBarrier::new();

        loop {
            // If a value already exists, we're done
            let map = self.map.read();
            if let Some((key, value)) = map.get_raw_entry(hash, &key) {
                return Ok(on_occupied(data, key, value));
            }

            // Else try to register ourselves as willing to write
            let mut writing = self.waiters.lock();

            drop(map);

            match writing.get_raw_entry_mut(hash, &key) {
                hash_map::RawEntryMut::Occupied(entry) => {
                    // Somebody is already writing this value ! Wait until it
                    // is done, then start again.

                    // Safety: We call `prepare_wait` before dropping the mutex
                    // guard, so the barrier is guarantied to be valid for the
                    // wait even if it was removed from the map.
                    let barrier = unsafe { entry.get().0.as_ref() };
                    let waiter = barrier.prepare_waiting();

                    // Ensure that other threads will be able to use the mutex
                    // while we wait for the value's writing to complete
                    drop(writing);
                    waiter.wait();
                    continue;
                }
                hash_map::RawEntryMut::Vacant(entry) => {
                    // We're the first ! Register our barrier so other can wait
                    // on it.
                    let key_ref = ValidPtr(NonNull::from(&key));
                    let barrier_ref = ValidPtr(NonNull::from(&barrier));
                    entry.insert_hashed_nocheck(hash, key_ref, barrier_ref);
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
        match writing.get_raw_entry_mut(hash, &key) {
            hash_map::RawEntryMut::Occupied(e) => {
                let b = e.remove();
                debug_assert!(core::ptr::eq(b.0.as_ptr(), &barrier));
            }
            hash_map::RawEntryMut::Vacant(_) => debug_assert!(false),
        }

        // We have just done the cleanup manually
        core::mem::forget(guard);

        // We can finally insert the value in the map.
        match map.get_raw_entry_mut(hash, &key) {
            hash_map::RawEntryMut::Vacant(entry) => {
                entry.insert_hashed_nocheck(hash, key, value);
            }
            hash_map::RawEntryMut::Occupied(_) => panic!("re-entrant init"),
        }
        Ok(ret)

        // Leaving the function will wake up waiting threads.
    }

    pub fn contains_key<Q>(&self, hash: u64, key: &Q) -> bool
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        self.map.read().get_raw_entry(hash, key).is_some()
    }

    pub fn remove_entry<Q>(&mut self, hash: u64, key: &Q) -> Option<(K, V)>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        match self.map.get_mut().get_raw_entry_mut(hash, key) {
            hash_map::RawEntryMut::Occupied(entry) => Some(entry.remove_entry()),
            hash_map::RawEntryMut::Vacant(_) => None,
        }
    }
}

pub struct OnceMap<K, V, S = hash_map::DefaultHashBuilder> {
    shards: Box<[Shard<K, V, S>]>,
    hash_builder: S,
}

impl<K, V> OnceMap<K, V> {
    pub fn new() -> Self {
        Self::with_hasher(hash_map::DefaultHashBuilder::new())
    }

    #[cfg(test)]
    pub(crate) fn with_single_shard() -> Self {
        let hash_builder = hash_map::DefaultHashBuilder::new();
        let shards = Box::new([Shard::new(hash_builder.clone())]);
        Self {
            shards,
            hash_builder,
        }
    }
}

impl<K, V, S> OnceMap<K, V, S>
where
    S: Clone,
{
    pub fn with_hasher(hash_builder: S) -> Self {
        let shards = (0..32).map(|_| Shard::new(hash_builder.clone())).collect();
        Self {
            shards,
            hash_builder,
        }
    }

    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.map.read().len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.shards.iter().all(|s| s.map.read().is_empty())
    }

    pub fn clear(&mut self) {
        self.shards.iter_mut().for_each(|s| s.map.get_mut().clear());
    }

    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }
}

impl<K, V, S> OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn hash_one<Q>(&self, key: &Q) -> u64
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        crate::hash_one(&self.hash_builder, key)
    }

    fn get_shard(&self, hash: u64) -> &Shard<K, V, S> {
        let len = self.shards.len();
        &self.shards[(len - 1) & (hash as usize)]
    }

    fn get_shard_mut(&mut self, hash: u64) -> &mut Shard<K, V, S> {
        let len = self.shards.len();
        &mut self.shards[(len - 1) & (hash as usize)]
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        let hash = self.hash_one(key);
        self.get_shard(hash).contains_key(hash, key)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        let hash = self.hash_one(key);
        let (_, v) = self.get_shard_mut(hash).remove_entry(hash, key)?;
        Some(v)
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
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
    pub fn get<Q>(&self, key: &Q) -> Option<&V::Target>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        self.map_get(key, |_, v| unsafe { extend_lifetime(v) })
    }

    pub fn insert_with(&self, key: K, make_val: impl FnOnce(&K) -> V) -> &V::Target {
        self.map_insert_with(key, make_val, |_, v| unsafe { extend_lifetime(v) })
    }

    pub fn try_insert_with<E>(
        &self,
        key: K,
        make_val: impl FnOnce(&K) -> Result<V, E>,
    ) -> Result<&V::Target, E> {
        self.map_try_insert_with(key, make_val, |_, v| unsafe { extend_lifetime(v) })
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
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        self.map_get(key, |_, v| v.clone())
    }

    pub fn insert_cloned(&self, key: K, value: V) -> V {
        self.map_insert(key, value, |_, v| v.clone())
    }

    pub fn insert_with_cloned(&self, key: K, make_val: impl FnOnce(&K) -> V) -> V {
        self.map_insert_with(key, make_val, |_, v| v.clone())
    }

    pub fn try_insert_with_cloned<E>(
        &self,
        key: K,
        make_val: impl FnOnce(&K) -> Result<V, E>,
    ) -> Result<V, E> {
        self.map_try_insert_with(key, make_val, |_, v| v.clone())
    }
}

impl<K, V, S> OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn map_get<Q, T>(&self, key: &Q, with_result: impl FnOnce(&K, &V) -> T) -> Option<T>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        let hash = self.hash_one(key);
        self.get_shard(hash).get(hash, key, with_result)
    }

    pub fn map_insert<T>(&self, key: K, value: V, with_result: impl FnOnce(&K, &V) -> T) -> T {
        self.map_insert_with(key, |_| value, with_result)
    }

    pub fn map_insert_with<T>(
        &self,
        key: K,
        make_val: impl FnOnce(&K) -> V,
        with_result: impl FnOnce(&K, &V) -> T,
    ) -> T {
        self.map_try_insert_with(key, |k| Ok(make_val(k)), with_result)
            .unwrap_infallible()
    }

    pub fn map_insert_with_ref<Q, T>(
        &self,
        key: &Q,
        make_key: impl FnOnce(&Q) -> K,
        make_val: impl FnOnce(&K) -> V,
        with_result: impl FnOnce(&K, &V) -> T,
    ) -> T
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        self.map_try_insert_with_ref(key, make_key, |k| Ok(make_val(k)), with_result)
            .unwrap_infallible()
    }

    pub fn map_try_insert_with<T, E>(
        &self,
        key: K,
        make_val: impl FnOnce(&K) -> Result<V, E>,
        with_result: impl FnOnce(&K, &V) -> T,
    ) -> Result<T, E> {
        self.get_or_try_insert_with(
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

    pub fn map_try_insert_with_ref<Q, T, E>(
        &self,
        key: &Q,
        make_key: impl FnOnce(&Q) -> K,
        make_val: impl FnOnce(&K) -> Result<V, E>,
        with_result: impl FnOnce(&K, &V) -> T,
    ) -> Result<T, E>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        self.get_or_try_insert_with_ref(
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

    pub fn get_or_insert_with<T, U>(
        &self,
        key: K,
        data: T,
        on_vacant: impl FnOnce(T, &K) -> (V, U),
        on_occupied: impl FnOnce(T, &K, &V) -> U,
    ) -> U {
        self.get_or_try_insert_with(key, data, |data, k| Ok(on_vacant(data, k)), on_occupied)
            .unwrap_infallible()
    }

    pub fn get_or_try_insert_with<T, U, E>(
        &self,
        key: K,
        data: T,
        on_vacant: impl FnOnce(T, &K) -> Result<(V, U), E>,
        on_occupied: impl FnOnce(T, &K, &V) -> U,
    ) -> Result<U, E> {
        let hash = self.hash_one(&key);
        self.get_shard(hash)
            .get_or_try_insert_with(hash, key, data, on_vacant, on_occupied)
    }

    pub fn get_or_try_insert_with_ref<Q, T, U, E>(
        &self,
        key: &Q,
        data: T,
        make_key: impl FnOnce(&Q) -> K,
        on_vacant: impl FnOnce(T, &K) -> Result<(V, U), E>,
        on_occupied: impl FnOnce(T, &K, &V) -> U,
    ) -> Result<U, E>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        let hash = self.hash_one(key);
        let shard = self.get_shard(hash);

        // Try to get the value from the map as a fast-path, to avoid having to
        // compute an owned version of the key.
        shard.try_get(hash, key, data, on_occupied).or_else(
            #[cold]
            |(data, on_occupied)| {
                shard.get_or_try_insert_with(hash, make_key(key), data, on_vacant, on_occupied)
            },
        )
    }
}

impl<K, V> Default for OnceMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V, S> fmt::Debug for OnceMap<K, V, S>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut map = f.debug_map();

        for shard in &*self.shards {
            map.entries(&*shard.map.read());
        }

        map.finish()
    }
}

pub struct LazyMap<K, V, S = hash_map::DefaultHashBuilder, F = fn(&K) -> V> {
    map: OnceMap<K, V, S>,
    init: F,
}

impl<K, V, F> LazyMap<K, V, hash_map::DefaultHashBuilder, F> {
    pub fn new(f: F) -> Self {
        Self::with_hasher(hash_map::DefaultHashBuilder::new(), f)
    }
}

impl<K, V, S, F> LazyMap<K, V, S, F>
where
    S: Clone,
{
    pub fn with_hasher(hash_builder: S, f: F) -> Self {
        Self {
            map: OnceMap::with_hasher(hash_builder),
            init: f,
        }
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
        Q: Eq + Hash + ToOwned<Owned = K>,
        K: Borrow<Q>,
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
        Q: Eq + Hash + ToOwned<Owned = K>,
        K: Borrow<Q>,
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
        Q: Eq + Hash + ToOwned<Owned = K>,
        K: Borrow<Q>,
    {
        self.map
            .map_insert_with_ref(key, Q::to_owned, &self.init, with_result)
    }
}
