use crate::{HashMapExt, InfallibleResult};
use core::{
    borrow::Borrow,
    cell::RefCell,
    fmt,
    hash::{BuildHasher, Hash},
};
use alloc::borrow::ToOwned;
use hashbrown::{hash_map, HashMap};
use stable_deref_trait::StableDeref;

unsafe fn extend_lifetime<'a, T: StableDeref>(ptr: &T) -> &'a T::Target {
    &*(&**ptr as *const T::Target)
}

pub struct OnceMap<K, V, S = hash_map::DefaultHashBuilder> {
    map: RefCell<HashMap<K, V, S>>,
}

impl<K, V> OnceMap<K, V> {
    pub fn new() -> Self {
        Self::with_hasher(hash_map::DefaultHashBuilder::new())
    }
}

impl<K, V, S> OnceMap<K, V, S>
where
    S: Clone,
{
    pub fn with_hasher(hash_builder: S) -> Self {
        let map = RefCell::new(HashMap::with_hasher(hash_builder));
        Self { map }
    }

    pub fn len(&self) -> usize {
        self.map.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.borrow().is_empty()
    }

    pub fn clear(&mut self) {
        self.map.get_mut().clear();
    }
}

impl<K, V, S> OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        self.map.borrow().contains_key(key)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        self.map.get_mut().remove(key)
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        self.map.get_mut().remove_entry(key)
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

    pub fn insert(&self, key: K, make_val: impl FnOnce(&K) -> V) -> &V::Target {
        self.map_insert(key, make_val, |_, v| unsafe { extend_lifetime(v) })
    }

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
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
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
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        let map = self.map.borrow();
        let (key, value) = map.get_key_value(key)?;
        Some(with_result(key, value))
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
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
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
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
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
        let map = self.map.borrow();
        let hash = crate::hash_one(map.hasher(), &key);

        if let Some((key, value)) = map.get_raw_entry(hash, &key) {
            return Ok(on_occupied(data, key, value));
        }
        drop(map);

        // We must not borrow `self.map` here
        let (value, ret) = on_vacant(data, &key)?;

        self.raw_insert(hash, key, value);
        Ok(ret)
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
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
    {
        let map = self.map.borrow();
        let hash = crate::hash_one(map.hasher(), key);

        if let Some((key, value)) = map.get_raw_entry(hash, key) {
            return Ok(on_occupied(data, key, value));
        }
        drop(map);

        // We must not borrow `self.map` here
        let key = make_key(key);
        let (value, ret) = on_vacant(data, &key)?;

        self.raw_insert(hash, key, value);
        Ok(ret)
    }

    fn raw_insert(&self, hash: u64, key: K, value: V) {
        let mut map = self.map.borrow_mut();
        match map.get_raw_entry_mut::<K>(hash, &key) {
            hash_map::RawEntryMut::Vacant(entry) => {
                entry.insert_hashed_nocheck(hash, key, value);
            }
            hash_map::RawEntryMut::Occupied(_) => panic!("re-entrant init"),
        }
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
        self.map.borrow().fmt(f)
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
            .map_insert_ref(key, Q::to_owned, &self.init, with_result)
    }
}
