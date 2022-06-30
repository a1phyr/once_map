use crate::{HashMapExt, InfallibleResult};
use core::{
    borrow::Borrow,
    cell::RefCell,
    fmt,
    hash::{BuildHasher, Hash},
};
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

    pub fn insert_with<M>(&self, key: K, make_val: M) -> &V::Target
    where
        M: FnOnce(&K) -> V,
    {
        self.map_insert_with(key, make_val, |_, v| unsafe { extend_lifetime(v) })
    }

    pub fn try_insert_with<M, E>(&self, key: K, make_val: M) -> Result<&V::Target, E>
    where
        M: FnOnce(&K) -> Result<V, E>,
    {
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

    pub fn insert_with_cloned<M>(&self, key: K, make_val: M) -> V
    where
        M: FnOnce(&K) -> V,
    {
        self.map_insert_with(key, make_val, |_, v| v.clone())
    }

    pub fn try_insert_with_cloned<M, E>(&self, key: K, make_val: M) -> Result<V, E>
    where
        M: FnOnce(&K) -> Result<V, E>,
    {
        self.map_try_insert_with(key, make_val, |_, v| v.clone())
    }
}

impl<K, V, S> OnceMap<K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn map_get<Q, F, T>(&self, key: &Q, with_result: F) -> Option<T>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
        F: FnOnce(&K, &V) -> T,
    {
        let map = self.map.borrow();
        let (key, value) = map.get_key_value(key)?;
        Some(with_result(key, value))
    }

    pub fn map_insert<F, T>(&self, key: K, value: V, with_result: F) -> T
    where
        F: FnOnce(&K, &V) -> T,
    {
        self.map_insert_with(key, |_| value, with_result)
    }

    pub fn map_insert_with<M, F, T>(&self, key: K, make_val: M, with_result: F) -> T
    where
        M: FnOnce(&K) -> V,
        F: FnOnce(&K, &V) -> T,
    {
        self.map_try_insert_with(key, |k| Ok(make_val(k)), with_result)
            .unwrap_infallible()
    }

    pub fn map_insert_with_ref<Q, MK, M, F, T>(
        &self,
        key: &Q,
        make_key: MK,
        make_val: M,
        with_result: F,
    ) -> T
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
        MK: FnOnce(&Q) -> K,
        M: FnOnce(&K) -> V,
        F: FnOnce(&K, &V) -> T,
    {
        self.map_try_insert_with_ref(key, make_key, |k| Ok(make_val(k)), with_result)
            .unwrap_infallible()
    }

    pub fn map_try_insert_with<M, E, F, T>(
        &self,
        key: K,
        make_val: M,
        with_result: F,
    ) -> Result<T, E>
    where
        M: FnOnce(&K) -> Result<V, E>,
        F: FnOnce(&K, &V) -> T,
    {
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

    pub fn map_try_insert_with_ref<Q, MK, M, E, F, T>(
        &self,
        key: &Q,
        make_key: MK,
        make_val: M,
        with_result: F,
    ) -> Result<T, E>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
        MK: FnOnce(&Q) -> K,
        M: FnOnce(&K) -> Result<V, E>,
        F: FnOnce(&K, &V) -> T,
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

    pub fn get_or_insert_with<F, G, T, U>(&self, key: K, data: T, on_vacant: F, on_occupied: G) -> U
    where
        F: FnOnce(T, &K) -> (V, U),
        G: FnOnce(T, &K, &V) -> U,
    {
        self.get_or_try_insert_with(key, data, |data, k| Ok(on_vacant(data, k)), on_occupied)
            .unwrap_infallible()
    }

    pub fn get_or_try_insert_with<F, G, E, T, U>(
        &self,
        key: K,
        data: T,
        on_vacant: F,
        on_occupied: G,
    ) -> Result<U, E>
    where
        F: FnOnce(T, &K) -> Result<(V, U), E>,
        G: FnOnce(T, &K, &V) -> U,
    {
        let map = self.map.borrow();
        let hash = crate::hash_one(map.hasher(), &key);

        if let Some((key, value)) = map.get_raw_entry(hash, &key) {
            return Ok(on_occupied(data, key, value));
        }
        drop(map);

        // We must not borrow `self.map` here
        let (value, ret) = on_vacant(data, &key)?;

        self.insert(hash, key, value);
        Ok(ret)
    }

    pub fn get_or_try_insert_with_ref<Q, F, G, MK, E, T, U>(
        &self,
        key: &Q,
        data: T,
        make_key: MK,
        on_vacant: F,
        on_occupied: G,
    ) -> Result<U, E>
    where
        Q: Eq + Hash + ?Sized,
        K: Borrow<Q>,
        MK: FnOnce(&Q) -> K,
        F: FnOnce(T, &K) -> Result<(V, U), E>,
        G: FnOnce(T, &K, &V) -> U,
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

        self.insert(hash, key, value);
        Ok(ret)
    }

    fn insert(&self, hash: u64, key: K, value: V) {
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
    pub fn map_get<Q, G, T>(&self, key: &Q, with_result: G) -> T
    where
        Q: Eq + Hash + ToOwned<Owned = K>,
        K: Borrow<Q>,
        G: FnOnce(&K, &V) -> T,
    {
        self.map
            .map_insert_with_ref(key, Q::to_owned, &self.init, with_result)
    }
}
