use crate::{map, map::HashMap, Equivalent, InfallibleResult, ToOwnedEquivalent};
use core::{
    cell::RefCell,
    fmt,
    hash::{BuildHasher, Hash},
};
use stable_deref_trait::StableDeref;

unsafe fn extend_lifetime<'a, T: StableDeref>(ptr: &T) -> &'a T::Target {
    &*(&**ptr as *const T::Target)
}

pub struct OnceMap<K, V, S = crate::RandomState> {
    map: RefCell<HashMap<K, V>>,
    hash_builder: S,
}

impl<K, V> OnceMap<K, V> {
    /// Creates an empty `OnceMap`.
    pub fn new() -> Self {
        Self::with_hasher(crate::RandomState::new())
    }
}

impl<K, V, S> OnceMap<K, V, S> {
    /// Creates an empty `OnceMap` which will use the given hash builder to hash keys.
    pub const fn with_hasher(hash_builder: S) -> Self {
        let map = RefCell::new(HashMap::new());
        Self { map, hash_builder }
    }

    pub fn len(&self) -> usize {
        self.map.borrow().len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.borrow().is_empty()
    }

    /// Returns a reference to the map's [`BuildHasher`].
    pub fn hasher(&self) -> &S {
        &self.hash_builder
    }

    /// Locks the whole map for reading.
    ///
    /// This enables more methods, such as iterating on the maps, but will cause
    /// a panic if trying to insert values in the map while the view is live.
    pub fn read_only_view(&self) -> ReadOnlyView<K, V, S> {
        ReadOnlyView::new(self)
    }

    /// Removes all key-value pairs from the map, but keeps the allocated memory.
    pub fn clear(&mut self) {
        self.map.get_mut().clear();
    }

    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.map.get_mut().values_mut()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.map.get_mut().iter_mut()
    }

    #[allow(clippy::should_implement_trait)]
    pub fn into_iter(self) -> impl Iterator<Item = (K, V)> {
        self.map.into_inner().into_iter()
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
        self.map.into_inner().into_par_iter()
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

    /// Returns `true` if the map contains a value for the specified key.
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.map.borrow().contains_key(hash, key)
    }

    pub fn remove<Q>(&mut self, key: &Q) -> Option<V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.map.get_mut().remove(hash, key)
    }

    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.map.get_mut().remove_entry(hash, key)
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
        let map = self.map.borrow();
        let hash = self.hash_one(key);
        let (key, value) = map.get_key_value(hash, key)?;
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
        let map = self.map.borrow();
        let hash = self.hash_one(&key);

        if let Some((key, value)) = map.get_key_value(hash, &key) {
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
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let map = self.map.borrow();
        let hash = self.hash_one(key);

        if let Some((key, value)) = map.get_key_value(hash, key) {
            return Ok(on_occupied(data, key, value));
        }
        drop(map);

        // We must not borrow `self.map` here
        let owned_key = make_key(key);
        debug_assert!(key.equivalent(&owned_key));
        let (value, ret) = on_vacant(data, &owned_key)?;

        self.raw_insert(hash, owned_key, value);
        Ok(ret)
    }

    fn raw_insert(&self, hash: u64, key: K, value: V) {
        let mut map = self.map.borrow_mut();
        match map.entry(hash, &key, &self.hash_builder) {
            map::Entry::Vacant(entry) => {
                entry.insert(key, value);
            }
            map::Entry::Occupied(_) => panic!("re-entrant init"),
        }
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
        self.map.borrow().fmt(f)
    }
}

pub struct ReadOnlyView<'a, K, V, S = crate::RandomState> {
    map: core::cell::Ref<'a, HashMap<K, V>>,
    hash_builder: &'a S,
}

impl<'a, K, V, S> ReadOnlyView<'a, K, V, S> {
    fn new(map: &'a OnceMap<K, V, S>) -> Self {
        Self {
            map: map.map.borrow(),
            hash_builder: &map.hash_builder,
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    pub fn hasher(&self) -> &S {
        self.hash_builder
    }

    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.map.iter()
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.map.keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.map.values()
    }
}

impl<'a, K, V, S> ReadOnlyView<'a, K, V, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    fn hash_one<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        crate::hash_one(self.hash_builder, key)
    }

    pub fn get<Q>(&self, key: &Q) -> Option<&V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.map.get(hash, key)
    }

    pub fn get_key_value<Q>(&self, key: &Q) -> Option<(&K, &V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.map.get_key_value(hash, key)
    }

    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        let hash = self.hash_one(key);
        self.map.contains_key(hash, key)
    }
}

#[cfg(feature = "rayon")]
#[cfg_attr(docsrs, doc(cfg(feature = "rayon")))]
impl<'a, K, V, S> ReadOnlyView<'a, K, V, S>
where
    K: Sync,
    V: Sync,
{
    pub fn par_iter(&self) -> impl rayon::iter::ParallelIterator<Item = (&K, &V)> {
        self.map.par_iter()
    }

    pub fn par_keys(&self) -> impl rayon::iter::ParallelIterator<Item = &K> {
        self.map.par_keys()
    }

    pub fn par_values(&self) -> impl rayon::iter::ParallelIterator<Item = &V> {
        self.map.par_values()
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

impl<'a, K, V, S> fmt::Debug for ReadOnlyView<'a, K, V, S>
where
    K: fmt::Debug,
    V: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadOnlyView")
            .field("map", &self.map)
            .finish()
    }
}

/// A map where values are automatically filled at access.
///
/// This type has less overhead than [`crate::sync::LazyMap`] but it cannot be
/// shared across threads.
///
/// ```
/// let map = once_map::unsync::LazyMap::new(|x: &i32| x.to_string());
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
    pub const fn with_hasher(hash_builder: S, f: F) -> Self {
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

/// Creates a `LazyMap` that fills all values with `V::default()`.
impl<K, V: Default, S: Default> Default for LazyMap<K, V, S> {
    fn default() -> Self {
        Self::with_hasher(S::default(), |_| V::default())
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
