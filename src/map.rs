use crate::Equivalent;
use core::{fmt, hash::Hash, marker::PhantomData};
use hashbrown::raw::{Bucket, RawIter, RawTable};

#[inline]
fn equivalent<Q, K, V>(key: &Q) -> impl Fn(&(K, V)) -> bool + '_
where
    Q: Hash + Equivalent<K> + ?Sized,
{
    |(k, _)| key.equivalent(k)
}

/// This is just like std's `HashMap`, but it does not store its `BuildHasher`,
/// so it has to be provided (or a hash) for each operation.
pub struct HashMap<K, V>(RawTable<(K, V)>);

impl<K, V> HashMap<K, V> {
    #[inline]
    pub const fn new() -> Self {
        Self(RawTable::new())
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
    pub fn iter(&self) -> Iter<K, V> {
        Iter {
            iter: unsafe { self.0.iter() },
            _lt: PhantomData,
        }
    }

    #[inline]
    pub fn keys(&self) -> Keys<K, V> {
        Keys {
            iter: unsafe { self.0.iter() },
            _lt: PhantomData,
        }
    }

    #[inline]
    pub fn values(&self) -> Values<K, V> {
        Values {
            iter: unsafe { self.0.iter() },
            _lt: PhantomData,
        }
    }

    #[inline]
    pub fn clear(&mut self) {
        self.0.clear();
    }
}

impl<K, V> HashMap<K, V>
where
    K: Eq + Hash,
{
    #[inline]
    pub fn get<Q>(&self, hash: u64, k: &Q) -> Option<&V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        match self.0.get(hash, equivalent(k)) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    #[inline]
    pub fn get_key_value<Q>(&self, hash: u64, k: &Q) -> Option<(&K, &V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        match self.0.get(hash, equivalent(k)) {
            Some((k, v)) => Some((k, v)),
            None => None,
        }
    }

    #[inline]
    pub fn contains_key<Q>(&self, hash: u64, k: &Q) -> bool
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.0.get(hash, equivalent(k)).is_some()
    }

    #[inline]
    pub fn entry<Q>(&mut self, hash: u64, k: &Q) -> Entry<K, V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        match self.0.find(hash, equivalent(k)) {
            Some(bucket) => Entry::Occupied(OccupiedEntry { map: self, bucket }),
            None => Entry::Vacant(VacantEntry { map: self, hash }),
        }
    }

    #[inline]
    pub fn remove<Q>(&mut self, hash: u64, k: &Q) -> Option<V>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        match self.0.remove_entry(hash, equivalent(k)) {
            Some((_, v)) => Some(v),
            None => None,
        }
    }

    #[inline]
    pub fn remove_entry<Q>(&mut self, hash: u64, k: &Q) -> Option<(K, V)>
    where
        Q: Hash + Equivalent<K> + ?Sized,
    {
        self.0.remove_entry(hash, equivalent(k))
    }
}

impl<'a, K, V> IntoIterator for &'a HashMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
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

pub struct OccupiedEntry<'a, K, V> {
    map: &'a mut HashMap<K, V>,
    bucket: Bucket<(K, V)>,
}

impl<'a, K, V> OccupiedEntry<'a, K, V> {
    #[inline]
    pub fn get(&self) -> &V {
        unsafe { &self.bucket.as_ref().1 }
    }

    #[inline]
    pub fn remove(self) -> V {
        unsafe { self.map.0.remove(self.bucket).1 }
    }

    #[inline]

    pub fn remove_entry(self) -> (K, V) {
        unsafe { self.map.0.remove(self.bucket) }
    }
}

pub struct VacantEntry<'a, K, V> {
    map: &'a mut HashMap<K, V>,
    hash: u64,
}

impl<'a, K, V> VacantEntry<'a, K, V>
where
    K: Hash,
{
    #[inline]
    pub fn insert<S>(self, key: K, value: V, hasher: &S) -> &'a mut (K, V)
    where
        S: std::hash::BuildHasher,
    {
        let hash_one = |(k, _): &(K, V)| crate::hash_one(hasher, k);
        self.map.0.insert_entry(self.hash, (key, value), hash_one)
    }
}

pub struct Iter<'a, K, V> {
    _lt: PhantomData<&'a HashMap<K, V>>,
    iter: RawIter<(K, V)>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (k, v) = unsafe { self.iter.next()?.as_ref() };
        Some((k, v))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub struct Keys<'a, K, V> {
    _lt: PhantomData<&'a HashMap<K, V>>,
    iter: RawIter<(K, V)>,
}

impl<'a, K, V> Iterator for Keys<'a, K, V> {
    type Item = &'a K;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (k, _) = unsafe { self.iter.next()?.as_ref() };
        Some(k)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub struct Values<'a, K, V> {
    _lt: PhantomData<&'a HashMap<K, V>>,
    iter: RawIter<(K, V)>,
}

impl<'a, K, V> Iterator for Values<'a, K, V> {
    type Item = &'a V;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let (_, v) = unsafe { self.iter.next()?.as_ref() };
        Some(v)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}
