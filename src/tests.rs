use crate::*;
use core::cell::Cell;
use std::{thread, time};

#[test]
fn smoke_test() {
    let store = OnceMap::new();
    let val = store.insert(String::from("aaa"), |_| String::from("bbb"));
    assert_eq!(val, store.get("aaa").unwrap());
}

#[test]
#[cfg_attr(miri, ignore)]
fn concurrent_init() {
    let store = OnceMap::new();
    let count = parking_lot::Mutex::new(0);

    std::thread::scope(|s| {
        s.spawn(|| {
            thread::sleep(time::Duration::from_millis(50));
            store.insert(String::from("aaa"), |_| {
                thread::sleep(time::Duration::from_millis(50));
                *count.lock() += 1;
                String::from("bbb")
            })
        });

        s.spawn(|| {
            thread::sleep(time::Duration::from_millis(50));
            store.insert(String::from("aaa"), |_| {
                thread::sleep(time::Duration::from_millis(50));
                *count.lock() += 1;
                String::from("bbb")
            })
        });

        store
            .try_insert(String::from("aaa"), |_| {
                thread::sleep(time::Duration::from_millis(200));
                *count.lock() += 2;
                Err(())
            })
            .unwrap_err();
    });

    assert_eq!(*count.lock(), 3);
    assert_eq!(store.get("aaa").unwrap(), "bbb");
}

#[test]
fn reentrant_init() {
    let store = OnceMap::with_single_shard();

    let res = store.insert(String::from("aaa"), |_| {
        let x = store.insert_cloned(String::from("bbb"), |_| String::from("x"));
        let y = store.insert(String::from("ccc"), |_| String::from("y"));
        assert!(store.get("aaa").is_none());
        x + y
    });

    assert_eq!(res, "xy");
}

#[test]
fn panic_init() {
    let store = OnceMap::new();

    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        store.insert(0, |_| panic!())
    }));
    assert!(res.is_err());

    assert!(store.insert(0, |x| x.to_string()) == "0");
}

#[test]
fn lazy() {
    let init_count = Cell::new(0);
    let int_map = LazyMap::new(|n: &i32| {
        init_count.set(init_count.get() + 1);
        n.to_string()
    });

    assert_eq!(&int_map[&3], "3");
    assert_eq!(&int_map[&12], "12");
    assert_eq!(&int_map[&3], "3");
    assert_eq!(init_count.get(), 2)
}

#[cfg(feature = "rayon")]
#[test]
fn rayon() {
    use rayon::prelude::*;

    let map: OnceMap<_, _> = (0..1000)
        .into_par_iter()
        .map(|n| (n, n.to_string()))
        .collect();

    let view = map.read_only_view();

    assert_eq!(
        view.par_values().map(|s| s.len()).sum::<usize>(),
        view.values().map(|s| s.len()).sum()
    );
}

/// https://github.com/a1phyr/once_map/issues/3
#[test]
fn issue_3() {
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::Mutex,
    };

    #[derive(PartialEq, Eq, Debug)]
    struct H(u32);

    impl std::hash::Hash for H {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            if PANIC_ON.lock().unwrap().as_ref() == Some(&self.0) {
                panic!();
            }
            0_u32.hash(state);
        }
    }

    static PANIC_ON: Mutex<Option<u32>> = Mutex::new(None);

    let mut map = crate::unsync::OnceMap::new();
    for i in 1..=28 {
        map.insert(H(i), |k| {
            if *k == H(28) {
                String::from("Hello World!")
            } else {
                String::new()
            }
        });
    }
    for i in 1..=27 {
        map.remove(&H(i));
    }

    let hello_world = map.get(&H(28)).unwrap();

    assert!(hello_world == "Hello World!");

    let _ = catch_unwind(AssertUnwindSafe(|| {
        *PANIC_ON.lock().unwrap() = Some(28);
        map.insert(H(1), |_| String::new());
    }));

    assert!(hello_world == "Hello World!");
}
