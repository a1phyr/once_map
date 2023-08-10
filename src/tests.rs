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

    crossbeam_utils::thread::scope(|s| {
        s.spawn(|_| {
            thread::sleep(time::Duration::from_millis(50));
            store.insert(String::from("aaa"), |_| {
                thread::sleep(time::Duration::from_millis(50));
                *count.lock() += 1;
                String::from("bbb")
            })
        });

        s.spawn(|_| {
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
    })
    .unwrap();

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
