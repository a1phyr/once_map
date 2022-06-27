use crate::*;
use std::{thread, time};

#[test]
fn smoke_test() {
    let store = OnceMap::new();
    let val = store.insert_with(String::from("aaa"), |_| String::from("bbb"));
    assert_eq!(val, store.get("aaa").unwrap());
}

#[test]
fn concurrent_init() {
    let store = OnceMap::new();
    let count = parking_lot::Mutex::new(0);

    crossbeam_utils::thread::scope(|s| {
        s.spawn(|_| {
            thread::sleep(time::Duration::from_millis(50));
            store.insert_with(String::from("aaa"), |_| {
                thread::sleep(time::Duration::from_millis(50));
                *count.lock() += 1;
                String::from("bbb")
            })
        });

        s.spawn(|_| {
            thread::sleep(time::Duration::from_millis(50));
            store.insert_with(String::from("aaa"), |_| {
                thread::sleep(time::Duration::from_millis(50));
                *count.lock() += 1;
                String::from("bbb")
            })
        });

        store
            .try_insert_with(String::from("aaa"), |_| {
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

    let res = store.insert_with(String::from("aaa"), |_| {
        let x = store.insert_with_cloned(String::from("bbb"), |_| String::from("x"));
        let y = store.insert_with(String::from("ccc"), |_| String::from("y"));
        assert!(store.get("aaa").is_none());
        x + y
    });

    assert_eq!(res, "xy");
}
