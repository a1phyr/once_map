# `once_map`

[![Crates.io](https://img.shields.io/crates/v/once_map.svg)](https://crates.io/crates/once_map)
[![Docs.rs](https://docs.rs/once_map/badge.svg)](https://docs.rs/once_map/)
![Minimum rustc version](https://img.shields.io/badge/rustc-1.63+-lightgray.svg)

This crate provides `OnceMap`, a type of `HashMap` where entries can be written with a shared reference,
but can be written ony once. This is similar to [`once_cell`], but with a map.
This enables to reference values inside the map for the lifetime of the map, without the need
of further locks.

This makes this type perfect for implementation of caches. A type `LazyMap` is provided for such cases.

This crate provides such a map heavily optimized for concurrent use, but also a single-threaded version.

[`once_cell`]: https://docs.rs/once_cell

# Example

```rust
let map = OnceMap::new();

// All these are `&str` pointing directly in the map.
// Note that we don't need a mutable reference, so we can have several of
// them at the same time.
let roses = map.insert(String::from("rose"), |_| String::from("red"));
let violets = map.insert(String::from("violets"), |_| String::from("blue"));
let sugar = map.insert(String::from("sugar"), |_| String::from("sweet"));

assert_eq!(roses, "red");
assert_eq!(violets, "blue");
assert_eq!(sugar, "sweet");

// The closure is never run here, because we already have a value for "rose"
let roses = map.insert(String::from("rose"), |_| String::from("green"));
// The old value did not change
assert_eq!(roses, "red");
```
