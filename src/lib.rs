#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(any(feature = "std", feature = "spin"))]
pub mod sync;

#[cfg(any(feature = "std", feature = "spin"))]
pub use sync::OnceMap;

#[cfg(test)]
mod tests;
