#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[cfg(feature = "std")]
pub mod sync;

#[cfg(feature = "std")]
pub use sync::OnceMap;

#[cfg(test)]
mod tests;
