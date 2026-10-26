//! Content hashing helpers. Hashes identify content; they do not by themselves
//! prove that a computation can be replayed.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use serde::Serialize;
use sha2::{Digest, Sha256};

fn to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Lowercase hex SHA-256 of a byte slice.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    to_hex(digest.as_slice())
}

/// Streams a file and returns `(sha256_hex, byte_len)`.
pub fn sha256_file(path: impl AsRef<Path>) -> io::Result<(String, u64)> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut total = 0u64;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        total += n as u64;
        hasher.update(&buf[..n]);
    }
    Ok((to_hex(hasher.finalize().as_slice()), total))
}

/// SHA-256 of the canonical JSON form of a value (object keys sorted).
pub fn canonical_hash<T: Serialize + ?Sized>(value: &T) -> String {
    // serde_json::Value uses a BTreeMap for objects (the `preserve_order` feature is
    // not enabled anywhere in the workspace), so keys come out sorted.
    let v = serde_json::to_value(value).expect("contract types always serialize");
    let bytes = serde_json::to_vec(&v).expect("json value always serializes");
    sha256_hex(&bytes)
}

/// Short, human-friendly prefix of a hash.
pub fn short(hash: &str) -> &str {
    &hash[..hash.len().min(12)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn canonical_hash_ignores_insertion_order() {
        let a = serde_json::json!({"b": 1, "a": [1, 2]});
        let b = serde_json::json!({"a": [1, 2], "b": 1});
        assert_eq!(canonical_hash(&a), canonical_hash(&b));
    }
}
