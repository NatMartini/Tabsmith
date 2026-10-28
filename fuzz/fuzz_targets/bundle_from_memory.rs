//! Loads arbitrary bytes as an in-memory bundle and, if that succeeds, runs the
//! bundle's self-check. Loading may fail; it must never panic, hang or abort.
//!
//! Input layout (so a real bundle can seed the corpus, see `fuzz/README.md`):
//! one flags byte (bit 0: verify hashes), then entries of
//! `[u16 LE name length][name][u32 LE data length][data]`.
#![no_main]

use std::collections::BTreeMap;

use libfuzzer_sys::fuzz_target;
use tabsmith_runtime::{LoadOptions, Predictor};

fn take<'a>(data: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
    if data.len() < n {
        return None;
    }
    let (head, rest) = data.split_at(n);
    *data = rest;
    Some(head)
}

fn parse(mut data: &[u8]) -> Option<(bool, BTreeMap<String, Vec<u8>>)> {
    let flags = take(&mut data, 1)?[0];
    let mut files = BTreeMap::new();
    while !data.is_empty() {
        let name_len = u16::from_le_bytes(take(&mut data, 2)?.try_into().ok()?) as usize;
        let name = String::from_utf8_lossy(take(&mut data, name_len)?).into_owned();
        let len = u32::from_le_bytes(take(&mut data, 4)?.try_into().ok()?) as usize;
        let available = len.min(data.len());
        let bytes = take(&mut data, available)?;
        files.insert(name, bytes.to_vec());
    }
    Some((flags & 1 == 1, files))
}

fuzz_target!(|data: &[u8]| {
    let Some((verify, files)) = parse(data) else {
        return;
    };
    let opts = LoadOptions::cpu().with_hash_verification(verify);
    if let Ok(predictor) = Predictor::from_memory("fuzz", files, opts) {
        let _ = predictor.self_check();
    }
});
