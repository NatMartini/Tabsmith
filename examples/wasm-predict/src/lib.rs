//! A Tabsmith bundle predicting in a web page: `tabsmith-runtime` compiled to
//! WebAssembly, with the same verification and the same transform as everywhere else.
//!
//! The page passes the bundle as one byte array in the layout `fuzz/seed.py`
//! writes: one flags byte, then for every file
//! `[u16 LE name length][name][u32 LE data length][data]`. Hashes are always verified.

use std::collections::BTreeMap;

use tabsmith_runtime::{LoadOptions, Predictor};
use wasm_bindgen::prelude::*;

/// Splits a packed bundle into its files.
pub fn unpack(mut data: &[u8]) -> Result<BTreeMap<String, Vec<u8>>, String> {
    fn take<'a>(data: &mut &'a [u8], n: usize) -> Result<&'a [u8], String> {
        if data.len() < n {
            return Err("the packed bundle is truncated".into());
        }
        let (head, rest) = data.split_at(n);
        *data = rest;
        Ok(head)
    }
    take(&mut data, 1)?;
    let mut files = BTreeMap::new();
    while !data.is_empty() {
        let n = take(&mut data, 2)?;
        let name_len = u16::from_le_bytes([n[0], n[1]]) as usize;
        let name = String::from_utf8(take(&mut data, name_len)?.to_vec()).map_err(|_| "a file name is not UTF-8")?;
        let n = take(&mut data, 4)?;
        let len = u32::from_le_bytes([n[0], n[1], n[2], n[3]]) as usize;
        let bytes = take(&mut data, len)?.to_vec();
        if files.contains_key(&name) {
            return Err(format!("the packed bundle holds `{name}` twice"));
        }
        files.insert(name, bytes);
    }
    Ok(files)
}

fn js_error(e: tabsmith_runtime::TabsmithError) -> JsError {
    let next = e.next_action.map(|n| format!(" (next: {n})")).unwrap_or_default();
    JsError::new(&format!("[{}] {}{next}", e.code, e.message))
}

/// A verified bundle, ready to predict.
#[wasm_bindgen]
pub struct Bundle {
    predictor: Predictor,
}

#[wasm_bindgen]
impl Bundle {
    /// Loads a packed bundle and verifies it: hashes, versions, consistency, and
    /// the example predictions (self-check).
    #[wasm_bindgen(constructor)]
    pub fn new(packed: &[u8]) -> Result<Bundle, JsError> {
        let files = unpack(packed).map_err(|e| JsError::new(&e))?;
        let opts = LoadOptions::cpu().with_self_check(true);
        let predictor = Predictor::from_memory("bundle", files, opts).map_err(js_error)?;
        Ok(Bundle { predictor })
    }

    /// The manifest, input and output contracts, fitted transform, model,
    /// evidence and example records, as JSON.
    pub fn describe(&self) -> String {
        let b = self.predictor.bundle();
        serde_json::json!({
            "manifest": b.manifest,
            "input": b.input_schema,
            "output": b.output_schema,
            "transform": b.transform,
            "model": b.model.config(),
            "evaluation": b.evaluation,
            "examples": b.examples.records,
        })
        .to_string()
    }

    /// Predicts a JSON record, or an array of records; returns the predictions as JSON.
    pub fn predict(&self, records: &str) -> Result<String, JsError> {
        let value: serde_json::Value = serde_json::from_str(records).map_err(|e| JsError::new(&format!("not JSON: {e}")))?;
        let preds = self.predictor.predict_json(&value).map_err(js_error)?;
        serde_json::to_string(&preds).map_err(|e| JsError::new(&e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::unpack;

    fn pack(files: &[(&str, &[u8])]) -> Vec<u8> {
        let mut packed = vec![1u8];
        for (name, data) in files {
            packed.extend((name.len() as u16).to_le_bytes());
            packed.extend(name.as_bytes());
            packed.extend((data.len() as u32).to_le_bytes());
            packed.extend(*data);
        }
        packed
    }

    #[test]
    fn unpacks_the_seed_layout() {
        let packed = pack(&[("manifest.json", b"{}"), ("examples/input.json", b"[1]")]);
        let files = unpack(&packed).unwrap();
        assert_eq!(files["manifest.json"], b"{}");
        assert_eq!(files["examples/input.json"], b"[1]");
        assert!(unpack(&packed[..packed.len() - 1]).is_err());
    }

    #[test]
    fn a_file_named_twice_is_refused() {
        let packed = pack(&[("transform.json", b"{}"), ("transform.json", b"[]")]);
        assert!(unpack(&packed).unwrap_err().contains("twice"));
    }
}
