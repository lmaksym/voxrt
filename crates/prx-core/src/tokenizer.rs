//! Detokenizer for the model's `tokenizer.json` (HF tokenizers, BPE vocab + Metaspace decoder, byte fallback).
//! Mirrors `ParakeetTokenizer.decode(ids, skip_special_tokens=True)` (PLAN 2.11).
use std::path::Path;

use serde::Deserialize;

use crate::{Error, Result};

#[derive(Deserialize)]
struct AddedToken {
    id: usize,
    content: String,
    #[serde(default)]
    special: bool,
}
#[derive(Deserialize)]
struct Model {
    vocab: std::collections::HashMap<String, usize>,
}
#[derive(Deserialize)]
struct Decoder {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    replacement: Option<String>,
}
#[derive(Deserialize)]
struct TokenizerJson {
    added_tokens: Vec<AddedToken>,
    decoder: Option<Decoder>,
    model: Model,
}

pub struct Tokenizer {
    pieces: Vec<String>, // id → piece ("" for ids beyond the vocab, e.g. blank)
    special: Vec<bool>,
    replacement: char,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let j: TokenizerJson = serde_json::from_slice(&std::fs::read(path)?)?;
        let n = j.model.vocab.values().copied().max().map_or(0, |m| m + 1);
        let mut pieces = vec![String::new(); n];
        for (p, &i) in &j.model.vocab {
            pieces[i] = p.clone();
        }
        let mut special = vec![false; n];
        for a in &j.added_tokens {
            if a.id < n {
                if pieces[a.id].is_empty() {
                    pieces[a.id] = a.content.clone();
                }
                special[a.id] = a.special;
            }
        }
        let replacement = match j.decoder {
            Some(d) if d.kind == "Metaspace" => d
                .replacement
                .and_then(|r| r.chars().next())
                .unwrap_or('\u{2581}'),
            Some(d) => {
                return Err(Error::Format(format!(
                    "unsupported tokenizer decoder {}",
                    d.kind
                )))
            }
            None => '\u{2581}',
        };
        Ok(Self {
            pieces,
            special,
            replacement,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    pub fn piece(&self, id: usize) -> Option<&str> {
        self.pieces.get(id).map(String::as_str)
    }

    /// Decode ids to text: skip special tokens and out-of-vocab ids (blank), merge byte-fallback runs,
    /// `▁` → space with the leading space removed (Metaspace, prepend_scheme "always").
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();
        let mut bytes: Vec<u8> = Vec::new();
        let flush = |bytes: &mut Vec<u8>, out: &mut String| {
            if !bytes.is_empty() {
                out.push_str(&String::from_utf8_lossy(bytes));
                bytes.clear();
            }
        };
        for &id in ids {
            let Some(p) = self.pieces.get(id) else {
                continue;
            };
            if p.is_empty() || self.special[id] {
                continue;
            }
            if p.len() == 6 && p.starts_with("<0x") && p.ends_with('>') {
                if let Ok(b) = u8::from_str_radix(&p[3..5], 16) {
                    bytes.push(b);
                    continue;
                }
            }
            flush(&mut bytes, &mut out);
            out.push_str(p);
        }
        flush(&mut bytes, &mut out);
        let s: String = out
            .chars()
            .map(|c| if c == self.replacement { ' ' } else { c })
            .collect();
        s.strip_prefix(' ').map(str::to_string).unwrap_or(s)
    }
}
