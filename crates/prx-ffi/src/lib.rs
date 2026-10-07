//! C ABI over `prx-core` (PLAN 5.1). Plain C structs with borrowed, NUL-terminated UTF-8 strings that live as long
//! as the `PrxTranscript` that owns them. Every function is safe to call from any thread; a `PrxTranscriber` may be
//! shared across threads for concurrent transcription (it is immutable after load). Errors: functions that return a
//! pointer return NULL and set a thread-local message readable with `prx_last_error`.
#![allow(clippy::missing_safety_doc)]
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::ptr;

use prx_core::longform::LongForm;
use prx_core::transcriber::{Segment, Transcriber};

/// Opaque handle to a loaded model (`prx_new` / `prx_free`).
pub struct PrxTranscriber(Transcriber);

/// Opaque streaming session (`prx_stream_new` / `prx_stream_free`); must not outlive its transcriber.
pub struct PrxStream {
    inner: LongForm<'static>,
}

/// A word with its time span in seconds (from TDT emission frames, 80 ms grid).
#[repr(C)]
pub struct PrxWord {
    pub start: f64,
    pub end: f64,
    /// NUL-terminated UTF-8, owned by the transcript.
    pub word: *const c_char,
}

/// A segment (one VAD-cut piece of long-form audio, or the whole clip for short audio).
#[repr(C)]
pub struct PrxSegment {
    pub start: f64,
    pub end: f64,
    /// NUL-terminated UTF-8, owned by the transcript.
    pub text: *const c_char,
    pub words: *const PrxWord,
    pub n_words: usize,
}

/// Result of a transcription. Free with `prx_transcript_free`; all pointers inside are owned by it.
#[repr(C)]
pub struct PrxTranscript {
    /// Full text, NUL-terminated UTF-8.
    pub text: *const c_char,
    pub segments: *const PrxSegment,
    pub n_segments: usize,
    /// Audio duration in seconds.
    pub duration: f64,
    // private storage keeping the pointers above alive
    _storage: *mut TranscriptStorage,
}

struct TranscriptStorage {
    _text: CString,
    _strings: Vec<CString>,
    _words: Vec<PrxWord>,
    _segments: Vec<PrxSegment>,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(msg: impl std::fmt::Display) {
    let s = CString::new(msg.to_string().replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = s);
}

/// The last error message on this thread (empty string if none). Valid until the next failing call on this thread.
#[no_mangle]
pub extern "C" fn prx_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Library version, e.g. "0.1.0".
#[no_mangle]
pub extern "C" fn prx_version() -> *const c_char {
    static V: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    V.as_ptr() as *const c_char
}

/// Load a model directory (the `moondream/parakeet-redux` repo layout). Returns NULL on error.
#[no_mangle]
pub unsafe extern "C" fn prx_new(model_dir: *const c_char) -> *mut PrxTranscriber {
    if model_dir.is_null() {
        set_error("model_dir is NULL");
        return ptr::null_mut();
    }
    let dir = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error("model_dir is not UTF-8");
            return ptr::null_mut();
        }
    };
    match std::panic::catch_unwind(|| Transcriber::load(dir)) {
        Ok(Ok(t)) => Box::into_raw(Box::new(PrxTranscriber(t))),
        Ok(Err(e)) => {
            set_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic while loading the model");
            ptr::null_mut()
        }
    }
}

/// Free a transcriber. All streams created from it must be freed first.
#[no_mangle]
pub unsafe extern "C" fn prx_free(t: *mut PrxTranscriber) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

fn build_transcript(text: String, segments: Vec<Segment>, duration: f64) -> *mut PrxTranscript {
    let mut strings = Vec::new();
    let mut words_all: Vec<PrxWord> = Vec::new();
    // first pass: count words so the word vector never reallocates after we take pointers into it
    let n_words: usize = segments.iter().map(|s| s.words.len()).sum();
    words_all.reserve_exact(n_words);
    let mut segs: Vec<PrxSegment> = Vec::with_capacity(segments.len());
    let cstr = |s: &str| CString::new(s.replace('\0', " ")).unwrap_or_default();
    for s in &segments {
        let w0 = words_all.len();
        for w in &s.words {
            let c = cstr(&w.word);
            words_all.push(PrxWord {
                start: w.start,
                end: w.end,
                word: c.as_ptr(),
            });
            strings.push(c);
        }
        let c = cstr(&s.text);
        segs.push(PrxSegment {
            start: s.start,
            end: s.end,
            text: c.as_ptr(),
            words: ptr::null(),
            n_words: s.words.len(),
        });
        strings.push(c);
        let _ = w0;
    }
    // second pass: word pointers (the vector is final now)
    let mut off = 0;
    for (seg, s) in segs.iter_mut().zip(&segments) {
        seg.words = if s.words.is_empty() {
            ptr::null()
        } else {
            unsafe { words_all.as_ptr().add(off) }
        };
        off += s.words.len();
    }
    let text = cstr(&text);
    let storage = Box::new(TranscriptStorage {
        _text: text,
        _strings: strings,
        _words: words_all,
        _segments: segs,
    });
    let out = PrxTranscript {
        text: storage._text.as_ptr(),
        segments: storage._segments.as_ptr(),
        n_segments: storage._segments.len(),
        duration,
        _storage: Box::into_raw(storage),
    };
    Box::into_raw(Box::new(out))
}

/// Transcribe PCM samples (mono f32, any sample rate; resampled to 16 kHz). Long audio is VAD-segmented
/// automatically. Returns NULL on error.
#[no_mangle]
pub unsafe extern "C" fn prx_transcribe(
    t: *const PrxTranscriber,
    samples: *const f32,
    n_samples: usize,
    sample_rate: u32,
) -> *mut PrxTranscript {
    if t.is_null() || (samples.is_null() && n_samples > 0) {
        set_error("NULL argument");
        return ptr::null_mut();
    }
    let tr = &(*t).0;
    let x = std::slice::from_raw_parts(samples, n_samples);
    let owned;
    let x: &[f32] = if sample_rate == 16_000 {
        x
    } else {
        owned = prx_core::audio::resample(x, sample_rate, 16_000);
        &owned
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| tr.transcribe_long(x))) {
        Ok(r) => build_transcript(r.text, r.segments, x.len() as f64 / 16_000.0),
        Err(_) => {
            set_error("panic during transcription");
            ptr::null_mut()
        }
    }
}

/// Transcribe a WAV file (PCM 8/16/24/32-bit or float, any rate, any channel count). Returns NULL on error.
#[no_mangle]
pub unsafe extern "C" fn prx_transcribe_file(
    t: *const PrxTranscriber,
    path: *const c_char,
) -> *mut PrxTranscript {
    if t.is_null() || path.is_null() {
        set_error("NULL argument");
        return ptr::null_mut();
    }
    let p = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error("path is not UTF-8");
            return ptr::null_mut();
        }
    };
    let audio = match prx_core::audio::read_wav(p) {
        Ok(a) => a,
        Err(e) => {
            set_error(e);
            return ptr::null_mut();
        }
    };
    prx_transcribe(
        t,
        audio.samples.as_ptr(),
        audio.samples.len(),
        audio.sample_rate,
    )
}

/// Free a transcript.
#[no_mangle]
pub unsafe extern "C" fn prx_transcript_free(r: *mut PrxTranscript) {
    if r.is_null() {
        return;
    }
    let r = Box::from_raw(r);
    drop(Box::from_raw(r._storage));
}

/// Start a streaming session. Feed 16 kHz mono f32 with `prx_stream_push`; segments are returned as soon as the
/// following pause is known (offline and streaming produce the same cuts). The transcriber must outlive the stream.
#[no_mangle]
pub unsafe extern "C" fn prx_stream_new(t: *const PrxTranscriber) -> *mut PrxStream {
    if t.is_null() {
        set_error("NULL transcriber");
        return ptr::null_mut();
    }
    // SAFETY: the caller guarantees the transcriber outlives the stream (documented contract).
    let tr: &'static Transcriber = &*(&(*t).0 as *const Transcriber);
    Box::into_raw(Box::new(PrxStream {
        inner: LongForm::new(tr),
    }))
}

/// Push 16 kHz mono samples; returns the segments finalised by this call (possibly none: `n_segments == 0`).
#[no_mangle]
pub unsafe extern "C" fn prx_stream_push(
    s: *mut PrxStream,
    samples: *const f32,
    n_samples: usize,
) -> *mut PrxTranscript {
    if s.is_null() || (samples.is_null() && n_samples > 0) {
        set_error("NULL argument");
        return ptr::null_mut();
    }
    let st = &mut *s;
    let x = std::slice::from_raw_parts(samples, n_samples);
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| st.inner.push(x))) {
        Ok(segs) => {
            let text = segs
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            build_transcript(text, segs, n_samples as f64 / 16_000.0)
        }
        Err(_) => {
            set_error("panic during streaming push");
            ptr::null_mut()
        }
    }
}

/// End of stream: returns the remaining segments.
#[no_mangle]
pub unsafe extern "C" fn prx_stream_flush(s: *mut PrxStream) -> *mut PrxTranscript {
    if s.is_null() {
        set_error("NULL stream");
        return ptr::null_mut();
    }
    let st = &mut *s;
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| st.inner.flush())) {
        Ok(segs) => {
            let text = segs
                .iter()
                .map(|s| s.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            build_transcript(text, segs, 0.0)
        }
        Err(_) => {
            set_error("panic during streaming flush");
            ptr::null_mut()
        }
    }
}

/// Free a streaming session.
#[no_mangle]
pub unsafe extern "C" fn prx_stream_free(s: *mut PrxStream) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}
