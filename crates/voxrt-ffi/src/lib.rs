//! C ABI over `voxrt`. Plain C structs with borrowed, NUL-terminated UTF-8 strings that live as long
//! as the `VoxrtTranscript` that owns them. Every function is safe to call from any thread; a `VoxrtTranscriber` may be
//! shared across threads for concurrent transcription (it is immutable after load). Errors: functions that return a
//! pointer return NULL and set a thread-local message readable with `voxrt_last_error`.
#![allow(clippy::missing_safety_doc)]
use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::ptr;

use voxrt_core::longform::LongForm;
use voxrt_core::transcriber::{Segment, Transcriber};

/// Opaque handle to a loaded model (`voxrt_new` / `voxrt_free`).
pub struct VoxrtTranscriber(Transcriber);

/// Opaque streaming session (`voxrt_stream_new` / `voxrt_stream_free`); must not outlive its transcriber.
pub struct VoxrtStream {
    inner: LongForm<'static>,
}

/// A word with its time span in seconds (from TDT emission frames, 80 ms grid).
#[repr(C)]
pub struct VoxrtWord {
    pub start: f64,
    pub end: f64,
    /// NUL-terminated UTF-8, owned by the transcript.
    pub word: *const c_char,
}

/// A segment (one VAD-cut piece of long-form audio, or the whole clip for short audio).
#[repr(C)]
pub struct VoxrtSegment {
    pub start: f64,
    pub end: f64,
    /// NUL-terminated UTF-8, owned by the transcript.
    pub text: *const c_char,
    pub words: *const VoxrtWord,
    pub n_words: usize,
}

/// Result of a transcription. Free with `voxrt_transcript_free`; all pointers inside are owned by it.
#[repr(C)]
pub struct VoxrtTranscript {
    /// Full text, NUL-terminated UTF-8.
    pub text: *const c_char,
    pub segments: *const VoxrtSegment,
    pub n_segments: usize,
    /// Audio duration in seconds.
    pub duration: f64,
    // private storage keeping the pointers above alive
    _storage: *mut TranscriptStorage,
}

struct TranscriptStorage {
    _text: CString,
    _strings: Vec<CString>,
    _words: Vec<VoxrtWord>,
    _segments: Vec<VoxrtSegment>,
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
pub extern "C" fn voxrt_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Library version, e.g. "0.1.0".
#[no_mangle]
pub extern "C" fn voxrt_version() -> *const c_char {
    static V: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    V.as_ptr() as *const c_char
}

/// Load a model directory (the `moondream/parakeet-redux` repo layout). Returns NULL on error.
#[no_mangle]
pub unsafe extern "C" fn voxrt_new(model_dir: *const c_char) -> *mut VoxrtTranscriber {
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
        Ok(Ok(t)) => Box::into_raw(Box::new(VoxrtTranscriber(t))),
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
pub unsafe extern "C" fn voxrt_free(t: *mut VoxrtTranscriber) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

fn build_transcript(text: String, segments: Vec<Segment>, duration: f64) -> *mut VoxrtTranscript {
    let mut strings = Vec::new();
    let mut words_all: Vec<VoxrtWord> = Vec::new();
    // first pass: count words so the word vector never reallocates after we take pointers into it
    let n_words: usize = segments.iter().map(|s| s.words.len()).sum();
    words_all.reserve_exact(n_words);
    let mut segs: Vec<VoxrtSegment> = Vec::with_capacity(segments.len());
    let cstr = |s: &str| CString::new(s.replace('\0', " ")).unwrap_or_default();
    for s in &segments {
        let w0 = words_all.len();
        for w in &s.words {
            let c = cstr(&w.word);
            words_all.push(VoxrtWord {
                start: w.start,
                end: w.end,
                word: c.as_ptr(),
            });
            strings.push(c);
        }
        let c = cstr(&s.text);
        segs.push(VoxrtSegment {
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
    let out = VoxrtTranscript {
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
pub unsafe extern "C" fn voxrt_transcribe(
    t: *const VoxrtTranscriber,
    samples: *const f32,
    n_samples: usize,
    sample_rate: u32,
) -> *mut VoxrtTranscript {
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
        owned = voxrt_core::audio::resample(x, sample_rate, 16_000);
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
pub unsafe extern "C" fn voxrt_transcribe_file(
    t: *const VoxrtTranscriber,
    path: *const c_char,
) -> *mut VoxrtTranscript {
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
    let audio = match voxrt_core::audio::read_wav(p) {
        Ok(a) => a,
        Err(e) => {
            set_error(e);
            return ptr::null_mut();
        }
    };
    voxrt_transcribe(
        t,
        audio.samples.as_ptr(),
        audio.samples.len(),
        audio.sample_rate,
    )
}

/// Free a transcript.
#[no_mangle]
pub unsafe extern "C" fn voxrt_transcript_free(r: *mut VoxrtTranscript) {
    if r.is_null() {
        return;
    }
    let r = Box::from_raw(r);
    drop(Box::from_raw(r._storage));
}

/// Start a streaming session. Feed 16 kHz mono f32 with `voxrt_stream_push`; segments are returned as soon as the
/// following pause is known (offline and streaming produce the same cuts). The transcriber must outlive the stream.
#[no_mangle]
pub unsafe extern "C" fn voxrt_stream_new(t: *const VoxrtTranscriber) -> *mut VoxrtStream {
    if t.is_null() {
        set_error("NULL transcriber");
        return ptr::null_mut();
    }
    // SAFETY: the caller guarantees the transcriber outlives the stream (documented contract).
    let tr: &'static Transcriber = &*(&(*t).0 as *const Transcriber);
    Box::into_raw(Box::new(VoxrtStream {
        inner: LongForm::new(tr),
    }))
}

/// Push 16 kHz mono samples; returns the segments finalised by this call (possibly none: `n_segments == 0`).
#[no_mangle]
pub unsafe extern "C" fn voxrt_stream_push(
    s: *mut VoxrtStream,
    samples: *const f32,
    n_samples: usize,
) -> *mut VoxrtTranscript {
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
pub unsafe extern "C" fn voxrt_stream_flush(s: *mut VoxrtStream) -> *mut VoxrtTranscript {
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
pub unsafe extern "C" fn voxrt_stream_free(s: *mut VoxrtStream) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}
