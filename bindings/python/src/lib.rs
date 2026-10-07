//! PyO3 bindings (PLAN 5.2): `prx.Transcriber(model_dir).transcribe(path | samples)` and `prx.Stream`.
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use prx_core::longform::LongForm;
use prx_core::transcriber::{Segment, Transcriber};

fn err(e: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

fn segments_py<'py>(py: Python<'py>, segs: &[Segment]) -> PyResult<Vec<Bound<'py, PyDict>>> {
    segs.iter()
        .map(|s| {
            let d = PyDict::new(py);
            d.set_item("start", s.start)?;
            d.set_item("end", s.end)?;
            d.set_item("text", &s.text)?;
            let words: Vec<Bound<'py, PyDict>> = s
                .words
                .iter()
                .map(|w| {
                    let wd = PyDict::new(py);
                    wd.set_item("start", w.start)?;
                    wd.set_item("end", w.end)?;
                    wd.set_item("word", &w.word)?;
                    Ok(wd)
                })
                .collect::<PyResult<_>>()?;
            d.set_item("words", words)?;
            Ok(d)
        })
        .collect()
}

/// Result of a transcription: `.text`, `.segments` (list of dicts with start/end/text/words), `.duration`.
#[pyclass(name = "Transcript", frozen)]
pub struct PyTranscript {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    duration: f64,
    segments: Vec<Segment>,
}

#[pymethods]
impl PyTranscript {
    #[getter]
    fn segments<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        segments_py(py, &self.segments)
    }
    fn __repr__(&self) -> String {
        format!(
            "Transcript({:?}, {} segments, {:.1} s)",
            self.text,
            self.segments.len(),
            self.duration
        )
    }
    fn __str__(&self) -> String {
        self.text.clone()
    }
}

/// Samples arrive as a path, or as raw little-endian float32 bytes (the Python wrapper converts arrays; this keeps
/// the extension on the stable ABI, where the buffer protocol is unavailable).
fn samples_from(audio: &Bound<'_, PyAny>, sample_rate: u32) -> PyResult<Vec<f32>> {
    if let Ok(p) = audio.extract::<String>() {
        let a = prx_core::audio::read_wav(&p).map_err(err)?;
        return Ok(a.samples);
    }
    let b: &[u8] = audio.extract().map_err(|_| {
        PyValueError::new_err("audio must be a path or float32 sample bytes (use prx.Transcriber.transcribe with an array)")
    })?;
    if b.len() % 4 != 0 {
        return Err(PyValueError::new_err(
            "sample bytes length is not a multiple of 4",
        ));
    }
    let x: Vec<f32> = b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(if sample_rate == 16_000 {
        x
    } else {
        prx_core::audio::resample(&x, sample_rate, 16_000)
    })
}

/// A loaded model. `Transcriber(model_dir=None)` uses `PRX_MODEL_DIR` or the repo default when `model_dir` is None.
#[pyclass(name = "Transcriber", frozen)]
pub struct PyTranscriber {
    inner: Transcriber,
}

#[pymethods]
impl PyTranscriber {
    #[new]
    #[pyo3(signature = (model_dir=None))]
    fn new(py: Python<'_>, model_dir: Option<String>) -> PyResult<Self> {
        let dir = model_dir
            .map(std::path::PathBuf::from)
            .unwrap_or_else(prx_core::format::default_model_dir);
        let inner = py.allow_threads(|| Transcriber::load(&dir)).map_err(err)?;
        Ok(Self { inner })
    }

    /// Transcribe a WAV path or an array of mono samples (`sample_rate` for arrays; resampled to 16 kHz).
    /// Long audio is VAD-segmented automatically. Releases the GIL while running.
    #[pyo3(signature = (audio, sample_rate=16000))]
    fn transcribe(
        &self,
        py: Python<'_>,
        audio: &Bound<'_, PyAny>,
        sample_rate: u32,
    ) -> PyResult<PyTranscript> {
        let x = samples_from(audio, sample_rate)?;
        let r = py.allow_threads(|| self.inner.transcribe_long(&x));
        Ok(PyTranscript {
            text: r.text,
            duration: x.len() as f64 / 16_000.0,
            segments: r.segments,
        })
    }

    /// Streaming session over this model (see `Stream`).
    fn stream(slf: Py<Self>, py: Python<'_>) -> PyStream {
        let tr: &'static Transcriber = {
            let r = slf.borrow(py);
            // SAFETY: `PyStream` holds a strong reference to the transcriber object, so it outlives the borrow.
            unsafe { &*(&r.inner as *const Transcriber) }
        };
        PyStream {
            _owner: slf,
            inner: std::sync::Mutex::new(LongForm::new(tr)),
        }
    }

    fn __repr__(&self) -> String {
        format!("Transcriber(frame={:.2}s)", self.inner.frame_seconds)
    }
}

/// Streaming: `push(samples)` with 16 kHz mono float32 returns the segments finalised so far; `flush()` the rest.
/// Segments are closed only once the following pause is known, so streaming output equals offline output.
#[pyclass(name = "Stream", frozen)]
pub struct PyStream {
    _owner: Py<PyTranscriber>,
    inner: std::sync::Mutex<LongForm<'static>>,
}

#[pymethods]
impl PyStream {
    fn push<'py>(
        &self,
        py: Python<'py>,
        samples: &Bound<'py, PyAny>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let x = samples_from(samples, 16_000)?;
        let segs = py.allow_threads(|| self.inner.lock().unwrap().push(&x));
        segments_py(py, &segs)
    }
    fn flush<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let segs = py.allow_threads(|| self.inner.lock().unwrap().flush());
        segments_py(py, &segs)
    }
}

#[pymodule]
fn _prx(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTranscriber>()?;
    m.add_class::<PyTranscript>()?;
    m.add_class::<PyStream>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
