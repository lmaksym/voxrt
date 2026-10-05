use std::io::{BufRead, Write};
use std::path::PathBuf;

use prx_core::format::{default_model_dir, ModelFile};
use prx_core::transcriber::Transcriber;

fn usage() -> ! {
    eprintln!("usage: prx <version|inspect [DIR]|transcribe FILE.wav [--model DIR]|batch MANIFEST.jsonl OUT.jsonl [--model DIR]|trace GOLDEN_UTT_DIR [--model DIR]>");
    std::process::exit(2)
}

fn model_dir(args: &[String]) -> PathBuf {
    args.iter()
        .position(|a| a == "--model")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(default_model_dir)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("version") | None => println!("prx {}", env!("CARGO_PKG_VERSION")),
        Some("inspect") => {
            let dir = args
                .get(2)
                .filter(|a| !a.starts_with("--"))
                .map(PathBuf::from)
                .unwrap_or_else(default_model_dir);
            let m = ModelFile::open(&dir)?;
            let infos = m.tensor_infos();
            println!(
                "{}: format {} · {} tensors · {} ternary modules · {} ternary / {} dense params",
                dir.display(),
                m.ternary.format,
                infos.len(),
                m.ternary.quantized_modules.len(),
                m.ternary.n_quantized_params,
                m.ternary.n_dense_params
            );
            let problems = m.validate();
            if problems.is_empty() {
                println!("validate: OK");
            } else {
                for p in &problems {
                    println!("validate: {p}");
                }
                anyhow::bail!("{} problems", problems.len());
            }
        }
        Some("transcribe") => {
            let file = args.get(2).unwrap_or_else(|| usage());
            let t0 = std::time::Instant::now();
            let tr = Transcriber::load(model_dir(&args))?;
            let load = t0.elapsed().as_secs_f32();
            let t1 = std::time::Instant::now();
            let a = prx_core::audio::read_wav(file)?;
            let r = tr.transcribe_samples(&a.samples);
            let secs = t1.elapsed().as_secs_f32();
            println!("{}", r.text);
            eprintln!("load {load:.2}s · {:.2}s audio in {secs:.2}s ({:.1}× real time) · {} frames · {} tokens", a.samples.len() as f32 / 16000.0, a.samples.len() as f32 / 16000.0 / secs, r.n_frames, r.emissions.len());
        }
        Some("batch") => {
            let (manifest, out) = (
                args.get(2).unwrap_or_else(|| usage()),
                args.get(3).unwrap_or_else(|| usage()),
            );
            let tr = Transcriber::load(model_dir(&args))?;
            let root = PathBuf::from(manifest)
                .parent()
                .and_then(|p| p.parent())
                .map(PathBuf::from)
                .unwrap_or_default(); // manifests live in <root>/data/manifests
            let done: std::collections::HashSet<String> = std::fs::read_to_string(out)
                .map(|s| {
                    s.lines()
                        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
                        .filter_map(|v| v["id"].as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default();
            let mut w = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(out)?;
            let reader = std::io::BufReader::new(std::fs::File::open(manifest)?);
            let (mut n, mut audio_s, t0) = (0usize, 0f32, std::time::Instant::now());
            for line in reader.lines() {
                let v: serde_json::Value = serde_json::from_str(&line?)?;
                let id = v["id"].as_str().unwrap().to_string();
                if done.contains(&id) {
                    continue;
                }
                let wav = root
                    .join("..")
                    .join(v["audio"].as_str().unwrap())
                    .with_extension("wav");
                let a = prx_core::audio::read_wav(&wav)?;
                let r = tr.transcribe_samples(&a.samples);
                audio_s += a.samples.len() as f32 / 16000.0;
                let em: Vec<[usize; 3]> = r
                    .emissions
                    .iter()
                    .map(|e| [e.token, e.frame, e.duration])
                    .collect();
                writeln!(
                    w,
                    "{}",
                    serde_json::json!({"id": id, "text": r.text, "n_frames": r.n_frames, "emissions": em})
                )?;
                n += 1;
                if n % 50 == 0 {
                    eprintln!(
                        "{n} utts · {:.0}s · {:.1}× real time",
                        t0.elapsed().as_secs_f32(),
                        audio_s / t0.elapsed().as_secs_f32()
                    );
                }
            }
            eprintln!(
                "done: {n} new utts, {:.1}× real time",
                audio_s / t0.elapsed().as_secs_f32().max(1e-6)
            );
        }
        Some("trace") => {
            // per-stage error table against one golden utterance directory (docs/golden.md)
            let gdir = PathBuf::from(args.get(2).unwrap_or_else(|| usage()));
            let tr = Transcriber::load(model_dir(&args))?;
            let a = prx_core::audio::read_wav(gdir.join("audio.wav"))?;
            let (feats, t) = tr.frontend.compute(&a.samples);
            let mut outs = Vec::new();
            let mut l0 = prx_core::encoder::LayerTrace::default();
            let (enc, te) = tr
                .encoder
                .forward(&feats, t, Some(&mut outs), Some(&mut l0));
            let cmp = |name: &str, ours: &[f32]| {
                let g = npy_f32(&gdir.join(format!("{name}.npy")));
                let n = ours.len().min(g.len());
                let (mut da, mut scale) = (0f32, 0f32);
                for i in 0..n {
                    da = da.max((ours[i] - g[i]).abs());
                    scale = scale.max(g[i].abs());
                }
                println!(
                    "{name:28} max|Δ|={da:.2e}  rel={:.2e}",
                    da / scale.max(1e-12)
                );
            };
            cmp("features", &feats);
            cmp("layer00_norm_ff1_out", &l0.norm_ff1);
            cmp("layer00_ff1_out", &l0.ff1);
            cmp("layer00_attn_out", &l0.attn);
            cmp("layer00_conv_out", &l0.conv);
            cmp("layer00_ff2_out", &l0.ff2);
            for (i, o) in outs.iter().enumerate() {
                cmp(&format!("layer{i:02}_out"), o);
            }
            cmp("encoder_out", &enc);
            let proj = tr.decoder.project_encoder(&enc, te);
            cmp("encoder_proj", &proj);
            let em = tr.decoder.greedy(&proj, te, None);
            let ge = npy_f32(&gdir.join("emissions.npy"));
            let ours: Vec<f32> = em
                .iter()
                .flat_map(|e| [e.token as f32, e.frame as f32, e.duration as f32])
                .collect();
            println!(
                "emissions: {} ours vs {} golden → {}",
                em.len(),
                ge.len() / 3,
                if ours == ge { "EXACT" } else { "DIFFER" }
            );
            println!(
                "{}",
                tr.tokenizer
                    .decode(&em.iter().map(|e| e.token).collect::<Vec<_>>())
            );
        }
        Some(_) => usage(),
    }
    Ok(())
}

/// Minimal little-endian f32/i32 .npy reader for `trace`.
fn npy_f32(path: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (hlen, start) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[start..start + hlen]).unwrap();
    let body = &b[start + hlen..];
    if header.contains("'<i4'") {
        body.chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect()
    } else {
        body.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}
