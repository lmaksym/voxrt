"""Reference inference for the dequantized checkpoint: HF encoder + explicit NeMo-semantics TDT greedy loop.
Everything is exposed step by step so dump_golden.py can hook any tensor."""
import json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import ParakeetForTDT, ParakeetTokenizer, ParakeetFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]
FP = ROOT / "models/parakeet-redux-fp"

@dataclass
class Emission:
    token: int
    frame: int
    duration: int

@dataclass
class Result:
    text: str
    emissions: list[Emission]
    n_frames: int
    steps: list[dict] = field(default_factory=list)   # filled only when trace=True

class Reference:
    def __init__(self, device: str = "cpu", model_dir: Path = FP):
        self.device = torch.device(device)
        self.model = ParakeetForTDT.from_pretrained(model_dir, dtype=torch.float32).to(self.device).eval()
        self.cfg = self.model.config
        self.tok = ParakeetTokenizer.from_pretrained(model_dir)
        self.fe = ParakeetFeatureExtractor.from_pretrained(model_dir)
        self.blank = self.cfg.blank_token_id          # 8192
        self.durations = list(self.cfg.durations)      # [0,1,2,3,4]
        self.max_symbols = self.cfg.max_symbols_per_step  # 10
        self.V = self.cfg.vocab_size                   # 8193

    @staticmethod
    def load_audio(path) -> np.ndarray:
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        assert sr == 16000, sr
        return x.mean(1)

    @torch.no_grad()
    def features(self, wav: np.ndarray):
        f = self.fe(wav, sampling_rate=16000, return_tensors="pt")
        return f["input_features"].to(self.device), f["attention_mask"].to(self.device)

    @torch.no_grad()
    def encode(self, feats, mask):
        enc = self.model.encoder(input_features=feats, attention_mask=mask)
        h = enc.last_hidden_state                       # [1, T, 1024]
        T = int(enc.attention_mask.sum()) if enc.attention_mask is not None else h.shape[1]
        proj = self.model.encoder_projector(h)          # [1, T, 640]
        return h[0, :T], proj[0, :T]

    @torch.no_grad()
    def pred_step(self, last_token: int, state):
        dec = self.model.decoder
        ids = torch.tensor([[last_token]], device=self.device)
        emb = dec.embedding(ids)                        # blank row is zeros → SOS
        out, (h, c) = dec.lstm(emb, state)
        return dec.decoder_projector(out)[0, 0], (h, c)  # [640]

    @torch.no_grad()
    def joint(self, enc_t, g):
        return self.model.joint.head(self.model.joint.activation(enc_t + g))  # [8198]

    @torch.no_grad()
    def greedy_tdt(self, proj, trace: bool = False) -> tuple[list[Emission], list[dict]]:
        """NeMo GreedyTDTInfer semantics."""
        T = proj.shape[0]
        emissions, steps = [], []
        state, last = None, self.blank
        t = 0
        while t < T:
            symbols = 0
            while True:
                g, new_state = self.pred_step(last, state)
                logits = self.joint(proj[t], g)
                k = int(torch.argmax(logits[: self.V]))
                d = int(torch.argmax(logits[self.V:]))
                dur = self.durations[d]
                if trace:
                    steps.append({"t": t, "token": k, "dur": dur, "logits": logits.cpu().numpy(),
                                  "h": new_state[0].cpu().numpy(), "c": new_state[1].cpu().numpy()})
                if k != self.blank:
                    emissions.append(Emission(k, t, dur))
                    state, last = new_state, k
                symbols += 1
                t += dur
                if dur != 0 or symbols >= self.max_symbols:
                    break
                if k == self.blank:        # blank with dur 0: state/logits unchanged → NeMo spins to the cap, then +1
                    symbols = self.max_symbols
                    break
            if dur == 0:
                t += 1
        return emissions, steps

    def decode_text(self, emissions) -> str:
        return self.decode_ids([e.token for e in emissions])

    def decode_ids(self, ids) -> str:
        """SentencePiece/Metaspace decode WITHOUT HF's CTC-style collapsing of consecutive duplicate ids
        (ParakeetTokenizer.decode dedups repeats, which is wrong for transducer output: "▁an ▁an ne x" = "an annex")."""
        pieces = self.tok.convert_ids_to_tokens([i for i in ids if i != self.blank])
        special = set(self.tok.all_special_tokens)
        text = "".join(p for p in pieces if p not in special).replace("\u2581", " ")
        return text[1:] if text.startswith(" ") else text

    @torch.no_grad()
    def transcribe(self, path, trace: bool = False) -> Result:
        wav = self.load_audio(path)
        feats, mask = self.features(wav)
        _, proj = self.encode(feats, mask)
        em, steps = self.greedy_tdt(proj, trace=trace)
        return Result(self.decode_text(em), em, proj.shape[0], steps)

if __name__ == "__main__":
    import sys, time
    ref = Reference(device=sys.argv[1] if len(sys.argv) > 1 else "cpu")
    rows = [json.loads(l) for l in (ROOT / "data/manifests/dev50.jsonl").read_text().splitlines()][:3]
    for r in rows:
        t0 = time.time(); res = ref.transcribe(ROOT / r["audio"]); dt = time.time() - t0
        print(f"{r['id']}  {r['seconds']:.1f}s  frames={res.n_frames}  tokens={len(res.emissions)}  {dt:.2f}s")
        print("  REF:", r["text"]); print("  HYP:", res.text)
