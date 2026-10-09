"""Regenerate the README assets (logo and benchmark chart, light and dark): python3 assets/make_assets.py
Numbers come from bench/README.md (dev50, processing time only, same method for every runtime)."""
from pathlib import Path

OUT = Path(__file__).parent
THEMES = {
    "light": dict(fg="#1f2328", muted="#59636e", grid="#d1d9e0", bar="#c9d1d9", accent="#2da44e", bg="none"),
    "dark": dict(fg="#f0f6fc", muted="#9198a1", grid="#3d444d", bar="#3d444d", accent="#3fb950", bg="none"),
}

# (runtime, × real time, teacher model?)
PANELS = [
    ("Apple M4 Max · 14 cores", [("voxrt", 78, False), ("Photon (Moondream)", 70, False),
                                 ("sherpa-onnx int8", 28, True), ("onnx-asr int8", 18, True),
                                 ("parakeet.cpp q8_0", 13, True)]),
    ("AMD EPYC Turin · 8 cores / 16 threads", [("voxrt", 104, False), ("Photon (Moondream)", 98, False),
                                                ("parakeet.cpp q8_0", 42, True)]),
]


def logo(t):
    c = THEMES[t]
    # ternary "waveform": bar heights at three levels (the weights are trits)
    levels = [0.45, 1.0, 0.7, 0.3, 0.85, 0.55, 0.95]
    bars = "".join(
        f'<rect x="{6 + i * 13}" y="{40 - h * 30:.1f}" width="8" height="{h * 60:.1f}" rx="4" '
        f'fill="{c["accent"] if i in (1, 4) else c["fg"]}"/>'
        for i, h in enumerate(levels)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="420" height="80" viewBox="0 0 420 80">'
        f"{bars}"
        f'<text x="108" y="56" font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" '
        f'font-size="54" font-weight="700" letter-spacing="-1" fill="{c["fg"]}">voxrt</text></svg>'
    )


def chart(t):
    c = THEMES[t]
    font = "-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif"
    W, label_w, bar_w, row, gap = 760, 170, 470, 26, 34
    xmax = 110
    y = 30
    parts = [f'<text x="0" y="16" font-family="{font}" font-size="14" font-weight="600" fill="{c["fg"]}">'
             f'Speed on LibriSpeech dev50 (× real time, CPU, higher is better)</text>']
    for title, rows in PANELS:
        y += 22
        parts.append(f'<text x="0" y="{y}" font-family="{font}" font-size="12.5" fill="{c["muted"]}">{title}</text>')
        y += 8
        for name, v, teacher in rows:
            w = bar_w * v / xmax
            col = c["accent"] if name == "voxrt" else c["bar"]
            weight = "700" if name == "voxrt" else "400"
            parts.append(f'<text x="{label_w - 8}" y="{y + 15}" text-anchor="end" font-family="{font}" font-size="12.5" '
                         f'font-weight="{weight}" fill="{c["fg"]}">{name}{"†" if teacher else ""}</text>')
            parts.append(f'<rect x="{label_w}" y="{y + 3}" width="{w:.1f}" height="{row - 8}" rx="3" fill="{col}"/>')
            parts.append(f'<text x="{label_w + w + 6:.1f}" y="{y + 15}" font-family="{font}" font-size="12.5" '
                         f'font-weight="{weight}" fill="{c["fg"]}">{v}×</text>')
            y += row
        y += gap - row + 6
    y += 4
    parts.append(f'<text x="0" y="{y}" font-family="{font}" font-size="11" fill="{c["muted"]}">'
                 f'† runs the 0.6B teacher model (parakeet-tdt-0.6b-v3), not the ternary student. Processing time only, '
                 f'model load excluded.</text>')
    H = y + 8
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">'
            + "".join(parts) + "</svg>")


for t in THEMES:
    (OUT / f"logo-{t}.svg").write_text(logo(t))
    (OUT / f"bench-{t}.svg").write_text(chart(t))
print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
