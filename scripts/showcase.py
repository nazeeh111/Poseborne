"""Render the Poseborne signal path with deterministic synthetic I/Q, never pose inference.

The numerical work calls the retained amplitude, alignment, window and motion
energy routines. Pillow only draws those results. No network or model is used.
New files use the repository's LICENSE-branding terms.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "host"), str(ROOT / "rt")]


def signal_trace():
    import numpy as np
    from csi_pipe.align import LinkStream, amplitude, cut_windows, fill_gaps, grid_block
    from csi_rt.presence import MotionEnergy

    rng = np.random.default_rng(20261014)
    t = np.arange(401, dtype=np.int64) * 10_000_000
    seconds = t / 1e9
    carrier = np.arange(56)
    tensor = np.empty((401, 56, 3, 3), dtype=np.float32)
    mask = np.zeros((401, 3, 3), dtype=bool)
    means = np.empty((401, 9), dtype=np.float32)
    energies = np.zeros(401)
    motion = MotionEnergy(window_s=0.5)
    iq_hash = hashlib.sha256()
    links = []
    # An intentionally missing burst on RX2 / TX2 demonstrates the quality gate.
    keep = np.ones(401, dtype=bool)
    keep[180:186] = False
    raw = []
    for rx in range(3):
        for tx in range(3):
            k = rx * 3 + tx
            envelope = np.sin(np.pi * np.clip((seconds - .5) / 2.7, 0, 1)) ** 2
            level = (38 + 5 * np.cos(carrier[None] / 7 + k)
                     + 9 * envelope[:, None] * np.cos(seconds[:, None] * 5 + k / 2)
                     + 18 * envelope[:, None] * np.sin(2 * np.pi * seconds[:, None] * 2
                                                       + carrier[None] / 5 + k / 4))
            phase = seconds[:, None] * .6 + carrier[None] * .17 + k * .2
            iq = np.stack((level * np.cos(phase), level * np.sin(phase)), axis=-1)
            iq = np.clip(np.rint(iq + rng.normal(0, .35, iq.shape)), -127, 127).astype(np.int8)
            iq_hash.update(iq.tobytes())
            a = amplitude(iq)
            available = keep if k == 8 else np.ones(401, dtype=bool)
            t2, a2, interpolated, breaks = fill_gaps(t[available], np.arange(401)[available], a[available])
            aligned, bad = grid_block(LinkStream(t2, a2, interpolated, breaks), t)
            tensor[:, :, rx, tx], mask[:, rx, tx] = aligned, bad
            means[:, k] = aligned.mean(axis=1)
            raw.append((rx, tx, a, available))
            links.append({"rx": rx, "tx": tx, "packets": int(available.sum()),
                          "breaks": [list(b) for b in breaks]})
    for i in range(401):
        for rx, tx, a, available in raw:
            if available[i]:
                motion.add(rx, tx, int(t[i]), a[i])
        energies[i] = motion.energy()
    starts = np.arange(5, 401, 5) - 5
    X, valid = cut_windows(tensor, mask, starts)
    metrics = {"kind": "synthetic signal walkthrough", "seed": 20261014,
               "duration_s": 4, "grid_hz": 100, "links": links,
               "input_iq_sha256": iq_hash.hexdigest(), "windows": len(X),
               "valid_windows": int(valid.sum()), "rejected_windows": int((~valid).sum()),
               "window_shape": list(X.shape[1:]), "window_dtype": str(X.dtype),
               "window_sha256": hashlib.sha256(X.tobytes()).hexdigest(),
               "motion_energy_range": [float(energies.min()), float(energies.max())],
               "pose_inference": False, "hardware_measurement": False,
               "routines": ["csi_pipe.align.amplitude", "csi_pipe.align.fill_gaps",
                            "csi_pipe.align.grid_block", "csi_pipe.align.cut_windows",
                            "csi_rt.presence.MotionEnergy"]}
    return tensor, means, energies, mask, starts, X, valid, metrics


def frame_sample(index, count=64):
    """The source timestamp shown by a frame, independent of playback duration."""
    return 5 * int(round((5 + index * 395 / (count - 1)) / 5))


def render_frames(trace, *, font_path=None, compact=False):
    """Draw executed arrays as diagnostics; compact mode has its own readable layout."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    tensor, means, energies, mask, starts, X, valid, _ = trace
    w, h = (420, 650) if compact else (1120, 700)
    ink, muted, rule = "#20252b", "#58616c", "#c9cfd6"
    cobalt, warning = "#1558d6", "#c22f32"
    def font(size):
        return ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size=size)
    fonts = {s: font(s) for s in (12, 13, 14, 15, 16, 17, 18, 20, 22, 24, 28, 32)}
    def heat(values):
        z = np.clip((values - 18) / 43, 0, 1)[..., None]
        lo, hi = np.array([242, 247, 253]), np.array([21, 88, 214])
        return Image.fromarray(np.uint8(lo + (hi - lo) * z), "RGB")
    frames = []
    for n in range(64):
        sample = frame_sample(n)
        ix = sample // 5 - 1
        good = bool(valid[ix])
        im = Image.new("RGB", (w, h), "white")
        d = ImageDraw.Draw(im)
        def txt(x, y, value, size=17, fill=ink):
            d.text((x, y), value, font=fonts[size], fill=fill)
        def line(points, fill=rule, width=1):
            d.line(points, fill=fill, width=width)
        def spectrum(x, y, width, height):
            im.paste(heat(means.T).resize((width, height), Image.Resampling.NEAREST), (x, y))
            for k in range(9):
                txt(14 if compact else 32, y + k * height / 9 + 2,
                    f"{k // 3 + 1}/{k % 3 + 1}", 12 if compact else 16, muted)
            cursor = x + sample * (width-1) / 400
            line((cursor, y-4, cursor, y+height+4), cobalt, 2)
            gx = x + 180 * (width-1) / 400
            d.rectangle((gx, y+8*height/9, gx+6*(width-1)/400, y+height-1), fill=warning)
            for tick in range(5):
                px=x+tick*(width-1)/4
                line((px,y+height,px,y+height+5),muted)
                txt(px-4,y+height+8,str(tick),14,muted)
            txt(x+width-36,y+height+29,"time / s",12,muted)
        def energy(x, y, width, height):
            for frac in (0,.5,1):
                yy=y+height*(1-frac)
                line((x,yy,x+width,yy))
            points=[(x+j*width/400,y+height-float(energies[j])*height/max(energies.max(),1))
                    for j in range(sample+1)]
            if len(points)>1:line(points,cobalt,2)
        txt(24 if compact else 32, 20, "Poseborne", 28 if compact else 32)
        if compact:
            txt(24,57,"SIGNAL DIAGNOSTIC",13,muted)
            txt(24,83,"Synthetic input · no pose model",16,cobalt)
            line((24,113,396,113),ink)
            txt(24,126,f"{sample/100:.2f} s",28)
            txt(160,132,"VALID" if good else "REJECTED",22,ink if good else warning)
            txt(24,166,"9 links / mean carrier amplitude",16)
            txt(14,190,"RX/TX",12,muted)
            spectrum(54,210,342,126)
            txt(24,389,"50 ms feature window",20)
            txt(24,419,"5 samples x 56 carriers x 9 links",14,muted)
            matrix=X[ix].reshape(5,56,9).transpose(2,0,1).reshape(9,280)
            im.paste(heat(matrix).resize((372,80),Image.Resampling.NEAREST),(24,447))
            txt(24,536,"MODEL-FREE MOTION ENERGY",12,muted)
            txt(24,558,f"{energies[sample]:.3f}",22,cobalt)
            energy(160,550,236,40)
            txt(24,609,"6 missing packets injected in RX3/TX3",14,warning)
            txt(24,630,"Executed transforms; no hardware measurement.",12,muted)
        else:
            txt(250,29,"SIGNAL DIAGNOSTIC",16,muted)
            txt(32,74,"Synthetic I/Q · 9 links · 56 carriers · 100 Hz",20)
            txt(815,25,f"t = {sample/100:.2f} s",28)
            txt(815,68,"VALID WINDOW" if good else "REJECTED WINDOW",20,ink if good else warning)
            line((32,112,1080,112),ink)
            txt(32,130,"RX/TX",14,muted)
            txt(100,130,"Mean carrier amplitude",18)
            txt(650,130,"Display scale: 18 to 61 amplitude units",15,muted)
            spectrum(100,168,980,225)
            txt(100,446,"50 ms feature window",22)
            txt(100,480,"5 samples x 56 carriers x 9 links = (280, 3, 3)",16,muted)
            matrix=X[ix].reshape(5,56,9).transpose(2,0,1).reshape(9,280)
            im.paste(heat(matrix).resize((570,114),Image.Resampling.NEAREST),(100,516))
            txt(740,446,"Model-free motion energy",22)
            txt(740,480,f"{energies[sample]:.3f} amplitude units",16,cobalt)
            energy(740,516,340,114)
            line((32,655,1080,655),ink)
            txt(32,671,"6 missing packets injected in RX3/TX3",16,warning)
            txt(570,671,"Executed transforms / no hardware / no pose model",16,muted)
        frames.append(im)
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "showcase-output",
                    help="new output directory; existing paths are never overwritten")
    ap.add_argument("--font", type=Path, help="optional local TrueType/OpenType font")
    args = ap.parse_args(argv)
    if args.out.exists():
        ap.error(f"output already exists: {args.out}; choose a new --out directory")
    if args.font and not args.font.is_file():
        ap.error(f"font not found: {args.font}")
    try:
        trace = signal_trace()
        frames = render_frames(trace, font_path=args.font)
        compact_frames = render_frames(trace, font_path=args.font, compact=True)
    except ImportError as e:
        raise SystemExit(f"showcase dependency unavailable: {e}. Install requirements-showcase.txt.") from e
    args.out.mkdir(parents=True, exist_ok=False)
    from PIL import Image
    # Use a single palette to prevent GIF frame-to-frame color flicker.
    palette = frames[24].quantize(colors=128)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
    rejected_frames = [n for n in range(len(frames))
                       if not trace[6][frame_sample(n, len(frames)) // 5 - 1]]
    durations = [900 if n in rejected_frames else 70 for n in range(len(frames))]
    indexed[0].save(args.out / "signal-path.gif", save_all=True, append_images=indexed[1:],
                    duration=durations, loop=0, optimize=True, disposal=1)
    still = rejected_frames[0] if rejected_frames else 28
    frames[still].save(args.out / "signal-path.png")
    compact_palette = compact_frames[24].quantize(colors=128)
    compact_indexed = [im.quantize(palette=compact_palette, dither=Image.Dither.NONE) for im in compact_frames]
    compact_indexed[0].save(args.out / "signal-path-mobile.gif", save_all=True,
                            append_images=compact_indexed[1:], duration=durations,
                            loop=0, optimize=True, disposal=1)
    compact_frames[still].save(args.out / "signal-path-mobile.png")
    metrics = trace[-1]
    metrics["animation"] = {"frames": len(frames), "frame_duration_ms": durations,
                            "rejected_frame_hold_ms": 900,
                            "rejected_frame_indices": rejected_frames,
                            "playback_duration_ms": sum(durations),
                            "size": list(frames[0].size),
                            "compact_size": list(compact_frames[0].size),
                            "still_frame_index": still}
    (args.out / "signal-path.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"{metrics['valid_windows']}/{metrics['windows']} windows accepted; "
          f"{metrics['rejected_windows']} rejected. Synthetic input, no pose inference.\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
