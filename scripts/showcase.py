"""Render the WiPose signal path with deterministic synthetic I/Q, never pose inference.

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


def render_frames(trace, *, font_path=None):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    tensor, means, energies, mask, starts, X, valid, _ = trace
    w, h = 1120, 660
    navy, panel, border = "#0c1625", "#122236", "#263a51"
    ink, muted, teal, amber = "#edf4f7", "#9bafc3", "#66d8c4", "#ffcb74"
    def font(size):
        return ImageFont.truetype(str(font_path), size) if font_path else ImageFont.load_default(size=size)
    fonts = {s: font(s) for s in (13, 15, 17, 20, 25, 42)}
    def color_matrix(values):
        z = np.clip((values - 18) / 43, 0, 1)
        lo, mid, hi = np.array([18, 40, 61]), np.array([67, 183, 165]), np.array([255, 204, 113])
        c = np.where((z < .65)[..., None], lo + (mid - lo) * (z / .65)[..., None],
                     mid + (hi - mid) * ((z - .65) / .35)[..., None])
        return Image.fromarray(np.uint8(np.clip(c, 0, 255)), "RGB")
    frames = []
    for n in range(64):
        sample = 5 * int(round((5 + n * 395 / 63) / 5))
        im = Image.new("RGB", (w, h), navy)
        d = ImageDraw.Draw(im)
        def txt(x, y, text, size=17, fill=ink):
            d.text((x, y), text, font=fonts[size], fill=fill)
        txt(38, 24, "WiPose", 42)
        txt(40, 78, "Inside the signal. Before the skeleton.", 20, muted)
        d.rounded_rectangle((760, 35, 1080, 69), 8, fill=panel, outline=border)
        txt(777, 43, "SYNTHETIC I/Q  /  NO POSE MODEL", 13, amber)
        cards = [("01", "I/Q packets", "9 links / 56 carriers"),
                 ("02", "Amplitude", "sqrt(I^2 + Q^2)"),
                 ("03", "Align + gate", "100 Hz / mask gaps"),
                 ("04", "Window tensor", "50 ms / 280 x 3 x 3"),
                 ("05", "Pose inference", "trained weights required")]
        for k, (num, name, detail) in enumerate(cards):
            x = 40 + k * 211
            active = k < 4
            d.rounded_rectangle((x, 128, x + 194, 222), 10, fill=panel, outline=border)
            txt(x + 13, 139, num, 13, teal if active else muted)
            txt(x + 13, 162, name, 20, ink if active else muted)
            txt(x + 13, 195, detail, 13, muted)
            if k < 4:
                d.line((x + 195, 176, x + 210, 176), fill=border, width=2)
                if k < 3:
                    travel = (n / 8 + k / 3) % 1
                    px = x + 196 + 13 * travel
                    d.ellipse((px - 3, 173, px + 3, 179), fill=teal)
        d.rounded_rectangle((40, 247, 704, 531), 12, fill=panel, outline=border)
        txt(59, 264, "9 synthetic links / mean carrier amplitude", 20)
        txt(59, 294, "Deterministic input / actual alignment routine", 15, muted)
        heat = color_matrix(means.T).resize((591, 157), Image.Resampling.NEAREST)
        im.paste(heat, (91, 328))
        for k in range(9):
            txt(59, 328 + k * 17, f"{k // 3 + 1}{k % 3 + 1}", 13, muted)
        d.line((91 + sample * 590 / 400, 325, 91 + sample * 590 / 400, 489), fill=ink, width=2)
        # Missing samples are shown as a marked gap, rather than invented measurements.
        gx = 91 + 180 * 590 / 400
        d.rectangle((gx, 328 + 8 * 17.44, gx + 6 * 590 / 400, 485), fill=amber)
        txt(91, 499, "0 s", 13, muted)
        txt(625, 499, "4 s", 13, muted)
        txt(330, 499, "6 missing packets / RX3-TX3", 13, amber)
        d.rounded_rectangle((723, 247, 1080, 531), 12, fill=panel, outline=border)
        ix = sample // 5 - 1
        good = bool(valid[ix])
        txt(743, 264, "50 ms feature window", 20)
        txt(743, 294, "5 samples x 56 carriers x 9 links", 15, muted)
        matrix = X[ix].reshape(5, 56, 9).transpose(2, 0, 1).reshape(9, 280)
        im.paste(color_matrix(matrix).resize((316, 127), Image.Resampling.NEAREST), (743, 328))
        d.rounded_rectangle((743, 474, 1059, 512), 7, fill=navy)
        txt(757, 484, "VALID / model input ready" if good else "REJECTED / missing-data gate", 15,
            teal if good else amber)
        d.rounded_rectangle((40, 550, 1080, 614), 10, fill=panel, outline=border)
        txt(60, 564, f"{sample / 100:04.2f} s", 25)
        txt(194, 565, "MODEL-FREE MOTION ENERGY", 13, muted)
        txt(194, 585, f"{energies[sample]:.3f} amplitude units", 15, teal)
        points = [(475 + j * 565 / 400, 603 - float(energies[j]) * 36 / max(energies.max(), 1))
                  for j in range(sample + 1)]
        if len(points) >= 2:
            d.line(points, fill=teal, width=2)
        txt(40, 633, "Executed transforms, not RF measurements or inferred human motion. Reproduce: python wipose.py showcase", 13, muted)
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
    except ImportError as e:
        raise SystemExit(f"showcase dependency unavailable: {e}. Install requirements-showcase.txt.") from e
    args.out.mkdir(parents=True, exist_ok=False)
    from PIL import Image
    # Use a single palette to prevent GIF frame-to-frame color flicker.
    palette = frames[24].quantize(colors=128)
    indexed = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
    indexed[0].save(args.out / "signal-path.gif", save_all=True, append_images=indexed[1:],
                    duration=70, loop=0, optimize=True, disposal=1)
    frames[28].save(args.out / "signal-path.png")
    metrics = trace[-1]
    metrics["animation"] = {"frames": len(frames), "frame_duration_ms": 70, "size": list(frames[0].size)}
    (args.out / "signal-path.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"{metrics['valid_windows']}/{metrics['windows']} windows accepted; "
          f"{metrics['rejected_windows']} rejected. Synthetic input, no pose inference.\n{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
