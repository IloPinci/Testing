"""
make_comparison_video.py
------------------------
Website / presentation helpers built on the videos zeroshot_eval.py records.

1) Side-by-side comparison of the SAME episode (same matched initial state,
   same noise seed) across conditions or horizons:

    python make_comparison_video.py --episode 3 --stream presentation \\
        --panel "Open loop (k=50)=runs/paper-v2/eval/k50/wtilt+10" \\
        --panel "Closed loop (k=10)=runs/paper-v2/eval/k10/wtilt+10" \\
        --out site/videos/wtilt10_ep003_ol_vs_cl.mp4

   Streams: presentation | witness/context | witness/top | witness/wrist | policy/combined.
   Shorter clips freeze on their last frame until the longest one ends.

2) Static HTML gallery of every recorded episode in a campaign (relative
   links, so the folder can be copied to a web server as is):

    python make_comparison_video.py --gallery runs/paper-v2 --out runs/paper-v2/gallery.html
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from video_utils import _Writer, label_tile, letterbox  # noqa: E402


def find_clip(run_dir: Path, stream: str, episode: int) -> Path:
    hits = sorted((run_dir / "videos" / stream).glob(f"ep{episode:03d}_*.mp4"))
    hits = [h for h in hits if not h.name.endswith(".part.mp4")]
    if not hits:
        sys.exit(f"[error] no {stream} clip for episode {episode} in {run_dir}/videos "
                 f"(was it in --video_episodes? replay it with --episodes {episode})")
    return hits[0]


def side_by_side(panels, stream, episode, out, tile_w, cols, crf):
    import imageio.v2 as iio2
    readers, labels, fps = [], [], None
    for label, d in panels:
        clip = find_clip(Path(d), stream, episode)
        r = iio2.get_reader(str(clip))
        fps = fps or r.get_meta_data().get("fps", 25)
        outcome = "PASS" if clip.stem.endswith("_pass") else "FAIL"
        readers.append(r)
        labels.append(f"{label}  [{outcome}]")
    first = [r.get_data(0) for r in readers]
    h0, w0 = first[0].shape[:2]
    tile_h = int(round(tile_w * h0 / w0 / 2) * 2)
    rows = int(np.ceil(len(readers) / cols))
    iters = [iter(r) for r in readers]
    last = list(first)
    w = _Writer(Path(out), fps, crf)
    while True:
        alive = False
        for i, it in enumerate(iters):
            try:
                last[i] = next(it)
                alive = True
            except StopIteration:
                pass
        if not alive:
            break
        tiles = [label_tile(letterbox(f, tile_w, tile_h), lab) for f, lab in zip(last, labels)]
        tiles += [np.zeros_like(tiles[0])] * (rows * cols - len(tiles))
        grid = np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], 1) for r in range(rows)], 0)
        w.add(grid)
    w.close()
    from PIL import Image
    Image.fromarray(w.first).save(Path(out).with_suffix(".jpg"), quality=88)
    print(f"[ok] {out}  ({w.n} frames @ {fps} fps)")


def gallery(root: Path, out: Path, stream: str):
    manifests = sorted(root.glob("eval/k*/*/videos/manifest.json"))
    if not manifests:
        sys.exit(f"[error] no video manifests under {root}/eval/k*/<slug>/videos/")
    by_cond: dict[str, dict[int, dict]] = {}
    for m in manifests:
        data = json.loads(m.read_text())
        k = int(m.parent.parent.parent.name.lstrip("k"))
        by_cond.setdefault(m.parent.parent.name, {})[k] = {"dir": m.parent, "data": data}
    out.parent.mkdir(parents=True, exist_ok=True)
    rel = lambda p: os.path.relpath(p, out.parent)
    parts = ["<!doctype html><meta charset='utf-8'><title>Rollout gallery</title>",
             "<style>body{font-family:system-ui,sans-serif;margin:24px;background:#fafafa}"
             "h2{margin-top:32px;border-bottom:1px solid #ddd}"
             ".row{display:flex;gap:16px;overflow-x:auto}.card{background:#fff;border:1px solid #e5e5e5;"
             "border-radius:8px;padding:8px;min-width:420px}video{width:420px;border-radius:4px}"
             ".ok{color:#2a7d2a;font-weight:600}.bad{color:#b22;font-weight:600}"
             "small{color:#666}</style>",
             f"<h1>Rollout gallery</h1><p><small>{html.escape(str(root))} · stream: {stream} · "
             f"same episode index = same initial state in every condition</small></p>"]
    for cond in sorted(by_cond, key=lambda c: (c != "nominal", c)):
        parts.append(f"<h2>{html.escape(cond)}</h2>")
        for k in sorted(by_cond[cond], reverse=True):
            info = by_cond[cond][k]
            parts.append(f"<h3>k={k} {'(open loop)' if k >= 50 else '(closed loop)'}</h3><div class='row'>")
            for e in info["data"]["episodes"]:
                f = e["files"].get(stream)
                if not f:
                    continue
                vid, poster = info["dir"] / f["video"], info["dir"] / f["poster"]
                ok = e["success"]
                parts.append(
                    f"<div class='card'><video controls preload='none' poster='{rel(poster)}' "
                    f"src='{rel(vid)}'></video><div>episode {e['episode']} · "
                    f"<span class='{'ok' if ok else 'bad'}'>{'success' if ok else 'failure'}</span> · "
                    f"<small>{html.escape(e['failure_mode'])}, {e['steps']} steps</small></div></div>")
            parts.append("</div>")
    out.write_text("\n".join(parts))
    print(f"[ok] {out}  ({sum(len(v) for v in by_cond.values())} runs)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", action="append", default=[], help='"Label=run_dir" (repeatable)')
    ap.add_argument("--episode", type=int)
    ap.add_argument("--stream", default="presentation")
    ap.add_argument("--cols", type=int, default=None)
    ap.add_argument("--tile_w", type=int, default=960)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--gallery", default=None, help="campaign root, e.g. runs/paper-v2")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.gallery:
        gallery(Path(a.gallery), Path(a.out), a.stream)
        return
    if not a.panel or a.episode is None:
        ap.error("need --panel (x2+) and --episode, or --gallery")
    panels = [tuple(p.split("=", 1)) for p in a.panel]
    side_by_side(panels, a.stream, a.episode, a.out, a.tile_w, a.cols or min(len(panels), 2), a.crf)


if __name__ == "__main__":
    main()