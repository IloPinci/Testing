"""
video_utils.py
--------------
Video recording helpers shared by zeroshot_eval.py and make_comparison_video.py.
No Genesis / torch imports, so this module is testable on its own.

Design choices (and why)
------------------------
* STREAMING writers. Frames go straight into an ffmpeg pipe as they are
  rendered. The old code kept every frame of every camera in Python lists
  until the episode ended (a 6 s episode of three 1280x720 witness streams
  at 25 fps is ~1.2 GB of RAM).
* WEB-READY output: H.264, yuv420p, `-movflags +faststart` (the moov atom
  is at the front, so browsers start playing before the download ends),
  plus a JPEG poster frame per video for <video poster="...">.
* REAL-TIME playback. Policy-camera videos are written at the policy rate
  (10 Hz -> 10 fps). Witness videos are sampled every `witness_every`
  simulator steps (default 4 x 10 ms = 25 fps) and written at that rate, so
  1 s of video is 1 s of simulated time. The old setup sampled at 2 Hz and
  played at 15 fps (7.5x fast-forward).
* CLEAN + ANNOTATED. Per-camera witness streams are clean (no text) so you
  can cut them freely for a website. The `presentation` stream is a 2x2
  grid (context / top / wrist witness + what the policy actually received)
  with a banner, sub-goal status chips and a "NEW CHUNK" marker whenever the
  policy re-plans — this makes open- vs closed-loop behaviour visible.
* Files are written as <tag>.part.mp4 and renamed to <tag>_<outcome>.mp4
  on success, so an interrupted run never leaves a file that looks complete.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID_W, GRID_H = 1280, 720
TILE_W, TILE_H = GRID_W // 2, GRID_H // 2

_COLORS = {
    "on": (46, 160, 67), "off": (90, 90, 90), "fail": (200, 50, 50),
    "replan": (255, 140, 0), "text": (255, 255, 255), "banner": (0, 0, 0),
}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)      # Pillow >= 10.1
    except TypeError:                                  # older Pillow
        return ImageFont.load_default()


_F_SMALL, _F_MED = _font(14), _font(20)


def _to_uint8_rgb(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame)
    if f.dtype != np.uint8:
        f = np.clip(f, 0, 255).astype(np.uint8)
    if f.ndim == 2:
        f = np.repeat(f[..., None], 3, axis=2)
    return f[..., :3]


def resize(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(_to_uint8_rgb(frame)).resize((w, h), Image.BILINEAR))


def letterbox(frame: np.ndarray, w: int, h: int, bg=(0, 0, 0)) -> np.ndarray:
    """Fit frame into w x h keeping aspect ratio, padding with bg."""
    f = _to_uint8_rgb(frame)
    fh, fw = f.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = max(1, int(round(fw * s))), max(1, int(round(fh * s)))
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(Image.fromarray(f).resize((nw, nh), Image.BILINEAR), ((w - nw) // 2, (h - nh) // 2))
    return np.asarray(canvas)


def placeholder(w: int, h: int, text: str, bg=(60, 60, 60)) -> np.ndarray:
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=_F_MED)
    d.text(((w - tw) / 2, h / 2 - 10), text, fill=_COLORS["text"], font=_F_MED)
    return np.asarray(img)


def label_tile(frame: np.ndarray, text: str, border: Optional[tuple] = None,
               y: int = 0) -> np.ndarray:
    img = Image.fromarray(_to_uint8_rgb(frame).copy())
    d = ImageDraw.Draw(img)
    tw = d.textlength(text, font=_F_SMALL)
    d.rectangle([0, y, tw + 10, y + 20], fill=(0, 0, 0))
    d.text((5, y + 3), text, fill=_COLORS["text"], font=_F_SMALL)
    if border is not None:
        w, h = img.size
        d.rectangle([1, 1, w - 2, h - 2], outline=border, width=4)
    return np.asarray(img)


def policy_strip(frames: dict, status: dict) -> np.ndarray:
    """
    Horizontal strip of what the POLICY received: context | wrist | top.
    frames[role] is an HxWx3 image, or None if the camera was absent (tensor
    dropped). status[role] in {"ok", "blank", "absent"} labels each tile.
    """
    tiles = []
    for role in ("context", "wrist", "top"):
        f = frames.get(role)
        st = status.get(role, "ok")
        if f is None:
            tiles.append(placeholder(256, 256, "ABSENT"))
            continue
        tile = resize(f, 256, 256) if f.shape[:2] != (256, 256) else _to_uint8_rgb(f)
        tiles.append(label_tile(tile, f"{role}{' (blanked)' if st == 'blank' else ''}"))
    return np.concatenate(tiles, axis=1)


def annotate_grid(grid: np.ndarray, title: str, info: str, chips: dict,
                  replan: bool, outcome: Optional[str] = None) -> np.ndarray:
    """Banner with title/info, sub-goal chips, replan marker and final outcome."""
    img = Image.fromarray(grid.copy())
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, GRID_W, 30], fill=_COLORS["banner"])
    d.text((8, 5), title, fill=_COLORS["text"], font=_F_MED)
    iw = d.textlength(info, font=_F_MED)
    d.text((GRID_W - iw - 8, 5), info, fill=_COLORS["text"], font=_F_MED)
    x = 8
    y = GRID_H - 28
    for name, on in chips.items():
        tw = d.textlength(name, font=_F_SMALL)
        d.rounded_rectangle([x, y, x + tw + 16, y + 22], radius=6,
                            fill=_COLORS["on"] if on else _COLORS["off"])
        d.text((x + 8, y + 4), name, fill=_COLORS["text"], font=_F_SMALL)
        x += tw + 24
    if replan:
        txt = "NEW CHUNK"
        tw = d.textlength(txt, font=_F_SMALL)
        d.rounded_rectangle([x, y, x + tw + 16, y + 22], radius=6, fill=_COLORS["replan"])
        d.text((x + 8, y + 4), txt, fill=_COLORS["text"], font=_F_SMALL)
    if outcome is not None:
        ok = outcome.lower().startswith("success")
        txt = outcome.upper()
        tw = d.textlength(txt, font=_F_MED)
        d.rounded_rectangle([GRID_W - tw - 28, y - 6, GRID_W - 8, y + 24], radius=8,
                            fill=_COLORS["on"] if ok else _COLORS["fail"])
        d.text((GRID_W - tw - 18, y - 1), txt, fill=_COLORS["text"], font=_F_MED)
    return np.asarray(img)


def presentation_grid(w_context, w_top, w_wrist, policy_view, replan: bool) -> np.ndarray:
    """2x2: witness context | witness top / witness wrist | policy input."""
    # top-row labels start below the 30 px banner drawn by annotate_grid()
    tl = label_tile(letterbox(w_context, TILE_W, TILE_H), "scene (context view)", y=32)
    tr = label_tile(letterbox(w_top, TILE_W, TILE_H), "scene (top view)", y=32)
    bl = label_tile(letterbox(w_wrist, TILE_W, TILE_H), "scene (wrist view)")
    br = label_tile(letterbox(policy_view, TILE_W, TILE_H), "policy input: context | wrist | top",
                    border=_COLORS["replan"] if replan else None)
    return np.concatenate([np.concatenate([tl, tr], 1), np.concatenate([bl, br], 1)], 0)


class _Writer:
    def __init__(self, path: Path, fps: float, crf: int = 20):
        import imageio.v2 as iio2
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.first = None
        self.n = 0
        self._w = iio2.get_writer(
            str(path), format="FFMPEG", mode="I", fps=fps, codec="libx264",
            pixelformat="yuv420p", macro_block_size=2,
            ffmpeg_params=["-crf", str(crf), "-preset", "veryfast", "-movflags", "+faststart"],
        )

    def add(self, frame):
        f = _to_uint8_rgb(frame)
        h, w = f.shape[:2]
        if h % 2 or w % 2:                      # yuv420p needs even dimensions
            f = f[: h - h % 2, : w - w % 2]
        if self.first is None:
            self.first = f
        self._w.append_data(f)
        self.n += 1

    def close(self):
        self._w.close()


class VideoRecorder:
    """
    One recorder per recorded episode. Streams are opened lazily on first
    frame, e.g. rec.add("witness/context", frame, fps=25).
    """

    def __init__(self, video_root: Path, ep_tag: str, crf: int = 20):
        self.root = Path(video_root)
        self.tag = ep_tag
        self.crf = crf
        self._writers: dict[str, _Writer] = {}

    def add(self, stream: str, frame: np.ndarray, fps: float):
        w = self._writers.get(stream)
        if w is None:
            w = _Writer(self.root / stream / f"{self.tag}.part.mp4", fps, self.crf)
            self._writers[stream] = w
        w.add(frame)

    def last_frame(self, stream: str) -> Optional[np.ndarray]:
        w = self._writers.get(stream)
        return None if w is None else w.first

    def finalize(self, outcome: str) -> dict:
        """Close, rename <tag>.part.mp4 -> <tag>_<outcome>.mp4, write posters."""
        files = {}
        for stream, w in self._writers.items():
            w.close()
            final = w.path.with_name(f"{self.tag}_{outcome}.mp4")
            if final.exists():
                final.unlink()
            w.path.rename(final)
            poster = final.with_suffix(".jpg")
            if w.first is not None:
                Image.fromarray(w.first).save(poster, quality=88)
            files[stream] = {"video": str(final.relative_to(self.root)),
                             "poster": str(poster.relative_to(self.root)),
                             "frames": w.n}
        self._writers.clear()
        return files

    def abort(self):
        for w in self._writers.values():
            try:
                w.close()
            finally:
                w.path.unlink(missing_ok=True)
        self._writers.clear()


def append_manifest(video_root: Path, entry: dict, run_info: dict):
    """Maintain videos/manifest.json (rewritten after every episode, so a
    crash never loses the entries of episodes that did finish)."""
    p = Path(video_root) / "manifest.json"
    data = json.loads(p.read_text()) if p.exists() else {"run": run_info, "episodes": []}
    data["run"] = run_info
    data["episodes"] = [e for e in data["episodes"] if e["episode"] != entry["episode"]] + [entry]
    data["episodes"].sort(key=lambda e: e["episode"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))