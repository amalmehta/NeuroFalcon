"""Download FALCON M2 (DANDI 000953) as slim NWB files.

Each M2 file is 96% raw 30 kHz voltage (acquisition/fullband, ~2.3 GB per
held-in run) that the FALCON harness never reads: its M2 loader uses only
units/spike_times, finger_vel, eval_mask and trials. We stream each file over
HTTP, drop `fullband`, and export the rest with pynwb, so the whole M2
benchmark fits in ~100 MB instead of 15.9 GB.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import remfile
import requests
import h5py
from pynwb import NWBHDF5IO

DANDISET = "000953"
API = "https://api.dandiarchive.org/api"
SPLITS = ("held-in-calib", "held-out-calib", "held-in-minival")
DROP = ("fullband",)
DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def list_assets() -> list[dict]:
    url = f"{API}/dandisets/{DANDISET}/versions/draft/assets/"
    out, params = [], {"page_size": 200}
    while url:
        r = requests.get(url, params=params, timeout=60)
        r.raise_for_status()
        page = r.json()
        out += page["results"]
        url, params = page["next"], None
    return out


def local_path(dandi_path: str, root: Path = DATA_ROOT) -> Path:
    # Mirror DANDI layout: data/000953/sub-MonkeyN-held-in-calib/<file>.nwb
    # (the harness finds M2 via DANDI_FOLDER_MAP['m2'] == '000953').
    return root / DANDISET / dandi_path


def slim_copy(asset: dict, dest: Path) -> None:
    url = requests.head(f"{API}/assets/{asset['asset_id']}/download/",
                        allow_redirects=True, timeout=60).url
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".partial.nwb")
    with h5py.File(remfile.File(url), "r") as h5, NWBHDF5IO(file=h5, mode="r", load_namespaces=True) as src:
        nwb = src.read()
        for name in DROP:
            if name in nwb.acquisition:
                nwb.acquisition.pop(name)
        nwb.set_modified()
        with NWBHDF5IO(str(tmp), mode="w") as out:
            # link_data=False: copy the bytes; the default writes external links back to the remote file.
            out.export(src_io=src, nwbfile=nwb, write_args={"link_data": False})
    tmp.rename(dest)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Download slim FALCON M2 NWB files.")
    p.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    p.add_argument("--root", type=Path, default=DATA_ROOT)
    args = p.parse_args(argv)
    assets = [a for a in list_assets() if any(f"-{s}/" in a["path"] for s in args.splits)]
    for a in sorted(assets, key=lambda a: a["path"]):
        dest = local_path(a["path"], args.root)
        if dest.exists():
            print(f"have  {dest.name}")
            continue
        print(f"fetch {a['path']} ({a['size'] / 1e6:.0f} MB on DANDI)", flush=True)
        slim_copy(a, dest)
        print(f"  -> {dest.stat().st_size / 1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
