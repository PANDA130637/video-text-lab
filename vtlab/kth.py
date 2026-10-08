"""Download a small official KTH subset using HTTP ranges, preserving subject splits."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
import zipfile

import requests

BASE = "https://www.csc.kth.se/cvap/actions/"
ACTIONS = {"boxing": "A person is boxing.", "handclapping": "A person is clapping their hands.",
           "handwaving": "A person is waving their hands."}
SUBJECT_SPLITS = {"train": [11, 12, 13, 14, 15, 16, 17, 18],
                  "val": [19, 20, 21, 23, 24, 25, 1, 4],
                  "test": [22, 2, 3, 5, 6, 7, 8, 9, 10]}


class HTTPRangeFile(io.RawIOBase):
    """Read-only seekable ZIP input; every response must honor the requested range."""
    def __init__(self, url):
        super().__init__()
        self.url, self.position, self.downloaded = url, 0, 0
        self.session = requests.Session()
        response = self.session.head(url, timeout=30, allow_redirects=True)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.memo = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError("Negative ZIP offset")
        self.position = position
        return position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if size <= 0:
            return b""
        start, end = self.position, self.position + size - 1
        key = (start, end)
        if key not in self.memo:
            for attempt in range(3):
                try:
                    with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}",
                                         "Accept-Encoding": "identity"}, timeout=(15, 45), stream=True) as response:
                        response.raise_for_status()
                        expected = f"bytes {start}-{end}/{self.size}"
                        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                            raise RuntimeError("Server did not honor byte range; full download was not attempted.")
                        payload = response.content
                        if len(payload) != size:
                            raise IOError("Truncated range response")
                        self.downloaded += len(payload)
                        self.memo[key] = payload
                        break
                except requests.RequestException:
                    if attempt == 2:
                        raise
                    time.sleep(1 + attempt)
        payload = self.memo.pop(key) if size > 1024 * 1024 else self.memo[key]
        self.position += len(payload)
        return payload

    def close(self):
        self.session.close()
        super().close()


def record_for(filename: str, action: str) -> dict:
    match = re.fullmatch(r"person(\d{2})_([a-z]+)_d1_uncomp\.avi", filename)
    if not match or match[2] != action:
        raise ValueError(f"Unexpected KTH subset filename: {filename}")
    subject = int(match[1])
    split = next((name for name, people in SUBJECT_SPLITS.items() if subject in people), None)
    if split is None:
        raise ValueError(f"Unexpected subject: {subject}")
    return dict(id=filename[:-4], group_id=f"person{subject:02d}", split=split,
                video_path=f"videos/{filename}", text=ACTIONS[action], semantic_id=action,
                pair_id="", source="KTH_official_d1_three_actions")


def _download_action(action: str, output: Path) -> tuple[list, dict]:
    records = []
    url = BASE + action + ".zip"
    receipt_path = output / f"{action}.receipt.json"
    old = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    checksums = dict(old.get("sha256", {}))
    print(f"Inspecting official archive: {action}", flush=True)
    with HTTPRangeFile(url) as remote, zipfile.ZipFile(remote) as archive:
        selected = sorted((entry for entry in archive.infolist()
                           if re.fullmatch(rf"person\d{{2}}_{action}_d1_uncomp\.avi", Path(entry.filename).name)),
                          key=lambda entry: entry.filename)
        if len(selected) != 25:
            raise ValueError(f"Expected 25 d1 videos in {action}; found {len(selected)}")
        for index, entry in enumerate(selected):
            filename = Path(entry.filename).name
            destination = output / "videos" / filename
            if entry.file_size > 32 * 1024 * 1024 or entry.compress_size > 32 * 1024 * 1024:
                raise ValueError(f"Unexpectedly large subset video: {filename}")
            verified = False
            if destination.exists() and filename in old.get("sha256", {}):
                digest = hashlib.sha256(destination.read_bytes()).hexdigest()
                verified = digest == old["sha256"][filename]
            if destination.exists() and not verified:
                raise FileExistsError(f"Existing video has no matching receipt; choose a new output directory: {destination}")
            if not verified:
                # Read a complete local ZIP entry in one range, then use zipfile
                # to check decompression and CRC; no archive paths are extracted.
                remote.seek(entry.header_offset)
                header = remote.read(30)
                name_size, extra_size = struct.unpack_from("<HH", header, 26)
                remote.seek(entry.header_offset)
                raw = remote.read(30 + name_size + extra_size + entry.compress_size)
                local = zipfile.ZipInfo(filename=entry.filename)
                local.compress_type, local.CRC = entry.compress_type, entry.CRC
                local.file_size, local.compress_size = entry.file_size, entry.compress_size
                local.flag_bits, local.header_offset = entry.flag_bits, 0
                # ZipExtFile uses the already parsed info and a local data stream.
                with zipfile.ZipExtFile(io.BytesIO(raw[30 + name_size + extra_size:]), "r", local) as stream:
                    payload = stream.read()
                if len(payload) != entry.file_size:
                    raise IOError(f"Invalid video length: {filename}")
                temporary = destination.with_suffix(".part")
                temporary.write_bytes(payload)
                temporary.replace(destination)
                digest = hashlib.sha256(payload).hexdigest()
            checksums[filename] = digest
            records.append(record_for(filename, action))
            if (index + 1) % 5 == 0:
                print(f"{action}: {index + 1}/25 videos", flush=True)
            receipt_path.write_text(json.dumps({"url": url, "sha256": checksums}, indent=2), encoding="utf-8")
        return records, {"url": url, "downloaded_bytes": remote.downloaded,
                         "archive_bytes": remote.size, "sha256": checksums}


def prepare_kth(output: Path) -> dict:
    """Three classes, d1 only; labels are template descriptions, not free captions."""
    output = Path(output).resolve()
    (output / "videos").mkdir(parents=True, exist_ok=True)
    response = requests.get(BASE + "00sequences.txt", timeout=30)
    response.raise_for_status()
    (output / "official_sequences.txt").write_text(response.text, encoding="utf-8")
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda action: _download_action(action, output), ACTIONS))
    records = sorted([row for rows, _ in results for row in rows], key=lambda row: row["id"])
    manifest = output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    provenance = dict(dataset="KTH d1 three-action subset", source=BASE,
                      license="Publicly available for non-commercial use; cite Schuldt, Laptev, Caputo ICPR 2004.",
                      protocol="Official person split; only d1 of boxing, handclapping, handwaving; whole AVI clips.",
                      captions="Template descriptions derived from action labels; no order-pair annotations.",
                      videos=len(records), subject_splits=SUBJECT_SPLITS,
                      downloads=[receipt for _, receipt in results])
    (output / "source.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    from .data import validate_manifest
    validate_manifest(manifest)
    return {"manifest": str(manifest), "videos": len(records),
            "downloaded_mb": sum(r[1]["downloaded_bytes"] for r in results) / 1_000_000,
            "source": BASE}
