"""Regenerate the packaged SONICOM hash registry for supported files."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from urllib.request import Request, urlopen

DATABASE_URL = "https://ecosystem.sonicom.eu/databases/76/download?type=json"
OUTPUT_PATH = Path("src/irdl/registry/sonicom_hashes.json")
DATASET_PREFIX = "hutubs"


def _manifest_entries() -> list[tuple[str, str]]:
    request = Request(DATABASE_URL, headers={"User-Agent": "irdl-hash-generator"})  # noqa: S310
    with urlopen(request, timeout=120) as response:  # noqa: S310
        payload = json.load(response)

    entries: list[tuple[str, str]] = []
    for item in payload["data"]:
        file_name = item["Datafile Name"]
        if not file_name.endswith(".sofa"):
            continue
        path_key = f"{DATASET_PREFIX}/{file_name}"
        entries.append((path_key, item["Datafile URL"]))
    return entries


def _hash_static_file(path_key: str, file_url: str) -> tuple[str, str]:
    request = Request(file_url, headers={"User-Agent": "irdl-hash-generator"})  # noqa: S310
    digest = sha256()
    with urlopen(request, timeout=120) as response:  # noqa: S310
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
    return path_key, f"sha256:{digest.hexdigest()}"


def main() -> None:
    """Regenerate the flat provider-relative-path hash registry."""
    jobs = _manifest_entries()
    registry: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {executor.submit(_hash_static_file, path_key, file_url): path_key for path_key, file_url in jobs}
        for future in as_completed(futures):
            path_key, digest = future.result()
            registry[path_key] = digest
            print(f"hashed {path_key}")  # noqa: T201

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dict(sorted(registry.items())), indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")  # noqa: T201


if __name__ == "__main__":
    main()
