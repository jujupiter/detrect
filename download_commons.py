"""Download the Wikimedia Commons test photos listed in data/commons_*/sources.csv.

The photos themselves are not stored in git. Each sources.csv records, per photo: its title and page on Commons,
licence and author (needed to credit it), the thumbnail that was used, and checksums to verify the download.

Every photo is fetched from the first source that works:
  1. the recorded thumbnail URL
  2. the Commons API, by title. This follows renames, and if the file has been re-uploaded since,
     looks for the version that was used in its history.
  3. the Internet Archive's copy of the recorded thumbnail URL (if it was saved there, see --archive)
If none works, the photo is reported as missing and skipped: benchmark.py runs on the photos that are present.
A photo that downloads but does not match the recorded checksum is kept, with a warning.

usage: python download_commons.py             download missing or changed photos
       python download_commons.py --record    fill in the URL and checksum columns for newly added rows
       python download_commons.py --archive   ask the Internet Archive to keep a copy of every thumbnail
"""
import argparse
import hashlib
import io
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
from PIL import Image

FOLDERS = ["data/commons_studio", "data/commons_outdoor"]
# Wikimedia refuses requests whose user agent has no contact information
USER_AGENT = "detrect/0.1 (https://github.com/jujupiter/detrect)"
API = "https://commons.wikimedia.org/w/api.php"
THUMB_WIDTH = 640  # Commons answers with its nearest standard size (960 px wide), or the original if smaller
WAYBACK = "https://web.archive.org/web/2026id_/"  # id_ = the archived bytes, unmodified


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=60).read()


def file_info(title, history=False):
    """imageinfo of a Commons file: its current version, or every version with history=True."""
    params = dict(action="query", titles="File:" + title, prop="imageinfo", iiprop="url|sha1", iiurlwidth=THUMB_WIDTH,
                  redirects=1, format="json", iilimit=50 if history else 1)
    page = next(iter(json.loads(fetch(API + "?" + urllib.parse.urlencode(params)))["query"]["pages"].values()))
    return page.get("imageinfo", [])


def strip_query(url):
    return url.split("?")[0]


def process(data, processing):
    """`original`: the thumbnail as served. `white_jpeg`: transparency flattened onto white, saved as JPEG."""
    if processing == "original":
        return data
    im = Image.open(io.BytesIO(data))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, "white")
        bg.alpha_composite(im)
        im = bg
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=92)
    return buf.getvalue()


def sources(row):
    """(name, function returning the thumbnail bytes), in order of preference."""
    def api():
        versions = file_info(row.title, history=True)
        match = next((v for v in versions if v["sha1"] == row.commons_sha1), None)
        if match is None:
            raise LookupError("the recorded version is not in the file's history")
        if "thumburl" not in match:
            raise LookupError("no thumbnail for the recorded version")
        return fetch(match["thumburl"])

    return [("recorded URL", lambda: fetch(row.image_url)),
            ("Commons API", api),
            ("Internet Archive", lambda: fetch(WAYBACK + row.image_url))]


def download(row):
    """Returns (status, message). status: ok, changed or missing."""
    errors = []
    for name, get in sources(row):
        try:
            data = process(get(), row.processing)
            Image.open(io.BytesIO(data)).verify()  # an error page is not an image
        except Exception as e:
            errors.append(f"{name}: {e}")
            continue
        Path(row.path).parent.mkdir(parents=True, exist_ok=True)
        Path(row.path).write_bytes(data)
        if hashlib.sha256(data).hexdigest() != row.sha256:
            return "changed", f"from {name}, but it differs from the recorded file"
        return "ok", f"from {name}"
    return "missing", "; ".join(errors)


def record(df):
    """Fill image_url, commons_sha1 and sha256 for rows where they are empty, from the local photos."""
    for i, row in df[df.sha256.isna()].iterrows():
        info = file_info(row.title)[0]
        thumb = fetch(info["thumburl"])
        local = Path(row.path).read_bytes()
        processing = "original" if thumb == local else "white_jpeg"
        if process(thumb, processing) != local:
            sys.exit(f"{row.path}: the local photo does not match its Commons thumbnail")
        df.loc[i, ["image_url", "commons_sha1", "sha256", "processing"]] = [
            strip_query(info["thumburl"]), info["sha1"], hashlib.sha256(local).hexdigest(), processing]
        print("recorded", row.path)
        time.sleep(0.5)
    return df


def archive(df):
    for url in df.image_url:
        try:
            fetch("https://web.archive.org/save/" + url)
            print("archived", url)
        except Exception as e:
            print("failed", url, e)
        time.sleep(5)  # the Save Page Now service is rate-limited


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--record", action="store_true", help="fill in URL and checksums for new rows")
    mode.add_argument("--archive", action="store_true", help="save every thumbnail to the Internet Archive")
    args = parser.parse_args()

    missing = []
    for folder in FOLDERS:
        csv = Path(folder, "sources.csv")
        df = pd.read_csv(csv)
        for col in ["image_url", "commons_sha1", "sha256", "processing"]:
            if col not in df:
                df[col] = None
        if args.record:
            record(df).to_csv(csv, index=False)
            continue
        if args.archive:
            archive(df)
            continue
        for row in df.itertuples():
            path = Path(row.path)
            if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256:
                continue
            status, message = download(row)
            print(f"{status:<8} {row.path}  {message}")
            if status == "missing":
                missing.append(row.path)
            time.sleep(0.5)

    if missing:
        print(f"\n{len(missing)} photo(s) could not be downloaded; benchmark.py will skip them:")
        print("\n".join("  " + p for p in missing))
    elif not (args.record or args.archive):
        print("All Commons photos present.")


if __name__ == "__main__":
    main()
