"""Collect candidate photos from Wikimedia Commons for a large test set, to be checked by hand with review_commons.py.

For each row of data/commons_queries.csv (a J-code and a Commons category), list the photos in the category
(and its subcategories, down to `depth`), keep those whose licence allows reuse, and download a thumbnail.
Rows without a J-code are general litter categories: their photos are labelled from scratch during review.

Writes data/commons_candidates/candidates.csv (one row per photo, with the suggested class and J-code) and the
thumbnails in data/commons_candidates/img/ (not in git). Safe to run again: photos already collected are kept.

usage: python collect_commons.py               collect up to max_files photos per query row
       python collect_commons.py --dry-run     only count what each category would give
"""
import argparse
import hashlib
import html
import io
import json
import re
import time
import urllib.parse
from pathlib import Path

import pandas as pd
from PIL import Image

from download_commons import API, THUMB_WIDTH, fetch, strip_query

OUT = Path("data/commons_candidates")
QUERIES = Path("data/commons_queries.csv")
JLIST = Path("data/jlist.csv")
EXISTING = ["data/commons_studio/sources.csv", "data/commons_outdoor/sources.csv"]

# Licences whose terms allow any use with credit. Commons hosts free files only, but GFDL-only files
# are awkward to credit and non-free logos can slip in, so anything else is left out.
FREE = re.compile(r"^(CC0|Public domain|PD\b|CC BY(-SA)? \d|CC BY(-SA)?$|Attribution$)", re.I)
SKIP_TITLE = re.compile(r"logo|icon|diagram|map\b|poster|advert|label|chart|drawing|illustration|sign\b", re.I)
MIN_SIDE = 400  # px, original size

# Suggested trashnet class for a J-code, consistent with TACO_CLASSES in prepare_extra_data.py: foam, film,
# wrappers, cigarettes, hygiene items, textiles, rubber, wood and food are "trash"; paper cups, drink cartons
# and cigarette packets are ambiguous and get no class.
TRASH_PLASTIC = {"J25", "J27", "J30", "J40", "J41", "J49", "J58", "J82", "J83", "J95", "J96", "J98", "J101", "J144",
                 "J211", "J224", "J226", "J236", "J237", "J239", "J240", "J242", "J252", "J253", "J256", "J257"}
CARDBOARD = {"J148", "J245", "J247"}
NO_CLASS = {"J150", "J151", "J152", "J244"}
MATERIAL_CLASS = {"plastic": "plastic", "metal": "metal", "glass/ceramics": "glass", "paper": "paper"}


def suggested_label(jcodes, materials):
    if not jcodes:
        return ""
    labels = set()
    for code in jcodes.split():
        if code in NO_CLASS:
            return ""
        if code in TRASH_PLASTIC:
            labels.add("trash")
        elif code in CARDBOARD:
            labels.add("cardboard")
        else:
            labels.add(MATERIAL_CLASS.get(materials[code], "trash"))
    return labels.pop() if len(labels) == 1 else ""


def api(**params):
    params.update(format="json")
    return json.loads(fetch(API + "?" + urllib.parse.urlencode(params)))


def category_files(category, depth):
    """{page id: title} of the files in a category and its subcategories, down to `depth` levels."""
    files, seen, todo = {}, set(), [("Category:" + category, 0)]
    while todo:
        cat, level = todo.pop(0)
        if cat in seen:
            continue
        seen.add(cat)
        params = dict(action="query", list="categorymembers", cmtitle=cat, cmtype="file|subcat", cmlimit=500)
        while True:
            r = api(**params)
            for m in r.get("query", {}).get("categorymembers", []):
                if m["ns"] == 6:
                    files[m["pageid"]] = m["title"][5:]
                elif m["ns"] == 14 and level < depth:
                    todo.append((m["title"], level + 1))
            if "continue" not in r:
                break
            params.update(r["continue"])
            time.sleep(0.2)
        time.sleep(0.2)
    return files


def plain(text):
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def file_infos(ids):
    """imageinfo for up to 50 page ids: thumbnail URL, size, type, sha1 and licence metadata."""
    r = api(action="query", pageids="|".join(map(str, ids)), prop="imageinfo",
            iiprop="url|size|mime|sha1|extmetadata", iiurlwidth=THUMB_WIDTH,
            iiextmetadatafilter="LicenseShortName|Artist|Credit")
    for page in r.get("query", {}).get("pages", {}).values():
        if not page.get("imageinfo"):
            continue
        info = page["imageinfo"][0]
        meta = info.get("extmetadata", {})
        yield dict(id=page["pageid"], title=page["title"][5:], mime=info["mime"], width=info["width"],
                   height=info["height"], license=plain(meta.get("LicenseShortName", {}).get("value")),
                   artist=plain(meta.get("Artist", {}).get("value")) or plain(meta.get("Credit", {}).get("value")),
                   source=info["descriptionurl"], image_url=strip_query(info.get("thumburl", info["url"])),
                   commons_sha1=info["sha1"])


def usable(info):
    if info["mime"] != "image/jpeg" or min(info["width"], info["height"]) < MIN_SIDE:
        return False
    return bool(FREE.match(info["license"])) and not SKIP_TITLE.search(info["title"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="only count the usable photos per category")
    args = parser.parse_args()

    jlist = pd.read_csv(JLIST)
    materials = dict(zip(jlist.jcode, jlist.material))
    queries = pd.read_csv(QUERIES, dtype=str).fillna("")
    csv = OUT / "candidates.csv"
    old = pd.read_csv(csv, dtype=str).fillna("") if csv.exists() else pd.DataFrame(columns=["title"])
    taken = set(old.title)
    for f in EXISTING:  # photos already in the small test sets
        taken |= set(pd.read_csv(f).title)

    rows = []
    for q in queries.itertuples():
        files = {i: t for i, t in category_files(q.category, int(q.depth)).items() if t not in taken}
        # a fixed pseudo-random order, so that a large category is not sampled alphabetically
        ids = sorted(files, key=lambda i: hashlib.md5(files[i].encode()).hexdigest())
        infos = []
        for i in range(0, len(ids), 50):
            infos += [x for x in file_infos(ids[i:i + 50]) if usable(x)]
            if len(infos) >= int(q.max_files):
                break
            time.sleep(0.2)
        infos = sorted(infos, key=lambda x: hashlib.md5(x["title"].encode()).hexdigest())[:int(q.max_files)]
        print(f"{q.jcode or '(any)':<10} {q.category:<55} {len(files):>5} files, {len(infos):>3} kept")
        for x in infos:
            taken.add(x["title"])
            rows.append(dict(x, query_category=q.category, suggested_jcode=q.jcode,
                             suggested_label=suggested_label(q.jcode, materials)))

    if args.dry_run:
        print(f"\n{len(rows)} new candidates (dry run, nothing downloaded)")
        return

    (OUT / "img").mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    ok = []
    for i, x in enumerate(new.itertuples()):
        path = OUT / "img" / f"{x.id}.jpg"
        if not path.exists():
            try:
                data = fetch(x.image_url)
                Image.open(io.BytesIO(data)).verify()
                path.write_bytes(data)
            except Exception as e:
                print("skip", x.title, e)
                ok.append(False)
                continue
            time.sleep(0.3)
        ok.append(True)
        if (i + 1) % 100 == 0:
            print(f"downloaded {i + 1}/{len(new)}")
    if len(new):
        new = new[ok].drop(columns=["mime"])
        new["path"] = [str(OUT / "img" / f"{i}.jpg") for i in new.id]
    out = pd.concat([old, new.astype(str)], ignore_index=True) if len(old) else new
    out.to_csv(csv, index=False)
    print(f"\n{len(new)} new candidates, {len(out)} in total, in {csv}")


if __name__ == "__main__":
    main()
