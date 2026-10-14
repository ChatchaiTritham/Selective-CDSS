"""Download PhysioNet/CinC Challenge 2019 v1.0.0 training sets A and B from the
official PhysioNet AWS Open Data mirror (s3://physionet-open), verify each file
against its S3 ETag (MD5 for single-part objects) and write a SHA-256 manifest.

Usage: python -I download_data.py <out_dir>
"""
import hashlib, json, sys, re, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "challenge-2019/1.0.0/"


def list_keys(prefix):
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        xml = urllib.request.urlopen(BUCKET + "/?" + urllib.parse.urlencode(q), timeout=120).read().decode()
        for k, e, s in re.findall(r"<Key>([^<]+)</Key>.*?<ETag>&quot;([0-9a-f-]+)&quot;</ETag><Size>(\d+)</Size>", xml):
            keys.append((k, e, int(s)))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def fetch(item, out):
    key, etag, size = item
    dest = out / key[len(PREFIX):]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and hashlib.md5(dest.read_bytes()).hexdigest() == etag:
        data = dest.read_bytes()
    else:
        for _ in range(5):
            try:
                data = urllib.request.urlopen(BUCKET + "/" + urllib.parse.quote(key), timeout=120).read()
                break
            except Exception:
                continue
        else:
            raise RuntimeError("failed " + key)
        if "-" not in etag and hashlib.md5(data).hexdigest() != etag:
            raise RuntimeError("md5 mismatch " + key)
        dest.write_bytes(data)
    return key[len(PREFIX):], hashlib.sha256(data).hexdigest(), len(data)


def main():
    out = Path(sys.argv[1])
    items = []
    for s in ("A", "B"):
        items += list_keys(PREFIX + f"training/training_set{s}/")
    items += [k for k in list_keys(PREFIX) if k[0].endswith(("LICENSE.txt", "SHA256SUMS.txt"))]
    with ThreadPoolExecutor(32) as ex:
        rows = sorted(ex.map(lambda i: fetch(i, out), items))
    agg = hashlib.sha256("".join(f"{p} {h}\n" for p, h, _ in rows).encode()).hexdigest()
    man = {"source": BUCKET + "/" + PREFIX, "dataset": "PhysioNet/CinC Challenge 2019, v1.0.0",
           "license": "CC BY 4.0",
           "n_setA": sum(p.startswith("training/training_setA/") for p, _, _ in rows),
           "n_setB": sum(p.startswith("training/training_setB/") for p, _, _ in rows),
           "total_bytes": sum(n for _, _, n in rows), "manifest_sha256": agg}
    (out / "MANIFEST_sha256.txt").write_text("".join(f"{h}  {p}\n" for p, h, _ in rows))
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=2))
    print(json.dumps(man))


if __name__ == "__main__":
    main()
