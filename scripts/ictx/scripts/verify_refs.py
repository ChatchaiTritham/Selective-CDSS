import json, sys, urllib.request, urllib.parse, time
Q = [l.strip() for l in open(sys.argv[1], encoding="utf8") if l.strip()]
out = []
for q in Q:
    url = "https://api.crossref.org/works?rows=2&select=DOI,title,author,issued,container-title,volume,page,type&query.bibliographic=" + urllib.parse.quote(q)
    try:
        d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "refcheck (mailto:chatchai.t@mail.rmutk.ac.th)"}), timeout=30))
        items = d["message"]["items"]
    except Exception as e:
        items = []; print("ERR", q, e)
    for it in items[:1]:
        au = ", ".join((a.get("family") or "") for a in it.get("author", [])[:6])
        yr = it.get("issued", {}).get("date-parts", [[None]])[0][0]
        print(f"Q: {q}\n  -> {it.get('title',[''])[0]} | {au} | {yr} | {(it.get('container-title') or [''])[0]} | {it.get('volume')} | {it.get('page')} | {it['DOI']} | {it.get('type')}")
        out.append({"q": q, "doi": it["DOI"], "title": it.get("title", [""])[0], "year": yr})
    time.sleep(0.3)
json.dump(out, open(sys.argv[2], "w"), indent=1)
