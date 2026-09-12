"""Build an India knowledge base from Wikidata and Wikipedia.

Why this is a knowledge BASE and not a training set
---------------------------------------------------
Measured earlier in this project: fine-tuning 22 facts moved validation loss from
1.6301 to 0.0680 - a 24x improvement - and still answered "what is Karnataka's
capital" wrong, because Karnataka was not one of the 22. Facts do not generalise
from the weights they were trained into.

The same corpus in a retrieval store answers Karnataka correctly, cites its
source, and can be corrected by editing one row. So this writes a store.

Two sources, deliberately:

  Wikidata  - structured (state, capital) pairs. Machine-readable, unambiguous,
              and the right source for "what is X's capital" style questions.
  Wikipedia - one-paragraph summaries, for context and for questions structure
              cannot answer. Every passage keeps its article URL so answers cite.

"Latest" is bounded by what these sources say today; both are live APIs, so
re-running this refreshes the store without retraining anything.

    python -m serve.build_india_kb --out india_kb.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List

UA = {"User-Agent": "training_for_practice/0.1 (educational RAG build; contact via GitHub)"}
WIKIDATA = "https://query.wikidata.org/sparql"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# Q12443800 = state of India, Q2072238 = union territory of India
SPARQL_ADMIN = """
SELECT ?place ?kindLabel ?placeLabel ?capitalLabel WHERE {
  VALUES ?kind { wd:Q12443800 wd:Q2072238 }
  ?place wdt:P31 ?kind ; wdt:P36 ?capital .
  FILTER NOT EXISTS { ?place wdt:P576 ?dissolved . }
  FILTER NOT EXISTS { ?capital wdt:P576 ?capDissolved . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""

# Two constraints here are load-bearing, and both were added after the first
# build produced documents that were plainly false:
#
#   P31 type list - the original query asked only for "anything in India with a
#   population", which admitted STATES and REGIONS. It produced "Karnataka is a
#   major city in India, India" and two copies of "Central National Capital
#   Region is a major city". A city query must require the thing to be a city.
#
#   no state join - resolving a city's state needs P131 transitive (or even
#   P131/P131?), and every variant of that returns 504 on the public endpoint.
#   Rather than guess, the city sentence simply omits the state. A shorter true
#   sentence beats a longer one that might name the wrong state.
SPARQL_BIG_CITIES = """
SELECT DISTINCT ?cityLabel ?pop WHERE {
  VALUES ?ctype {
    wd:Q515 wd:Q1549591 wd:Q1637706 wd:Q3957 wd:Q2039348 wd:Q1093829
    wd:Q174844 wd:Q200250 wd:Q11271835
  }
  ?city wdt:P31 ?ctype ; wdt:P17 wd:Q668 ; wdt:P1082 ?pop .
  FILTER(?pop > 900000)
  FILTER NOT EXISTS { ?city wdt:P576 ?d . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY DESC(?pop)
LIMIT 80
"""

# Districts are the layer people actually ask about ("which district is X in",
# "what is the headquarters of Y district") and the layer the state/capital query
# cannot reach. Unlike the city query this one CAN join to a state cheaply,
# because a district's P131 parent IS the state - no transitive walk, no 504.
SPARQL_DISTRICTS = """
SELECT DISTINCT ?districtLabel ?stateLabel ?hqLabel WHERE {
  ?district wdt:P31 wd:Q1149652 ; wdt:P131 ?state .
  ?state wdt:P31 ?skind . VALUES ?skind { wd:Q12443800 wd:Q2072238 }
  FILTER NOT EXISTS { ?district wdt:P576 ?d . }
  FILTER NOT EXISTS { ?state wdt:P576 ?sd . }
  OPTIONAL { ?district wdt:P36 ?hq . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
ORDER BY ?stateLabel ?districtLabel
"""

# Topics worth a prose summary: national context a news assistant is asked about.
SUMMARY_TOPICS = [
    "India", "Government_of_India", "States_and_union_territories_of_India",
    "Andhra_Pradesh", "Telangana", "Karnataka", "Tamil_Nadu", "Kerala",
    "Maharashtra", "Uttar_Pradesh", "West_Bengal", "Gujarat", "Rajasthan",
    "Madhya_Pradesh", "Bihar", "Punjab,_India", "Odisha", "Delhi",
    "Hyderabad", "Bengaluru", "Chennai", "Mumbai", "Kolkata", "Visakhapatnam",
    "Amaravati", "Indian_Parliament", "President_of_India", "Prime_Minister_of_India",
    "Economy_of_India", "Indian_Railways", "Geography_of_India",
    "Andhra_Pradesh_Reorganisation_Act,_2014",
]


def sparql(query: str) -> List[dict]:
    url = f"{WIKIDATA}?format=json&query={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["results"]["bindings"]


def wiki_summary(title: str) -> dict | None:
    req = urllib.request.Request(WIKI_SUMMARY + urllib.parse.quote(title), headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.load(r)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="india_kb.json")
    ap.add_argument("--skip-summaries", action="store_true")
    args = ap.parse_args()

    docs: Dict[str, dict] = {}

    # ---- structured: states, union territories and their capitals ----
    print("[1/4] Wikidata: states and union territories")
    caps: Dict[str, list] = {}
    kinds: Dict[str, str] = {}
    uris: Dict[str, str] = {}
    for row in sparql(SPARQL_ADMIN):
        place = row["placeLabel"]["value"]
        cap = row["capitalLabel"]["value"]
        kind = row["kindLabel"]["value"]
        if place.startswith("Q") or cap.startswith("Q"):
            continue          # unlabelled entity
        # A label like "Andhra Pradesh (1956-2014)" is a former entity whose
        # capital has since changed. Including it would make the store assert a
        # fact that is no longer true.
        if "(" in place and any(ch.isdigit() for ch in place):
            continue
        if place.endswith(" State") and place not in ("Telangana State",):
            continue
        kinds[place] = kind
        # Keep the entity URI. Special:EntityPage takes a Q-id, so building the
        # link from the LABEL produced dead URLs with raw spaces in them
        # ("Special:EntityPage/Andhra Pradesh") in a field meant for citation.
        uris.setdefault(place, row["place"]["value"])
        if cap not in caps.setdefault(place, []):
            caps[place].append(cap)

    multi = 0
    for place, cl in sorted(caps.items()):
        kind = kinds[place]
        kind_phrase = kind if "india" in kind.lower() else f"{kind} of India"
        key = f"adm_{len(docs):04d}"
        if len(cl) == 1:
            text = f"{cl[0]} is the capital of {place}, a {kind_phrase}."
        else:
            # Karnataka carries both Bengaluru and Belgaum at NormalRank with no
            # qualifier distinguishing them, so there is no field that says which
            # is primary. Emitting two "X is THE capital" documents would put a
            # direct contradiction in the store and let ranking pick the winner
            # by accident. One document, stating exactly what the source states.
            multi += 1
            joined = " and ".join(cl)
            text = (f"{joined} are both recorded as capitals of {place}, "
                    f"a {kind_phrase}.")
        docs[key] = {
            "text": text,
            "source": uris.get(place, "https://www.wikidata.org/"),
            "kind": "structured",
        }
    print(f"      {len(caps)} states/UTs with capitals ({multi} with more than one)")

    # ---- structured: large cities ----
    # Each stage is independent: a public SPARQL endpoint returning 504 should
    # cost us one section, not the whole build.
    print("[2/4] Wikidata: large cities")
    try:
        city_rows = sparql(SPARQL_BIG_CITIES)
    except Exception as e:
        print(f"      SKIPPED ({type(e).__name__}: {str(e)[:60]}) - keeping what we have")
        city_rows = []
    # A city with two P1082 statements (census year vs estimate) returns twice -
    # Vadodara came back at both 3,100,260 and 2,065,771. Keep the largest and
    # emit one document, or the store contradicts itself about the same city.
    best: Dict[str, int] = {}
    for row in city_rows:
        city = row["cityLabel"]["value"]
        if city.startswith("Q"):
            continue
        pop = int(float(row["pop"]["value"]))
        best[city] = max(best.get(city, 0), pop)
    for city, pop in sorted(best.items(), key=lambda kv: -kv[1]):
        key = f"cty_{len(docs):04d}"
        docs[key] = {
            "text": f"{city} is a major city in India, with a population of "
                    f"about {pop:,}.",
            "source": "https://www.wikidata.org/",
            "kind": "structured",
        }
    print(f"      {len(best)} cities (deduplicated from {len(city_rows)} rows)")

    # ---- structured: districts and their headquarters ----
    print("[3/4] Wikidata: districts")
    try:
        dist_rows = sparql(SPARQL_DISTRICTS)
    except Exception as e:
        print(f"      SKIPPED ({type(e).__name__}: {str(e)[:60]})")
        dist_rows = []

    # Same shape of problem as the state capitals: Annamayya district comes back
    # with two headquarters (Madanapalle and Rayachoti) and nothing distinguishes
    # them. Group first, decide once, never emit two contradicting sentences.
    dhq: Dict[tuple, list] = {}
    for row in dist_rows:
        name = row["districtLabel"]["value"]
        state = row["stateLabel"]["value"]
        if name.startswith("Q") or state.startswith("Q"):
            continue
        if "(" in name and any(ch.isdigit() for ch in name):
            continue
        name = " ".join(name.split())          # "YSR  Kadapa district" -> single spaces
        hq = row.get("hqLabel", {}).get("value")
        entry = dhq.setdefault((name, state), [])
        if hq and not hq.startswith("Q") and hq not in entry:
            entry.append(hq)

    n_hq = n_multi = 0
    for (name, state), hqs in sorted(dhq.items()):
        label = name if name.lower().endswith("district") else f"{name} district"
        key = f"dis_{len(docs):04d}"
        if len(hqs) == 1:
            text = (f"{hqs[0]} is the administrative headquarters of {label}, "
                    f"in {state}, India.")
            n_hq += 1
        elif len(hqs) > 1:
            n_multi += 1
            text = (f"{' and '.join(hqs)} are both recorded as the administrative "
                    f"headquarters of {label}, in {state}, India.")
        else:
            text = f"{label} is a district of {state}, India."
        docs[key] = {
            "text": text,
            "source": "https://www.wikidata.org/",
            "kind": "structured",
        }
    print(f"      {len(dhq)} districts across "
          f"{len({st for _, st in dhq})} states/UTs "
          f"({n_hq} with a headquarters, {n_multi} with more than one)")

    # ---- prose: Wikipedia summaries ----
    if not args.skip_summaries:
        print(f"[4/4] Wikipedia summaries for {len(SUMMARY_TOPICS)} topics")
        got = 0
        for t in SUMMARY_TOPICS:
            s = wiki_summary(t)
            time.sleep(0.15)                      # be polite to the API
            if not s or not s.get("extract"):
                print(f"      miss {t}")
                continue
            # First two sentences: enough to answer, short enough to cite cleanly.
            extract = s["extract"]
            parts = extract.split(". ")
            text = ". ".join(parts[:2]).strip()
            if not text.endswith("."):
                text += "."
            key = f"wik_{len(docs):04d}"
            docs[key] = {
                "text": text,
                "source": s.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "kind": "prose",
                "title": s.get("title", t),
            }
            got += 1
        print(f"      {got} summaries")

    out = Path(args.out)
    out.write_text(json.dumps(
        {"built": time.strftime("%Y-%m-%d"), "count": len(docs), "docs": docs},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(docs)} documents -> {out}")
    for k in list(docs)[:3]:
        print(f"  {k}: {docs[k]['text'][:88]}")


if __name__ == "__main__":
    main()
