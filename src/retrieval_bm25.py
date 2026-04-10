# ===============================
# Build BM25 index from LongEval Sci docs
# ===============================
import re
import math
from collections import Counter, defaultdict
from ir_datasets_longeval import load
from pathlib import Path

PACE_BASE = Path("/storage/project/ps-clef2026_longeval-0")
LOCAL_DATASET_ROOT = PACE_BASE / "temp"
LOCAL_QUERIES_PATH = LOCAL_DATASET_ROOT / "temp"
LOCAL_QRELS_PATH = PACE_BASE / "temp" / "temp"

# load dataset
def load_sci_dataset(use_local_path=True):
    dataset = load(str(LOCAL_DATASET_ROOT))
    queries_path = LOCAL_QUERIES_PATH
    qrels_path = LOCAL_QRELS_PATH
    return dataset, queries_path, qrels_path

dataset, queries_override_path, qrels_override_path = load_sci_dataset()

# normalization helpers

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text: str):
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())

def normalize_authors(authors):
    if not authors:
        return ""
    if isinstance(authors, (list, tuple)):
        out = []
        for a in authors:
            if isinstance(a, dict):
                out.append(a.get("name", ""))
            else:
                out.append(str(a))
        return " ".join(x for x in out if x).strip()
    return str(authors)

def normalize_doc(doc):
    published = (
        getattr(doc, "publishedDate", None)
        or getattr(doc, "createdDate", None)
        or getattr(doc, "updatedDate", None)
        or getattr(doc, "year", None)
    )
    return {
        "doc_id": str(getattr(doc, "doc_id")),
        "title": getattr(doc, "title", "") or "",
        "abstract": getattr(doc, "abstract", "") or "",
        "authors_text": normalize_authors(getattr(doc, "authors", None)),
        "publishedDate": published,
    }

def doc_to_text(doc_dict):
    return "\n".join(
        x for x in [
            doc_dict["title"],
            doc_dict["abstract"],
            doc_dict["authors_text"],
        ] if x
    )

def build_bm25_index(dataset, max_docs=None, k1=1.2, b=0.75):
    postings = defaultdict(list)   # term -> [(doc_idx, tf), ...]
    df = Counter()                 # term -> doc freq
    doc_len = []                   # doc_idx -> length
    doc_meta = []                  # doc_idx -> metadata
    doc_store = {}                 # doc_id -> full normalized doc

    doc_idx = 0
    for doc in dataset.docs_iter():
        if max_docs is not None and doc_idx >= max_docs:
            break

        d = normalize_doc(doc)
        text = doc_to_text(d)
        tokens = tokenize(text)
        if not tokens:
            continue

        doc_meta.append({
            "doc_id": d["doc_id"],
            "title": d["title"],
            "abstract": d["abstract"],   # important for dense rerank
            "publishedDate": d["publishedDate"],
        })
        doc_store[d["doc_id"]] = d
        doc_len.append(len(tokens))

        tf = Counter(tokens)
        for term, freq in tf.items():
            postings[term].append((doc_idx, freq))
        for term in tf.keys():
            df[term] += 1

        doc_idx += 1

    N = len(doc_len)
    avgdl = sum(doc_len) / max(1, N)
    idf = {
        term: math.log(1 + (N - dfi + 0.5) / (dfi + 0.5))
        for term, dfi in df.items()
    }
    params = {"k1": k1, "b": b, "N": N, "avgdl": avgdl}
    return postings, idf, doc_len, doc_meta, doc_store, params

# Set max_docs=None for full corpus. You can use a small value while debugging.
MAX_DOCS = 500000

postings, idf, doc_len, doc_meta, doc_store, bm25_params = build_bm25_index(
    dataset=dataset,
    max_docs=MAX_DOCS,
)

print("BM25 params:", bm25_params)
print("Docs indexed:", len(doc_meta))
print("Vocab size:", len(postings))
print("Example stored doc:", next(iter(doc_store.items()))[0] if doc_store else None)