"""
preprocess.py
=============
Tiền xử lý dữ liệu theo chuẩn SASRec paper:
  - 5-core filter cả user lẫn item (chuẩn SASRec paper)
  - Sắp xếp theo thứ tự thời gian
  - Leave-one-out split:
      test:  item cuối cùng  (item_N)      — positive để đánh giá
      valid: item áp chót    (item_{N-1})  — positive để chọn model
      train: item kế tiếp    (item_{N-2})  — positive để huấn luyện

Supported datasets: beauty, sports, ml-1m, steam

Xử lý Cold-start users:
  5-core đảm bảo mỗi user có >= 5 tương tác trong raw data.
  Tuy nhiên sau khi bỏ duplicate liên tiếp và lọc, một số user
  có thể còn lại ít hơn CONTEXT_SIZE + 3 items trong sequence.

  MIN_SEQ_LEN = 3: giữ lại những user còn >= 3 items sau dedup,
  ngay cả khi không đủ CONTEXT_SIZE items để làm query đầy đủ.

  Cold-start context sẽ ngắn hơn CONTEXT_SIZE, dùng tất cả item có sẵn:

    seq = [i1, i2, i3]  (N=3)
      train → context=[],       positive=i1  ← không có history
      valid → context=[i1],     positive=i2
      test  → context=[i1, i2], positive=i3

    seq = [i1, i2, i3, i4]  (N=4)
      train → context=[i1],          positive=i2
      valid → context=[i1, i2],      positive=i3
      test  → context=[i1, i2, i3],  positive=i4

    seq = [i1..i7]  (N >= CONTEXT_SIZE+3 = 6, đủ history)
      train → context=[i3, i4],      positive=i5  (capped CONTEXT_SIZE)
      valid → context=[i4, i5],      positive=i6
      test  → context=[i5, i6],      positive=i7

Lưu ý về context:
  CONTEXT_SIZE chỉ dùng cho Tevatron query (k item gần nhất).
  RecBole/SASRec dùng TOÀN BỘ history và tự học attention.
"""

import ast
import json
import os
import gzip
import pandas as pd
from collections import defaultdict
from tqdm import tqdm


# ── Config ──────────────────────────────────────────────────────────────────

DATASETS = {
    "beauty": {
        "type":        "amazon",
        "review_file": "raw/reviews_Beauty_5.json.gz",
        "meta_file":   "raw/meta_Beauty.json.gz",
        "name":        "Amazon Beauty",
    },
    "sports": {
        "type":        "amazon",
        "review_file": "raw/reviews_Sports_and_Outdoors_5.json.gz",
        "meta_file":   "raw/meta_Sports_and_Outdoors.json.gz",
        "name":        "Amazon Sports",
    },
    "ml-1m": {
        "type":     "movielens",
        "data_dir": "raw/ml-1m/",
        "name":     "MovieLens-1M",
    },
    "steam": {
        "type":        "steam",
        "review_file": "raw/steam_reviews.json.gz",
        "meta_file":   "raw/steam_games.json.gz",
        "name":        "Steam",
    },
}

MIN_INTERACTIONS = 5   # 5-core filter (CHỈ áp dụng cho item, không lọc user)
CONTEXT_SIZE     = 3   # số item gần nhất dùng làm query trong Tevatron
MIN_SEQ_LEN      = 4   # tối thiểu 3 item để tạo được 3 split (train/valid/test)


# ── Helpers ─────────────────────────────────────────────────────────────────

def parse_line(line: str) -> dict:
    """Parse một dòng dữ liệu Amazon — hỗ trợ cả JSON (2018) lẫn Python dict (2014)."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return ast.literal_eval(line)


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_amazon(review_file: str, meta_file: str):
    """Load Amazon dataset từ raw gzip files.
    Hỗ trợ cả hai format:
      - JSON (dấu nháy kép)  — Amazon 2018
      - Python dict (dấu nháy đơn) — Amazon 2014
    """
    print(f"  Loading reviews : {review_file}")
    interactions = []
    with gzip.open(review_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = parse_line(line)
                interactions.append({
                    "user_id":   d["reviewerID"],
                    "item_id":   d["asin"],
                    "timestamp": int(d.get("unixReviewTime", 0)),
                })
            except Exception:
                continue

    print(f"  Loading metadata: {meta_file}")
    item_meta = {}
    with gzip.open(meta_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = parse_line(line)
                asin  = d.get("asin", "")
                title = d.get("title", "")
                if asin and title:
                    # categories là list of lists trong Amazon 2014
                    cats = d.get("categories", [])
                    if cats and isinstance(cats[0], list):
                        cat_str = " ".join(cat for sub in cats for cat in sub)
                    else:
                        cat_str = " ".join(str(c) for c in cats)

                    item_meta[asin] = {
                        "title":       str(title),
                        "brand":       str(d.get("brand", "") or ""),
                        "categories":  cat_str,
                        "description": str(d.get("description") or "")[:200],
                    }
            except Exception:
                continue

    return pd.DataFrame(interactions), item_meta


def load_movielens(data_dir: str):
    """Load MovieLens 1M dataset."""
    print(f"  Loading MovieLens from: {data_dir}")
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"),
        sep="::", header=None, engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["item_id"] = ratings["item_id"].astype(str)

    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"),
        sep="::", header=None, engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )
    movies["item_id"] = movies["item_id"].astype(str)

    item_meta = {
        str(row["item_id"]): {
            "title":       row["title"],
            "brand":       "",
            "categories":  row["genres"].replace("|", " "),
            "description": "",
        }
        for _, row in movies.iterrows()
    }

    return ratings[["user_id", "item_id", "timestamp"]], item_meta


def load_steam(review_file: str, meta_file: str):
    """Load Steam dataset."""
    print(f"  Loading Steam reviews : {review_file}")
    interactions = []
    with gzip.open(review_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                interactions.append({
                    "user_id":   str(d.get("username", d.get("user_id", ""))),
                    "item_id":   str(d.get("product_id", "")),
                    "timestamp": int(d.get("date", 0)),
                })
            except Exception:
                continue

    item_meta = {}
    with gzip.open(meta_file, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                item_id = str(d.get("id", ""))
                if item_id:
                    item_meta[item_id] = {
                        "title":       d.get("app_name", ""),
                        "brand":       d.get("developer", ""),
                        "categories":  " ".join(d.get("genres", [])),
                        "description": (d.get("desc_snippet") or "")[:200],
                    }
            except Exception:
                continue

    return pd.DataFrame(interactions), item_meta


# ── Core processing ──────────────────────────────────────────────────────────

def apply_kcore(df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    """
    Lọc lặp đến khi tất cả user VÀ item đều có >= k tương tác (chuẩn SASRec paper).
    Sau khi lọc, một số user vẫn có thể có ít item trong sequence nếu
    MIN_SEQ_LEN < k — đây là các cold-start users được giữ lại.
    """
    print(f"  Applying {k}-core filter (users + items)...")
    prev_size = 0
    iteration = 0
    while len(df) != prev_size:
        prev_size = len(df)
        iteration += 1
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= k].index)]
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= k].index)]
    print(
        f"  After {iteration} iterations: "
        f"{df['user_id'].nunique()} users, "
        f"{df['item_id'].nunique()} items, "
        f"{len(df)} interactions"
    )
    return df


def build_sequences(df: pd.DataFrame) -> dict:
    """
    Build chuỗi tương tác theo thứ tự thời gian cho mỗi user.
    Bỏ duplicate liên tiếp và lọc chuỗi quá ngắn.
    """
    df = df.sort_values(["user_id", "timestamp"])
    sequences = {}
    for user_id, group in df.groupby("user_id"):
        items = group["item_id"].tolist()
        # Bỏ duplicate liên tiếp
        deduped = [items[0]]
        for item in items[1:]:
            if item != deduped[-1]:
                deduped.append(item)
        if len(deduped) >= MIN_SEQ_LEN:
            sequences[user_id] = deduped
    return sequences


def split_sequences(sequences: dict) -> dict:
    """
    Leave-one-out split theo SASRec paper với hỗ trợ cold-start.

    Với seq = [i1, i2, ..., i_N] (1-indexed):
      test  → positive = i_N,     context = [i_{N-CONTEXT_SIZE}..i_{N-1}]
      valid → positive = i_{N-1}, context = [i_{N-CONTEXT_SIZE-1}..i_{N-2}]
      train → positive = i_{N-2}, context = [i_{N-CONTEXT_SIZE-2}..i_{N-3}]

    Cold-start: nếu không đủ CONTEXT_SIZE item trước positive,
    dùng tất cả item có sẵn (context ngắn hơn CONTEXT_SIZE).

    Ví dụ cold-start N=3: [i1, i2, i3]
      train → context=[],       positive=i1
      valid → context=[i1],     positive=i2
      test  → context=[i1, i2], positive=i3

    Ví dụ đủ history N=7, CONTEXT_SIZE=3: [i1..i7]
      train → context=[i3, i4], positive=i5  (capped)
      valid → context=[i4, i5], positive=i6
      test  → context=[i5, i6], positive=i7

    Fields:
      tevatron_context : list[item_id] — dùng cho Tevatron query
      positive         : item_id       — item cần predict
      full_seq         : list[item_id] — toàn bộ history (dùng cho RecBole .inter)
    """
    splits = {"train": {}, "valid": {}, "test": {}}

    for user_id, seq in sequences.items():
        N = len(seq)  # N >= MIN_SEQ_LEN = 3

        # ── Positive items (1-indexed: item_N, item_{N-1}, item_{N-2}) ──────
        test_pos  = seq[N - 1]   # item_N   — last
        valid_pos = seq[N - 2]   # item_{N-1}
        train_pos = seq[N - 3]   # item_{N-2}

        # ── Context: tất cả item TRƯỚC positive, capped tới CONTEXT_SIZE ────
        # test  context = items trước item_N     = seq[:N-1][-CONTEXT_SIZE:]
        # valid context = items trước item_{N-1} = seq[:N-2][-CONTEXT_SIZE:]
        # train context = items trước item_{N-2} = seq[:N-3][-CONTEXT_SIZE:]
        test_ctx  = seq[:N - 1][-CONTEXT_SIZE:]   # có thể ngắn nếu cold-start
        valid_ctx = seq[:N - 2][-CONTEXT_SIZE:]
        train_ctx = seq[:N - 3][-CONTEXT_SIZE:]   # = [] nếu N == 3

        splits["test"][user_id] = {
            "tevatron_context": test_ctx,
            "positive":         test_pos,
            "full_seq":         seq,           # full sequence cho RecBole
        }
        splits["valid"][user_id] = {
            "tevatron_context": valid_ctx,
            "positive":         valid_pos,
            "full_seq":         seq[:N - 1],
        }
        splits["train"][user_id] = {
            "tevatron_context": train_ctx,
            "positive":         train_pos,
            "full_seq":         seq[:N - 2],
        }

    return splits


def build_negative_pool(sequences: dict, all_items: list) -> dict:
    """
    Không pre-compute toàn bộ negatives (tốn RAM).
    Chỉ trả về all_items_set để export_tevatron sample on-the-fly.
    """
    return set(all_items)


def build_item_text(item_id: str, item_meta: dict) -> str:
    """
    Tạo text mô tả item từ metadata.
    Chỉ dùng title của item.
    """
    meta = item_meta.get(item_id, {})
    title = meta.get("title") or ""
    if isinstance(title, list):
        title = " ".join(str(t) for t in title)
    title = str(title).strip()
    return title if title else item_id


# ── Main entry point ─────────────────────────────────────────────────────────

def preprocess(dataset_name: str):
    """
    Pipeline chính. Trả về:
      splits             – dict {train/valid/test → {user_id → data}}
      all_items          – list of item_id sau filter
      item_meta          – dict {item_id → {title, brand, categories, description}}
      negatives_per_user – dict {user_id → [negative item_ids]}
      sequences          – dict {user_id → [item_ids theo thứ tự thời gian]}
    """
    cfg = DATASETS.get(dataset_name)
    if cfg is None:
        raise ValueError(
            f"Dataset '{dataset_name}' không được hỗ trợ. "
            f"Chọn trong: {list(DATASETS.keys())}"
        )

    print(f"\n{'=' * 60}")
    print(f"Processing: {cfg['name']}")
    print(f"{'=' * 60}")

    # Load raw data
    if cfg["type"] == "amazon":
        df, item_meta = load_amazon(cfg["review_file"], cfg["meta_file"])
    elif cfg["type"] == "movielens":
        df, item_meta = load_movielens(cfg["data_dir"])
    elif cfg["type"] == "steam":
        df, item_meta = load_steam(cfg["review_file"], cfg["meta_file"])
    else:
        raise ValueError(f"Unknown dataset type: {cfg['type']}")

    print(
        f"  Raw: {df['user_id'].nunique()} users, "
        f"{df['item_id'].nunique()} items, "
        f"{len(df)} interactions"
    )

    # 5-core filter
    df = apply_kcore(df, k=MIN_INTERACTIONS)

    # Build sequences
    sequences = build_sequences(df)
    print(f"  Valid sequences (len >= {MIN_SEQ_LEN}): {len(sequences)} "
          f"(dropped {df['user_id'].nunique() - len(sequences)} users with < {MIN_SEQ_LEN} items)")

    if not sequences:
        raise RuntimeError(
            "Không có sequence nào hợp lệ sau khi filter. "
            "Kiểm tra lại MIN_SEQ_LEN hoặc MIN_INTERACTIONS."
        )

    # Split
    splits = split_sequences(sequences)

    # All items còn lại sau filter
    all_items = df["item_id"].unique().tolist()

    # Negative pool
    negatives_per_user = build_negative_pool(sequences, all_items)

    # Thống kê
    seq_lens = [len(s) for s in sequences.values()]
    cold_start = sum(1 for l in seq_lens if l < CONTEXT_SIZE + 3)
    print(f"\n  {'─' * 40}")
    print(f"  Train queries : {len(splits['train'])}")
    print(f"  Valid queries : {len(splits['valid'])}")
    print(f"  Test  queries : {len(splits['test'])}")
    print(f"  Corpus items  : {len(all_items)}")
    print(f"  Seq len       : avg={sum(seq_lens)/len(seq_lens):.1f}, "
          f"min={min(seq_lens)}, max={max(seq_lens)}")
    print(f"  Cold-start    : {cold_start} users (seq < {CONTEXT_SIZE + 3} items, "
          f"{100*cold_start/len(seq_lens):.1f}%)")
    print(f"    len=3 : {sum(1 for l in seq_lens if l == 3)} users (context=[] for train)")
    print(f"    len=4 : {sum(1 for l in seq_lens if l == 4)} users (context=[1] for train)")
    print(f"    len=5 : {sum(1 for l in seq_lens if l == 5)} users (context=[2] for train)")
    print(f"  {'─' * 40}")

    return splits, all_items, item_meta, negatives_per_user, sequences


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "beauty"
    preprocess(dataset)
    print("\nPreprocessing OK.")