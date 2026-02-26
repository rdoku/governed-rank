"""Category-level NMF moment discovery and back-mapping to products."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_instacart_catalog(data_dir: str) -> Tuple[List[int], Dict[int, dict]]:
    """Load Instacart catalog with aisle/department names."""
    data_path = Path(data_dir)

    aisles: Dict[int, str] = {}
    with (data_path / "aisles.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aisles[int(row["aisle_id"])] = row["aisle"]

    departments: Dict[int, str] = {}
    with (data_path / "departments.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            departments[int(row["department_id"])] = row["department"]

    products: Dict[int, dict] = {}
    product_ids: List[int] = []
    with (data_path / "products.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["product_id"])
            product_ids.append(pid)
            products[pid] = {
                "product_id": pid,
                "name": row["product_name"],
                "aisle": aisles.get(int(row["aisle_id"]), "unknown"),
                "department": departments.get(int(row["department_id"]), "unknown"),
            }
    product_ids.sort()
    return product_ids, products


def build_token_index(
    products: Dict[int, dict],
    token_mode: str = "hybrid",
) -> Tuple[Dict[str, int], List[str]]:
    """Build token index for departments/aisles."""
    token_mode = token_mode.lower()
    token_set = set()
    for info in products.values():
        if token_mode in ("dept", "hybrid"):
            token_set.add(f"dept:{info['department']}")
        if token_mode in ("aisle", "hybrid"):
            token_set.add(f"aisle:{info['aisle']}")
    token_list = sorted(token_set)
    token_index = {token: idx for idx, token in enumerate(token_list)}
    return token_index, token_list


def build_product_tokens(
    product_ids: List[int],
    products: Dict[int, dict],
    token_index: Dict[str, int],
    token_mode: str = "hybrid",
) -> Dict[int, List[int]]:
    """Map product_ids to token indices."""
    token_mode = token_mode.lower()
    product_tokens: Dict[int, List[int]] = {}
    for pid in product_ids:
        info = products.get(pid)
        if not info:
            continue
        tokens: List[int] = []
        if token_mode in ("dept", "hybrid"):
            tokens.append(token_index[f"dept:{info['department']}"])
        if token_mode in ("aisle", "hybrid"):
            tokens.append(token_index[f"aisle:{info['aisle']}"])
        product_tokens[pid] = tokens
    return product_tokens


def build_order_token_matrix(
    data_dir: str,
    product_tokens: Dict[int, List[int]],
    token_index: Dict[str, int],
    max_orders: Optional[int] = None,
    min_basket_size: int = 2,
    binary_token_counts: bool = True,
) -> Tuple["csr_matrix", List[int], int]:
    """Build order x token CSR matrix from Instacart data."""
    from scipy.sparse import csr_matrix

    data_path = Path(data_dir)
    order_ids: List[int] = []
    with (data_path / "orders.csv").open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            order_ids.append(int(row["order_id"]))
            if max_orders and len(order_ids) >= max_orders:
                break

    order_id_to_row = {oid: idx for idx, oid in enumerate(order_ids)}
    target_orders = len(order_ids)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []

    def flush_order(row_idx: Optional[int], counter: Counter, item_count: int) -> None:
        if row_idx is None or item_count < min_basket_size:
            return
        for col_idx, count in counter.items():
            rows.append(row_idx)
            cols.append(col_idx)
            vals.append(1.0 if binary_token_counts else float(count))

    order_products_total = 0
    processed_orders = 0
    stop_early = False
    for filename in ["order_products__prior.csv", "order_products__train.csv"]:
        path = data_path / filename
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            current_order_id: Optional[int] = None
            current_row_idx: Optional[int] = None
            token_counts: Counter = Counter()
            item_count = 0

            for row in reader:
                order_id = int(row["order_id"])
                if order_id != current_order_id:
                    if current_order_id is not None:
                        flush_order(current_row_idx, token_counts, item_count)
                        if current_row_idx is not None:
                            processed_orders += 1
                            if max_orders and processed_orders >= target_orders:
                                stop_early = True
                                break
                    current_order_id = order_id
                    current_row_idx = order_id_to_row.get(order_id)
                    token_counts = Counter()
                    item_count = 0

                if current_row_idx is None:
                    continue

                pid = int(row["product_id"])
                tokens = product_tokens.get(pid)
                if not tokens:
                    continue
                item_count += 1
                order_products_total += 1
                for token_idx in tokens:
                    token_counts[token_idx] += 1

            if not stop_early:
                flush_order(current_row_idx, token_counts, item_count)
                if current_row_idx is not None:
                    processed_orders += 1

        if stop_early or (max_orders and processed_orders >= target_orders):
            break

    matrix = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(order_ids), len(token_index)),
        dtype=np.float32,
    )
    return matrix, order_ids, order_products_total


def fit_nmf(
    matrix: "csr_matrix",
    k: int,
    init: str = "nndsvda",
    max_iter: int = 800,
    random_state: int = 42,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit NMF to an order x token matrix."""
    from sklearn.decomposition import NMF

    nmf = NMF(
        n_components=k,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        tol=tol,
        l1_ratio=0.0,
    )
    W = nmf.fit_transform(matrix)
    H = nmf.components_
    return W, H


def backmap_moment2vec(
    H: np.ndarray,
    product_ids: List[int],
    product_tokens: Dict[int, List[int]],
    smoothing: float = 1e-3,
) -> np.ndarray:
    """Back-map token moments to products and normalize."""
    k = H.shape[0]
    moment2vec = np.zeros((len(product_ids), k), dtype=np.float32)
    for idx, pid in enumerate(product_ids):
        tokens = product_tokens.get(pid)
        if not tokens:
            moment2vec[idx] = 1.0 / k
            continue
        weights = H[:, tokens].mean(axis=1)
        moment2vec[idx] = weights

    if smoothing > 0:
        moment2vec += float(smoothing)

    row_sums = moment2vec.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-12)
    moment2vec = moment2vec / row_sums
    return moment2vec


def build_moment_meta(
    moment2vec: np.ndarray,
    H: np.ndarray,
    token_list: List[str],
    product_ids: List[int],
    products: Dict[int, dict],
    top_n_items: int = 10,
    top_n_tokens: int = 5,
) -> List[dict]:
    """Build moment metadata from token weights and item affinities."""
    def format_label(name: str) -> str:
        name = name.replace("_", " ").strip()
        if not name:
            return name
        return name[0].upper() + name[1:]

    moments = []
    k = H.shape[0]

    token_names = token_list

    for m in range(k):
        token_weights = H[m]
        token_order = np.argsort(token_weights)[-top_n_tokens:][::-1]

        dept_weights: Dict[str, float] = {}
        aisle_weights: Dict[str, float] = {}
        for idx in token_order:
            token = token_names[idx]
            weight = float(token_weights[idx])
            if token.startswith("dept:"):
                name = token.split("dept:", 1)[1]
                dept_weights[name] = dept_weights.get(name, 0.0) + weight
            elif token.startswith("aisle:"):
                name = token.split("aisle:", 1)[1]
                aisle_weights[name] = aisle_weights.get(name, 0.0) + weight

        top_depts = sorted(dept_weights.items(), key=lambda x: x[1], reverse=True)[:3]
        top_aisles = sorted(aisle_weights.items(), key=lambda x: x[1], reverse=True)[:3]

        if top_depts and top_aisles:
            if top_depts[0][1] >= 1.2 * top_aisles[0][1]:
                label_source = top_depts[0][0]
            else:
                label_source = top_aisles[0][0]
        elif top_depts:
            label_source = top_depts[0][0]
        elif top_aisles:
            label_source = top_aisles[0][0]
        else:
            label_source = f"Moment {m + 1}"
        auto_label = format_label(label_source)

        top_indices = np.argsort(moment2vec[:, m])[-top_n_items:][::-1]
        top_items = [product_ids[idx] for idx in top_indices]

        keywords = [name for name, _ in top_aisles] + [name for name, _ in top_depts]
        moments.append({
            "id": m,
            "auto_label": auto_label,
            "keywords": keywords[:5],
            "top_items": top_items,
            "top_categories": [name for name, _ in top_depts],
            "time_signature": "Unknown",
        })

    return moments
