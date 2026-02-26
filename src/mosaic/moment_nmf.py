"""NMF-based moment discovery from basket co-occurrence data.

This module discovers shopping "moments" (missions/intents) from real order data
using Non-negative Matrix Factorization. Key insight: items that frequently
appear together in the same basket share an underlying shopping mission.

Approach:
1. Build item co-occurrence matrix from baskets
2. Apply NMF to discover K latent topics (moments)
3. Each product gets a distribution over moments (moment2vec)
4. Auto-label moments based on top products in each topic
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def load_instacart_baskets(
    data_dir: str,
    max_orders: Optional[int] = None,
    min_basket_size: int = 2,
) -> Tuple[List[List[int]], Dict[int, dict], int]:
    """Load Instacart order data as baskets.

    Args:
        data_dir: Path to instacart_market_analysis directory
        max_orders: Maximum orders to load (None for all)
        min_basket_size: Minimum items per basket (skip smaller)

    Returns:
        baskets: List of baskets, each basket is list of product_ids
        products: Dict mapping product_id to product info
        num_products: Total number of unique products
    """
    import csv
    from collections import defaultdict

    data_path = Path(data_dir)

    # Load products
    products = {}
    with open(data_path / "products.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["product_id"])
            products[pid] = {
                "product_id": pid,
                "name": row["product_name"],
                "aisle_id": int(row["aisle_id"]),
                "department_id": int(row["department_id"]),
            }

    # Load aisles
    aisles = {}
    with open(data_path / "aisles.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            aisles[int(row["aisle_id"])] = row["aisle"]

    # Load departments
    departments = {}
    with open(data_path / "departments.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            departments[int(row["department_id"])] = row["department"]

    # Enrich products with aisle/department names
    for pid, p in products.items():
        p["aisle"] = aisles.get(p["aisle_id"], "unknown")
        p["department"] = departments.get(p["department_id"], "unknown")

    # Load order-product mappings
    order_items: Dict[int, List[int]] = defaultdict(list)
    orders_loaded = 0

    for filename in ["order_products__prior.csv", "order_products__train.csv"]:
        filepath = data_path / filename
        if not filepath.exists():
            continue

        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                order_id = int(row["order_id"])
                product_id = int(row["product_id"])
                order_items[order_id].append(product_id)

                if max_orders and len(order_items) >= max_orders:
                    break

            if max_orders and len(order_items) >= max_orders:
                break

    # Filter to baskets with min size
    baskets = [items for items in order_items.values() if len(items) >= min_basket_size]

    logger.info(f"Loaded {len(baskets)} baskets from {len(order_items)} orders, {len(products)} products")

    return baskets, products, len(products)


def build_cooccurrence_matrix(
    baskets: List[List[int]],
    num_products: int,
    product_id_to_idx: Dict[int, int],
) -> np.ndarray:
    """Build item co-occurrence matrix from baskets.

    Uses basket-level co-occurrence: two items co-occur if they appear
    in the same basket. Returns a symmetric matrix.

    Args:
        baskets: List of baskets (each basket is list of product_ids)
        num_products: Number of unique products
        product_id_to_idx: Mapping from product_id to matrix index

    Returns:
        Co-occurrence matrix of shape (num_products, num_products)
    """
    from scipy.sparse import lil_matrix

    cooc = lil_matrix((num_products, num_products), dtype=np.float32)

    for basket in baskets:
        # Get indices for products in this basket
        indices = []
        for pid in basket:
            if pid in product_id_to_idx:
                indices.append(product_id_to_idx[pid])

        # Add co-occurrence for all pairs
        for i, idx1 in enumerate(indices):
            for idx2 in indices[i:]:
                cooc[idx1, idx2] += 1
                if idx1 != idx2:
                    cooc[idx2, idx1] += 1

    return cooc.tocsr()


def discover_moments_nmf(
    baskets: List[List[int]],
    products: Dict[int, dict],
    k: int = 8,
    max_iter: int = 300,
    random_state: int = 42,
    init: str = "random",
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, List[int], Dict[int, int]]:
    """Discover moments using NMF on item co-occurrence.

    Args:
        baskets: List of baskets
        products: Product metadata
        k: Number of moments to discover
        max_iter: Maximum NMF iterations
        random_state: Random seed
        init: NMF initialization method
        tol: NMF tolerance

    Returns:
        moment2vec: (num_products, k) - product distribution over moments
        H: (k, num_products) - moment-item weights (for labeling)
        idx_to_product_id: Mapping from matrix index to product_id
        product_id_to_idx: Mapping from product_id to matrix index
    """
    from sklearn.decomposition import NMF

    # Build product ID mappings
    all_product_ids = sorted(products.keys())
    product_id_to_idx = {pid: idx for idx, pid in enumerate(all_product_ids)}
    idx_to_product_id = {idx: pid for pid, idx in product_id_to_idx.items()}
    num_products = len(all_product_ids)

    logger.info(f"Building co-occurrence matrix for {num_products} products...")
    cooc = build_cooccurrence_matrix(baskets, num_products, product_id_to_idx)

    logger.info(f"Running NMF with k={k}...")
    nmf = NMF(
        n_components=k,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
        tol=tol,
        l1_ratio=0.0,
    )

    # W: (num_products, k) - product loadings on moments
    # H: (k, num_products) - moment weights for products
    W = nmf.fit_transform(cooc)
    H = nmf.components_

    # Normalize W to get probability distribution (moment2vec)
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)  # Avoid division by zero
    moment2vec = W / row_sums

    # Fill zero rows with uniform distribution (products not in any basket)
    zero_rows = np.where(row_sums.flatten() < 1e-9)[0]
    if len(zero_rows) > 0:
        moment2vec[zero_rows] = 1.0 / k
        logger.info(f"Filled {len(zero_rows)} products with no basket data using uniform prior")

    logger.info(f"NMF reconstruction error: {nmf.reconstruction_err_:.4f}")

    return moment2vec, H, list(idx_to_product_id.values()), product_id_to_idx


def auto_label_moments(
    H: np.ndarray,
    products: Dict[int, dict],
    idx_to_product_id: List[int],
    top_n: int = 10,
) -> List[dict]:
    """Auto-generate moment labels from top products in each topic.

    Args:
        H: (k, num_products) moment-item weights
        products: Product metadata
        idx_to_product_id: Mapping from index to product_id
        top_n: Number of top products to consider per moment

    Returns:
        List of moment label dicts with name, top_products, top_aisles, top_departments
    """
    from collections import Counter

    k = H.shape[0]
    labels = []

    for moment_idx in range(k):
        weights = H[moment_idx]
        top_indices = np.argsort(weights)[-top_n:][::-1]

        top_products_info = []
        aisle_counter = Counter()
        dept_counter = Counter()

        for idx in top_indices:
            pid = idx_to_product_id[idx]
            if pid in products:
                p = products[pid]
                top_products_info.append({
                    "product_id": pid,
                    "name": p["name"],
                    "weight": float(weights[idx]),
                })
                aisle_counter[p["aisle"]] += 1
                dept_counter[p["department"]] += 1

        # Generate label from top aisles
        top_aisles = [a for a, _ in aisle_counter.most_common(3)]
        top_depts = [d for d, _ in dept_counter.most_common(2)]

        # Simple heuristic label
        if top_aisles:
            label_name = top_aisles[0].replace("_", " ").title()
        else:
            label_name = f"Moment {moment_idx + 1}"

        labels.append({
            "moment_id": moment_idx,
            "label": label_name,
            "top_aisles": top_aisles,
            "top_departments": top_depts,
            "top_products": top_products_info[:5],
        })

    return labels


def compute_moment2vec_stats(moment2vec: np.ndarray, threshold: float = 0.35) -> dict:
    """Compute statistics on moment2vec distribution quality.

    Args:
        moment2vec: (num_products, k) distribution matrix
        threshold: Confidence threshold for "peaked" distribution

    Returns:
        Dict with stats: max_weight, mean_max, pct_above_threshold, entropy stats
    """
    max_weights = moment2vec.max(axis=1)

    # Entropy per product
    eps = 1e-10
    entropy = -np.sum(moment2vec * np.log(moment2vec + eps), axis=1)
    max_entropy = np.log(moment2vec.shape[1])
    normalized_entropy = entropy / max_entropy

    # Margin (top1 - top2)
    sorted_weights = np.sort(moment2vec, axis=1)
    margins = sorted_weights[:, -1] - sorted_weights[:, -2]

    return {
        "num_products": moment2vec.shape[0],
        "num_moments": moment2vec.shape[1],
        "max_weight": {
            "min": float(max_weights.min()),
            "mean": float(max_weights.mean()),
            "max": float(max_weights.max()),
            "std": float(max_weights.std()),
        },
        "pct_above_threshold": float((max_weights >= threshold).mean() * 100),
        "entropy": {
            "mean": float(normalized_entropy.mean()),
            "std": float(normalized_entropy.std()),
        },
        "margin": {
            "mean": float(margins.mean()),
            "std": float(margins.std()),
        },
    }


def build_and_save_moment2vec(
    data_dir: str,
    output_dir: str,
    k: int = 8,
    max_orders: Optional[int] = None,
    min_basket_size: int = 2,
) -> dict:
    """Full pipeline: load data, discover moments, save artifacts.

    Args:
        data_dir: Path to instacart_market_analysis
        output_dir: Where to save moment2vec.npy, labels.json, etc.
        k: Number of moments
        max_orders: Limit orders for testing
        min_basket_size: Minimum basket size

    Returns:
        Summary dict with stats and file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    baskets, products, num_products = load_instacart_baskets(
        data_dir, max_orders=max_orders, min_basket_size=min_basket_size
    )

    # Discover moments
    moment2vec, H, idx_to_product_id, product_id_to_idx = discover_moments_nmf(
        baskets, products, k=k
    )

    # Auto-label
    labels = auto_label_moments(H, products, idx_to_product_id)

    # Compute stats
    stats = compute_moment2vec_stats(moment2vec)

    # Compute checksum
    checksum = hashlib.sha256(moment2vec.tobytes()).hexdigest()[:16]

    # Save artifacts
    np.save(output_path / "moment2vec.npy", moment2vec)
    np.save(output_path / "H_components.npy", H)

    with open(output_path / "product_id_mapping.json", "w") as f:
        json.dump({
            "idx_to_product_id": idx_to_product_id,
            "product_id_to_idx": product_id_to_idx,
        }, f)

    with open(output_path / "moment_labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    meta = {
        "k": k,
        "num_products": moment2vec.shape[0],
        "num_baskets": len(baskets),
        "checksum": checksum,
        "stats": stats,
        "labels": [l["label"] for l in labels],
    }
    with open(output_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved moment2vec artifacts to {output_path}")
    logger.info(f"Stats: {stats['pct_above_threshold']:.1f}% products above {0.35} threshold")

    return {
        "output_dir": str(output_path),
        "checksum": checksum,
        "stats": stats,
        "labels": labels,
    }


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    max_orders = int(sys.argv[4]) if len(sys.argv) > 4 else None

    result = build_and_save_moment2vec(
        data_dir=data_dir,
        output_dir=output_dir,
        k=k,
        max_orders=max_orders,
    )

    print("\n" + "="*60)
    print("MOMENT DISCOVERY RESULTS")
    print("="*60)
    print(f"\nDiscovered {k} moments from {result['stats']['num_products']} products")
    print(f"\nConfidence stats:")
    print(f"  Max weight: mean={result['stats']['max_weight']['mean']:.3f}, max={result['stats']['max_weight']['max']:.3f}")
    print(f"  % above 0.35 threshold: {result['stats']['pct_above_threshold']:.1f}%")
    print(f"  Margin (top1-top2): mean={result['stats']['margin']['mean']:.3f}")
    print(f"\nMoment labels:")
    for label in result['labels']:
        print(f"  {label['moment_id']}: {label['label']}")
        print(f"      Top aisles: {', '.join(label['top_aisles'][:3])}")
