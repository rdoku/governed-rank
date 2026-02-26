"""Patterna platform integration loaders.

This module provides model/data loading functions used when MOSAIC is deployed
within the full PatternaAi server stack. It depends on the ``patterna`` and
``server`` packages which are **not** required for standalone governed-rank
usage. If those packages are absent the guarded imports gracefully degrade.
"""
import os, json, hashlib
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import torch

from .state import S, ROOT
from ..utils import l2n

from patterna.model import PatternaUnified
from patterna.config import PatternaConfig

# These modules live in the server package, not in mosaic.
# Import them at call site or guard with try/except.
try:
    from server.core.market import MomentPriceController, MomentMarketConfig
    from server.core.shop_assets import get_shop_assets
except ImportError:
    MomentPriceController = None
    MomentMarketConfig = None
    get_shop_assets = None


def load_arrays():
    if S.item2vec is None:
        S.item2vec = l2n(np.load(os.environ.get('PATTERNA_ITEM2VEC', os.path.join(ROOT, 'models', 'item2vec.npy'))).astype(np.float32))
    if S.moment2vec is None:
        A = np.load(os.environ.get('PATTERNA_MOMENT2VEC', os.path.join(ROOT, 'models', 'lightgcn_moment2vec.npy'))).astype(np.float32)
        S.moment2vec = A / (A.sum(axis=1, keepdims=True) + 1e-9)
        try:
            S.moment2vec_checksum = hashlib.sha256(S.moment2vec.tobytes()).hexdigest()
        except Exception:
            S.moment2vec_checksum = None


def _compute_moment_space_id(model: Optional[PatternaUnified]) -> str:
    model_version = os.environ.get('PATTERNA_MODEL_VERSION', 'unknown')
    k = int(S.moment2vec.shape[1]) if S.moment2vec is not None else 0
    checksum = S.moment2vec_checksum or 'none'
    head_version = type(model.moments).__name__ if model is not None and hasattr(model, 'moments') else 'unknown'
    payload = f"{model_version}|{k}|{checksum}|{head_version}"
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _moment_artifact_dir(shop_id: Optional[str]) -> str:
    space_id = S.moment_space_id or 'default'
    if not shop_id:
        return os.path.join(ROOT, 'artifacts', space_id)
    return os.path.join(ROOT, 'artifacts', shop_id, space_id)


def _moment_labels_path(shop_id: Optional[str]) -> str:
    if not shop_id:
        return S.moment_labels_path
    return os.path.join(_moment_artifact_dir(shop_id), 'moment_labels.json')


def _moment_meta_path(shop_id: Optional[str]) -> str:
    return os.path.join(_moment_artifact_dir(shop_id), 'moment_meta.json')


def _moment_policies_path(shop_id: Optional[str]) -> str:
    return os.path.join(_moment_artifact_dir(shop_id), 'moment_policies.json')


def load_model():
    if S.model is not None:
        return
    load_arrays()
    num_items = int(max(S.item2vec.shape[0], S.moment2vec.shape[0]))
    S.num_items = num_items
    S.persona_start = num_items + 64
    cfg = PatternaConfig(
        vocab_size=max(num_items + 200, S.persona_start + 100),
        embedding_dim=64,
        context_dim=32,
        num_personas=12,
        hist_len_norm=9,
        use_item_features=True,
        num_depts=100,
        num_aisles=200,
        return_aux_info=True,
        use_practical_time=True, use_grocery_moments=True, use_social_influence=True,
        use_advanced_context=True, advanced_context_weight=0.35,
        num_latent_moments=S.moment2vec.shape[1],
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PatternaUnified(cfg, num_items=num_items, persona_start=S.persona_start).to(device)
    ckpt_path = os.path.join(ROOT, 'checkpoints', 'last_model.pt')
    ckpt_path = os.environ.get('PATTERNA_CHECKPOINT', ckpt_path)
    if os.path.exists(ckpt_path):
        try:
            sd = torch.load(ckpt_path, map_location=device)
            if isinstance(sd, dict) and 'state_dict' in sd:
                sd = sd['state_dict']
            model.load_state_dict(sd, strict=False)
        except Exception as e:
            print('[warn] failed to load checkpoint:', e)
    try:
        if getattr(cfg, 'use_item_features', True) and hasattr(model, 'item_features'):
            itf = model.item_features
            n_brand = int(getattr(cfg, 'n_brand_buckets', 0))
            n_size = int(getattr(cfg, 'n_size_buckets', 0))
            item2dept = (torch.arange(num_items, dtype=torch.long) % max(1, getattr(cfg, 'num_depts', 100))).to(device)
            item2aisle = (torch.arange(num_items, dtype=torch.long) % max(1, getattr(cfg, 'num_aisles', 200))).to(device)
            item2brand = (torch.arange(num_items, dtype=torch.long) % max(1, n_brand)).to(device) if n_brand > 0 else torch.zeros(num_items, dtype=torch.long, device=device)
            item2size = (torch.arange(num_items, dtype=torch.long) % max(1, n_size)).to(device) if n_size > 0 else torch.zeros(num_items, dtype=torch.long, device=device)
            try:
                itf.load_mappings(item2dept=item2dept, item2aisle=item2aisle, item2brand=item2brand, item2size=item2size, dense_blocks=None, graph_matrix=None, device=device)
            except TypeError:
                try:
                    itf.load_mappings(item2dept=item2dept, item2aisle=item2aisle, item2brand=item2brand, item2size=item2size, dense_blocks=None, graph_matrix=None)
                except Exception:
                    pass
    except Exception as e:
        print('[warn] item feature mapping load failed:', e)
    S.model, S.device = model, device
    if S.moment_space_id is None:
        try:
            S.moment_space_id = _compute_moment_space_id(model)
        except Exception:
            S.moment_space_id = None
    if S.market_enable and S.market is None:
        try:
            S.market = MomentPriceController(K=S.moment2vec.shape[1], cfg=MomentMarketConfig())
            print('[market] controller initialized with K=', S.moment2vec.shape[1])
        except Exception as e:
            print('[warn] market init failed:', e)
    if S.catalog is None:
        load_catalog()
    if S.moment_labels is None:
        load_moment_labels()
    try:
        os.makedirs(S.audit_dir, exist_ok=True)
    except Exception:
        pass


def load_catalog():
    try:
        import csv
        # Use Instacart-style files by default; adapt as needed
        inst = os.path.join(ROOT, 'Instacart')
        prod_path = os.path.join(inst, 'products.csv')
        aisle_path = os.path.join(inst, 'aisles.csv')
        dept_path = os.path.join(inst, 'departments.csv')
        if not (os.path.exists(prod_path) and os.path.exists(aisle_path) and os.path.exists(dept_path)):
            # Try data/catalog_mapped.csv
            alt = os.path.join(ROOT, 'data', 'catalog_mapped.csv')
            if os.path.exists(alt):
                cat = {}
                with open(alt, 'r', encoding='utf-8') as f:
                    for r in csv.DictReader(f):
                        try:
                            i = int(r.get('model_item_id') or r.get('item_id'))
                            cat[i] = { 'name': r.get('name',''), 'aisle': r.get('aisle',''), 'dept': r.get('dept','') }
                        except Exception:
                            continue
                S.catalog = cat
                print(f"[catalog] loaded {len(cat)} products (alt)")
            return
        aisles = {}
        with open(aisle_path, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                try: aisles[int(r['aisle_id'])] = r.get('aisle','')
                except: pass
        depts = {}
        with open(dept_path, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                try: depts[int(r['department_id'])] = r.get('department','')
                except: pass
        cat = {}
        with open(prod_path, 'r', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                try:
                    i = int(r['product_id'])
                    name = r.get('product_name', f'Item {i}')
                    a = aisles.get(int(r.get('aisle_id', '0') or '0'), '')
                    d = depts.get(int(r.get('department_id', '0') or '0'), '')
                    cat[i] = {'name': name, 'aisle': a, 'dept': d}
                except Exception:
                    continue
        S.catalog = cat
        print(f"[catalog] loaded {len(cat)} products")
    except Exception as e:
        print('[warn] catalog load failed:', e)


def load_moment_labels(shop_id: Optional[str] = None) -> Dict[int, str]:
    path = _moment_labels_path(shop_id)
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                labels = {int(k): str(v) for k, v in raw.items()}
        else:
            labels = {}
    except Exception as e:
        print('[warn] moment labels load failed:', e)
        labels = {}

    if shop_id:
        S.moment_labels_by_shop[shop_id] = labels
    else:
        S.moment_labels = labels
    return labels


def save_moment_labels(shop_id: Optional[str] = None):
    path = _moment_labels_path(shop_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        labels = S.moment_labels_by_shop.get(shop_id, {}) if shop_id else (S.moment_labels or {})
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({int(k): v for k, v in labels.items()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print('[warn] moment labels save failed:', e)


def get_moment_labels(shop_id: Optional[str] = None) -> Dict[int, str]:
    if shop_id:
        cached = S.moment_labels_by_shop.get(shop_id)
        if cached is not None:
            return cached
        return load_moment_labels(shop_id)
    if S.moment_labels is None:
        return load_moment_labels()
    return S.moment_labels


def load_moment_meta(shop_id: Optional[str] = None) -> Dict[int, Dict]:
    if shop_id and shop_id in S.moment_meta_by_shop:
        return S.moment_meta_by_shop[shop_id]
    path = _moment_meta_path(shop_id)
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        else:
            meta = {}
    except Exception as e:
        print('[warn] moment meta load failed:', e)
        meta = {}
    if shop_id:
        S.moment_meta_by_shop[shop_id] = meta
    return meta


def load_moment_policies(shop_id: Optional[str] = None) -> Dict[int, Dict]:
    if shop_id and shop_id in S.moment_policies_by_shop:
        return S.moment_policies_by_shop[shop_id]
    path = _moment_policies_path(shop_id)
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                policies = {int(k): v for k, v in raw.items()}
        else:
            policies = {}
    except Exception as e:
        print('[warn] moment policies load failed:', e)
        policies = {}
    if shop_id:
        S.moment_policies_by_shop[shop_id] = policies
    return policies


def save_moment_meta(shop_id: Optional[str], meta: Dict) -> None:
    path = _moment_meta_path(shop_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print('[warn] moment meta save failed:', e)


def save_moment_policies(shop_id: Optional[str], policies: Dict) -> None:
    path = _moment_policies_path(shop_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({int(k): v for k, v in policies.items()}, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print('[warn] moment policies save failed:', e)


def get_moment_policies(shop_id: Optional[str] = None) -> Dict[int, Dict]:
    if shop_id:
        cached = S.moment_policies_by_shop.get(shop_id)
        if cached is not None:
            return cached
        return load_moment_policies(shop_id)
    return {}


def query_moment_from_ctx(model: PatternaUnified, ctx: Dict) -> np.ndarray:
    with torch.no_grad():
        u_hist = torch.zeros(1, model.cfg.embedding_dim, device=S.device)
        u_ctx = model.build_context(ctx, u_hist)
        p = model.moments(u_ctx, ctx_dict=ctx)
        return p.squeeze(0).detach().cpu().numpy().astype(np.float32)


def moment_default_name(i: int) -> str:
    presets = [
        'Explore new', 'Everyday staples', 'Healthy picks', 'Quick meals', 'Family dinner',
        'Party & treats', 'Cozy comfort', 'On-the-go', 'Fresh produce', 'Breakfast',
        'Budget-friendly', 'Premium picks', 'Seasonal', 'Social gathering', 'Local favorites'
    ]
    return presets[i % len(presets)]


def compute_moment_keywords(
    moment_idx: int,
    top_n: int = 4,
    moment2vec: Optional[np.ndarray] = None,
    catalog: Optional[Dict] = None,
) -> List[Dict]:
    """Compute topic-like keywords (departments/aisles) for a moment.

    Returns list of {name, weight, type} sorted by weight descending.
    """
    moment2vec = moment2vec if moment2vec is not None else S.moment2vec
    catalog = catalog if catalog is not None else S.catalog
    if moment2vec is None or catalog is None:
        return []

    A = moment2vec
    N = A.shape[0]
    weights = A[:, moment_idx]

    # Aggregate by department
    from collections import defaultdict
    dept_weight = defaultdict(float)

    for iid in range(N):
        w = weights[iid]
        if w < 0.01:
            continue
        meta = catalog.get(iid)
        if meta:
            dept = meta.get('dept', 'other')
            if dept:
                dept_weight[dept] += w

    # Normalize and return top keywords
    total = sum(dept_weight.values()) or 1.0
    keywords = []
    for dept, w in sorted(dept_weight.items(), key=lambda x: -x[1])[:top_n]:
        keywords.append({
            'name': dept.replace('_', ' ').title(),
            'weight': round(w / total, 2),
            'type': 'dept'
        })

    return keywords


def suggest_moment_labels(moment_idx: int, keywords: List[Dict], top_hours: List[int], top_dows: List[int]) -> List[Dict]:
    """Suggest multiple label options for a moment based on its characteristics.

    Returns list of {label, confidence, reason} sorted by confidence.
    """
    suggestions = []

    # Extract info
    kw_names = [k['name'].lower() for k in keywords]
    top_kw = keywords[0]['name'] if keywords else 'General'
    second_kw = keywords[1]['name'] if len(keywords) > 1 else ''
    kw_weight = keywords[0]['weight'] if keywords else 0

    hours = set(top_hours)
    dows = set(top_dows)

    is_morning = any(6 <= h <= 11 for h in hours)
    is_evening = any(17 <= h <= 21 for h in hours)
    is_night = any(h >= 22 or h <= 4 for h in hours)
    is_weekend = any(d in (0, 5, 6) for d in dows)  # Sun, Fri, Sat
    is_weekday = any(d in (1, 2, 3, 4) for d in dows)

    def has_kw(*terms):
        return any(any(t in k for t in terms) for k in kw_names)

    # Time + category based suggestions
    if 0 in dows and is_morning:
        if has_kw('bakery', 'dairy', 'breakfast'):
            suggestions.append({'label': 'Sunday Brunch', 'confidence': 0.9, 'reason': 'Sunday morning + breakfast items'})
        else:
            suggestions.append({'label': 'Sunday Morning', 'confidence': 0.7, 'reason': 'Sunday morning pattern'})

    if is_weekend and is_evening:
        if has_kw('snack', 'beverage', 'frozen'):
            suggestions.append({'label': 'Party Time', 'confidence': 0.85, 'reason': 'Weekend evening + party categories'})
        suggestions.append({'label': 'Weekend Treats', 'confidence': 0.7, 'reason': 'Weekend evening pattern'})

    if is_weekday and is_evening:
        if has_kw('produce', 'meat', 'seafood', 'deli'):
            suggestions.append({'label': 'Dinner Prep', 'confidence': 0.85, 'reason': 'Weeknight + dinner ingredients'})
        if has_kw('household', 'pantry'):
            suggestions.append({'label': 'Weeknight Restock', 'confidence': 0.8, 'reason': 'Weeknight + household items'})

    if is_morning:
        if has_kw('dairy', 'bakery', 'cereal', 'breakfast'):
            suggestions.append({'label': 'Breakfast Run', 'confidence': 0.8, 'reason': 'Morning + breakfast categories'})

    if is_night:
        suggestions.append({'label': 'Late Night', 'confidence': 0.6, 'reason': 'Late night shopping pattern'})

    # Category-dominant suggestions
    if has_kw('produce'):
        suggestions.append({'label': 'Fresh Picks', 'confidence': 0.75, 'reason': 'Produce-heavy'})
    if has_kw('snack'):
        suggestions.append({'label': 'Snack Attack', 'confidence': 0.7, 'reason': 'Snacks prominent'})
    if has_kw('health', 'personal care', 'vitamin'):
        suggestions.append({'label': 'Self Care', 'confidence': 0.7, 'reason': 'Health/personal care items'})
    if has_kw('frozen'):
        suggestions.append({'label': 'Freezer Stock', 'confidence': 0.65, 'reason': 'Frozen items prominent'})
    if has_kw('beverage'):
        suggestions.append({'label': 'Drink Up', 'confidence': 0.6, 'reason': 'Beverages prominent'})

    # Fallback: combine top keywords
    if kw_weight > 0.3:
        suggestions.append({'label': f'{top_kw} Time', 'confidence': 0.5, 'reason': f'Dominated by {top_kw.lower()}'})
    elif second_kw:
        suggestions.append({'label': f'{top_kw} & {second_kw}', 'confidence': 0.4, 'reason': 'Mixed categories'})
    else:
        suggestions.append({'label': 'General Mix', 'confidence': 0.3, 'reason': 'No dominant pattern'})

    # Sort by confidence and dedupe
    seen = set()
    unique = []
    for s in sorted(suggestions, key=lambda x: -x['confidence']):
        if s['label'] not in seen:
            seen.add(s['label'])
            unique.append(s)

    return unique[:5]  # Return top 5 suggestions


def moment_summaries(top_n_items: int = 3, shop_id: Optional[str] = None):
    load_model()
    shop_assets = get_shop_assets(shop_id) if shop_id else None
    moment2vec = shop_assets.moment2vec if shop_assets and shop_assets.moment2vec is not None else S.moment2vec
    K = int(moment2vec.shape[1]) if moment2vec is not None else 0
    timing_path = os.path.join(ROOT, 'artifacts', 'timing_sweep.csv')
    timing = None
    if os.path.exists(timing_path):
        try:
            import csv
            timing = []
            with open(timing_path, 'r', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                for r in rdr:
                    timing.append(r)
        except Exception:
            timing = None
    hour_freq = [dict() for _ in range(K)]
    dow_freq = [dict() for _ in range(K)]
    if timing:
        for r in timing:
            try:
                h = int(r.get('hour', -1)); d = int(r.get('dow', -1))
            except Exception:
                continue
            if h < 0 or d < 0: continue
            dm = r.get('dominant_moment')
            m = int(dm) if dm not in (None, '') else None
            if m is None:
                vals = [float(r.get(f'm{j}', '0') or '0') for j in range(K)]
                m = int(max(range(K), key=lambda j: vals[j])) if K else 0
            hour_freq[m][h] = hour_freq[m].get(h, 0) + 1
            dow_freq[m][d] = dow_freq[m].get(d, 0) + 1
    A = moment2vec
    moments = []
    labels = get_moment_labels(shop_id)
    for j in range(K):
        try:
            col = A[:, j]
            idx = np.argsort(col)[-top_n_items:][::-1]
            items = []
            for iid in idx:
                iid = int(iid)
                meta = S.catalog.get(iid) if S.catalog else None
                items.append({'id': iid, 'name': (meta.get('name') if meta else f'Item {iid}'), 'dept': (meta.get('dept') if meta else '')})
        except Exception:
            items = []
        def top_keys(freq: Dict[int,int], n: int):
            if not freq: return []
            return [k for k, _ in sorted(freq.items(), key=lambda kv: -kv[1])[:n]]
        top_hours = top_keys(hour_freq[j], 3)
        top_dows = top_keys(dow_freq[j], 2)

        # NEW: Add keywords and suggestions
        keywords = compute_moment_keywords(j, top_n=4, moment2vec=moment2vec, catalog=S.catalog)
        suggestions = suggest_moment_labels(j, keywords, top_hours, top_dows)

        custom_label = labels.get(j)
        auto_label = suggestions[0]["label"] if suggestions else moment_default_name(j)
        label = custom_label or auto_label
        label_source = "merchant" if custom_label else "auto"
        moments.append({
            'id': j,
            'label': label,
            'label_source': label_source,
            'has_custom_label': bool(custom_label),
            'auto_label': auto_label,
            'top_hours': top_hours,
            'top_dows': top_dows,
            'items': items,
            'keywords': keywords,  # NEW: topic-like keywords
            'suggestions': suggestions,  # NEW: multiple label options
        })
    return moments


def generate_moment_meta(shop_id: Optional[str], moments: List[Dict]) -> Dict:
    def format_time_signature(dows: List[int], hours: List[int]) -> str:
        if not dows and not hours:
            return "Unknown"
        day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        day_part = ", ".join(day_names[d] for d in dows if 0 <= d < 7) if dows else "Any day"
        hour_part = ", ".join(f"{h}:00" for h in hours) if hours else "Any time"
        return f"{day_part} • {hour_part}"

    meta = {
        "moments": [],
        "K": len(moments),
        "generated_at": datetime.utcnow().isoformat(),
        "moment_space_id": S.moment_space_id,
    }
    for m in moments:
        keywords = m.get("keywords") or []
        meta["moments"].append({
            "id": m.get("id"),
            "auto_label": m.get("auto_label") or m.get("label"),
            "keywords": [k.get("name") for k in keywords if k.get("name")],
            "top_items": [it.get("id") for it in (m.get("items") or []) if it.get("id") is not None],
            "top_categories": [k.get("name") for k in keywords if k.get("type") == "dept"],
            "time_signature": format_time_signature(m.get("top_dows") or [], m.get("top_hours") or []),
        })
    return meta
