import os
import re
import json
import time
from copy import deepcopy
from typing import Dict, Any, Optional


def slugify_theme(theme: str) -> str:
    """Convert a theme name to a filesystem-safe slug"""
    return re.sub(r'[^\w\-]', '_', theme.lower())


def save_json(data: Dict[str, Any], filepath: str):
    """Save a JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_cost_report(themed_output_dir: str,
                    theme: str,
                    generator_summary: Dict[str, Any],
                    validator_summary: Dict[str, Any],
                    improver_summary: Dict[str, Any]):
    """Save cost report"""
    print("\n" + "=" * 80)
    print("💰 Total cost summary")
    print("=" * 80)

    total_cost_usd = (generator_summary['total_cost_usd'] +
                      validator_summary['total_cost_usd'] +
                      improver_summary['total_cost_usd'])
    total_cost_cny = total_cost_usd * 7.3
    total_tokens = (generator_summary['total_tokens'] +
                    validator_summary['total_tokens'] +
                    improver_summary['total_tokens'])

    print(f"Generator Agent: ${generator_summary['total_cost_usd']:.4f} ({generator_summary['total_calls']} calls)")
    print(f"Validator Agent: ${validator_summary['total_cost_usd']:.4f} ({validator_summary['total_calls']} calls)")
    print(f"Improver Agent: ${improver_summary['total_cost_usd']:.4f} ({improver_summary['total_calls']} calls)")
    print(f"\nTotal cost: ${total_cost_usd:.4f} (¥{total_cost_cny:.2f})")
    print(f"Total tokens: {total_tokens:,}")
    print("=" * 80 + "\n")

    cost_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theme": theme,
        "generator": generator_summary,
        "validator": validator_summary,
        "improver": improver_summary,
        "total": {
            "cost_usd": total_cost_usd,
            "cost_cny": total_cost_cny,
            "tokens": total_tokens
        }
    }

    cost_report_path = f"{themed_output_dir}/cost_report.json"
    save_json(cost_report, cost_report_path)
    print(f"💾 Cost report saved: {cost_report_path}\n")


DEFAULT_COMPLEXITY_PROFILE: Dict[str, Any] = {
    "interceptors": {
        "cookie": False,
        "permissions": [],
        "captcha": False,
        "register_correction": False,
    },
    "pages": {"min": 10, "max": 20},
    "actions": {"per_page_min": 3, "per_page_max": 6},
    "terminals": {"count": 3},
    "path_length": {"len": [4, 8], "shortest_only": True, "count_interceptors": False, "strict": False},
    "home_nav_variants": ["direct"],
    "home_nav_coverage": "each",
    "list_filters": {
        "enabled": False,
        "fields_max": 3,
        "two_step_parity": True,
        "allowed_shapes": ["text_inputs", "checkboxes", "sliders", "sort"],
        "require_open_filtered": False,
    },
    "policy": {
        "home_policy": "navigation_first",
        "back_policy": "recommended",
        "interceptors": {
            "cookie_placement": "home",
            "permission_placement": "first_relevant_non_home",
        },
    },
}


def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def normalize_complexity_profile(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Normalize and backfill profile fields for prompt/runtime consistency."""
    if profile is None:
        merged = deepcopy(DEFAULT_COMPLEXITY_PROFILE)
    elif not isinstance(profile, dict):
        raise ValueError("complexity_profile must be a dict or None")
    else:
        merged = _deep_merge_dict(DEFAULT_COMPLEXITY_PROFILE, profile)

    # Normalize known enum-like fields with safe defaults.
    if merged.get("home_nav_coverage") not in {"each", "any"}:
        merged["home_nav_coverage"] = DEFAULT_COMPLEXITY_PROFILE["home_nav_coverage"]

    allowed_home_variants = {"direct", "hover", "menu", "scroll"}
    variants = [v for v in (merged.get("home_nav_variants") or []) if v in allowed_home_variants]
    merged["home_nav_variants"] = variants or deepcopy(DEFAULT_COMPLEXITY_PROFILE["home_nav_variants"])

    allowed_filter_shapes = {"text_inputs", "checkboxes", "sliders", "sort"}
    list_filters = merged.setdefault("list_filters", {})
    shapes = [s for s in (list_filters.get("allowed_shapes") or []) if s in allowed_filter_shapes]
    list_filters["allowed_shapes"] = shapes or deepcopy(DEFAULT_COMPLEXITY_PROFILE["list_filters"]["allowed_shapes"])

    # Backward compatibility: derive policy from old interceptors fields when policy is absent.
    policy = merged.setdefault("policy", {})
    interceptor_policy = policy.setdefault("interceptors", {})
    if "cookie_placement" not in interceptor_policy:
        interceptor_policy["cookie_placement"] = "home" if merged.get("interceptors", {}).get("cookie") else "auto"
    if "permission_placement" not in interceptor_policy:
        interceptor_policy["permission_placement"] = "first_relevant_non_home"

    # Enforce bool default for strict.
    path_length = merged.setdefault("path_length", {})
    path_length["strict"] = bool(path_length.get("strict", False))

    return merged

