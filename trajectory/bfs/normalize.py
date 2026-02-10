# normalize_fsm.py
# 用法：
#   python normalize_fsm.py --input ./perfect_fsm_email_100.json --output ./fsm_norm.json
import json
import argparse
from typing import Dict, Any, List

SUPPORTED_WRITE_OPS = {"set", "inc", "clear", "append_unique"}

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def dump_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def assert_true(cond: bool, msg: str):
    if not cond:
        raise ValueError(msg)

def _flatten_schema(prefix: str, schema: Any, out: Dict[str, str]):
    # 将 signature_schema 展平成 JSONPath → 类型字符串
    if isinstance(schema, dict):
        for k, v in schema.items():
            key_path = f"{prefix}.{k}" if prefix else f"$.{k}"
            _flatten_schema(key_path, v, out)
    else:
        out[prefix if prefix else "$"] = str(schema)

def build_schema_lookup(pages: List[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for p in pages:
        pid = p.get("id")
        sig_schema = p.get("signature_schema", {})
        page_map: Dict[str, str] = {}
        _flatten_schema("$", sig_schema, page_map)
        lookup[pid] = page_map
    return lookup

def collect_reads_preconditions(action: Dict[str, Any]) -> List[str]:
    reads = []
    for c in action.get("preconditions", []) or []:
        path = c.get("path")
        if isinstance(path, str):
            reads.append(path)
    # 去重保序
    seen, out = set(), []
    for r in reads:
        if r not in seen:
            seen.add(r); out.append(r)
    return out

def collect_writes_effects(action: Dict[str, Any]) -> List[str]:
    writes = []
    for ef in action.get("effects", []) or []:
        op, path = ef.get("op"), ef.get("path")
        if isinstance(path, str) and (op in SUPPORTED_WRITE_OPS):
            writes.append(path)
    seen, out = set(), []
    for w in writes:
        if w not in seen:
            seen.add(w); out.append(w)
    return out

def collect_params_needed(action: Dict[str, Any]) -> List[str]:
    params = action.get("parameters", {}) or {}
    return list(params.keys()) if isinstance(params, dict) else []

def normalize_fsm(fsm: Dict[str, Any]) -> Dict[str, Any]:
    meta = fsm.get("meta", {})
    initial = meta.get("initial_page_id")
    terminals = meta.get("terminal_pages", [])
    pages: List[Dict[str, Any]] = fsm.get("pages", [])

    assert_true(isinstance(initial, str) and initial, "meta.initial_page_id 缺失或非法")
    assert_true(isinstance(terminals, list) and len(terminals) > 0, "meta.terminal_pages 必须是非空数组")
    assert_true(isinstance(pages, list) and len(pages) > 0, "pages 必须是非空数组")

    page_ids = [p.get("id") for p in pages]
    assert_true(all(isinstance(x, str) and x for x in page_ids), "每个 page.id 必须是非空字符串")
    assert_true(len(set(page_ids)) == len(page_ids), "page.id 必须唯一")

    page_index: Dict[str, Dict[str, Any]] = {p["id"]: p for p in pages}
    schema_lookup = build_schema_lookup(pages)

    actions_summary, edges = [], []
    for pid in page_ids:
        page_obj = page_index[pid]
        for act in page_obj.get("actions", []) or []:
            aid = act.get("id")
            frm = act.get("from", pid)
            to = act.get("to", pid)
            is_nav = bool(act.get("is_navigation", False))
            assert_true(isinstance(aid, str) and aid, f"动作缺少有效 id（page={pid}）")
            reads = collect_reads_preconditions(act)
            writes = collect_writes_effects(act)
            params_needed = collect_params_needed(act)

            actions_summary.append({
                "action_id": aid,
                "name": act.get("name", ""),
                "from": frm,
                "to": to,
                "is_navigation": is_nav,
                "params_needed": params_needed,
                "reads": reads,
                "writes": writes
            })
            edges.append({
                "from": frm, "to": to, "is_navigation": is_nav, "action_id": aid
            })

    return {
        "meta": {"initial_page_id": initial, "terminal_pages": terminals},
        "pages": sorted(page_ids),
        "schema_lookup": schema_lookup,
        "actions": actions_summary,
        "edges": edges
    }

def main():
    ap = argparse.ArgumentParser(description="Normalize FSM for downstream steps.")
    ap.add_argument("--input", required=True, help="Path to fsm.json")
    ap.add_argument("--output", required=True, help="Path to write normalized json")
    args = ap.parse_args()
    fsm = load_json(args.input)
    norm = normalize_fsm(fsm)
    dump_json(norm, args.output)

if __name__ == "__main__":
    main()
