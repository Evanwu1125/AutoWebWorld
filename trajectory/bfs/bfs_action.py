import json, argparse
from collections import deque, defaultdict
from copy import deepcopy
from typing import Dict, Any, Tuple, List
from .general_params_provider import type_category

# ---------- IO ----------
def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ---------- 默认签名 ----------
def _default_from_type(tp_tuple: Tuple[str,str]):
    kind, spec = tp_tuple
    if kind == 'array':   return []
    if spec == 'boolean': return False
    if spec == 'integer_ge1': return 1
    if spec == 'integer': return 0
    if spec == 'number':  return 0.0
    if spec in ('string','any'): return ""
    return None

def default_signature_for_page(page_id: str, schema_lookup: Dict[str, Dict[str,str]]):
    sig = {}
    for path, tp in schema_lookup.get(page_id, {}).items():
        if path == '$':  # 根
            continue
        keys = path[2:].split('.')  # 去掉 "$."
        cur = sig
        for k in keys[:-1]:
            cur = cur.setdefault(k, {})
        # 检查是否是 nullable 类型 (如 "boolean|null")
        if '|null' in tp.lower():
            cur[keys[-1]] = None
        else:
            cur[keys[-1]] = _default_from_type(type_category(tp))
    return sig


# ---------- HOME 一跳直达子页面收集 ----------
def direct_nav_targets_from_home(norm: Dict[str, Any]):
    init = norm["meta"]["initial_page_id"]
    return sorted({
        e.get("to") for e in norm.get("edges", [])
        if e.get("is_navigation") and e.get("from") == init
    })

# ---------- JSONPath 辅助 ----------
def read_path(sig, path):
    cur = sig
    for k in path.replace("$.","").split("."):
        if k == "":
            continue
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k, None)
    return cur

def write_path(sig, path, value):
    keys = path.replace("$.","").split(".")
    cur = sig
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value

def clear_value(current):
    if isinstance(current, list):  return []
    if isinstance(current, str) or current is None: return None
    if isinstance(current, bool):  return False
    if isinstance(current, int):   return 0
    if isinstance(current, float): return 0.0
    return None

# ---------- 前置 & 效果 ----------
def cond_ok(val, cond, expect):
    try:
        if cond == 'eq':        return val == expect
        if cond == 'neq':       return val != expect
        if cond == 'not_null':  return val is not None
        if cond == 'length_gt': return (val is not None) and (len(val) > expect)
        if cond == 'in':        return val in (expect or [])
        if cond == 'gt':        return val > expect
        if cond == 'lt':        return val < expect
        if cond == 'gte':       return val >= expect
        if cond == 'lte':       return val <= expect
    except Exception:
        return False
    return False

def preconditions_ok(sig, pres):
    for c in pres or []:
        if not cond_ok(read_path(sig, c.get('path')), c.get('cond'), c.get('value')):
            return False
    return True

def resolve_ref(params, ref):
    return params.get(ref.strip("{} "))

def apply_effects(sig, effects, params):
    new_sig = deepcopy(sig)
    for ef in effects or []:
        op   = ef.get('op')
        path = ef.get('path')
        if op == 'set':
            val = ef.get('value')
            if 'value_ref' in ef:
                val = resolve_ref(params, ef['value_ref'])
            write_path(new_sig, path, val)
        elif op == 'inc':
            by  = ef.get('by', ef.get('value', 1))
            cur = read_path(new_sig, path)
            if cur is None: cur = 0
            write_path(new_sig, path, cur + by)
        elif op == 'clear':
            cur = read_path(new_sig, path)
            write_path(new_sig, path, clear_value(cur))
        elif op == 'append_unique':
            cur = read_path(new_sig, path)
            if cur is None: cur = []
            v = resolve_ref(params, ef.get('value_ref')) if ef.get('value_ref') else ef.get('value')
            arr = list(cur)
            if v not in arr:
                arr.append(v)
            write_path(new_sig, path, arr)
        # 其它 GUI-only 操作忽略
    return new_sig

def transition(page_id: str, sig: Dict[str,Any], act: Dict[str,Any], params: Dict[str,Any],
               schema_lookup: Dict[str, Dict[str,str]]) -> Tuple[str, Dict[str,Any]]:
    cur = apply_effects(sig, act.get('effects', []), params)
    if act.get('is_navigation'):
        to_page = act['to']
        base = default_signature_for_page(to_page, schema_lookup)
        # “目标页默认签名 + 同名字段合并”
        merged = {**base, **{k: v for k, v in cur.items() if k in base}}
        return to_page, merged
    else:
        return page_id, cur

def state_key(page_id, sig):
    return page_id + "::" + json.dumps(sig, sort_keys=True, ensure_ascii=False)

# ---------- Interceptor 处理逻辑 ----------
# 弹窗类 interceptor 强制第一个处理，captcha 排除（后面强制最后）
INTERCEPTOR_PATTERNS = ['accepted', 'permission', 'granted']
INTERCEPTOR_EXCLUDE = ['captcha']

def get_interceptor_fields_from_schema(page_schema: Dict[str, str]) -> List[str]:
    """从页面 schema 中识别 interceptor 字段（排除 captcha）"""
    interceptor_fields = []
    for field_path, field_type in page_schema.items():
        field_name = field_path.replace('$.', '')
        field_lower = field_name.lower()
        if any(ex in field_lower for ex in INTERCEPTOR_EXCLUDE):
            continue
        if any(p in field_lower for p in INTERCEPTOR_PATTERNS):
            if 'null' in field_type.lower():
                interceptor_fields.append(field_name)
    return interceptor_fields

def has_unhandled_interceptors(sig: Dict[str, Any], page_schema: Dict[str, str]) -> bool:
    """检查是否有未处理的 interceptor"""
    for field in get_interceptor_fields_from_schema(page_schema):
        if sig.get(field) is None:
            return True
    return False

def is_interceptor_action(action: Dict[str, Any]) -> bool:
    """判断是否是 interceptor 处理动作（排除 captcha）"""
    for eff in action.get('effects', []):
        if eff.get('op') == 'set':
            path = eff.get('path', '').lower()
            if any(ex in path for ex in INTERCEPTOR_EXCLUDE):
                continue
            if any(p in path for p in INTERCEPTOR_PATTERNS):
                return True
    return False

# ---------- CAPTCHA 强制最后处理 ----------
def is_captcha_action(action: Dict[str, Any]) -> bool:
    """判断是否是 CAPTCHA 动作"""
    return 'captcha' in action.get('id', '').lower()

def is_submit_action(action: Dict[str, Any]) -> bool:
    """判断是否是提交动作"""
    return 'submit' in action.get('id', '').lower()

def is_navigation_action(action: Dict[str, Any]) -> bool:
    """判断是否是导航动作（离开当前页面）"""
    aid = action.get('id', '').upper()
    return 'BACK' in aid or 'NAV' in aid

def action_still_needed(sig: Dict[str, Any], action: Dict[str, Any]) -> bool:
    """检查动作是否还需要执行（effect 对应的字段还是空的）"""
    for eff in action.get('effects', []):
        if eff.get('op') == 'set':
            path = eff.get('path', '')
            if path.startswith('$.'):
                field = path[2:].split('.')[0]
                val = sig.get(field)
                # 字段为空或 None，说明还需要执行
                if val is None or val == '' or val == []:
                    return True
    return False

def get_action_effect_field(action: Dict[str, Any]) -> str:
    """获取动作 effect 设置的字段名"""
    for eff in action.get('effects', []):
        if eff.get('op') == 'set':
            path = eff.get('path', '')
            if path.startswith('$.'):
                return path[2:].split('.')[0]
    return None

def get_submit_required_fields(actions: List[Dict[str, Any]]) -> set:
    """获取 SUBMIT 动作的 preconditions 依赖的字段"""
    required = set()
    for act in actions:
        if is_submit_action(act):
            for pre in act.get('preconditions', []):
                path = pre.get('path', '')
                if path.startswith('$.'):
                    field = path[2:].split('.')[0]
                    required.add(field)
            break
    return required

def has_pending_form_actions(sig: Dict[str, Any], actions: List[Dict[str, Any]],
                              disabled_actions: set = None) -> bool:
    """检查是否有待完成的必须表单动作（只检查被 SUBMIT 依赖的字段）"""
    # 获取 SUBMIT 依赖的必须字段
    required_fields = get_submit_required_fields(actions)

    for act in actions:
        if disabled_actions and act.get('id') in disabled_actions:
            continue
        if is_captcha_action(act):
            continue
        if is_submit_action(act):
            continue
        if is_navigation_action(act):
            continue
        if not preconditions_ok(sig, act.get('preconditions')):
            continue

        # 只检查必须动作（effect 字段被 SUBMIT 依赖的）
        effect_field = get_action_effect_field(act)
        if effect_field and effect_field not in required_fields:
            continue  # 可选动作，跳过

        # 检查动作是否还需要执行
        if action_still_needed(sig, act):
            return True
    return False

# ---------- Item Access Method 识别与覆盖 ----------
# precondition 中的字段模式
PRECOND_PATTERNS = {
    'filter': ['filters_applied'],
    'search': ['has_searched'],
    'scroll': ['viewport_anchor_id'],
}

# action ID 中的名称模式（用于识别没有 precondition 的捷径 action）
# 注意：scroll 只匹配 ACT_SCROLL_*_INTO_VIEW，不匹配 ACT_NAV_EXPLORE_SCROLL 等导航 action
ACTION_NAME_PATTERNS = {
    'filter': ['FILTERED'],
    'search': ['SEARCH'],
    'scroll': ['SCROLL_', '_ANY_'],  # SCROLL_ 确保匹配 SCROLL_xxx_INTO_VIEW 而非 NAV_xxx_SCROLL
}


def get_access_method_from_preconditions(preconditions: List[Dict]) -> str:
    """根据 precondition 判断 OPEN 动作属于哪种查找方式"""
    for pre in preconditions or []:
        path = pre.get('path', '').lower()
        for method, patterns in PRECOND_PATTERNS.items():
            if any(p in path for p in patterns):
                return method
    return None


def get_access_method_from_action_id(action_id: str) -> str:
    """根据 action ID 名称判断查找方式（用于没有 precondition 的捷径）"""
    action_id_upper = action_id.upper()
    for method, patterns in ACTION_NAME_PATTERNS.items():
        if any(p in action_id_upper for p in patterns):
            return method
    return None


def identify_item_access_actions(fsm: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    分析 FSM，识别所有 item 查找方式及其对应的"入口动作"
    包括：
    1. 有 precondition 的 OPEN 动作（如 ACT_OPEN_FILTERED_PROJECT）
    2. 没有 precondition 但名称包含特征的导航动作（如 ACT_NAV_SCROLL_TO_xxx）
    返回: {'filter': [...], 'search': [...], 'scroll': [...]}
    """
    result = {'filter': [], 'search': [], 'scroll': []}

    for page in fsm.get('pages', []):
        for action in page.get('actions', []):
            # 只分析导航动作
            if not action.get('is_navigation'):
                continue

            action_id = action.get('id', '')
            preconditions = action.get('preconditions', [])

            # 优先通过 precondition 识别
            method = get_access_method_from_preconditions(preconditions)

            # 如果 precondition 没有识别出来，尝试通过 action ID 识别
            if not method:
                method = get_access_method_from_action_id(action_id)

            if method and method in result:
                if action_id not in result[method]:
                    result[method].append(action_id)

    return result


def get_competing_actions(method: str, all_methods: Dict[str, List[str]]) -> set:
    """获取与指定方式竞争的其他入口动作"""
    competing = set()
    for m, actions in all_methods.items():
        if m != method:
            competing.update(actions)
    return competing


# ---------- 多父指针 BFS：列出所有等长最短路径 ----------
def bfs_all_shortest_paths(fsm: Dict[str, Any],
                           schema_lookup: Dict[str, Dict[str, str]],
                           initial_page: str,
                           target_page: str,
                           seed: int = 42,
                           max_paths: int = 5000,
                           disabled_actions: set = None):
    page_index = {p['id']: p for p in fsm['pages']}
    def actions_from_page(pid):
        return page_index[pid].get('actions', [])

    start_sig = default_signature_for_page(initial_page, schema_lookup)
    start_key = state_key(initial_page, start_sig)

    dist = {start_key: 0}
    parents = defaultdict(list)   # child_key -> list[(parent_key, action_record)]
    state_by_key = {start_key: (initial_page, start_sig)}

    q = deque([start_key])
    found_depth = None
    target_keys_at_depth: List[str] = []

    while q:
        cur_key = q.popleft()
        cur_depth = dist[cur_key]
        page_id, sig = state_by_key[cur_key]

        # 已知最短深度后，不再扩展更深层
        if found_depth is not None and cur_depth >= found_depth:
            continue

        page_actions = actions_from_page(page_id)
        for act in page_actions:
            # 0) 检查是否被禁用（用于强制特定查找方式）
            if disabled_actions and act.get('id') in disabled_actions:
                continue

            # 1) 检查前置
            if not preconditions_ok(sig, act.get('preconditions')):
                continue

            # 1.5) Interceptor 检查：弹窗类强制第一个处理
            page_schema = schema_lookup.get(page_id, {})
            if has_unhandled_interceptors(sig, page_schema):
                if not is_interceptor_action(act):
                    continue

            # 1.6) CAPTCHA 强制最后：如果还有待完成的表单动作，跳过 CAPTCHA
            if is_captcha_action(act):
                if has_pending_form_actions(sig, page_actions, disabled_actions):
                    continue

            # 2) 直接使用 FSM 中的参数（保留占位符）
            params = act.get('parameters', {})
            # 3) 过渡
            next_page, next_sig = transition(page_id, sig, act, params, schema_lookup)
            nxt_key = state_key(next_page, next_sig)
            step_depth = cur_depth + 1

            if nxt_key not in dist:
                dist[nxt_key] = step_depth
                state_by_key[nxt_key] = (next_page, next_sig)
                parents[nxt_key].append((cur_key, {
                    "id": act['id'],
                    "name": act.get('name',''),
                    "from": page_id,
                    "to": next_page,
                    "params": params
                }))
                q.append(nxt_key)

                if next_page == target_page:
                    if found_depth is None:
                        found_depth = step_depth
                    if step_depth == found_depth:
                        target_keys_at_depth.append(nxt_key)

            elif dist[nxt_key] == step_depth:
                # 另一条同长父指针
                parents[nxt_key].append((cur_key, {
                    "id": act['id'],
                    "name": act.get('name',''),
                    "from": page_id,
                    "to": next_page,
                    "params": params
                }))
                if next_page == target_page and (found_depth is not None) and found_depth == step_depth:
                    target_keys_at_depth.append(nxt_key)

        # 安全阈值（避免组合爆炸）
        if found_depth is not None and len(target_keys_at_depth) > max_paths:
            break

    if found_depth is None:
        return {"shortest_step_count": None, "paths": []}

    # 回溯生成所有等长最短路径（带缓存）
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def backtrack(key: str) -> List[List[Dict[str,Any]]]:
        if key == start_key:
            return [[]]
        res: List[List[Dict[str,Any]]] = []
        for pkey, actrec in parents[key]:
            for prefix in backtrack(pkey):
                res.append(prefix + [actrec])
        return res

    all_paths: List[List[Dict[str,Any]]] = []
    for end_key in set(target_keys_at_depth):
        for p in backtrack(end_key):
            all_paths.append(p)

    # 按动作 id 序列去重（不同签名导致的重复路径合并）
    seen_seq = set()
    unique_paths = []
    for p in all_paths:
        seq = tuple(a['id'] for a in p)
        if seq not in seen_seq:
            seen_seq.add(seq)
            unique_paths.append({"actions": p, "id_seq": list(seq)})

    return {"shortest_step_count": found_depth, "paths": unique_paths}


def bfs_with_access_coverage(fsm: Dict[str, Any],
                             schema_lookup: Dict[str, Dict[str, str]],
                             initial_page: str,
                             target_page: str,
                             seed: int = 42,
                             max_paths: int = 5000) -> Dict[str, Any]:
    """
    包装函数：确保覆盖所有 item 查找方式 (filter/search/scroll)
    1. 识别 FSM 中的所有查找方式
    2. 为每种方式分别生成路径（禁用其他方式的入口动作）
    3. 合并所有路径并去重
    """
    # 识别所有查找方式
    access_methods = identify_item_access_actions(fsm)
    active_methods = {m: acts for m, acts in access_methods.items() if acts}

    if not active_methods:
        # 没有识别到查找方式，直接用普通 BFS
        return bfs_all_shortest_paths(
            fsm, schema_lookup, initial_page, target_page, seed, max_paths
        )

    all_results = []
    min_step = None

    # 为每种查找方式分别生成路径
    for method, method_actions in active_methods.items():
        competing = get_competing_actions(method, active_methods)

        res = bfs_all_shortest_paths(
            fsm, schema_lookup, initial_page, target_page,
            seed, max_paths, disabled_actions=competing
        )

        if res["shortest_step_count"] is not None:
            all_results.append((method, res))
            if min_step is None or res["shortest_step_count"] < min_step:
                min_step = res["shortest_step_count"]

    if not all_results:
        return {"shortest_step_count": None, "paths": []}

    # 合并所有路径并去重
    seen_seq = set()
    merged_paths = []

    for method, res in all_results:
        for path_obj in res.get("paths", []):
            seq = tuple(path_obj.get("id_seq", []))
            if seq not in seen_seq:
                seen_seq.add(seq)
                # 添加查找方式标记
                path_obj["access_method"] = method
                merged_paths.append(path_obj)

    return {
        "shortest_step_count": min_step,
        "paths": merged_paths,
        "access_methods_covered": list(active_methods.keys())
    }


# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Action-level BFS (all shortest paths).")
    ap.add_argument("--fsm", required=True, help="Path to original FSM JSON")
    ap.add_argument("--norm", required=True, help="Path to normalized FSM JSON")
    ap.add_argument("--out", required=False, help="Output path for home_initial results (from initial_page_id)")
    ap.add_argument("--out_sub", required=False, help="Output path for sub_initial results (from HOME one-hop pages)")
    ap.add_argument("--seed", type=int, default=42, help="Deterministic seed for value generation")
    ap.add_argument("--max_paths", type=int, default=5000, help="Safety cap for number of shortest paths enumerated")
    args = ap.parse_args()

    if not args.out and not args.out_sub:
        raise ValueError("❌ You must specify at least one of --out or --out_sub")

    fsm  = load_json(args.fsm)
    norm = load_json(args.norm)

    initial   = norm["meta"]["initial_page_id"]
    terminals = norm["meta"]["terminal_pages"]
    schema_lk = norm["schema_lookup"]
    all_pages = norm["pages"]

    # 识别并打印查找方式
    access_methods = identify_item_access_actions(fsm)
    active_methods = {m: acts for m, acts in access_methods.items() if acts}
    if active_methods:
        print(f"🔍 Detected item access methods:")
        for method, actions in active_methods.items():
            print(f"   {method}: {actions}")

    # 1️⃣ 仅当提供 --out 时生成 home_initial
    if args.out:
        print(f"\n🏠 Generating home_initial paths from {initial}...")
        home_results = {"initial": initial, "terminals": []}
        for t in terminals:
            res = bfs_with_access_coverage(
                fsm, schema_lk, initial, t,
                seed=args.seed, max_paths=args.max_paths
            )
            home_results["terminals"].append({
                "terminal_page": t,
                **res
            })
        save_json(home_results, args.out)
        print(f"✅ Saved home_initial to {args.out}")

    # 2️⃣ 仅当提供 --out_sub 时生成 sub_initial
    if args.out_sub:
        print(f"\n📄 Generating sub_initial paths from HOME one-hop pages...")
        sub_pages = [
            p for p in direct_nav_targets_from_home(norm)
            if p != initial and p not in terminals
        ]

        sub_results = {"sub_initials": []}
        for sub_page in sub_pages:
            print(f"   Processing {sub_page}...")
            page_terminals = []
            for t in terminals:
                res = bfs_with_access_coverage(
                    fsm, schema_lk, sub_page, t,
                    seed=args.seed, max_paths=args.max_paths
                )
                if res["shortest_step_count"] is not None:
                    page_terminals.append({
                        "terminal_page": t,
                        **res
                    })
            if page_terminals:
                sub_results["sub_initials"].append({
                    "initial": sub_page,
                    "terminals": page_terminals
                })

        save_json(sub_results, args.out_sub)
        print(f"✅ Saved sub_initial to {args.out_sub}")
        print(f"   Total sub_initial pages: {len(sub_results['sub_initials'])}")

if __name__ == "__main__":
    main()
