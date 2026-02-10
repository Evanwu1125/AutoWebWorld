#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re, sys
from pathlib import Path
from itertools import combinations
from copy import deepcopy

# ---------- IO ----------
def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        sys.exit(f"[ERROR] 读取JSON失败: {p} -> {e}")

def dump_json(obj, p: Path):
    txt = json.dumps(obj, ensure_ascii=False, indent=2)
    p.write_text(txt, encoding="utf-8")

# ---------- FSM 索引 ----------
def index_actions_by_id(fsm: dict):
    idx = {}
    for page in fsm.get("pages", []):
        for a in page.get("actions", []):
            aid = a["id"]
            if aid in idx:
                # 同名覆盖以最后出现为准（通常无重复）
                pass
            idx[aid] = a
    return idx

# ---------- 参数替换 ----------
def lookup(d, dotted: str):
    cur = d
    for k in dotted.split("."):
        if k not in cur:
            raise KeyError(f"params 缺少键: {dotted}")
        cur = cur[k]
    return cur

def materialize_step(step_tpl: dict, params: dict):
    step = dict(step_tpl)  # 浅拷贝
    # value_ref -> value
    if "value_ref" in step:
        m = re.findall(r"\{([\w\.]+)\}", str(step["value_ref"]))
        if m:
            step["value"] = lookup(params, m[0])
        step.pop("value_ref", None)
    # attr_ref -> attr
    if "attr_ref" in step:
        attr = {}
        for k, v in step["attr_ref"].items():
            mm = re.findall(r"\{([\w\.]+)\}", str(v))
            attr[k] = str(lookup(params, mm[0])) if mm else v
        step["attr"] = attr
        step.pop("attr_ref", None)
    return step

# ---------- 可选动作识别与组合 ----------
def get_submit_required_fields(submit_action: dict) -> set:
    """获取 SUBMIT 动作的 preconditions 依赖的字段"""
    required = set()
    for pre in submit_action.get('preconditions', []):
        path = pre.get('path', '')
        if path.startswith('$.'):
            field = path[2:].split('.')[0]
            required.add(field)
    return required

def get_action_effect_field(action: dict) -> str:
    """获取动作 effect 设置的字段名"""
    for eff in action.get('effects', []):
        if eff.get('op') == 'set':
            path = eff.get('path', '')
            if path.startswith('$.'):
                return path[2:].split('.')[0]
    return None

def find_optional_actions(page_id: str, fsm: dict, path_action_ids: set) -> list:
    """找到页面中的可选动作（不在路径中、不被 SUBMIT 依赖、非特殊动作）"""
    page = None
    for p in fsm.get('pages', []):
        if p.get('id') == page_id:
            page = p
            break
    if not page:
        return []

    actions = page.get('actions', [])

    # 找 SUBMIT 动作
    submit_action = None
    for a in actions:
        if 'submit' in a.get('id', '').lower():
            submit_action = a
            break
    if not submit_action:
        return []

    required_fields = get_submit_required_fields(submit_action)

    # 筛选可选动作
    optional = []
    for a in actions:
        aid = a.get('id', '')
        aid_lower = aid.lower()
        # 排除已在路径中的动作
        if aid in path_action_ids:
            continue
        # 排除特殊动作（ID 中包含关键词）
        if any(x in aid_lower for x in ['submit', 'captcha', 'back', 'nav']):
            continue
        # 排除导航动作（会离开当前页面）
        if a.get('is_navigation') and a.get('to') != page_id:
            continue
        # 排除被 SUBMIT 依赖的动作
        effect_field = get_action_effect_field(a)
        if effect_field and effect_field in required_fields:
            continue
        optional.append(a)
    return optional

def generate_optional_combinations(optional_actions: list) -> list:
    """生成所有可选动作组合（包括空集）"""
    result = [[]]
    for r in range(1, len(optional_actions) + 1):
        for combo in combinations(optional_actions, r):
            result.append(list(combo))
    return result

def find_captcha_index(actions: list) -> int:
    """找到 CAPTCHA 动作的索引"""
    for i, a in enumerate(actions):
        if 'captcha' in a.get('id', '').lower():
            return i
    return -1

def find_form_start_index(actions: list, form_page: str) -> int:
    """找到表单页面第一个动作的索引"""
    for i, a in enumerate(actions):
        if a.get('from') == form_page:
            return i
    return 0

def generate_insertion_permutations(opt_actions: list, slot_count: int) -> list:
    """
    生成可选动作在插入槽位中的所有排列方式
    slot_count: 可插入的位置数（表单动作之间 + CAPTCHA 之前）
    返回: [(positions, permutation), ...] 其中 positions 是插入位置列表
    """
    from itertools import permutations as perms
    results = []
    n = len(opt_actions)

    if n == 0:
        return [([], [])]

    # 生成所有可选动作的排列
    for perm in perms(opt_actions):
        # 生成所有可能的位置组合（带重复，递增）
        def gen_positions(count, max_pos, current=[]):
            if count == 0:
                results.append((list(current), list(perm)))
                return
            start = current[-1] if current else 0
            for pos in range(start, max_pos):
                gen_positions(count - 1, max_pos, current + [pos])

        gen_positions(n, slot_count)

    return results

def expand_path_with_optionals(path_obj: dict, fsm: dict) -> list:
    """将可选动作插入到路径中的任意位置（CAPTCHA 之前），生成多条扩展路径"""
    actions = path_obj.get('actions', [])
    if not actions:
        return [path_obj]

    captcha_idx = find_captcha_index(actions)
    if captcha_idx == -1:
        return [path_obj]

    captcha_page = actions[captcha_idx].get('from')
    path_action_ids = {a.get('id') for a in actions}
    optional_actions = find_optional_actions(captcha_page, fsm, path_action_ids)

    if not optional_actions:
        return [path_obj]

    # 找到表单页面的起始位置
    form_start = find_form_start_index(actions, captcha_page)
    # 可插入的槽位数 = 表单动作数 + 1（包括 CAPTCHA 之前）
    slot_count = captcha_idx - form_start + 1

    expanded = []

    # 生成所有可选动作组合
    combos = generate_optional_combinations(optional_actions)

    for combo in combos:
        if not combo:
            # 空组合，保持原路径
            expanded.append(deepcopy(path_obj))
            continue

        # 生成该组合的所有插入排列
        insert_perms = generate_insertion_permutations(combo, slot_count)

        for positions, perm in insert_perms:
            new_path = deepcopy(path_obj)
            new_actions = new_path['actions']

            # 按位置倒序插入，避免索引偏移
            sorted_inserts = sorted(zip(positions, perm), key=lambda x: -x[0])
            for pos, opt_action in sorted_inserts:
                insert_idx = form_start + pos
                action_record = {
                    'id': opt_action['id'],
                    'name': opt_action.get('name', ''),
                    'from': captcha_page,
                    'to': captcha_page,
                    'params': {}
                }
                new_actions.insert(insert_idx, action_record)

            expanded.append(new_path)

    return expanded


# ---------- 单条路径编译（严格按你的 bfs.json 结构） ----------
# 按你的需求：返回“顶层为 actions 数组”，每个元素是大操作及其已实参化的 gui_procedure
def compile_one_path(path_obj: dict, act_index: dict):
    actions = path_obj.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("路径对象缺少 actions[]")

    actions_out = []
    for a in actions:
        aid = a["id"]
        params = a.get("params", {}) or {}
        action_def = act_index.get(aid)
        if not action_def:
            raise KeyError(f"FSM 未找到 action.id: {aid}")
        proc = action_def.get("gui_procedure", [])
        if not proc:
            raise ValueError(f"FSM action 无 gui_procedure: {aid}")
        # 将该大操作的步骤按参数实参化
        steps = [materialize_step(step_tpl, params) for step_tpl in proc]
        actions_out.append({
            "id": aid,
            "name": a.get("name", ""),
            "from": a["from"],
            "to": a["to"],
            "params": params,
            "gui_procedure": steps
        })

    return actions_out

# ---------- 主函数（仅 --fsm / --bfs / --out） ----------
def main():
    parser = argparse.ArgumentParser(
        description="按 email_communication_fsm.json 与 email_communication_allshortest.json 的结构，将最短路径编译为 GUI 宏。"
    )
    parser.add_argument("--fsm", required=True, type=Path, help="FSM JSON 路径（含 pages[].actions[].gui_procedure）")
    parser.add_argument("--bfs", required=True, type=Path, help="BFS 最短路径 JSON 路径（含 terminals[] 或 sub_initials[]）")
    parser.add_argument("--out", required=True, type=Path, help="输出目录")
    args = parser.parse_args()

    fsm = load_json(args.fsm)
    bfs = load_json(args.bfs)
    act_index = index_actions_by_id(fsm)

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0

    # 支持两种格式：
    # 1. home_initial: {"initial": "HOME", "terminals": [...]}
    # 2. sub_initial: {"sub_initials": [{"initial": "PAGE_X", "terminals": [...]}, ...]}

    if "terminals" in bfs:
        # home_initial 格式
        terminals = bfs.get("terminals")
        if not isinstance(terminals, list) or not terminals:
            sys.exit("[ERROR] bfs.json 顶层未找到 terminals[]")

        for term in terminals:
            terminal_page = term.get("terminal_page", "UNKNOWN")
            paths = term.get("paths", [])
            if not isinstance(paths, list):
                continue

            # 每个 terminal 一个子目录，便于分组
            subdir = args.out / terminal_page
            subdir.mkdir(exist_ok=True)

            for i, p in enumerate(paths):
                # 扩展可选动作组合
                expanded_paths = expand_path_with_optionals(p, fsm)
                for j, exp_p in enumerate(expanded_paths):
                    compiled = compile_one_path(exp_p, act_index)
                    start_from = exp_p["actions"][0]["from"]
                    end_to = exp_p["actions"][-1]["to"]
                    fname = f"macro_{i:03d}_{j:02d}_{start_from}__{end_to}.json"
                    dump_json(compiled, subdir / fname)
                    written += 1

    elif "sub_initials" in bfs:
        # sub_initial 格式
        sub_initials = bfs.get("sub_initials")
        if not isinstance(sub_initials, list):
            sys.exit("[ERROR] bfs.json 顶层未找到 sub_initials[]")

        for sub_init in sub_initials:
            initial_page = sub_init.get("initial", "UNKNOWN")
            terminals = sub_init.get("terminals", [])

            for term in terminals:
                terminal_page = term.get("terminal_page", "UNKNOWN")
                paths = term.get("paths", [])
                if not isinstance(paths, list):
                    continue

                # 每个 terminal 一个子目录，便于分组
                subdir = args.out / terminal_page
                subdir.mkdir(exist_ok=True)

                for i, p in enumerate(paths):
                    # 扩展可选动作组合
                    expanded_paths = expand_path_with_optionals(p, fsm)
                    for j, exp_p in enumerate(expanded_paths):
                        compiled = compile_one_path(exp_p, act_index)
                        start_from = exp_p["actions"][0]["from"]
                        end_to = exp_p["actions"][-1]["to"]
                        fname = f"macro_{i:03d}_{j:02d}_{start_from}__{end_to}.json"
                        dump_json(compiled, subdir / fname)
                        written += 1
    else:
        sys.exit("[ERROR] bfs.json 格式不正确，需要包含 'terminals' 或 'sub_initials'")

    print(f"[OK] 共写入 {written} 条宏到 {args.out.resolve()}")

if __name__ == "__main__":
    main()
