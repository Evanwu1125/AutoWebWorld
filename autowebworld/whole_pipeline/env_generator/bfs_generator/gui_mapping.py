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
        sys.exit(f"[ERROR] Failed to read JSON: {p} -> {e}")

def dump_json(obj, p: Path):
    txt = json.dumps(obj, ensure_ascii=False, indent=2)
    p.write_text(txt, encoding="utf-8")

def index_actions_by_id(fsm: dict):
    idx = {}
    for page in fsm.get("pages", []):
        for a in page.get("actions", []):
            aid = a["id"]
            if aid in idx:
                pass
            idx[aid] = a
    return idx

def lookup(d, dotted: str):
    cur = d
    for k in dotted.split("."):
        if k not in cur:
            raise KeyError(f"params is missing key: {dotted}")
        cur = cur[k]
    return cur

def materialize_step(step_tpl: dict, params: dict):
    step = dict(step_tpl)
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

def get_submit_required_fields(submit_action: dict) -> set:
    required = set()
    for pre in submit_action.get('preconditions', []):
        path = pre.get('path', '')
        if path.startswith('$.'):
            field = path[2:].split('.')[0]
            required.add(field)
    return required

def get_action_effect_field(action: dict) -> str:
    for eff in action.get('effects', []):
        if eff.get('op') == 'set':
            path = eff.get('path', '')
            if path.startswith('$.'):
                return path[2:].split('.')[0]
    return None

def find_optional_actions(page_id: str, fsm: dict, path_action_ids: set) -> list:
    page = None
    for p in fsm.get('pages', []):
        if p.get('id') == page_id:
            page = p
            break
    if not page:
        return []

    actions = page.get('actions', [])

    submit_action = None
    for a in actions:
        if 'submit' in a.get('id', '').lower():
            submit_action = a
            break
    if not submit_action:
        return []

    required_fields = get_submit_required_fields(submit_action)

    optional = []
    for a in actions:
        aid = a.get('id', '')
        aid_lower = aid.lower()
        if aid in path_action_ids:
            continue
        if any(x in aid_lower for x in ['submit', 'captcha', 'back', 'nav']):
            continue
        if a.get('is_navigation') and a.get('to') != page_id:
            continue
        effect_field = get_action_effect_field(a)
        if effect_field and effect_field in required_fields:
            continue
        optional.append(a)
    return optional

def generate_optional_combinations(optional_actions: list) -> list:
    result = [[]]
    for r in range(1, len(optional_actions) + 1):
        for combo in combinations(optional_actions, r):
            result.append(list(combo))
    return result

def find_captcha_index(actions: list) -> int:
    for i, a in enumerate(actions):
        if 'captcha' in a.get('id', '').lower():
            return i
    return -1

def find_form_start_index(actions: list, form_page: str) -> int:
    for i, a in enumerate(actions):
        if a.get('from') == form_page:
            return i
    return 0

def generate_insertion_permutations(opt_actions: list, slot_count: int) -> list:
    from itertools import permutations as perms
    results = []
    n = len(opt_actions)

    if n == 0:
        return [([], [])]

    for perm in perms(opt_actions):
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

    form_start = find_form_start_index(actions, captcha_page)
    slot_count = captcha_idx - form_start + 1

    expanded = []

    combos = generate_optional_combinations(optional_actions)

    for combo in combos:
        if not combo:
            expanded.append(deepcopy(path_obj))
            continue

        insert_perms = generate_insertion_permutations(combo, slot_count)

        for positions, perm in insert_perms:
            new_path = deepcopy(path_obj)
            new_actions = new_path['actions']

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


def compile_one_path(path_obj: dict, act_index: dict):
    actions = path_obj.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError("Path object is missing actions[]")

    actions_out = []
    for a in actions:
        aid = a["id"]
        params = a.get("params", {}) or {}
        action_def = act_index.get(aid)
        if not action_def:
            raise KeyError(f"FSM action.id not found: {aid}")
        proc = action_def.get("gui_procedure", [])
        if not proc:
            raise ValueError(f"FSM action has no gui_procedure: {aid}")
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

def main():
    parser = argparse.ArgumentParser(
        description="Compile shortest paths into GUI macros using the structure of email_communication_fsm.json and email_communication_allshortest.json."
    )
    parser.add_argument("--fsm", required=True, type=Path, help="Path to FSM JSON (including pages[].actions[].gui_procedure)")
    parser.add_argument("--bfs", required=True, type=Path, help="Path to BFS shortest-path JSON (including terminals[] or sub_initials[])")
    parser.add_argument("--out", required=True, type=Path, help="Output directory")
    args = parser.parse_args()

    fsm = load_json(args.fsm)
    bfs = load_json(args.bfs)
    act_index = index_actions_by_id(fsm)

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0

    # 1. home_initial: {"initial": "HOME", "terminals": [...]}
    # 2. sub_initial: {"sub_initials": [{"initial": "PAGE_X", "terminals": [...]}, ...]}

    if "terminals" in bfs:
        terminals = bfs.get("terminals")
        if not isinstance(terminals, list) or not terminals:
            sys.exit("[ERROR] terminals[] not found at the top level of bfs.json")

        for term in terminals:
            terminal_page = term.get("terminal_page", "UNKNOWN")
            paths = term.get("paths", [])
            if not isinstance(paths, list):
                continue

            subdir = args.out / terminal_page
            subdir.mkdir(exist_ok=True)

            for i, p in enumerate(paths):
                expanded_paths = expand_path_with_optionals(p, fsm)
                for j, exp_p in enumerate(expanded_paths):
                    compiled = compile_one_path(exp_p, act_index)
                    start_from = exp_p["actions"][0]["from"]
                    end_to = exp_p["actions"][-1]["to"]
                    fname = f"macro_{i:03d}_{j:02d}_{start_from}__{end_to}.json"
                    dump_json(compiled, subdir / fname)
                    written += 1

    elif "sub_initials" in bfs:
        sub_initials = bfs.get("sub_initials")
        if not isinstance(sub_initials, list):
            sys.exit("[ERROR] sub_initials[] not found at the top level of bfs.json")

        for sub_init in sub_initials:
            initial_page = sub_init.get("initial", "UNKNOWN")
            terminals = sub_init.get("terminals", [])

            for term in terminals:
                terminal_page = term.get("terminal_page", "UNKNOWN")
                paths = term.get("paths", [])
                if not isinstance(paths, list):
                    continue

                subdir = args.out / terminal_page
                subdir.mkdir(exist_ok=True)

                for i, p in enumerate(paths):
                    expanded_paths = expand_path_with_optionals(p, fsm)
                    for j, exp_p in enumerate(expanded_paths):
                        compiled = compile_one_path(exp_p, act_index)
                        start_from = exp_p["actions"][0]["from"]
                        end_to = exp_p["actions"][-1]["to"]
                        fname = f"macro_{i:03d}_{j:02d}_{start_from}__{end_to}.json"
                        dump_json(compiled, subdir / fname)
                        written += 1
    else:
        sys.exit("[ERROR] Invalid bfs.json format; expected 'terminals' or 'sub_initials'")

    print(f"[OK] Wrote {written} macros to {args.out.resolve()}")

if __name__ == "__main__":
    main()
