from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Deque, Optional, Set
from collections import deque, defaultdict
import copy
import json
import hashlib

@dataclass
class Action:
    id: str
    name: str
    is_navigation: bool
    src: str
    dst: str
    parameters: Dict[str, Any]
    preconditions: List[Dict[str, Any]]
    effects: List[Dict[str, Any]]

@dataclass
class Page:
    id: str
    signature_schema: Dict[str, Any]
    actions: List[Action]

@dataclass
class BFSNode:
    page_id: str
    signature: Dict[str, Any]
    path_actions: List[str]

def deep_get(obj: Dict[str, Any], path: str) -> Any:
    if not path or not path.startswith("$."):
        return None
    cur = obj
    for key in path[2:].split("."):
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur

def deep_set(obj: Dict[str, Any], path: str, value: Any) -> None:
    assert path.startswith("$.")
    keys = path[2:].split(".")
    cur = obj
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def deep_inc(obj: Dict[str, Any], path: str, by: int) -> None:
    v = deep_get(obj, path)
    if v is None: v = 0
    deep_set(obj, path, (v + by))

def ensure_list(obj: Dict[str, Any], path: str) -> List[Any]:
    v = deep_get(obj, path)
    if not isinstance(v, list):
        v = [] if v is None else list(v)
        deep_set(obj, path, v)
    return v

def append_unique(obj: Dict[str, Any], path: str, value: Any) -> None:
    lst = ensure_list(obj, path)
    if value not in lst:
        lst.append(value)

def hash_signature(sig: Dict[str, Any]) -> str:
    s = json.dumps(sig, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def default_from_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: default_from_schema(v) for k, v in schema.items()}
    if isinstance(schema, str):
        s = schema.replace(" ", "").lower()
        if s.startswith("string"):  return None if "null" in s else ""
        if s.startswith("integer"): return 1 if ">=1" in s else 0
        if s.startswith("number"):  return 0.0
        if s.startswith("boolean"): return None if "null" in s else False
        if s.startswith("array"):   return []
        if s.startswith("object"):  return None if "null" in s else {}
    return None

def build_initial_signature(page_schema: Dict[str, Any]) -> Dict[str, Any]:
    return default_from_schema(page_schema) or {}

def value_from_ref(signature: Dict[str, Any], params: Dict[str, Any], ref: Any) -> Any:
    if isinstance(ref, str):
        if ref.startswith("{") and ref.endswith("}"):
            return params.get(ref[1:-1])
        if ref.startswith("$."):
            return deep_get(signature, ref)
    return ref

def eval_precondition(signature: Dict[str, Any], prec: Dict[str, Any], params: Dict[str, Any]) -> bool:
    path, cond = prec.get("path"), prec.get("cond")
    val = prec.get("value", None)
    left = deep_get(signature, path) if path else None
    right = value_from_ref(signature, params, val)
    if cond == "exists":     return left is not None
    if cond == "not_null":   return left is not None
    if cond == "eq":         return left == right
    if cond == "neq":        return left != right
    if cond == "gt":         return left is not None and left > right
    if cond == "gte":        return left is not None and left >= right
    if cond == "lt":         return left is not None and left < right
    if cond == "lte":        return left is not None and left <= right
    if cond == "length_gt":  return hasattr(left, "__len__") and len(left) > int(right or 0)
    if cond == "in":         return right is not None and left in right
    if cond == "oneof":      return isinstance(right, (list, tuple, set)) and left in right
    return False

def apply_effects(signature: Dict[str, Any], effects: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    sig = copy.deepcopy(signature)
    for eff in effects or []:
        op, path = eff.get("op"), eff.get("path")
        if not path or not path.startswith("$."): continue
        if op == "set":
            v = value_from_ref(sig, params, eff.get("value_ref")) if "value_ref" in eff else eff.get("value")
            deep_set(sig, path, v)
        elif op == "inc":
            deep_inc(sig, path, int(eff.get("by", 1)))
        elif op == "append_unique":
            v = value_from_ref(sig, params, eff.get("value_ref")) if "value_ref" in eff else eff.get("value")
            append_unique(sig, path, v)
        elif op == "clear":
            deep_set(sig, path, [])
    return sig

class FSMBFS:
    def __init__(self, fsm: Dict[str, Any]):
        self.fsm = fsm
        self.meta = fsm.get("meta", {})
        self.pages: Dict[str, Page] = {}
        for p in fsm.get("pages", []):
            actions = []
            for a in p.get("actions", []):
                actions.append(Action(
                    id=a["id"],
                    name=a.get("name", a["id"]),
                    is_navigation=bool(a.get("is_navigation", False)),
                    src=a.get("from", p["id"]),
                    dst=a.get("to", p["id"]),
                    parameters=a.get("parameters", {}),
                    preconditions=a.get("preconditions", []),
                    effects=a.get("effects", []),
                ))
            self.pages[p["id"]] = Page(id=p["id"], signature_schema=p.get("signature_schema", {}), actions=actions)
        self.initial_page: str = self.meta.get("initial_page_id")
        self.terminal_pages: Set[str] = set(self.meta.get("terminal_pages", []))

    def initial_state(self) -> BFSNode:
        page = self.pages[self.initial_page]
        sig = build_initial_signature(page.signature_schema)
        return BFSNode(page_id=page.id, signature=sig, path_actions=[])

    def _applicable(self, action: Action, signature: Dict[str, Any]) -> bool:
        params = action.parameters or {}
        return all(eval_precondition(signature, prec, params) for prec in (action.preconditions or []))

    def _next_state(self, action: Action, cur: BFSNode) -> BFSNode:
        params = action.parameters or {}
        new_sig = apply_effects(cur.signature, action.effects, params)
        next_page = action.dst if action.is_navigation else cur.page_id
        if next_page != cur.page_id:
            base = build_initial_signature(self.pages[next_page].signature_schema)
            for k, v in list(new_sig.items()):
                if k in base: base[k] = v
            new_sig = base
        return BFSNode(page_id=next_page, signature=new_sig, path_actions=cur.path_actions + [action.id])

    def search(self, max_depth: int = 30) -> Dict[str, Any]:
        start = self.initial_state()
        q: Deque[BFSNode] = deque([start])
        visited: Set[Tuple[str, str]] = {(start.page_id, hash_signature(start.signature))}
        shortest_paths: Dict[str, List[str]] = {}
        expansions = 0
        while q:
            cur = q.popleft()
            if cur.page_id in self.terminal_pages and cur.page_id not in shortest_paths:
                shortest_paths[cur.page_id] = cur.path_actions
            if len(cur.path_actions) >= max_depth: continue
            for act in self.pages[cur.page_id].actions:
                if act.src != cur.page_id: continue
                if not self._applicable(act, cur.signature): continue
                nxt = self._next_state(act, cur)
                key = (nxt.page_id, hash_signature(nxt.signature))
                if key in visited: continue
                visited.add(key); q.append(nxt); expansions += 1
        unreachable_pages = [pid for pid in self.pages if all(pid != v[0] for v in visited)]
        return {
            "initial_page_id": self.initial_page,
            "terminal_pages": list(self.terminal_pages),
            "reached_terminals": {k: {"steps": len(v), "actions": v} for k,v in shortest_paths.items()},
            "unreached_terminals": [t for t in self.terminal_pages if t not in shortest_paths],
            "visited_states": len(visited),
            "expansions": expansions,
            "unreachable_pages": unreachable_pages,
        }

class FSMPathEnumerator(FSMBFS):
    def enumerate_paths(self, max_depth: int = 20, max_paths_per_terminal: int = 100):
        results = defaultdict(list)
        def sig_key(sig): return hash_signature(sig)
        def dfs(cur: BFSNode, path_states: Set[Tuple[str,str]]):
            if cur.page_id in self.terminal_pages:
                results[cur.page_id].append(cur.path_actions[:]); return
            if len(cur.path_actions) >= max_depth: return
            for act in self.pages[cur.page_id].actions:
                if act.src != cur.page_id: continue
                if not self._applicable(act, cur.signature): continue
                nxt = self._next_state(act, cur)
                key = (nxt.page_id, sig_key(nxt.signature))
                if key in path_states: continue
                if self.terminal_pages and all(len(results[t])>=max_paths_per_terminal for t in self.terminal_pages):
                    return
                path_states.add(key); dfs(nxt, path_states); path_states.remove(key)
        start = self.initial_state()
        dfs(start, {(start.page_id, hash_signature(start.signature))})
        return dict(results)

    @staticmethod
    def print_paths_tree(paths: Dict[str, List[List[str]]]) -> None:
        for term, plist in paths.items():
            print(f"\n=== Terminal: {term} (paths={len(plist)}) ===")
            for i, seq in enumerate(plist,1):
                print(f"  Path #{i}:")
                for step, aid in enumerate(seq,1):
                    print(f"    {step:02d}. {aid}")
