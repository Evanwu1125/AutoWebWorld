# general_params_provider.py

from typing import Dict, Any, Tuple, List
import hashlib
import random

def type_category(tp: str) -> Tuple[str, str]:
    t = str(tp).strip().lower()
    if t.startswith('array<string>'): return ('array','string')
    if t.startswith('array<number>'): return ('array','number')
    if t.startswith('array'):         return ('array','any')
    if t.startswith('boolean'):       return ('scalar','boolean')
    if t.startswith('integer>=1'):    return ('scalar','integer_ge1')
    if t.startswith('integer'):       return ('scalar','integer')
    if t.startswith('number'):        return ('scalar','number')
    if t.startswith('string'):        return ('scalar','string')
    return ('scalar','any')

def most_specific_type(types: List[Tuple[str,str]]) -> Tuple[str,str]:
    order = {
        ('array','string'):6, ('array','number'):6, ('array','any'):6,
        ('scalar','boolean'):5, ('scalar','integer_ge1'):4, ('scalar','integer'):3,
        ('scalar','number'):2, ('scalar','string'):1, ('scalar','any'):0
    }
    return max(types or [('scalar','any')], key=lambda x: order.get(x, -1))

def element_type_of(tp_tuple: Tuple[str,str]) -> Tuple[str,str]:
    if tp_tuple[0] != 'array': return ('scalar','any')
    return ('scalar', tp_tuple[1] if tp_tuple[1] in ('string','number') else 'any')

def _token(seed_bytes: bytes, n=8) -> str:
    return hashlib.sha256(seed_bytes).hexdigest()[:n]

def _gen_scalar(kind_spec: Tuple[str,str], uniq_key: str) -> Any:
    kind, spec = kind_spec
    h = _token(uniq_key.encode())
    if spec == 'boolean':      return True
    if spec == 'integer_ge1':  return 1
    if spec == 'integer':      return 0
    if spec == 'number':       return 0.0
    return f"s_{h}"  # string/any

def _gen_value_for_type(tp_tuple: Tuple[str,str], uniq_key: str) -> Any:
    if tp_tuple[0] == 'array':
        elem_type = element_type_of(tp_tuple)
        return [_gen_scalar(elem_type, uniq_key+"#0")]
    else:
        return _gen_scalar(tp_tuple, uniq_key)

def _map_param_expected_types(action: Dict[str, Any], page_schema_lookup: Dict[str,str]) -> Dict[str, Tuple[str,str]]:
    m: Dict[str, List[Tuple[str,str]]] = {}
    for ef in action.get('effects', []) or []:
        ref = ef.get('value_ref')
        if isinstance(ref, str) and ref.startswith('{') and ref.endswith('}'):
            pname = ref.strip('{} ').strip()
            path  = ef.get('path')
            tp    = type_category(page_schema_lookup.get(path, 'scalar:any'))
            m.setdefault(pname, []).append(tp)
    return {p: most_specific_type(tps) for p, tps in m.items()}

def params_for_action(action: Dict[str,Any],
                      page_id: str,
                      schema_lookup: Dict[str, Dict[str,str]],
                      signature: Dict[str,Any],
                      seed: int = 42) -> Dict[str,Any]:
    page_schema = schema_lookup.get(page_id, {})
    p2t = _map_param_expected_types(action, page_schema)
    out: Dict[str,Any] = {}
    for pname in (action.get('parameters') or {}).keys():
        tp_tuple = p2t.get(pname, ('scalar','any'))
        out[pname] = _gen_value_for_type(tp_tuple, uniq_key=f"{action.get('id','')}::{pname}")
    return out
