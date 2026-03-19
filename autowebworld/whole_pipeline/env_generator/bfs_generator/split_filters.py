#!/usr/bin/env python3
import json
import copy
from typing import Dict, List, Any

def split_filter_gui_procedures(gui_procedures: List[Dict]) -> List[List[Dict]]:
    atomic_groups = []
    i = 0

    while i < len(gui_procedures):
        step = gui_procedures[i]
        op = step.get('op')

        if op == 'click' and i + 1 < len(gui_procedures):
            next_step = gui_procedures[i + 1]
            if next_step.get('op') == 'type_text':
                atomic_groups.append([step, next_step])
                i += 2
                continue

        if op == 'click' and i + 1 < len(gui_procedures):
            next_step = gui_procedures[i + 1]
            if next_step.get('op') == 'drag':
                atomic_groups.append([step, next_step])
                i += 2
                continue

        if op == 'hover' and i + 1 < len(gui_procedures):
            next_step = gui_procedures[i + 1]
            if next_step.get('op') == 'click':
                atomic_groups.append([step, next_step])
                i += 2
                continue

        if op == 'click':
            atomic_groups.append([step])
            i += 1
            continue

        i += 1

    return atomic_groups

def create_atomic_filter_action(base_action: Dict, group_index: int, gui_group: List[Dict],
                                 filter_type: str) -> Dict:
    action = copy.deepcopy(base_action)

    base_id = base_action['id'].replace('ACT_APPLY_', 'ACT_FILTER_')

    first_op = gui_group[0].get('op')
    selector = gui_group[0].get('selector', '')

    if len(gui_group) == 2:
        second_op = gui_group[1].get('op')
        if first_op == 'click' and second_op == 'type_text':
            # click + type_text -> SEARCH
            suffix = 'SEARCH'
        elif first_op == 'click' and second_op == 'drag':
            # click + drag -> SLIDER
            suffix = 'SLIDER'
        elif first_op == 'hover' and second_op == 'click':
            # hover + click -> DROPDOWN
            suffix = 'DROPDOWN'
        else:
            suffix = f'FILTER_{group_index + 1}'
    elif first_op == 'click' and 'checkbox' in selector:
        suffix = 'CHECKBOX'
    else:
        suffix = f'FILTER_{group_index + 1}'

    action['id'] = f"{base_id}_{suffix}_{group_index + 1}"
    action['gui_procedure'] = gui_group

    return action

def normalize_keyboard_keys(fsm: Dict[str, Any]) -> Dict[str, Any]:
    modified_count = 0

    for page in fsm.get('pages', []):
        for action in page.get('actions', []):
            gui_procedure = action.get('gui_procedure', [])

            for step in gui_procedure:
                if step.get('op') == 'key_press':
                    key = step.get('key', '')
                    if 'Ctrl+' in key:
                        new_key = key.replace('Ctrl+', 'Control+')
                        step['key'] = new_key
                        modified_count += 1
                        print(f"  Normalized key: {key} -> {new_key}")

    if modified_count > 0:
        print(f"✅ Normalized {modified_count} keyboard keys in total")

    return fsm

def process_fsm(fsm: Dict[str, Any]) -> Dict[str, Any]:
    new_fsm = copy.deepcopy(fsm)
    
    for page in new_fsm.get('pages', []):
        new_actions = []
        
        for action in page.get('actions', []):
            action_id = action.get('id', '')

            if 'APPLY' in action_id and 'FILTER' in action_id:
                print(f"Splitting filter action: {action_id}")
                gui_procedures = action.get('gui_procedure', [])
                
                if len(gui_procedures) > 2:
                    atomic_groups = split_filter_gui_procedures(gui_procedures)
                    print(f"  Split into {len(atomic_groups)} atomic actions")
                    
                    for idx, group in enumerate(atomic_groups):
                        atomic_action = create_atomic_filter_action(
                            action, idx, group, page['id']
                        )
                        new_actions.append(atomic_action)
                        print(f"    - {atomic_action['id']}")
                else:
                    new_actions.append(action)
            else:
                new_actions.append(action)
        
        page['actions'] = new_actions
    
    return new_fsm

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Split filter actions in the FSM into atomic actions')
    parser.add_argument('--input', '-i', required=True, help='Input FSM file path')
    parser.add_argument('--output', '-o', required=True, help='Output FSM file path')
    args = parser.parse_args()

    print(f"Reading FSM: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        fsm = json.load(f)

    print("Normalizing keyboard keys...")
    fsm = normalize_keyboard_keys(fsm)

    new_fsm = process_fsm(fsm)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(new_fsm, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Processing completed! The new FSM has been saved to: {args.output}")

if __name__ == '__main__':
    main()

