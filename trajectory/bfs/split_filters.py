#!/usr/bin/env python3
"""拆分 FSM filter 操作为原子操作"""
import json
import copy
from typing import Dict, List, Any

def split_filter_gui_procedures(gui_procedures: List[Dict]) -> List[List[Dict]]:
    """拆分 filter 为原子操作: click+type_text(search) / click+drag(slider) / hover+click(dropdown) / click(checkbox)"""
    atomic_groups = []
    i = 0

    while i < len(gui_procedures):
        step = gui_procedures[i]
        op = step.get('op')

        # 模式 0: click + type_text (search 操作)
        if op == 'click' and i + 1 < len(gui_procedures):
            next_step = gui_procedures[i + 1]
            if next_step.get('op') == 'type_text':
                atomic_groups.append([step, next_step])
                i += 2
                continue

        # 模式 1: click + drag (slider)
        if op == 'click' and i + 1 < len(gui_procedures):
            next_step = gui_procedures[i + 1]
            if next_step.get('op') == 'drag':
                atomic_groups.append([step, next_step])
                i += 2
                continue

        # 模式 2: hover + click (dropdown)
        if op == 'hover' and i + 1 < len(gui_procedures):
            next_step = gui_procedures[i + 1]
            if next_step.get('op') == 'click':
                atomic_groups.append([step, next_step])
                i += 2
                continue

        # 模式 3: 单独的 click (checkbox)
        if op == 'click':
            atomic_groups.append([step])
            i += 1
            continue

        # 其他情况跳过
        i += 1

    return atomic_groups

def create_atomic_filter_action(base_action: Dict, group_index: int, gui_group: List[Dict],
                                 filter_type: str) -> Dict:
    """创建一个原子过滤操作"""
    action = copy.deepcopy(base_action)

    # 生成新的 action ID
    base_id = base_action['id'].replace('ACT_APPLY_', 'ACT_FILTER_')

    # 根据 GUI 操作推断过滤器类型
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
        # 单独的 click -> CHECKBOX
        suffix = 'CHECKBOX'
    else:
        suffix = f'FILTER_{group_index + 1}'

    # 为同类型的 filter 添加序号
    action['id'] = f"{base_id}_{suffix}_{group_index + 1}"
    action['gui_procedure'] = gui_group

    return action

def normalize_keyboard_keys(fsm: Dict[str, Any]) -> Dict[str, Any]:
    """
    规范化 FSM 中的键盘按键，将 Ctrl 替换为 Control（Playwright 兼容）

    遍历所有 pages -> actions -> gui_procedure，找到 op 为 key_press 的步骤，
    将 key 字段中的 "Ctrl+" 替换为 "Control+"
    """
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
                        print(f"  规范化按键: {key} -> {new_key}")

    if modified_count > 0:
        print(f"✅ 共规范化 {modified_count} 个键盘按键")

    return fsm

def process_fsm(fsm: Dict[str, Any]) -> Dict[str, Any]:
    """处理 FSM，拆分 filter 操作为原子操作"""
    new_fsm = copy.deepcopy(fsm)
    
    for page in new_fsm.get('pages', []):
        new_actions = []
        
        for action in page.get('actions', []):
            action_id = action.get('id', '')

            # 拆分 filter 操作
            if 'APPLY' in action_id and 'FILTER' in action_id:
                print(f"拆分 filter 操作: {action_id}")
                gui_procedures = action.get('gui_procedure', [])
                
                if len(gui_procedures) > 2:  # 只拆分复杂的 filter
                    atomic_groups = split_filter_gui_procedures(gui_procedures)
                    print(f"  拆分为 {len(atomic_groups)} 个原子操作")
                    
                    for idx, group in enumerate(atomic_groups):
                        atomic_action = create_atomic_filter_action(
                            action, idx, group, page['id']
                        )
                        new_actions.append(atomic_action)
                        print(f"    - {atomic_action['id']}")
                else:
                    # 简单的 filter 保持不变
                    new_actions.append(action)
            else:
                # 其他操作保持不变
                new_actions.append(action)
        
        page['actions'] = new_actions
    
    return new_fsm

def main():
    import argparse
    parser = argparse.ArgumentParser(description='拆分 FSM 中的 filter 操作为原子操作')
    parser.add_argument('--input', '-i', required=True, help='输入的 FSM 文件路径')
    parser.add_argument('--output', '-o', required=True, help='输出的 FSM 文件路径')
    args = parser.parse_args()

    # 读取 fsm.json
    print(f"读取 FSM: {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        fsm = json.load(f)

    # 规范化键盘按键（Ctrl -> Control）
    print("规范化键盘按键...")
    fsm = normalize_keyboard_keys(fsm)

    # 处理 FSM
    new_fsm = process_fsm(fsm)

    # 保存新的 FSM
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(new_fsm, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 处理完成！新的 FSM 已保存到: {args.output}")

if __name__ == '__main__':
    main()

