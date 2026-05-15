# Ideas & Discussion Log

## 2026-04-15: FSM gui_procedure 拆分

### 决定
- 将 FSM action 中的 `gui_procedure` 字段剥离，改为独立的 `action_type_mapping.json`
- FSM action 只保留 `name`（action type）+ `selector` + 可选 `ui_elements`
- mapping 文件为静态手写，所有项目共用

### 底层 op（13 种）
click, double_click, long_press, type_text, clear, key_press, hover, drag, scroll_to, focus, select_file, wait, right_click

### 上层 action type（18 种）
| action type | op 组合 |
|---|---|
| click | click |
| type | click + type_text |
| clear | click + clear |
| search | click + type_text + key_press |
| select | click |
| select_dropdown | click + wait + click |
| select_date_calendar | click + wait + click |
| select_date_input | click + type_text |
| toggle | click |
| slider | click + drag |
| hover_menu | hover + wait + click |
| scroll | scroll_to |
| upload_file | click + select_file |
| autocomplete | click + type_text + wait + click |
| drag_drop | drag |
| double_click_edit | double_click + type_text |
| modal_confirm | click + wait + click |
| keyboard_shortcut | key_press |

### 改动范围
- 第 1 层（核心 pipeline）：5 个文件
- 第 2 层（env_generator）：5 个文件
- 第 3 层（web_extractor）：2 个文件
