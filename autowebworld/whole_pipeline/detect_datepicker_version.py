#!/usr/bin/env python3

import json
import sys
import os


def detect_datepicker_version(fsm_file):
    if not os.path.exists(fsm_file):
        return "none", f"FSM file not found: {fsm_file}"
    
    try:
        with open(fsm_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        return "none", f"Failed to parse FSM JSON: {e}"
    
    has_datepicker = False
    has_hour_minute = False
    datepicker_count = 0
    
    if 'pages' not in data:
        return "none", "No pages found in FSM"
    
    for page in data['pages']:
        if 'actions' not in page:
            continue
        
        for action in page['actions']:
            if 'gui_procedure' not in action:
                continue
            
            gui_procedure = action['gui_procedure']
            
            procedure_text = json.dumps(gui_procedure)
            if 'date-picker' not in procedure_text:
                continue
            
            has_datepicker = True
            datepicker_count += 1
            
            for step in gui_procedure:
                if not isinstance(step, dict):
                    continue
                
                selector = step.get('selector', '')
                
                if '.hour-' in selector or '.minute-' in selector:
                    has_hour_minute = True
                    break
            
            if has_hour_minute:
                break
        
        if has_hour_minute:
            break
    
    if not has_datepicker:
        return "full", "No date-picker found in gui_procedures, defaulting to full version"
    elif has_hour_minute:
        return "full", f"Found {datepicker_count} date-picker(s) with hour/minute selectors"
    else:
        return "date_only", f"Found {datepicker_count} date-picker(s) with only date selectors"


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_datepicker_version.py <fsm_file>")
        sys.exit(1)
    
    fsm_file = sys.argv[1]
    version, reason = detect_datepicker_version(fsm_file)
    
    print(version)
    
    if len(sys.argv) > 2 and sys.argv[2] == "--verbose":
        print(f"Reason: {reason}", file=sys.stderr)


if __name__ == "__main__":
    main()

