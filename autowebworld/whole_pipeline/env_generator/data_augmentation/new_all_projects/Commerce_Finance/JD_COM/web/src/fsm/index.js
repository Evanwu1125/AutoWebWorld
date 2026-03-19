import { JSONPath } from 'jsonpath-plus';
import { get, set, cloneDeep } from 'lodash-es';

export class FSMRuntime {
  constructor(fsmData, context) {
    this.fsm = fsmData;
    this.context = context; // Should provide access to store
  }

  // Evaluate a condition against the store state
  evaluateCondition(condition, state) {
    const { path, cond, value } = condition;
    
    // Resolve path value from state
    // JSONPath returns an array, we usually want the first value for single fields
    const result = JSONPath({ path, json: state, wrap: false });
    const actualValue = result === undefined ? null : result;

    switch (cond) {
      case 'eq':
        return actualValue === value;
      case 'neq':
      case 'ne':
        return actualValue !== value;
      case 'exists':
        return actualValue !== null && actualValue !== undefined;
      case 'not_exists':
        return actualValue === null || actualValue === undefined;
      case 'gt':
        return actualValue > value;
      case 'lt':
        return actualValue < value;
      case 'length_gt':
        return (actualValue?.length || 0) > value;
      case 'length_lt':
        return (actualValue?.length || 0) < value;
      default:
        console.warn(`Unknown condition operator: ${cond}`);
        return false;
    }
  }

  // Check all preconditions for an action
  checkPreconditions(actionId, state) {
    // Find action in pages
    let action = null;
    for (const page of this.fsm.pages) {
      const found = page.actions.find(a => a.id === actionId);
      if (found) {
        action = found;
        break;
      }
    }

    if (!action) return true;
    if (!action.preconditions || action.preconditions.length === 0) return true;

    return action.preconditions.every(cond => this.evaluateCondition(cond, state));
  }
}

// Helper to set value by JSON Path (simplified for simple paths used in FSM)
export function setByPath(obj, pathString, value) {
    // Convert $.prop.subprop to prop.subprop
    const cleanPath = pathString.replace(/^\$\./, '');
    set(obj, cleanPath, value);
}