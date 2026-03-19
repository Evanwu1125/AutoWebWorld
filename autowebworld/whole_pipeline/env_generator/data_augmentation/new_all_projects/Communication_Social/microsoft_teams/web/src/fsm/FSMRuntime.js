import { JSONPath } from 'jsonpath-plus';
import { cloneDeep, set, get, isEqual } from 'lodash-es';

export class FSMRuntime {
  constructor(fsmData, signatureStore) {
    this.fsm = fsmData;
    this.store = signatureStore;
  }

  get currentPageId() {
    return this.store.currentPageId;
  }

  // Evaluate a condition against the store state
  evaluateCondition(condition, state) {
    const { path, cond, value } = condition;
    // JSONPath returns an array of matches. 
    // If we expect a single scalar, we usually take the first element or check existence.
    const currentValues = JSONPath({ path, json: state });
    const currentValue = currentValues && currentValues.length > 0 ? currentValues[0] : undefined;

    switch (cond) {
      case 'eq':
      case 'equals':
        return isEqual(currentValue, value);
      case 'ne':
      case 'neq':
        return !isEqual(currentValue, value);
      case 'exists':
      case 'not_null':
        return currentValue !== undefined && currentValue !== null;
      case 'not_exists':
        return currentValue === undefined || currentValue === null;
      case 'length_gt':
        return (currentValue?.length || 0) > value;
      case 'length_lt':
        return (currentValue?.length || 0) < value;
      default:
        console.warn(`Unknown condition operator: ${cond}`);
        return false;
    }
  }

  // Check all preconditions for an action
  checkPreconditions(actionId) {
    const page = this.fsm.pages.find(p => p.id === this.currentPageId);
    if (!page) return false;

    const action = page.actions.find(a => a.id === actionId);
    if (!action) return false;

    const state = this.store.$state; // Access raw state object for read
    
    // If no preconditions, allow
    if (!action.preconditions || action.preconditions.length === 0) return true;

    return action.preconditions.every(cond => this.evaluateCondition(cond, state));
  }

  // Execute effects of an action
  // Note: In this Vue implementation, we often handle effects directly in the component 
  // or store actions, but this helper can be used to interpret the FSM effects logic.
  // However, for complex logic like "value_ref" or "value_expr", custom handlers are often better.
  // This class is mainly for validation logic here.
}