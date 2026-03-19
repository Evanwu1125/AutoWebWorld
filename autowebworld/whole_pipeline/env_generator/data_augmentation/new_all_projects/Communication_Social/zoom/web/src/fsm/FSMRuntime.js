import { JSONPath } from 'jsonpath-plus';
import { get, set, unset, cloneDeep, has, isNil } from 'lodash-es';

export class FSMRuntime {
  constructor(fsmData, context) {
    this.fsm = fsmData;
    this.context = context; // { get currentPageId() }
  }

  getCurrentPage() {
    const pageId = this.context.currentPageId;
    return this.fsm.pages.find(p => p.id === pageId);
  }

  evaluateCondition(signature, condition) {
    const { path, cond, value } = condition;
    // Use wrap: false to get the value directly, returns undefined if not found
    const currentValue = JSONPath({ path, json: signature, wrap: false });

    switch (cond) {
      case 'eq': return currentValue === value;
      case 'ne': case 'neq': return currentValue !== value;
      case 'exists': return !isNil(currentValue);
      case 'not_exists': return isNil(currentValue);
      case 'length_gt': return (currentValue?.length || 0) > value;
      case 'length_lt': return (currentValue?.length || 0) < value;
      case 'gt': return Number(currentValue) > Number(value);
      case 'lt': return Number(currentValue) < Number(value);
      case 'gte': return Number(currentValue) >= Number(value);
      case 'lte': return Number(currentValue) <= Number(value);
      default: 
        console.warn(`Unknown condition operator: ${cond}`);
        return false;
    }
  }

  checkPreconditions(signature, action) {
    if (!action.preconditions || action.preconditions.length === 0) return true;
    return action.preconditions.every(cond => this.evaluateCondition(signature, cond));
  }

  // Helper to safely set value at JSONPath
  setAtPath(obj, path, value) {
    // Convert JSONPath ($.prop.nested) to lodash path (prop.nested)
    // Remove leading $.
    const lodashPath = path.replace(/^\$\.?/, '');
    set(obj, lodashPath, value);
  }

  // Helper to safely clear value at JSONPath
  clearAtPath(obj, path) {
    const lodashPath = path.replace(/^\$\.?/, '');
    // For arrays or objects, we might want to set to null or remove. 
    // FSM spec usually implies setting to null/undefined or removing.
    // Lodash unset removes the property.
    // If we want to set to null (as per typical FSM null usage):
    set(obj, lodashPath, null);
  }

  applyEffects(signature, action, parameters = {}) {
    if (!action.effects) return;

    // We operate on the signature object directly (it's reactive in store)
    action.effects.forEach(effect => {
      const { op, path, value, value_ref, value_expr } = effect;

      if (op === 'set') {
        let finalValue = value;
        
        if (value_ref) {
          // Resolve reference from parameters
          // value_ref might be "{item_id}" -> resolve to parameters.item_id
          const paramName = value_ref.replace(/^{|}$/g, '');
          if (parameters[paramName] !== undefined) {
            finalValue = parameters[paramName];
          } else {
             // If strictly {item_id}, and param is item_id.
             finalValue = value_ref; // fallback if not found (shouldn't happen in valid FSM)
          }
        } else if (value_expr) {
           // Simple expression evaluation if needed (FSM usually keeps it simple)
           // For this prompt, we assume basic value setting mostly.
           // If value_expr is "{count} + 1", we'd need a parser.
           // Current FSM spec uses value or value_ref mostly.
           finalValue = value_expr; // Placeholder logic
        }

        this.setAtPath(signature, path, finalValue);
      } else if (op === 'clear') {
        this.clearAtPath(signature, path);
      }
    });
  }
}