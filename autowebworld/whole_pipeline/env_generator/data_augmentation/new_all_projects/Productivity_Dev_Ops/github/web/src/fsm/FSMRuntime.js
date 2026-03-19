import { JSONPath } from 'jsonpath-plus';
import { get, set, unset, cloneDeep, isEqual, size } from 'lodash-es';

export class FSMRuntime {
  constructor(fsmData, context) {
    this.fsm = fsmData;
    this.context = context; // Access to store/router
  }

  // Helper to safely get value from path
  getValue(path, data) {
    // Handle {param} references
    if (path.startsWith('{') && path.endsWith('}')) {
      const paramName = path.slice(1, -1);
      return data[paramName];
    }

    // Handle JSONPath
    if (path.startsWith('$')) {
      try {
        // JSONPath returns an array, we usually want the first item or the array itself if expecting a list
        const result = JSONPath({ path, json: data, wrap: false });
        return result;
      } catch (e) {
        console.warn(`FSM: Error evaluating path ${path}`, e);
        return null;
      }
    }

    return path; // Literal value
  }

  // Helper to evaluate value expression (simple support)
  evaluateExpr(expr, data) {
    // Very basic eval for now, or just return value if not an expression
    // For security and simplicity, we might just support direct value references in the prototype
    return this.getValue(expr, data);
  }

  // Validate preconditions
  checkPreconditions(action, signature) {
    if (!action.preconditions || action.preconditions.length === 0) return true;

    for (const pre of action.preconditions) {
      const currentVal = this.getValue(pre.path, signature);
      let conditionMet = false;

      switch (pre.cond) {
        case 'eq':
        case 'equals':
          conditionMet = isEqual(currentVal, pre.value);
          break;
        case 'ne':
        case 'neq':
          conditionMet = !isEqual(currentVal, pre.value);
          break;
        case 'exists':
        case 'not_null':
          conditionMet = currentVal !== null && currentVal !== undefined;
          break;
        case 'not_exists':
          conditionMet = currentVal === null || currentVal === undefined;
          break;
        case 'length_gt':
          conditionMet = size(currentVal) > pre.value;
          break;
        case 'length_lt':
          conditionMet = size(currentVal) < pre.value;
          break;
        default:
          console.warn(`FSM: Unknown condition operator ${pre.cond}`);
          conditionMet = false;
      }

      if (!conditionMet) return false;
    }

    return true;
  }

  // Apply effects
  applyEffects(action, signature, params = {}) {
    if (!action.effects) return signature;

    // Create a deep clone to avoid direct mutation issues during processing
    const nextSignature = cloneDeep(signature);
    const combinedData = { ...nextSignature, ...params };

    for (const effect of action.effects) {
      const path = effect.path;
      // Convert JSONPath $.prop to prop for lodash set
      const lodashPath = path.replace(/^\$\./, ''); 

      switch (effect.op) {
        case 'set':
          let value;
          if (effect.value !== undefined) {
            value = effect.value;
          } else if (effect.value_ref) {
            value = this.getValue(effect.value_ref, params); // ref usually comes from params
          } else if (effect.value_expr) {
             // Simple eval handling: replace {var} with value
             let expr = effect.value_expr;
             // This is a simplified expression handler
             value = expr; 
          }
          set(nextSignature, lodashPath, value);
          break;
        case 'clear':
        case 'unset':
          // For our usage, clearing usually means setting to null or removing
          // FSM spec often implies setting to null for "clear"
          set(nextSignature, lodashPath, null);
          break;
        default:
          console.warn(`FSM: Unknown effect op ${effect.op}`);
      }
    }
    return nextSignature;
  }
}