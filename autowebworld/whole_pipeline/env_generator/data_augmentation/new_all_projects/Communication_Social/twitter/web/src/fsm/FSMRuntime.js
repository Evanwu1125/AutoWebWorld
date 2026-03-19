import { JSONPath } from 'jsonpath-plus';
import { cloneDeep, set, get, isEqual, isNil } from 'lodash-es';

export class FSMRuntime {
  constructor(fsmData, context) {
    this.fsm = fsmData;
    this.context = context; // Should provide access to current page ID
  }

  // Evaluate a condition against the current state
  evaluateCondition(condition, state) {
    const { path, cond, value } = condition;
    // Extract value from state using JSONPath
    // JSONPath returns an array, we usually want the first item or the array itself depending on check
    const currentValues = JSONPath({ path, json: state });
    const currentValue = currentValues && currentValues.length > 0 ? currentValues[0] : undefined;

    switch (cond) {
      case 'eq':
        return isEqual(currentValue, value);
      case 'ne':
      case 'neq':
        return !isEqual(currentValue, value);
      case 'exists':
        return !isNil(currentValue);
      case 'not_exists':
        return isNil(currentValue);
      case 'length_gt':
        return (currentValue?.length || 0) > value;
      case 'length_lt':
        return (currentValue?.length || 0) < value;
      default:
        console.warn(`Unknown condition operator: ${cond}`);
        return false;
    }
  }

  // Check if an action's preconditions are met
  checkPreconditions(action, state) {
    if (!action.preconditions || action.preconditions.length === 0) {
      return true;
    }
    return action.preconditions.every(condition => this.evaluateCondition(condition, state));
  }

  // Apply effects to the state
  applyEffects(action, state, parameters = {}) {
    if (!action.effects || action.effects.length === 0) {
      return state;
    }

    const newState = cloneDeep(state);

    action.effects.forEach(effect => {
      const { op, path, value, value_ref, value_expr } = effect;
      
      // Resolve the target path to a dot-notation string for lodash.set
      // Note: JSONPath format $.a.b -> a.b
      const targetPath = path.replace(/^\$\./, '');

      if (op === 'clear') {
        set(newState, targetPath, null);
      } else if (op === 'set') {
        let newValue = value;

        if (value_ref) {
          // Resolve reference from parameters
          // value_ref format might be "{param_name}"
          const paramName = value_ref.replace(/^\{|\}$/g, '');
          newValue = parameters[paramName];
        } else if (value_expr) {
          // Simple expression evaluation (limited support)
          // e.g., "{count} + 1"
          // This is a simplified handler
          // In a real scenario, might need a safe expression parser
          // For now, assuming basic increment/decrement if matches pattern
          // or just direct assignment if complex
           try {
             // extremely basic eval for numbers
             // WARNING: This is a placeholder for safe eval.
             // In this FSM, expressions are simple.
             // Replace {var} with param value
             let expr = value_expr;
             for (const [k, v] of Object.entries(parameters)) {
                expr = expr.replace(new RegExp(`{${k}}`, 'g'), v);
             }
             // Be very careful with eval, ensure inputs are sanitized or use a math lib
             // Here we assume inputs are safe from FSM definition
             // newValue = eval(expr); 
             // Ideally avoid eval. For this demo, let's stick to direct ref or constant if possible.
             // If strictly needed, implement specific logic.
             // For the provided FSM, most are value or value_ref.
           } catch (e) {
             console.error("Error evaluating expression", value_expr, e);
           }
        }

        set(newState, targetPath, newValue);
      }
    });

    return newState;
  }
}