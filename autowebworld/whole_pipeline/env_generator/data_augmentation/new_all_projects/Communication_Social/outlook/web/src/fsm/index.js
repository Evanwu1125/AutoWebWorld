import { JSONPath } from 'jsonpath-plus';
import { get, set, unset, cloneDeep } from 'lodash-es';

export class FSMRuntime {
  constructor(fsmData, context) {
    this.fsm = fsmData;
    this.context = context; // context provides access to current runtime state (e.g. currentPageId)
  }

  // Evaluate a value expression or reference
  evaluateValue(signature, valueExpr, valueRef, parameters) {
    if (valueExpr !== undefined) {
      // Simple expression evaluation (extensions can be added here)
      if (valueExpr === 'now()') return new Date().toISOString();
      // Basic arithmetic or logic could be added here if needed
      return valueExpr; 
    }
    
    if (valueRef !== undefined) {
      // Replace {param} placeholders with actual parameter values
      let resolvedValue = valueRef;
      if (typeof valueRef === 'string') {
        for (const [key, val] of Object.entries(parameters || {})) {
          resolvedValue = resolvedValue.replace(new RegExp(`{${key}}`, 'g'), val);
        }
        
        // Check if it's a JSONPath reference (starts with $.)
        if (resolvedValue.startsWith('$.')) {
           const result = JSONPath({ path: resolvedValue, json: signature, wrap: false });
           return result;
        }
      }
      return resolvedValue;
    }
    
    return undefined;
  }

  // Check if all preconditions are met
  checkPreconditions(preconditions, signature) {
    if (!preconditions || preconditions.length === 0) return true;

    return preconditions.every(condition => {
      const { path, cond, value } = condition;
      // Get value from signature using JSONPath
      const currentVal = JSONPath({ path, json: signature, wrap: false });

      switch (cond) {
        case 'eq':
        case 'equals':
          return currentVal === value;
        case 'ne':
        case 'neq':
          return currentVal !== value;
        case 'exists':
        case 'not_null':
          return currentVal !== undefined && currentVal !== null;
        case 'not_exists':
          return currentVal === undefined || currentVal === null;
        case 'length_gt':
          return (currentVal?.length || 0) > value;
        case 'length_lt':
          return (currentVal?.length || 0) < value;
        default:
          console.warn(`Unknown condition operator: ${cond}`);
          return false;
      }
    });
  }

  // Apply effects to the signature
  applyEffects(effects, signature, parameters) {
    if (!effects || effects.length === 0) return signature;

    const newSignature = cloneDeep(signature);

    effects.forEach(effect => {
      const { op, path, value, value_expr, value_ref } = effect;
      
      // Convert JSONPath $.a.b to dot notation a.b for lodash
      const dotPath = path.replace(/^\$\./, '');

      if (op === 'set') {
        let valToSet = value;
        
        if (value_expr || value_ref) {
          valToSet = this.evaluateValue(newSignature, value_expr, value_ref, parameters);
        }
        
        // Handle direct parameter substitution in value if it's a string
        if (typeof valToSet === 'string' && parameters) {
            for (const [key, paramVal] of Object.entries(parameters)) {
                valToSet = valToSet.replace(new RegExp(`{${key}}`, 'g'), paramVal);
            }
        }

        set(newSignature, dotPath, valToSet);
      } else if (op === 'clear') {
        // We set to null for 'clear' as per typical FSM behavior for fields
        set(newSignature, dotPath, null);
      }
    });

    return newSignature;
  }
}