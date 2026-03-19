import { JSONPath } from 'jsonpath-plus'
import { get, set, cloneDeep, isEqual } from 'lodash-es'

export class FSMRuntime {
  constructor(fsmData, context) {
    this.fsm = fsmData
    this.context = context
  }

  // Evaluate a condition against the store state
  evaluateCondition(condition, store) {
    // Resolve value from store path
    const path = condition.path
    // JSONPath returns array, get first item or null
    const result = JSONPath({ path, json: store.$state, wrap: false })
    const value = result === undefined ? null : result

    // Check operators
    const op = condition.cond
    const target = condition.value

    switch (op) {
      case 'eq':
        return isEqual(value, target)
      case 'ne':
      case 'neq':
        return !isEqual(value, target)
      case 'exists':
        return value !== null && value !== undefined
      case 'not_exists':
        return value === null || value === undefined
      case 'length_gt':
        return (value?.length || 0) > target
      case 'length_lt':
        return (value?.length || 0) < target
      default:
        console.warn(`Unknown condition operator: ${op}`)
        return false
    }
  }

  // Check if an action's preconditions are met
  checkPreconditions(actionId, store) {
    const page = this.fsm.pages.find(p => p.id === this.context.currentPageId)
    if (!page) return false

    const action = page.actions.find(a => a.id === actionId)
    if (!action) return false

    if (!action.preconditions || action.preconditions.length === 0) return true

    return action.preconditions.every(cond => this.evaluateCondition(cond, store))
  }
}