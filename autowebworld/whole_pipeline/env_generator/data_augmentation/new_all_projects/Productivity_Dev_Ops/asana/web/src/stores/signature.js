import { defineStore } from 'pinia'
import { JSONPath } from 'jsonpath-plus'
import { get, set, cloneDeep } from 'lodash-es'

export const useSignatureStore = defineStore('signature', {
  state: () => ({
    // Global State
    currentPageId: 'HOME',
    
    // HOME
    current_user_id: null,
    cookie_consent_given: null,

    // PROJECTS_LIST
    projects: [], // References to data store items or local copies if needed
    project_list_filters_applied: null,
    project_list_has_searched: null,
    project_list_viewport_anchor_id: null,
    matched_project_id: null,
    selected_project_id: null,
    location_permission_granted: null,

    // PROJECT_CREATE_FORM
    new_project_name: null,
    new_project_description: null,
    new_project_due_date: null,

    // PROJECT_CREATE_SUCCESS / SECTION_CREATE_SUCCESS / TASK_COMPLETE_SUCCESS / COMMENT_ADD_SUCCESS
    success_message: null,

    // PROJECT_BOARD
    board_filters_applied: null,
    board_has_searched: null,
    board_viewport_anchor_id: null,
    matched_task_id: null,
    selected_task_id: null,

    // TASK_CREATE_FORM
    new_task_name: null,
    new_task_description: null,
    new_task_due_date: null,
    new_task_assignee: null,

    // SECTION_CREATE_FORM
    new_section_name: null,

    // TASK_DETAIL
    task_completed: null,
    new_comment_text: null,

    // MY_TASKS_LIST
    my_tasks_filters_applied: null,
    my_tasks_has_searched: null,
    my_tasks_viewport_anchor_id: null,
    matched_my_task_id: null,
    selected_my_task_id: null,
    
    // Arrays for FSM effects (if they append to signature instead of data store)
    // The FSM effects like "append($.projects, ...)" imply signature store holds these,
    // but in a real app, these would be in the Data Store. 
    // However, strictly following FSM, we map these fields here.
    // We will sync with Data Store or just use Data Store for display.
  }),
  actions: {
    setCurrentPageId(pageId) {
      this.currentPageId = pageId
    },
    // Generic setter for FSM effects
    setPath(path, value) {
        // Convert $.field to field
        const cleanPath = path.startsWith('$.') ? path.substring(2) : path
        set(this, cleanPath, value)
    },
    getPath(path) {
        // Evaluate JSONPath against state
        // Note: JSONPath usually returns an array, we want the value.
        // For simple properties, direct access is better.
        try {
            const result = JSONPath({ path: path, json: this.$state, wrap: false })
            return result
        } catch (e) {
            console.error('JSONPath error', e)
            return null
        }
    }
  },
  persist: {
    storage: sessionStorage,
  },
})