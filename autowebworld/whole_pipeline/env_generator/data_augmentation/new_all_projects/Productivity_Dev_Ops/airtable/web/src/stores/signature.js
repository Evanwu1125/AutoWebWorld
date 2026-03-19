import { defineStore } from 'pinia'
import { ref } from 'vue'
import { useDataStore } from './data'

export const useSignatureStore = defineStore('signature', () => {
  // Global State (FSM Signature)
  const currentPageId = ref('HOME')
  
  // -- Auth & System --
  const current_user_id = ref('user_001')
  const cookie_consent_given = ref(null) // null, true, false
  const location_permission_granted = ref(null)

  // -- Bases Dashboard --
  const bases = ref([]) // loaded from data store
  const bases_dashboard_filters_applied = ref(false)
  const bases_dashboard_viewport_anchor_id = ref(null)
  const bases_dashboard_has_searched = ref(false)
  const matched_base_id = ref(null)
  const selected_base_id = ref(null)

  // -- Base Creation --
  const base_name_input = ref('')
  const base_color = ref('blue')
  const base_icon = ref('grid')
  const template_choice = ref('')
  const created_base_id = ref(null)

  // -- Base Workspace --
  const tables = ref([])
  const selected_table_id = ref(null)

  // -- Records (Grid/Kanban) --
  const records = ref([])
  const table_grid_filters_applied = ref(false)
  const table_grid_viewport_anchor_id = ref(null)
  const table_grid_has_searched = ref(false)
  const matched_record_id = ref(null)
  const selected_record_id = ref(null)
  
  // -- Kanban Specific --
  const kanban_viewport_anchor_id = ref(null)
  const kanban_filters_applied = ref(false)

  // -- Record Forms (Create/Edit) --
  const field_title_input = ref('')
  const field_status_select = ref('')
  const field_due_date = ref('')
  const created_record_id = ref(null)
  
  const edit_title_input = ref('')
  const edit_status_select = ref('')
  const updated_record_id = ref(null)

  // -- Automations --
  const automations = ref([])
  const automations_filters_applied = ref(false)
  const automations_viewport_anchor_id = ref(null)
  const trigger_type = ref('')
  const action_type = ref('')
  const email_recipient = ref('')
  const created_automation_id = ref(null)

  // -- Public Forms --
  const form_name = ref('')
  const form_email = ref('')
  const form_status = ref('')
  const submitted_record_id = ref(null)

  // Actions
  function setCurrentPageId(pageId) {
    currentPageId.value = pageId
  }

  // Helper to load initial data if array is empty
  function loadInitialData() {
    const dataStore = useDataStore()
    if (bases.value.length === 0) {
      bases.value = dataStore.bases
    }
    // We might load other lists dynamically based on selection, 
    // but for simplicity in this FSM mapping, we can initialize empty arrays 
    // and populate them when 'selected_base_id' changes or pages load.
  }

  return {
    currentPageId,
    setCurrentPageId,
    loadInitialData,

    // State Exports
    current_user_id,
    cookie_consent_given,
    location_permission_granted,
    
    bases,
    bases_dashboard_filters_applied,
    bases_dashboard_viewport_anchor_id,
    bases_dashboard_has_searched,
    matched_base_id,
    selected_base_id,
    
    base_name_input,
    base_color,
    base_icon,
    template_choice,
    created_base_id,
    
    tables,
    selected_table_id,
    
    records,
    table_grid_filters_applied,
    table_grid_viewport_anchor_id,
    table_grid_has_searched,
    matched_record_id,
    selected_record_id,
    
    kanban_viewport_anchor_id,
    kanban_filters_applied,
    
    field_title_input,
    field_status_select,
    field_due_date,
    created_record_id,
    
    edit_title_input,
    edit_status_select,
    updated_record_id,
    
    automations,
    automations_filters_applied,
    automations_viewport_anchor_id,
    trigger_type,
    action_type,
    email_recipient,
    created_automation_id,
    
    form_name,
    form_email,
    form_status,
    submitted_record_id
  }
}, {
  persist: {
    storage: sessionStorage
  }
})