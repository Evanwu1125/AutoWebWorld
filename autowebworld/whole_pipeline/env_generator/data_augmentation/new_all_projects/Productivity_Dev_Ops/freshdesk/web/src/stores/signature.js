import { defineStore } from 'pinia'

export const useSignatureStore = defineStore('signature', {
  state: () => ({
    // Core State
    currentPageId: 'HOME',
    
    // HOME
    current_agent_id: 'agent_1', // Default logged in agent
    cookie_consent_given: null,
    home_nav_target: null,

    // TICKETS
    tickets: [], // Will be populated from data store
    location_permission_granted: null,
    matched_ticket_id: null,
    selected_ticket_id: null,
    tickets_list_has_searched: null,
    tickets_list_viewport_anchor_id: null,
    tickets_list_filters_applied: null,

    // TICKET DETAIL & ACTIONS
    ticket_reply_draft: null,
    ticket_status: null,
    ticket_priority: null,
    ticket_assignee_id: null,
    ticket_merge_target_id: null,
    created_ticket_id: null,
    success_message: null,

    // NEW TICKET
    new_ticket_subject: null,
    new_ticket_description: null,
    new_ticket_priority: null,
    new_ticket_group: null,

    // MERGE
    merge_search_has_searched: null,
    merge_viewport_anchor_id: null,
    matched_merge_ticket_id: null,

    // CONTACTS
    contacts: [], // Will be populated from data store
    contacts_list_filters_applied: null,
    contacts_list_viewport_anchor_id: null,
    contacts_list_has_searched: null,
    matched_contact_id: null,
    selected_contact_id: null,
    created_contact_id: null,

    // NEW CONTACT
    new_contact_name: null,
    new_contact_email: null,
    new_contact_segment: null,

    // DASHBOARD
    dashboard_filters_applied: null
  }),
  actions: {
    setCurrentPageId(id) {
      this.currentPageId = id
    },
    // Generic setter for FSM effects
    set(path, value) {
      // Simple path traversal handling
      const parts = path.replace('$.', '').split('.')
      let target = this
      for (let i = 0; i < parts.length - 1; i++) {
        target = target[parts[i]]
      }
      target[parts[parts.length - 1]] = value
    }
  },
  persist: {
    storage: sessionStorage
  }
})