import { defineStore } from 'pinia'

export const useSignatureStore = defineStore('signature', {
  state: () => ({
    // Global
    current_page_id: 'HOME',
    current_user_id: 'user_123', // Default logged in user
    cookie_consent_given: null,

    // Permission
    location_permission_granted: null,

    // HOME
    // (no specific signature fields other than global ones used in HOME)

    // NOTEBOOK_LIST
    notebooks: [], // Populated from data store usually, but FSM defines it here
    NOTEBOOK_LIST_filters_applied: null,
    NOTEBOOK_LIST_viewport_anchor_id: null,
    NOTEBOOK_LIST_matched_notebook_id: null,
    NOTEBOOK_LIST_has_searched: null,
    selected_notebook_id: null,

    // NOTEBOOK_CREATE
    new_notebook_name: null,
    new_notebook_color: null,

    // SECTION_LIST
    sections: [],
    selected_section_id: null,
    SECTION_LIST_filters_applied: null,
    SECTION_LIST_viewport_anchor_id: null,
    SECTION_LIST_matched_section_id: null,
    SECTION_LIST_has_searched: null,

    // SECTION_CREATE
    new_section_name: null,

    // PAGE_LIST
    pages: [],
    selected_page_id: null,
    PAGE_LIST_filters_applied: null,
    PAGE_LIST_viewport_anchor_id: null,
    PAGE_LIST_matched_page_id: null,
    PAGE_LIST_has_searched: null,

    // NOTE_EDITOR
    note_title: null,
    note_body: null,
    note_tag_color: null,

    // NOTE_REVIEW
    // (uses selected_page_id, note_title, note_body from EDITOR usually, but defined here too)

    // NOTE_SHARE
    share_email: null,
    share_permission_level: null,

    // NOTE_DELETE_CONFIRM
    delete_confirmation_checked: null,

    // RECENT_NOTES
    recent_pages: [],
    RECENT_NOTES_filters_applied: null,
    RECENT_NOTES_viewport_anchor_id: null,
    RECENT_NOTES_matched_page_id: null,
    RECENT_NOTES_has_searched: null,

    // QUICK_NOTES
    quick_notes: [],
    QUICK_NOTES_viewport_anchor_id: null,
    selected_quick_note_id: null,

    // SETTINGS
    theme: null,
    sync_enabled: null,

    // Success Pages
    sign_up_message: null
  }),
  actions: {
    setCurrentPageId(pageId) {
      this.current_page_id = pageId
    }
  },
  persist: {
    storage: sessionStorage
  }
})