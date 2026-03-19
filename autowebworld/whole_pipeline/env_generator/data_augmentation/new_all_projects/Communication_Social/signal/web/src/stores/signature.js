import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Global State
  const currentPageId = ref('HOME')

  // HOME
  const current_user_id = ref(null)
  const cookie_consent_given = ref(null)

  // CHATS_LIST
  const chats = ref(null)
  const location_permission_granted = ref(null)
  const matched_chat_id = ref(null)
  const selected_chat_id = ref(null)
  const chats_list_has_searched = ref(null)
  const chats_list_viewport_anchor_id = ref(null)
  const chats_list_filters_applied = ref(null)

  // CHAT_THREAD & SEND_MESSAGE
  const draft_message_text = ref(null)
  const send_read_receipts = ref(null)
  const send_disappearing_timer_seconds = ref(null)

  // NEW_CHAT
  const contacts = ref(null)
  const matched_contact_id = ref(null)
  const selected_contact_id = ref(null)
  const new_chat_has_searched = ref(null)
  const new_chat_viewport_anchor_id = ref(null)
  const disappearing_timer_seconds = ref(null) // for NEW_CHAT_COMPOSE

  // CONTACTS_LIST
  const contacts_list_filters_applied = ref(null)
  const contacts_list_viewport_anchor_id = ref(null)
  const contacts_list_has_searched = ref(null)
  
  // BLOCK_USER
  const block_report_reason = ref(null)

  // GROUPS_LIST
  const groups = ref(null)
  const groups_list_filters_applied = ref(null)
  const groups_list_viewport_anchor_id = ref(null)
  const matched_group_id = ref(null)
  const groups_list_has_searched = ref(null)
  const selected_group_id = ref(null)

  // GROUP_CREATE
  const group_name = ref(null)
  const group_description = ref(null)
  const selected_member_ids = ref([]) // array<string>
  const add_members_viewport_anchor_id = ref(null)

  // CALL_HISTORY
  const calls = ref(null)
  const call_history_filters_applied = ref(null)
  const call_history_viewport_anchor_id = ref(null)
  const matched_call_id = ref(null)
  const call_history_has_searched = ref(null)
  const selected_call_id = ref(null)

  // START_CALL
  const call_type = ref(null)

  // SETTINGS
  const read_receipts_enabled = ref(null)
  const typing_indicators_enabled = ref(null)
  const notification_sound = ref(null)
  const notification_vibrate = ref(null)


  function setCurrentPageId(pageId) {
    currentPageId.value = pageId
  }

  // Generic setter for JSONPath updates (simplified for direct store usage)
  function setField(field, value) {
    // This is a helper; in Vue components we'll likely update refs directly
    // based on the mapping, but this can be useful for dynamic updates
    // For now, we rely on direct ref manipulation in components
  }

  return {
    currentPageId,
    setCurrentPageId,
    
    // Exports
    current_user_id,
    cookie_consent_given,
    chats,
    location_permission_granted,
    matched_chat_id,
    selected_chat_id,
    chats_list_has_searched,
    chats_list_viewport_anchor_id,
    chats_list_filters_applied,
    draft_message_text,
    send_read_receipts,
    send_disappearing_timer_seconds,
    contacts,
    matched_contact_id,
    selected_contact_id,
    new_chat_has_searched,
    new_chat_viewport_anchor_id,
    disappearing_timer_seconds,
    contacts_list_filters_applied,
    contacts_list_viewport_anchor_id,
    contacts_list_has_searched,
    block_report_reason,
    groups,
    groups_list_filters_applied,
    groups_list_viewport_anchor_id,
    matched_group_id,
    groups_list_has_searched,
    selected_group_id,
    group_name,
    group_description,
    selected_member_ids,
    add_members_viewport_anchor_id,
    calls,
    call_history_filters_applied,
    call_history_viewport_anchor_id,
    matched_call_id,
    call_history_has_searched,
    selected_call_id,
    call_type,
    read_receipts_enabled,
    typing_indicators_enabled,
    notification_sound,
    notification_vibrate
  }
}, {
  persist: {
    storage: sessionStorage
  }
})