import { defineStore } from 'pinia'
import { ref } from 'vue'
import { FSMRuntime } from '../fsm/runtime'
import fsmData from '../../fsm.json'

export const useSignatureStore = defineStore('signature', () => {
  // Global State matching fsm.signature_schema
  // Home Page State
  const current_user_id = ref(null)
  const cookie_consent_given = ref(null)

  // Workspace Page State
  const location_permission_granted = ref(null)
  const workspaces = ref(null)
  const selected_workspace_id = ref(null)

  // Channel List Page State
  const channels = ref(null)
  const dms = ref(null)
  const channel_list_filters_applied = ref(null)
  const channel_list_has_searched = ref(null)
  const channel_list_viewport_anchor_id = ref(null)
  const matched_channel_id = ref(null)
  const selected_channel_id = ref(null)
  const matched_dm_id = ref(null)
  const selected_dm_id = ref(null)

  // DM List Page State
  const dm_list_filters_applied = ref(null)
  const dm_list_has_searched = ref(null)
  const dm_list_viewport_anchor_id = ref(null)

  // Channel Detail & Settings
  const messages = ref(null)
  const channel_name = ref(null)
  const channel_description = ref(null)
  const channel_privacy = ref(null)

  // Message Compose
  const compose_text = ref(null)
  const compose_has_mention = ref(null)
  const compose_has_emoji = ref(null)

  // Message Schedule
  const schedule_date_time = ref(null)
  const schedule_text = ref(null)

  // DM Compose
  const dm_compose_text = ref(null)

  // Profile
  const profile_name = ref(null)
  const profile_title = ref(null)
  const profile_status = ref(null)

  // Success Pages
  const success_message = ref(null)

  // Current Page Tracker
  const currentPageId = ref('HOME')

  // FSM Runtime
  const runtime = new FSMRuntime(fsmData, {
    get currentPageId() {
      return currentPageId.value
    }
  })

  function setCurrentPageId(pageId) {
    currentPageId.value = pageId
  }

  return {
    // State refs
    current_user_id,
    cookie_consent_given,
    location_permission_granted,
    workspaces,
    selected_workspace_id,
    channels,
    dms,
    channel_list_filters_applied,
    channel_list_has_searched,
    channel_list_viewport_anchor_id,
    matched_channel_id,
    selected_channel_id,
    matched_dm_id,
    selected_dm_id,
    dm_list_filters_applied,
    dm_list_has_searched,
    dm_list_viewport_anchor_id,
    messages,
    channel_name,
    channel_description,
    channel_privacy,
    compose_text,
    compose_has_mention,
    compose_has_emoji,
    schedule_date_time,
    schedule_text,
    dm_compose_text,
    profile_name,
    profile_title,
    profile_status,
    success_message,
    
    // Core methods
    currentPageId,
    setCurrentPageId,
    runtime
  }
}, {
  persist: {
    storage: sessionStorage
  }
})