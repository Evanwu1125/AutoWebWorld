import { defineStore } from 'pinia';
import fsmData from '../../fsm.json';
import { FSMRuntime } from '../fsm/FSMRuntime';

export const useSignatureStore = defineStore('signature', {
  state: () => ({
    // Global State
    currentPageId: 'HOME',
    
    // HOME
    current_user_id: null,
    cookie_accepted: null, // boolean|null

    // DASHBOARD
    location_permission_granted: null,
    upcoming_meetings: [],
    past_meetings: [],
    dashboard_meetings_has_searched: null,
    dashboard_meetings_matched_id: null,
    dashboard_meetings_selected_id: null,
    dashboard_meetings_viewport_anchor_id: null,
    dashboard_meetings_filters_applied: null,

    // SCHEDULE_MEETING_FORM
    meeting_templates: [],
    selected_template_id: null,
    meeting_topic: '',
    meeting_description: '',
    meeting_date_time: '',
    meeting_duration_minutes: 30,
    meeting_password: '',
    waiting_room_enabled: false,
    host_video_on: false,

    // SCHEDULE_MEETING_REVIEW
    // Reusing fields from FORM where names match, or explicit if distinct context needed.
    // FSM reuses names usually.
    review_confirmed: null,

    // SCHEDULE_MEETING_SUCCESS
    scheduled_meeting_id: null,
    success_message: '',

    // JOIN_MEETING_FORM
    meeting_id_input: '',
    meeting_name_input: '',
    remember_name: false,

    // JOIN_MEETING_PREVIEW
    audio_join_with_computer: false,
    video_on: false,

    // INSTANT_MEETING_LOBBY
    audio_option: null,
    screen_share_ready: false,
    // video_on reused

    // PROFILE_OVERVIEW
    display_name: 'John Doe',
    email: 'john.doe@zoom.us',
    profile_has_searched: null,
    profile_matched_meeting_id: null,
    profile_selected_meeting_id: null,
    profile_viewport_anchor_id: null,
    profile_filters_applied: null,

    // PROFILE_RENAME_FORM
    current_display_name: 'John Doe',
    new_display_name: '',

    // PROFILE_CHANGE_PASSWORD_FORM
    old_password: '',
    new_password: '',
    confirm_password: '',

    // SETTINGS_GENERAL
    language: 'en',
    theme: 'light',

    // SETTINGS_VIDEO
    mirror_my_video: false,
    touch_up_appearance: false,

    // MEETINGS_LIST
    meetings: [],
    meetings_matched_id: null,
    meetings_selected_id: null,
    meetings_list_has_searched: null,
    meetings_list_viewport_anchor_id: null,
    meetings_list_filters_applied: null,

    // MEETING_DETAIL
    start_from_detail_confirmed: null,
  }),
  getters: {
    // Helper to get FSM runtime instance if needed, but actions usually suffice
  },
  actions: {
    setCurrentPageId(pageId) {
      this.currentPageId = pageId;
    },
    
    // Generic action handler to apply effects
    handleAction(actionId, parameters = {}) {
      // 1. Find action in FSM
      let action = null;
      const page = fsmData.pages.find(p => p.id === this.currentPageId);
      if (page) {
        action = page.actions.find(a => a.id === actionId);
      }

      if (!action) {
        console.error(`Action ${actionId} not found on page ${this.currentPageId}`);
        return false;
      }

      // 2. Check Preconditions
      const runtime = new FSMRuntime(fsmData, { currentPageId: this.currentPageId });
      // Pass state as plain object for evaluation if needed, or direct access
      // JSONPath works on objects.
      // We'll pass 'this' (the store state) but JSONPath might need a plain object or handle proxy.
      // Pinia state is a Proxy. safely clone or rely on JS compatibility.
      // Cloning is safer for JSONPath.
      const stateSnapshot = JSON.parse(JSON.stringify(this.$state));
      
      if (!runtime.checkPreconditions(stateSnapshot, action)) {
        console.warn(`Preconditions failed for ${actionId}`);
        return false;
      }

      // 3. Apply Effects
      runtime.applyEffects(this, action, parameters);
      
      return true;
    }
  },
  persist: {
    storage: sessionStorage, // Use sessionStorage as requested
    paths: undefined // persist all
  }
});