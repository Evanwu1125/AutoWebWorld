import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useSignatureStore = defineStore('signature', () => {
  // Global State
  const currentPageId = ref('HOME');
  
  // HOME
  const current_user_id = ref('user_001');
  const cookie_consent_given = ref(null);

  // BROWSE
  const location_permission_granted = ref(null);
  const browse_filters_applied = ref(null);
  const browse_has_searched = ref(null);
  const browse_viewport_anchor_id = ref(null);
  const matched_session_id = ref(null);
  const selected_session_id = ref(null);

  // SESSION
  const chosen_duration_minutes = ref(null);
  const session_notes = ref(null);
  const session_reminder_time = ref(null);
  const session_intention_text = ref(null);
  const session_environment_choice = ref(null);
  const session_started = ref(null);

  // COURSES
  const courses_filters_applied = ref(null);
  const courses_has_searched = ref(null);
  const courses_viewport_anchor_id = ref(null);
  const matched_course_id = ref(null);
  const selected_course_id = ref(null);
  const course_progress_percent = ref(null);
  const course_goal_text = ref(null);
  const course_reminder_time = ref(null);
  const enroll_reason_text = ref(null);
  const enroll_frequency_choice = ref(null);
  const course_enrolled = ref(null);

  // SLEEP
  const sleep_filters_applied = ref(null);
  const sleep_has_searched = ref(null);
  const sleep_viewport_anchor_id = ref(null);
  const matched_sleep_id = ref(null);
  const selected_sleep_id = ref(null);
  const sleep_volume_level = ref(null);
  const sleep_notes = ref(null);
  const sleep_bedtime_text = ref(null);
  const sleep_environment_choice = ref(null);
  const sleep_session_started = ref(null);

  // FOCUS
  const focus_filters_applied = ref(null);
  const focus_has_searched = ref(null);
  const focus_viewport_anchor_id = ref(null);
  const matched_focus_id = ref(null);
  const selected_focus_id = ref(null);
  const focus_volume_level = ref(null);
  const focus_notes = ref(null);
  const focus_task_text = ref(null);
  const focus_duration_choice = ref(null);
  const focus_session_started = ref(null);

  // REMINDER
  const reminder_label_text = ref(null);
  const reminder_time_choice = ref(null);
  const reminder_set = ref(null);

  function setCurrentPageId(id) {
    currentPageId.value = id;
  }

  return {
    currentPageId,
    current_user_id,
    cookie_consent_given,
    location_permission_granted,
    browse_filters_applied,
    browse_has_searched,
    browse_viewport_anchor_id,
    matched_session_id,
    selected_session_id,
    chosen_duration_minutes,
    session_notes,
    session_reminder_time,
    session_intention_text,
    session_environment_choice,
    session_started,
    courses_filters_applied,
    courses_has_searched,
    courses_viewport_anchor_id,
    matched_course_id,
    selected_course_id,
    course_progress_percent,
    course_goal_text,
    course_reminder_time,
    enroll_reason_text,
    enroll_frequency_choice,
    course_enrolled,
    sleep_filters_applied,
    sleep_has_searched,
    sleep_viewport_anchor_id,
    matched_sleep_id,
    selected_sleep_id,
    sleep_volume_level,
    sleep_notes,
    sleep_bedtime_text,
    sleep_environment_choice,
    sleep_session_started,
    focus_filters_applied,
    focus_has_searched,
    focus_viewport_anchor_id,
    matched_focus_id,
    selected_focus_id,
    focus_volume_level,
    focus_notes,
    focus_task_text,
    focus_duration_choice,
    focus_session_started,
    reminder_label_text,
    reminder_time_choice,
    reminder_set,
    setCurrentPageId
  };
}, {
  persist: {
    storage: sessionStorage
  }
});