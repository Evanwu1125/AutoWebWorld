import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Core State
  const currentPageId = ref('HOME')

  // Global Signature Schema
  const experiments = ref([])
  const audiences = ref([])
  const feature_flags = ref([])
  const selected_experiment_id = ref(null)
  const selected_audience_id = ref(null)
  const selected_feature_flag_id = ref(null)
  const location_permission_granted = ref(null)

  // Page Specific Fields
  // HOME
  const current_user_id = ref(null)
  const cookie_accepted = ref(null)

  // DASHBOARD
  const recent_activity_viewport_anchor_id = ref(null)
  const recent_activity_selected_item_id = ref(null)
  const dashboard_view_filters_applied = ref(null)
  const dashboard_location_permission_granted = ref(null)

  // EXPERIMENTS_LIST
  const experiments_list_matched_item_id = ref(null)
  const experiments_list_selected_item_id = ref(null)
  const experiments_list_has_searched = ref(null)
  const experiments_list_viewport_anchor_id = ref(null)
  const experiments_list_filters_applied = ref(null)

  // EXPERIMENT_CREATE / EDIT
  const new_experiment_name = ref(null)
  const new_experiment_url = ref(null)
  const new_experiment_type = ref(null)
  const variation_a_name = ref(null)
  const variation_b_name = ref(null)
  const traffic_allocation_slider_set = ref(null)
  const url_match_pattern = ref(null)
  const targeting_device_checkbox_set = ref(null)

  // EXPERIMENT_SCHEDULE
  const schedule_start_selected = ref(null)
  const schedule_end_selected = ref(null)
  const launch_immediately_checkbox = ref(null)
  
  // SUCCESS / ARCHIVE
  const success_message = ref(null)
  const archive_reason = ref(null)

  // AUDIENCES
  const audiences_list_matched_item_id = ref(null)
  const audiences_list_selected_item_id = ref(null)
  const audiences_list_has_searched = ref(null)
  const audiences_list_viewport_anchor_id = ref(null)
  const audiences_list_filters_applied = ref(null)
  const audience_name = ref(null)
  const audience_condition = ref(null)
  const audience_membership_slider_set = ref(null)

  // FEATURE FLAGS
  const feature_flags_list_viewport_anchor_id = ref(null)
  const feature_flags_list_selected_item_id = ref(null)
  const feature_flags_list_filters_applied = ref(null)
  const rollout_slider_set = ref(null)

  // RESULTS
  const results_viewport_anchor_id = ref(null)
  const results_selected_item_id = ref(null)

  // ACCOUNT & BILLING
  const account_name = ref(null)
  const notification_checkbox = ref(null)
  const card_number_set = ref(null)
  const card_expiry_set = ref(null)
  const billing_country_selected = ref(null)

  function setCurrentPageId(pageId) {
    currentPageId.value = pageId
  }

  return {
    currentPageId,
    experiments,
    audiences,
    feature_flags,
    selected_experiment_id,
    selected_audience_id,
    selected_feature_flag_id,
    location_permission_granted,
    current_user_id,
    cookie_accepted,
    recent_activity_viewport_anchor_id,
    recent_activity_selected_item_id,
    dashboard_view_filters_applied,
    dashboard_location_permission_granted,
    experiments_list_matched_item_id,
    experiments_list_selected_item_id,
    experiments_list_has_searched,
    experiments_list_viewport_anchor_id,
    experiments_list_filters_applied,
    new_experiment_name,
    new_experiment_url,
    new_experiment_type,
    variation_a_name,
    variation_b_name,
    traffic_allocation_slider_set,
    url_match_pattern,
    targeting_device_checkbox_set,
    schedule_start_selected,
    schedule_end_selected,
    launch_immediately_checkbox,
    success_message,
    archive_reason,
    audiences_list_matched_item_id,
    audiences_list_selected_item_id,
    audiences_list_has_searched,
    audiences_list_viewport_anchor_id,
    audiences_list_filters_applied,
    audience_name,
    audience_condition,
    audience_membership_slider_set,
    feature_flags_list_viewport_anchor_id,
    feature_flags_list_selected_item_id,
    feature_flags_list_filters_applied,
    rollout_slider_set,
    results_viewport_anchor_id,
    results_selected_item_id,
    account_name,
    notification_checkbox,
    card_number_set,
    card_expiry_set,
    billing_country_selected,
    setCurrentPageId
  }
}, {
  persist: {
    storage: sessionStorage
  }
})