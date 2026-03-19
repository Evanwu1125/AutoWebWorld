import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Global Signature Schema
  const location_permission_granted = ref(null)
  const current_user_id = ref(null)
  const accounts = ref(null)
  const cookie_consent_given = ref(null)

  // Campaigns List Page State
  const campaigns = ref(null)
  const matched_campaign_id = ref(null)
  const selected_campaign_id = ref(null)
  const campaigns_list_has_searched = ref(null)
  const campaigns_list_viewport_anchor_id = ref(null)
  const campaigns_list_filters_applied = ref(null)

  // Create Campaign Wizards State
  const selected_channel = ref(null) // email or sms
  
  // Email Campaign Basics
  const campaign_name = ref(null)
  const subject_line = ref(null)
  const from_name = ref(null)
  const from_email = ref(null)
  
  // Email Campaign Recipients
  const selected_list_id = ref(null)
  const selected_segment_id = ref(null)
  
  // Email Campaign Content
  const email_body_has_text = ref(null)
  const selected_email_template_id = ref(null)
  
  // Email Campaign Schedule
  const email_scheduled_datetime = ref(null)
  
  // SMS Campaign Basics
  const sms_campaign_name = ref(null)
  const sms_sender_id = ref(null)
  
  // SMS Campaign Recipients
  const sms_selected_list_id = ref(null)
  
  // SMS Campaign Content
  const sms_body_has_text = ref(null)
  
  // SMS Campaign Schedule
  const sms_scheduled_datetime = ref(null)
  
  // Success Messages
  const success_message = ref(null)

  // Flows List Page State
  const flows = ref(null)
  const flows_list_filters_applied = ref(null)
  const flows_list_viewport_anchor_id = ref(null)
  const flows_list_has_searched = ref(null)
  const matched_flow_id = ref(null)
  const selected_flow_id = ref(null)
  
  // Create Flow State
  const flow_trigger_type = ref(null)
  const flow_email_subject = ref(null)
  const flow_email_body_has_text = ref(null)
  const flow_is_activated = ref(null)

  // Lists & Segments Page State
  const lists = ref(null)
  const segments = ref(null)
  const lists_segments_filters_applied = ref(null)
  const lists_segments_viewport_anchor_id = ref(null)
  const lists_segments_has_searched = ref(null)
  const matched_segment_id = ref(null)
  
  // Create Segment State
  const segment_name = ref(null)
  const segment_condition_type = ref(null)
  const segment_condition_value = ref(null)

  // Signup Forms Page State
  const signup_forms = ref(null)
  const signup_forms_filters_applied = ref(null)
  const signup_forms_viewport_anchor_id = ref(null)
  const signup_forms_has_searched = ref(null)
  const matched_form_id = ref(null)
  const selected_form_id = ref(null)
  
  // Create Form State
  const form_name = ref(null)
  const form_field_email_enabled = ref(null)
  const form_call_to_action = ref(null)

  // Navigation Helper
  const currentPageId = ref('HOME')
  
  function setCurrentPageId(id) {
    currentPageId.value = id
  }

  return {
    location_permission_granted,
    current_user_id,
    accounts,
    cookie_consent_given,
    
    campaigns,
    matched_campaign_id,
    selected_campaign_id,
    campaigns_list_has_searched,
    campaigns_list_viewport_anchor_id,
    campaigns_list_filters_applied,
    
    selected_channel,
    campaign_name,
    subject_line,
    from_name,
    from_email,
    selected_list_id,
    selected_segment_id,
    email_body_has_text,
    selected_email_template_id,
    email_scheduled_datetime,
    
    sms_campaign_name,
    sms_sender_id,
    sms_selected_list_id,
    sms_body_has_text,
    sms_scheduled_datetime,
    
    success_message,
    
    flows,
    flows_list_filters_applied,
    flows_list_viewport_anchor_id,
    flows_list_has_searched,
    matched_flow_id,
    selected_flow_id,
    
    flow_trigger_type,
    flow_email_subject,
    flow_email_body_has_text,
    flow_is_activated,
    
    lists,
    segments,
    lists_segments_filters_applied,
    lists_segments_viewport_anchor_id,
    lists_segments_has_searched,
    matched_segment_id,
    
    segment_name,
    segment_condition_type,
    segment_condition_value,
    
    signup_forms,
    signup_forms_filters_applied,
    signup_forms_viewport_anchor_id,
    signup_forms_has_searched,
    matched_form_id,
    selected_form_id,
    
    form_name,
    form_field_email_enabled,
    form_call_to_action,
    
    currentPageId,
    setCurrentPageId
  }
}, {
  persist: {
    storage: sessionStorage
  }
})