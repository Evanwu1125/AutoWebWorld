import { defineStore } from 'pinia'

export const useSignatureStore = defineStore('signature', {
  state: () => ({
    // Global Navigation State
    currentPageId: 'HOME',
    
    // HOME
    current_user_id: null,
    cookie_consent_given: null,

    // LOGIN & PERMISSIONS
    location_permission_granted: null,
    login_email_entered: '',
    login_password_entered: '',
    login_can_submit: false,

    // DASHBOARD
    upcoming_appointments: [],
    conditions_list: [],
    dashboard_filters_applied: false,

    // VISIT FLOW
    selected_visit_type: null,

    // PROVIDER FLOW
    providers: [],
    provider_list_has_searched: false,
    provider_list_viewport_anchor_id: null,
    matched_provider_id: null,
    selected_provider_id: null,
    provider_list_filters_applied: false,
    selected_symptom_description: '',
    selected_reason_for_visit: '',
    schedule_selected_date: null,
    schedule_selected_slot: null,
    confirmation_number: null,

    // INSTANT VISIT FLOW
    triage_reason: '',
    triage_symptom_duration: '',
    queue_position: null,
    visit_started: false,

    // PRESCRIPTION FLOW
    prescriptions: [],
    matched_prescription_id: null,
    selected_prescription_id: null,
    prescription_list_has_searched: false,
    prescription_list_viewport_anchor_id: null,
    prescription_list_filters_applied: false,
    renewal_notes: '',
    renewal_confirmation: null,

    // MENTAL HEALTH FLOW
    therapists: [],
    matched_therapist_id: null,
    selected_therapist_id: null,
    mh_list_has_searched: false,
    mh_list_viewport_anchor_id: null,
    mh_list_filters_applied: false,
    mh_reason_for_visit: '',
    mh_schedule_date: null,
    mh_schedule_slot: null,
    mh_booking_confirmation: null,

    // APPOINTMENTS FLOW
    appointments: [],
    matched_appointment_id: null,
    selected_appointment_id: null,
    appts_list_has_searched: false,
    appts_list_viewport_anchor_id: null,
    appts_list_filters_applied: false,

    // BILLING FLOW
    bills: [],
    matched_bill_id: null,
    selected_bill_id: null,
    billing_list_has_searched: false,
    billing_list_viewport_anchor_id: null,
    billing_amount_due: 0,
    payment_method_selected: null,
    card_number_entered: '',
    card_cvv_entered: '',
    payment_confirmation: null,

    // BENEFITS FLOW
    plans: [],
    benefits_list_filters_applied: false,

    // SETTINGS FLOW
    full_name_entered: '',
    phone_number_entered: '',
    insurance_member_id_entered: ''
  }),
  actions: {
    setCurrentPageId(pageId) {
      this.currentPageId = pageId;
    }
  },
  persist: {
    storage: sessionStorage
  }
})