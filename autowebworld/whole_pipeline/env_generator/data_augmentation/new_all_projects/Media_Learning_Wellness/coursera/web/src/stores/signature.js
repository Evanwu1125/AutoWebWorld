import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // --- Global State ---
  const courses = ref([])
  const specializations = ref([])
  const professional_certs = ref([])

  // --- Page Specific State ---
  const currentPageId = ref('HOME')

  // HOME
  const current_user_id = ref(null)
  const cookie_consent_given = ref(null)

  // LOGIN
  const login_email_filled = ref(null)
  const login_password_filled = ref(null)
  const login_can_submit = ref(null)

  // COURSE_DISCOVERY
  const location_permission_granted = ref(null)
  const course_list_filters_applied = ref(null)
  const course_list_sort_applied = ref(null)
  const course_list_has_searched = ref(null)
  const course_list_viewport_anchor_id = ref(null)
  const matched_course_id = ref(null)
  const selected_course_id = ref(null)

  // COURSE_DETAIL
  const intended_enrollment_type = ref(null)

  // AUDIT_CONFIRM
  const audit_terms_confirmed = ref(null)

  // ENROLLMENT_OPTIONS
  const selected_pricing_option = ref(null)

  // PAYMENT_DETAILS (Shared with Specialization & Pro Cert)
  const card_number_filled = ref(null)
  const card_name_filled = ref(null)
  const card_cvv_filled = ref(null)
  const billing_country_selected = ref(null)

  // ENROLL_COURSE_SUCCESS / AUDIT_COURSE_SUCCESS / SPECIALIZATION_SUBSCRIBE_SUCCESS / ENROLL_PROFESSIONAL_CERT_SUCCESS
  const success_message = ref(null)

  // COURSE_RATING_FORM
  const rating_stars_selected = ref(null)
  const rating_text_filled = ref(null)

  // SPECIALIZATION_LIST
  const specialization_list_filters_applied = ref(null)
  const specialization_list_has_searched = ref(null)
  const specialization_list_viewport_anchor_id = ref(null)
  const matched_specialization_id = ref(null)
  const selected_specialization_id = ref(null)

  // SPECIALIZATION_DETAIL
  const specialization_enroll_type = ref(null)

  // PROFESSIONAL_CERT_LIST
  const pro_cert_list_has_searched = ref(null)
  const pro_cert_list_viewport_anchor_id = ref(null)
  const matched_pro_cert_id = ref(null)
  const selected_pro_cert_id = ref(null)

  // LEARNER_DASHBOARD
  const enrolled_courses = ref([])

  // Helper actions
  function setCurrentPageId(pageId) {
    currentPageId.value = pageId
  }

  return {
    courses,
    specializations,
    professional_certs,
    currentPageId,
    current_user_id,
    cookie_consent_given,
    login_email_filled,
    login_password_filled,
    login_can_submit,
    location_permission_granted,
    course_list_filters_applied,
    course_list_sort_applied,
    course_list_has_searched,
    course_list_viewport_anchor_id,
    matched_course_id,
    selected_course_id,
    intended_enrollment_type,
    audit_terms_confirmed,
    selected_pricing_option,
    card_number_filled,
    card_name_filled,
    card_cvv_filled,
    billing_country_selected,
    success_message,
    rating_stars_selected,
    rating_text_filled,
    specialization_list_filters_applied,
    specialization_list_has_searched,
    specialization_list_viewport_anchor_id,
    matched_specialization_id,
    selected_specialization_id,
    specialization_enroll_type,
    pro_cert_list_has_searched,
    pro_cert_list_viewport_anchor_id,
    matched_pro_cert_id,
    selected_pro_cert_id,
    enrolled_courses,
    setCurrentPageId
  }
}, {
  persist: {
    storage: sessionStorage
  }
})