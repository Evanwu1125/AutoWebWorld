import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Global State
  const currentPageId = ref('HOME')
  
  // Signature Fields from FSM
  const current_user_id = ref(null)
  const cookie_consent_given = ref(null)
  const location_permission_granted = ref(null)
  
  // Accounts
  const accounts = ref(null)
  const accounts_list_has_searched = ref(null)
  const accounts_matched_account_id = ref(null)
  const accounts_selected_account_id = ref(null)
  const accounts_viewport_anchor_id = ref(null)
  const accounts_filters_applied = ref(null)
  
  // Payments / Beneficiaries
  const beneficiaries = ref(null)
  const payments_list_has_searched = ref(null)
  const payments_matched_beneficiary_id = ref(null)
  const payments_selected_beneficiary_id = ref(null)
  const payments_viewport_anchor_id = ref(null)
  const payments_filters_applied = ref(null)
  
  // Transfer
  const from_account_id = ref(null)
  const to_beneficiary_id = ref(null)
  const transfer_amount = ref(null)
  const transfer_reference = ref(null)
  
  // Success Messages (Shared)
  const success_message = ref(null)
  
  // Exchange
  const exchange_pairs = ref(null)
  const exchange_list_has_searched = ref(null)
  const exchange_matched_pair_id = ref(null)
  const exchange_selected_pair_id = ref(null)
  const exchange_viewport_anchor_id = ref(null)
  const exchange_filters_applied = ref(null)
  const sell_amount = ref(null)
  const buy_amount = ref(null)
  
  // Cards
  const cards = ref(null)
  const cards_list_has_searched = ref(null)
  const cards_matched_card_id = ref(null)
  const cards_selected_card_id = ref(null)
  const cards_viewport_anchor_id = ref(null)
  const cards_filters_applied = ref(null)
  
  // Card Actions
  const freeze_reason = ref(null)
  const atm_limit = ref(null)
  const pos_limit = ref(null)
  
  // Topup
  const topup_methods = ref(null)
  const topup_filters_applied = ref(null)
  const topup_viewport_anchor_id = ref(null)
  const topup_selected_method_id = ref(null)
  const topup_amount = ref(null)

  function setCurrentPageId(id) {
    currentPageId.value = id
  }

  return {
    currentPageId,
    current_user_id,
    cookie_consent_given,
    location_permission_granted,
    accounts,
    accounts_list_has_searched,
    accounts_matched_account_id,
    accounts_selected_account_id,
    accounts_viewport_anchor_id,
    accounts_filters_applied,
    beneficiaries,
    payments_list_has_searched,
    payments_matched_beneficiary_id,
    payments_selected_beneficiary_id,
    payments_viewport_anchor_id,
    payments_filters_applied,
    from_account_id,
    to_beneficiary_id,
    transfer_amount,
    transfer_reference,
    success_message,
    exchange_pairs,
    exchange_list_has_searched,
    exchange_matched_pair_id,
    exchange_selected_pair_id,
    exchange_viewport_anchor_id,
    exchange_filters_applied,
    sell_amount,
    buy_amount,
    cards,
    cards_list_has_searched,
    cards_matched_card_id,
    cards_selected_card_id,
    cards_viewport_anchor_id,
    cards_filters_applied,
    freeze_reason,
    atm_limit,
    pos_limit,
    topup_methods,
    topup_filters_applied,
    topup_viewport_anchor_id,
    topup_selected_method_id,
    topup_amount,
    setCurrentPageId
  }
}, {
  persist: {
    storage: sessionStorage
  }
})