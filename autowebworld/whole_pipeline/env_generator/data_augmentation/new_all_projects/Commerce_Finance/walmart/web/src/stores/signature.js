import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Page Navigation State
  const currentPageId = ref('HOME')

  // HOME
  const cookie_consent_given = ref(null)

  // DEPARTMENTS
  const location_permission_granted = ref(null)

  // ELECTRONICS_CATEGORY
  const products = ref(null)
  const electronics_category_filters_applied = ref(null)
  const electronics_category_viewport_anchor_id = ref(null)
  const electronics_category_has_searched = ref(null)
  const matched_product_id = ref(null)
  const selected_product_id = ref(null)

  // PRODUCT_DETAIL
  const cart_items = ref([])
  const selected_quantity = ref(null)
  const selected_shipping_option = ref(null)
  const selected_pickup_store = ref(null)

  // ADD_TO_CART_OVERLAY uses cart_items

  // CART uses cart_items

  // CHECKOUT_SHIPPING
  const shipping_full_name = ref(null)
  const shipping_address_line1 = ref(null)
  const shipping_city = ref(null)
  const shipping_zip = ref(null)
  const shipping_method = ref(null)

  // CHECKOUT_PAYMENT
  const card_number = ref(null)
  const card_expiry = ref(null)
  const card_cvv = ref(null)
  const payment_method = ref(null)

  // CHECKOUT_REVIEW
  const review_terms_accepted = ref(null)

  // SUCCESS PAGES
  const order_id = ref(null)

  // GROCERY_CATEGORY
  // Reuses matched_product_id, selected_product_id from above but has its own fields too
  const grocery_category_filters_applied = ref(null)
  const grocery_category_viewport_anchor_id = ref(null)
  const grocery_category_has_searched = ref(null)

  // GROCERY_PRODUCT_DETAIL
  const grocery_cart_items = ref([])
  const grocery_selected_quantity = ref(null)
  const grocery_delivery_slot = ref(null)

  // GROCERY_CHECKOUT_REVIEW
  const grocery_review_terms_accepted = ref(null)

  // ORDER_HISTORY
  const orders = ref(null)
  const order_history_filters_applied = ref(null)
  const order_history_viewport_anchor_id = ref(null)
  const order_history_has_searched = ref(null)
  const matched_order_id = ref(null)
  const selected_order_id = ref(null)

  // ACCOUNT
  const account_name = ref(null)

  function setCurrentPageId(id) {
    currentPageId.value = id
  }

  return {
    currentPageId,
    setCurrentPageId,
    cookie_consent_given,
    location_permission_granted,
    products,
    electronics_category_filters_applied,
    electronics_category_viewport_anchor_id,
    electronics_category_has_searched,
    matched_product_id,
    selected_product_id,
    cart_items,
    selected_quantity,
    selected_shipping_option,
    selected_pickup_store,
    shipping_full_name,
    shipping_address_line1,
    shipping_city,
    shipping_zip,
    shipping_method,
    card_number,
    card_expiry,
    card_cvv,
    payment_method,
    review_terms_accepted,
    order_id,
    grocery_category_filters_applied,
    grocery_category_viewport_anchor_id,
    grocery_category_has_searched,
    grocery_cart_items,
    grocery_selected_quantity,
    grocery_delivery_slot,
    grocery_review_terms_accepted,
    orders,
    order_history_filters_applied,
    order_history_viewport_anchor_id,
    order_history_has_searched,
    matched_order_id,
    selected_order_id,
    account_name
  }
}, {
  persist: {
    storage: sessionStorage
  }
})