import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useSignatureStore = defineStore('signature', () => {
  // Global State - aggregated from all pages signature_schema
  const currentPageId = ref('HOME');
  
  // HOME
  const current_user_id = ref(null);
  const global_location_permission_granted = ref(null);
  const home_cookie_accepted = ref(null);

  // CATEGORY_ELECTRONICS
  const electronics_matched_item_id = ref(null);
  const electronics_selected_item_id = ref(null);
  const electronics_list_has_searched = ref(null);
  const electronics_list_viewport_anchor_id = ref(null);
  const electronics_list_filters_applied = ref(null);

  // CATEGORY_SUPERMARKET
  const supermarket_selected_item_id = ref(null);
  const supermarket_list_viewport_anchor_id = ref(null);
  const supermarket_list_filters_applied = ref(null);

  // SEARCH_RESULTS
  const search_matched_item_id = ref(null);
  const search_selected_item_id = ref(null);
  const search_list_has_searched = ref(null);
  const search_list_viewport_anchor_id = ref(null);
  const search_list_filters_applied = ref(null);

  // PRODUCT_DETAIL
  const product_selected_item_id = ref(null);
  const product_selected_sku_id = ref(null);
  const product_quantity = ref(null);

  // PRODUCT_REVIEWS
  const review_rating = ref(null);
  const review_text_entered = ref(null);

  // CART
  const cart_items = ref([]); // Initialize as array
  const cart_has_items = ref(null);

  // CHECKOUT
  const checkout_address_selected = ref(null);
  const checkout_payment_selected = ref(null);
  const checkout_bn_address_selected = ref(null);
  const checkout_bn_payment_selected = ref(null);

  // ORDERS
  const orders = ref([]);
  const orders_selected_item_id = ref(null);
  const orders_list_viewport_anchor_id = ref(null);
  const order_selected_item_id = ref(null);
  const order_can_apply_service = ref(null);

  // AFTER_SALE
  const after_sale_reason_selected = ref(null);
  const after_sale_description_entered = ref(null);

  // LOGIN/REGISTER
  const login_username_entered = ref(null);
  const login_password_entered = ref(null);
  const register_username_entered = ref(null);
  const register_password_entered = ref(null);
  const register_phone_entered = ref(null);

  // SUCCESS PAGES
  const success_message = ref(null);

  // USER CENTER
  const user_name = ref(null);

  // ADDRESS BOOK
  const addresses = ref([]);
  const address_form_name_entered = ref(null);
  const address_form_detail_entered = ref(null);

  // PAYMENT METHODS
  const payment_card_number_entered = ref(null);
  const payment_card_holder_entered = ref(null);

  function setCurrentPageId(id) {
    currentPageId.value = id;
  }

  return {
    currentPageId,
    current_user_id,
    global_location_permission_granted,
    home_cookie_accepted,
    electronics_matched_item_id,
    electronics_selected_item_id,
    electronics_list_has_searched,
    electronics_list_viewport_anchor_id,
    electronics_list_filters_applied,
    supermarket_selected_item_id,
    supermarket_list_viewport_anchor_id,
    supermarket_list_filters_applied,
    search_matched_item_id,
    search_selected_item_id,
    search_list_has_searched,
    search_list_viewport_anchor_id,
    search_list_filters_applied,
    product_selected_item_id,
    product_selected_sku_id,
    product_quantity,
    review_rating,
    review_text_entered,
    cart_items,
    cart_has_items,
    checkout_address_selected,
    checkout_payment_selected,
    checkout_bn_address_selected,
    checkout_bn_payment_selected,
    orders,
    orders_selected_item_id,
    orders_list_viewport_anchor_id,
    order_selected_item_id,
    order_can_apply_service,
    after_sale_reason_selected,
    after_sale_description_entered,
    login_username_entered,
    login_password_entered,
    register_username_entered,
    register_password_entered,
    register_phone_entered,
    success_message,
    user_name,
    addresses,
    address_form_name_entered,
    address_form_detail_entered,
    payment_card_number_entered,
    payment_card_holder_entered,
    setCurrentPageId
  };
}, {
  persist: {
    storage: sessionStorage,
  },
});