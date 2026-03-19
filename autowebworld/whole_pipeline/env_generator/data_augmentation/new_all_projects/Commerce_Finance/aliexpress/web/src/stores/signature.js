import { defineStore } from 'pinia'

export const useSignatureStore = defineStore('signature', {
  state: () => ({
    // Global
    currentPageId: 'HOME',
    current_user_id: null,
    cookie_accepted: null,
    location_permission_granted: null,
    
    // Navigation & Selection
    selected_item_id: null,
    matched_item_id: null,
    
    // Category List
    categories: [],
    CATEGORY_LIST_has_searched: null,
    CATEGORY_LIST_viewport_anchor_id: null,
    CATEGORY_LIST_filters_applied: null,
    
    // Deals List
    DEALS_LIST_has_searched: null,
    DEALS_LIST_viewport_anchor_id: null,
    DEALS_LIST_filters_applied: null,
    DEALS_LIST_sort_type: null,
    
    // Account Login
    login_email: "",
    login_password: "",
    login_error: null,
    
    // Product List
    products: [],
    PRODUCT_LIST_has_searched: null,
    PRODUCT_LIST_viewport_anchor_id: null,
    PRODUCT_LIST_filters_applied: null,
    
    // Product Detail
    selected_sku_id: null,
    quantity: 1,
    ship_to_country: "US",
    buy_option: null,
    
    // Product Reviews
    PRODUCT_REVIEWS_viewport_anchor_id: null,
    
    // Cart
    cart_items: [],
    CART_PAGE_viewport_anchor_id: null,
    
    // Checkout
    selected_shipping_address_id: null,
    selected_payment_method: null,
    order_id: null,
    success_message: null,
    
    // Payment Gateway
    paypal_email: "",
    card_number: "",
    card_holder: "",
    card_expiry: "",
    card_cvv: "",
    
    // Address Book
    addresses: [],
    ADDRESS_BOOK_viewport_anchor_id: null,
    selected_address_id: null,
    
    // Edit Address
    address_full_name: "",
    address_street: "",
    address_city: "",
    address_postcode: "",
    
    // Orders
    ORDERS_LIST_has_searched: null,
    ORDERS_LIST_viewport_anchor_id: null,
    ORDERS_LIST_filters_applied: null,
    
    // Contact Seller
    message_subject: "",
    message_body: ""
  }),
  actions: {
    setCurrentPageId(pageId) {
      this.currentPageId = pageId;
    }
  },
  persist: {
    storage: sessionStorage,
  },
})