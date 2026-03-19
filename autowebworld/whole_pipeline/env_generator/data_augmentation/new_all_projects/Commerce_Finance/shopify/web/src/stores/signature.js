import { defineStore } from 'pinia'

export const useSignatureStore = defineStore('signature', {
  state: () => ({
    // Global/System
    currentPageId: 'HOME',
    
    // HOME
    cookie_consent_given: null,
    
    // SHOP_ALL_COLLECTIONS
    products: [], // Loaded from data store
    location_permission_granted: null,
    collections_list_has_searched: null,
    collections_matched_product_id: null,
    collections_selected_product_id: null,
    collections_viewport_anchor_id: null,
    collections_filters_applied: null,
    
    // PRODUCT_DETAIL
    selected_product_id: null,
    selected_variant_sku: null,
    selected_quantity: null,
    
    // CART
    cart_items: [],
    cart_subtotal: 0,
    
    // CHECKOUT Information
    email: null,
    shipping_first_name: null,
    shipping_last_name: null,
    shipping_address1: null,
    shipping_city: null,
    shipping_postcode: null,
    
    // CHECKOUT Shipping
    shipping_method: null,
    
    // CHECKOUT Payment
    card_number: null,
    card_name: null,
    card_expiry: null,
    card_cvv: null,
    
    // CHECKOUT Express
    express_provider: null,
    
    // LOGIN
    password: null,
    
    // ACCOUNT Dashboard & Orders
    orders: [], // Loaded from data store
    orders_list_has_searched: null,
    orders_matched_order_id: null,
    orders_selected_order_id: null,
    orders_viewport_anchor_id: null,
    orders_filters_applied: null,
    
    // ORDER Detail
    selected_order_id: null,
    
    // ACCOUNT Addresses
    addresses: [], // Loaded from data store
    
    // ADDRESS Edit
    first_name: null, // Specific to address edit
    address1: null,   // Specific to address edit
    city: null,       // Specific to address edit
    postcode: null,   // Specific to address edit
    
    // SUCCESS
    order_id: null
  }),
  
  actions: {
    setCurrentPageId(pageId) {
      this.currentPageId = pageId;
    },
    
    // Helper to reset some transient state
    resetTransientState() {
      this.collections_list_has_searched = null;
      this.collections_matched_product_id = null;
      this.collections_filters_applied = null;
      this.orders_list_has_searched = null;
      this.orders_matched_order_id = null;
      this.orders_filters_applied = null;
    }
  },
  
  persist: {
    storage: sessionStorage
  }
})