# Development Checklist

## Pages (26 Total)
- [ ] HOME
- [ ] CATEGORY_LIST
- [ ] DEALS_LIST
- [ ] ACCOUNT_LOGIN
- [ ] ACCOUNT_OVERVIEW
- [ ] PRODUCT_LIST
- [ ] PRODUCT_DETAIL
- [ ] PRODUCT_REVIEWS
- [ ] ADD_TO_CART_CONFIRM
- [ ] CART_PAGE
- [ ] CART_CHECKOUT
- [ ] BUY_NOW_CHECKOUT
- [ ] PAYMENT_METHODS
- [ ] PAYPAL_PAYMENT_GATEWAY
- [ ] CARD_PAYMENT_GATEWAY
- [ ] ADDRESS_BOOK
- [ ] EDIT_ADDRESS_FORM
- [ ] ORDERS_LIST
- [ ] ORDER_DETAIL
- [ ] ORDER_TRACKING
- [ ] CONTACT_SELLER_FORM
- [ ] CHECKOUT_FROM_CART_SUCCESS
- [ ] CHECKOUT_BUYNOW_SUCCESS
- [ ] ORDER_PAYPAL_SUCCESS
- [ ] ORDER_CARD_SUCCESS
- [ ] CONTACT_SELLER_SUCCESS

## Core Infrastructure
- [ ] src/stores/signature.js (Pinia store with FSM state)
- [ ] src/stores/data.js (Mock data with sessionStorage)
- [ ] src/fsm/FSMRuntime.js (FSM engine)
- [ ] src/router/index.js (Routes for all pages)
- [ ] src/App.vue (Global interceptors: Cookie, Permissions)
- [ ] src/components/widgets/DateTimePicker.vue (Verify existence)

## Global Interceptors
- [ ] Cookie Consent Modal (HOME page)
- [ ] Location Permission Modal (CATEGORY_LIST page)

## Design & Assets
- [ ] style.txt (Visual guidelines)
- [ ] ImageGetter integration for realistic images