# Development Checklist

## 1. Infrastructure & Stores
- [ ] `src/stores/signature.js` (Pinia store for FSM state)
- [ ] `src/stores/data.js` (Mock data with session persistence, >15 items/collection)
- [ ] `src/router/index.js` (Routes for all 22 pages)
- [ ] `src/App.vue` (Main layout, global modals)

## 2. Global Components
- [ ] `src/components/PermissionModal.vue` (For Location Permission)
- [ ] `src/components/CookieConsentModal.vue` (For Cookie Consent)
- [ ] `src/components/NavBar.vue` (Optional: Reusable Header)

## 3. Pages (22 Total)
### Core Navigation
- [ ] `src/pages/HOME.vue`
- [ ] `src/pages/DEPARTMENTS.vue`
- [ ] `src/pages/ACCOUNT.vue`

### Electronics Flow
- [ ] `src/pages/ELECTRONICS_CATEGORY.vue`
- [ ] `src/pages/PRODUCT_DETAIL.vue`
- [ ] `src/pages/ADD_TO_CART_OVERLAY.vue`
- [ ] `src/pages/CART.vue`
- [ ] `src/pages/CHECKOUT_SHIPPING.vue`
- [ ] `src/pages/CHECKOUT_PAYMENT.vue`
- [ ] `src/pages/CHECKOUT_REVIEW.vue`
- [ ] `src/pages/CHECKOUT_DELIVERY_SUCCESS.vue`
- [ ] `src/pages/CHECKOUT_PICKUP_SUCCESS.vue`
- [ ] `src/pages/ORDER_PLACED_FROM_CART_SUCCESS.vue`
- [ ] `src/pages/ORDER_PLACED_BUY_NOW_SUCCESS.vue`

### Grocery Flow
- [ ] `src/pages/GROCERY_CATEGORY.vue`
- [ ] `src/pages/GROCERY_PRODUCT_DETAIL.vue`
- [ ] `src/pages/GROCERY_CART.vue`
- [ ] `src/pages/GROCERY_DELIVERY_SCHEDULING.vue`
- [ ] `src/pages/GROCERY_CHECKOUT_REVIEW.vue`
- [ ] `src/pages/CHECKOUT_GROCERY_SUCCESS.vue`

### Order Management
- [ ] `src/pages/ORDER_HISTORY.vue`
- [ ] `src/pages/ORDER_DETAIL.vue`

## 4. Validation
- [ ] Verify all selectors match `fsm.json`.
- [ ] Verify all actions are implemented.
- [ ] Run `pnpm run lint` and `pnpm run build`.