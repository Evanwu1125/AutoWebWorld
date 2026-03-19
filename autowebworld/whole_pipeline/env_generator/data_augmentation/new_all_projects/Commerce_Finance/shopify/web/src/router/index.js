import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature.js'

// Import all pages
const routes = [
  { path: '/', name: 'HOME', component: () => import('../pages/HOME.vue') },
  { path: '/collections', name: 'SHOP_ALL_COLLECTIONS', component: () => import('../pages/SHOP_ALL_COLLECTIONS.vue') },
  { path: '/product/:id', name: 'PRODUCT_DETAIL', component: () => import('../pages/PRODUCT_DETAIL.vue') },
  { path: '/product/:id/reviews', name: 'PRODUCT_REVIEWS', component: () => import('../pages/PRODUCT_REVIEWS.vue') },
  { path: '/cart', name: 'CART', component: () => import('../pages/CART.vue') },
  { path: '/checkout/information', name: 'CHECKOUT_INFORMATION_MAIN', component: () => import('../pages/CHECKOUT_INFORMATION_MAIN.vue') },
  { path: '/checkout/shipping', name: 'CHECKOUT_SHIPPING_MAIN', component: () => import('../pages/CHECKOUT_SHIPPING_MAIN.vue') },
  { path: '/checkout/payment', name: 'CHECKOUT_PAYMENT_MAIN', component: () => import('../pages/CHECKOUT_PAYMENT_MAIN.vue') },
  { path: '/checkout/buy-now/information', name: 'CHECKOUT_INFORMATION_BUY_NOW', component: () => import('../pages/CHECKOUT_INFORMATION_BUY_NOW.vue') },
  { path: '/checkout/buy-now/payment', name: 'CHECKOUT_PAYMENT_BUY_NOW', component: () => import('../pages/CHECKOUT_PAYMENT_BUY_NOW.vue') },
  { path: '/checkout/express', name: 'CHECKOUT_EXPRESS', component: () => import('../pages/CHECKOUT_EXPRESS.vue') },
  { path: '/checkout/guest/information', name: 'CHECKOUT_GUEST_INFORMATION', component: () => import('../pages/CHECKOUT_GUEST_INFORMATION.vue') },
  { path: '/checkout/guest/payment', name: 'CHECKOUT_GUEST_PAYMENT', component: () => import('../pages/CHECKOUT_GUEST_PAYMENT.vue') },
  { path: '/login', name: 'CUSTOMER_LOGIN', component: () => import('../pages/CUSTOMER_LOGIN.vue') },
  { path: '/account', name: 'ACCOUNT_DASHBOARD', component: () => import('../pages/ACCOUNT_DASHBOARD.vue') },
  { path: '/account/orders', name: 'ORDER_HISTORY', component: () => import('../pages/ORDER_HISTORY.vue') },
  { path: '/account/orders/:id', name: 'ORDER_DETAIL', component: () => import('../pages/ORDER_DETAIL.vue') },
  { path: '/account/orders/:id/summary', name: 'ORDER_SUMMARY', component: () => import('../pages/ORDER_SUMMARY.vue') },
  { path: '/account/addresses', name: 'ACCOUNT_ADDRESSES', component: () => import('../pages/ACCOUNT_ADDRESSES.vue') },
  { path: '/account/addresses/edit', name: 'ADDRESS_EDIT', component: () => import('../pages/ADDRESS_EDIT.vue') },
  { path: '/checkout/success', name: 'CHECKOUT_SUCCESS_MAIN', component: () => import('../pages/CHECKOUT_SUCCESS_MAIN.vue') },
  { path: '/checkout/success/buy-now', name: 'CHECKOUT_SUCCESS_BUY_NOW', component: () => import('../pages/CHECKOUT_SUCCESS_BUY_NOW.vue') },
  { path: '/checkout/success/guest', name: 'CHECKOUT_SUCCESS_GUEST', component: () => import('../pages/CHECKOUT_SUCCESS_GUEST.vue') },
  { path: '/checkout/success/express', name: 'CHECKOUT_SUCCESS_EXPRESS', component: () => import('../pages/CHECKOUT_SUCCESS_EXPRESS.vue') },
  { path: '/order/confirmation', name: 'ORDER_CONFIRMATION_SUCCESS', component: () => import('../pages/ORDER_CONFIRMATION_SUCCESS.vue') }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  signatureStore.setCurrentPageId(to.name)
  next()
})

export default router