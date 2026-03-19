import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

const routes = [
  { path: '/', redirect: '/home' },
  { path: '/home', name: 'HOME', component: () => import('../pages/HOME.vue') },
  { path: '/departments', name: 'DEPARTMENTS', component: () => import('../pages/DEPARTMENTS.vue') },
  { path: '/electronics', name: 'ELECTRONICS_CATEGORY', component: () => import('../pages/ELECTRONICS_CATEGORY.vue') },
  { path: '/product/:id?', name: 'PRODUCT_DETAIL', component: () => import('../pages/PRODUCT_DETAIL.vue') },
  { path: '/add-to-cart', name: 'ADD_TO_CART_OVERLAY', component: () => import('../pages/ADD_TO_CART_OVERLAY.vue') },
  { path: '/cart', name: 'CART', component: () => import('../pages/CART.vue') },
  { path: '/checkout/shipping', name: 'CHECKOUT_SHIPPING', component: () => import('../pages/CHECKOUT_SHIPPING.vue') },
  { path: '/checkout/payment', name: 'CHECKOUT_PAYMENT', component: () => import('../pages/CHECKOUT_PAYMENT.vue') },
  { path: '/checkout/review', name: 'CHECKOUT_REVIEW', component: () => import('../pages/CHECKOUT_REVIEW.vue') },
  { path: '/checkout/success-delivery', name: 'CHECKOUT_DELIVERY_SUCCESS', component: () => import('../pages/CHECKOUT_DELIVERY_SUCCESS.vue') },
  { path: '/checkout/success-pickup', name: 'CHECKOUT_PICKUP_SUCCESS', component: () => import('../pages/CHECKOUT_PICKUP_SUCCESS.vue') },
  { path: '/grocery', name: 'GROCERY_CATEGORY', component: () => import('../pages/GROCERY_CATEGORY.vue') },
  { path: '/grocery/product/:id?', name: 'GROCERY_PRODUCT_DETAIL', component: () => import('../pages/GROCERY_PRODUCT_DETAIL.vue') },
  { path: '/grocery/cart', name: 'GROCERY_CART', component: () => import('../pages/GROCERY_CART.vue') },
  { path: '/grocery/scheduling', name: 'GROCERY_DELIVERY_SCHEDULING', component: () => import('../pages/GROCERY_DELIVERY_SCHEDULING.vue') },
  { path: '/grocery/review', name: 'GROCERY_CHECKOUT_REVIEW', component: () => import('../pages/GROCERY_CHECKOUT_REVIEW.vue') },
  { path: '/grocery/success', name: 'CHECKOUT_GROCERY_SUCCESS', component: () => import('../pages/CHECKOUT_GROCERY_SUCCESS.vue') },
  { path: '/orders', name: 'ORDER_HISTORY', component: () => import('../pages/ORDER_HISTORY.vue') },
  { path: '/order/:id?', name: 'ORDER_DETAIL', component: () => import('../pages/ORDER_DETAIL.vue') },
  { path: '/account', name: 'ACCOUNT', component: () => import('../pages/ACCOUNT.vue') },
  { path: '/order-placed-cart', name: 'ORDER_PLACED_FROM_CART_SUCCESS', component: () => import('../pages/ORDER_PLACED_FROM_CART_SUCCESS.vue') },
  { path: '/order-placed-buy-now', name: 'ORDER_PLACED_BUY_NOW_SUCCESS', component: () => import('../pages/ORDER_PLACED_BUY_NOW_SUCCESS.vue') },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const store = useSignatureStore()
  if (to.name) {
    store.setCurrentPageId(to.name)
  }
  next()
})

export default router