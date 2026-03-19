import { createRouter, createWebHistory } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

// We will import pages lazily or directly. Given the instructions, simple imports work well.
// However, we need to make sure the files exist. Since I haven't created them yet, 
// I will setup the router structure but the app won't load until pages are created.

const routes = [
  { path: '/', name: 'HOME', component: () => import('../pages/HOME.vue') },
  { path: '/electronics', name: 'CATEGORY_ELECTRONICS', component: () => import('../pages/CATEGORY_ELECTRONICS.vue') },
  { path: '/supermarket', name: 'CATEGORY_SUPERMARKET', component: () => import('../pages/CATEGORY_SUPERMARKET.vue') },
  { path: '/search', name: 'SEARCH_RESULTS', component: () => import('../pages/SEARCH_RESULTS.vue') },
  { path: '/product/:id?', name: 'PRODUCT_DETAIL', component: () => import('../pages/PRODUCT_DETAIL.vue') },
  { path: '/product/:id?/reviews', name: 'PRODUCT_REVIEWS', component: () => import('../pages/PRODUCT_REVIEWS.vue') },
  { path: '/cart', name: 'CART', component: () => import('../pages/CART.vue') },
  { path: '/checkout/cart', name: 'CHECKOUT_CART_CONFIRM', component: () => import('../pages/CHECKOUT_CART_CONFIRM.vue') },
  { path: '/checkout/buy-now', name: 'CHECKOUT_BUY_NOW_CONFIRM', component: () => import('../pages/CHECKOUT_BUY_NOW_CONFIRM.vue') },
  { path: '/checkout/success/cart', name: 'CHECKOUT_FROM_CART_SUCCESS', component: () => import('../pages/CHECKOUT_FROM_CART_SUCCESS.vue') },
  { path: '/checkout/success/buy-now', name: 'CHECKOUT_BUY_NOW_SUCCESS', component: () => import('../pages/CHECKOUT_BUY_NOW_SUCCESS.vue') },
  { path: '/orders', name: 'ORDERS_LIST', component: () => import('../pages/ORDERS_LIST.vue') },
  { path: '/order/:id?', name: 'ORDER_DETAIL', component: () => import('../pages/ORDER_DETAIL.vue') },
  { path: '/after-sale', name: 'AFTER_SALE_APPLY', component: () => import('../pages/AFTER_SALE_APPLY.vue') },
  { path: '/after-sale/success', name: 'ORDER_SUBMITTED_SUCCESS', component: () => import('../pages/ORDER_SUBMITTED_SUCCESS.vue') },
  { path: '/login', name: 'LOGIN', component: () => import('../pages/LOGIN.vue') },
  { path: '/register', name: 'REGISTER', component: () => import('../pages/REGISTER.vue') },
  { path: '/register/success', name: 'ACCOUNT_CREATED_SUCCESS', component: () => import('../pages/ACCOUNT_CREATED_SUCCESS.vue') },
  { path: '/review/success', name: 'REVIEW_SUBMITTED_SUCCESS', component: () => import('../pages/REVIEW_SUBMITTED_SUCCESS.vue') },
  { path: '/user', name: 'USER_CENTER', component: () => import('../pages/USER_CENTER.vue') },
  { path: '/address-book', name: 'ADDRESS_BOOK', component: () => import('../pages/ADDRESS_BOOK.vue') },
  { path: '/payment-methods', name: 'PAYMENT_METHODS', component: () => import('../pages/PAYMENT_METHODS.vue') },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

router.beforeEach((to, from, next) => {
  const store = useSignatureStore();
  store.setCurrentPageId(to.name);
  next();
});

export default router;