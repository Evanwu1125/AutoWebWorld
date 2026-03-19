import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'HOME',
      component: () => import('../pages/HOME.vue')
    },
    {
      path: '/category',
      name: 'CATEGORY_LIST',
      component: () => import('../pages/CATEGORY_LIST.vue')
    },
    {
      path: '/deals',
      name: 'DEALS_LIST',
      component: () => import('../pages/DEALS_LIST.vue')
    },
    {
      path: '/login',
      name: 'ACCOUNT_LOGIN',
      component: () => import('../pages/ACCOUNT_LOGIN.vue')
    },
    {
      path: '/account',
      name: 'ACCOUNT_OVERVIEW',
      component: () => import('../pages/ACCOUNT_OVERVIEW.vue')
    },
    {
      path: '/products',
      name: 'PRODUCT_LIST',
      component: () => import('../pages/PRODUCT_LIST.vue')
    },
    {
      path: '/product/detail',
      name: 'PRODUCT_DETAIL',
      component: () => import('../pages/PRODUCT_DETAIL.vue')
    },
    {
      path: '/product/reviews',
      name: 'PRODUCT_REVIEWS',
      component: () => import('../pages/PRODUCT_REVIEWS.vue')
    },
    {
      path: '/cart/add-confirm',
      name: 'ADD_TO_CART_CONFIRM',
      component: () => import('../pages/ADD_TO_CART_CONFIRM.vue')
    },
    {
      path: '/cart',
      name: 'CART_PAGE',
      component: () => import('../pages/CART_PAGE.vue')
    },
    {
      path: '/cart/checkout',
      name: 'CART_CHECKOUT',
      component: () => import('../pages/CART_CHECKOUT.vue')
    },
    {
      path: '/buynow/checkout',
      name: 'BUY_NOW_CHECKOUT',
      component: () => import('../pages/BUY_NOW_CHECKOUT.vue')
    },
    {
      path: '/payment/methods',
      name: 'PAYMENT_METHODS',
      component: () => import('../pages/PAYMENT_METHODS.vue')
    },
    {
      path: '/payment/paypal',
      name: 'PAYPAL_PAYMENT_GATEWAY',
      component: () => import('../pages/PAYPAL_PAYMENT_GATEWAY.vue')
    },
    {
      path: '/payment/card',
      name: 'CARD_PAYMENT_GATEWAY',
      component: () => import('../pages/CARD_PAYMENT_GATEWAY.vue')
    },
    {
      path: '/addresses',
      name: 'ADDRESS_BOOK',
      component: () => import('../pages/ADDRESS_BOOK.vue')
    },
    {
      path: '/addresses/edit',
      name: 'EDIT_ADDRESS_FORM',
      component: () => import('../pages/EDIT_ADDRESS_FORM.vue')
    },
    {
      path: '/orders',
      name: 'ORDERS_LIST',
      component: () => import('../pages/ORDERS_LIST.vue')
    },
    {
      path: '/order/detail',
      name: 'ORDER_DETAIL',
      component: () => import('../pages/ORDER_DETAIL.vue')
    },
    {
      path: '/order/tracking',
      name: 'ORDER_TRACKING',
      component: () => import('../pages/ORDER_TRACKING.vue')
    },
    {
      path: '/contact',
      name: 'CONTACT_SELLER_FORM',
      component: () => import('../pages/CONTACT_SELLER_FORM.vue')
    },
    {
      path: '/checkout/cart/success',
      name: 'CHECKOUT_FROM_CART_SUCCESS',
      component: () => import('../pages/CHECKOUT_FROM_CART_SUCCESS.vue')
    },
    {
      path: '/checkout/buynow/success',
      name: 'CHECKOUT_BUYNOW_SUCCESS',
      component: () => import('../pages/CHECKOUT_BUYNOW_SUCCESS.vue')
    },
    {
      path: '/order/paypal/success',
      name: 'ORDER_PAYPAL_SUCCESS',
      component: () => import('../pages/ORDER_PAYPAL_SUCCESS.vue')
    },
    {
      path: '/order/card/success',
      name: 'ORDER_CARD_SUCCESS',
      component: () => import('../pages/ORDER_CARD_SUCCESS.vue')
    },
    {
      path: '/contact/success',
      name: 'CONTACT_SELLER_SUCCESS',
      component: () => import('../pages/CONTACT_SELLER_SUCCESS.vue')
    }
  ]
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  signatureStore.setCurrentPageId(to.name)
  next()
})

export default router