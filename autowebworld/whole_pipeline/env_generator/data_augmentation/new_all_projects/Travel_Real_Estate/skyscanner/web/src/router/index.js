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
      path: '/login',
      name: 'LOGIN',
      component: () => import('../pages/LOGIN.vue')
    },
    {
      path: '/account',
      name: 'ACCOUNT_OVERVIEW',
      component: () => import('../pages/ACCOUNT_OVERVIEW.vue')
    },
    {
      path: '/flights/search',
      name: 'FLIGHTS_SEARCH',
      component: () => import('../pages/FLIGHTS_SEARCH.vue')
    },
    {
      path: '/flights/multi-city',
      name: 'MULTI_CITY_SEARCH',
      component: () => import('../pages/MULTI_CITY_SEARCH.vue')
    },
    {
      path: '/flights/results',
      name: 'FLIGHTS_RESULTS',
      component: () => import('../pages/FLIGHTS_RESULTS.vue')
    },
    {
      path: '/flights/details/:id?',
      name: 'FLIGHT_DETAILS',
      component: () => import('../pages/FLIGHT_DETAILS.vue')
    },
    {
      path: '/flights/book/direct',
      name: 'BOOKING_FORM_DIRECT',
      component: () => import('../pages/BOOKING_FORM_DIRECT.vue')
    },
    {
      path: '/flights/review/direct',
      name: 'BOOKING_REVIEW_DIRECT',
      component: () => import('../pages/BOOKING_REVIEW_DIRECT.vue')
    },
    {
      path: '/flights/complete/direct',
      name: 'BOOKING_COMPLETE_DIRECT',
      component: () => import('../pages/BOOKING_COMPLETE_DIRECT.vue')
    },
    {
      path: '/flights/multi-city/results',
      name: 'MULTI_CITY_RESULTS',
      component: () => import('../pages/MULTI_CITY_RESULTS.vue')
    },
    {
      path: '/flights/multi-city/intro',
      name: 'MULTI_CITY_INTRO',
      component: () => import('../pages/MULTI_CITY_INTRO.vue')
    },
    {
      path: '/flights/book/multi',
      name: 'BOOKING_FORM_MULTI',
      component: () => import('../pages/BOOKING_FORM_MULTI.vue')
    },
    {
      path: '/flights/review/multi',
      name: 'BOOKING_REVIEW_MULTI',
      component: () => import('../pages/BOOKING_REVIEW_MULTI.vue')
    },
    {
      path: '/flights/complete/multi',
      name: 'BOOKING_COMPLETE_MULTI',
      component: () => import('../pages/BOOKING_COMPLETE_MULTI.vue')
    },
    {
      path: '/trip-summary',
      name: 'TRIP_SUMMARY',
      component: () => import('../pages/TRIP_SUMMARY.vue')
    },
    {
      path: '/alerts/create',
      name: 'PRICE_ALERT_FORM',
      component: () => import('../pages/PRICE_ALERT_FORM.vue')
    },
    {
      path: '/alerts/created',
      name: 'PRICE_ALERT_CREATED',
      component: () => import('../pages/PRICE_ALERT_CREATED.vue')
    },
    {
      path: '/alerts/list',
      name: 'PRICE_ALERTS_LIST',
      component: () => import('../pages/PRICE_ALERTS_LIST.vue')
    },
    {
      path: '/alerts/detail/:id?',
      name: 'ALERT_DETAIL',
      component: () => import('../pages/ALERT_DETAIL.vue')
    },
    {
      path: '/hotels/search',
      name: 'HOTELS_SEARCH',
      component: () => import('../pages/HOTELS_SEARCH.vue')
    },
    {
      path: '/hotels/results',
      name: 'HOTELS_RESULTS',
      component: () => import('../pages/HOTELS_RESULTS.vue')
    },
    {
      path: '/hotels/details/:id?',
      name: 'HOTEL_DETAILS',
      component: () => import('../pages/HOTEL_DETAILS.vue')
    },
    {
      path: '/hotels/complete',
      name: 'BOOKING_COMPLETE_HOTEL',
      component: () => import('../pages/BOOKING_COMPLETE_HOTEL.vue')
    },
    {
      path: '/cars/search',
      name: 'CARS_SEARCH',
      component: () => import('../pages/CARS_SEARCH.vue')
    },
    {
      path: '/cars/results',
      name: 'CARS_RESULTS',
      component: () => import('../pages/CARS_RESULTS.vue')
    },
    {
      path: '/cars/details/:id?',
      name: 'CAR_DETAILS',
      component: () => import('../pages/CAR_DETAILS.vue')
    },
    {
      path: '/cars/complete',
      name: 'BOOKING_COMPLETE_CAR',
      component: () => import('../pages/BOOKING_COMPLETE_CAR.vue')
    }
  ]
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  signatureStore.setCurrentPageId(to.name)
  next()
})

export default router