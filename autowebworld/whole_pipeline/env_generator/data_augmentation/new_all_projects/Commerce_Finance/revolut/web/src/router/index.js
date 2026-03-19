import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Import all pages
import HOME from '../pages/HOME.vue'
import ACCOUNTS_DASHBOARD from '../pages/ACCOUNTS_DASHBOARD.vue'
import ACCOUNT_DETAIL from '../pages/ACCOUNT_DETAIL.vue'
import PAYMENTS_LIST from '../pages/PAYMENTS_LIST.vue'
import BENEFICIARY_DETAIL from '../pages/BENEFICIARY_DETAIL.vue'
import TRANSFER_FORM from '../pages/TRANSFER_FORM.vue'
import TRANSFER_REVIEW from '../pages/TRANSFER_REVIEW.vue'
import TRANSFER_SUCCESS from '../pages/TRANSFER_SUCCESS.vue'
import EXCHANGE_DASHBOARD from '../pages/EXCHANGE_DASHBOARD.vue'
import EXCHANGE_FORM from '../pages/EXCHANGE_FORM.vue'
import EXCHANGE_REVIEW from '../pages/EXCHANGE_REVIEW.vue'
import EXCHANGE_SUCCESS from '../pages/EXCHANGE_SUCCESS.vue'
import CARDS_LIST from '../pages/CARDS_LIST.vue'
import CARD_DETAIL from '../pages/CARD_DETAIL.vue'
import CARD_FREEZE_FORM from '../pages/CARD_FREEZE_FORM.vue'
import CARD_FREEZE_SUCCESS from '../pages/CARD_FREEZE_SUCCESS.vue'
import CARD_LIMITS_FORM from '../pages/CARD_LIMITS_FORM.vue'
import CARD_LIMITS_SUCCESS from '../pages/CARD_LIMITS_SUCCESS.vue'
import TOPUP_METHOD_LIST from '../pages/TOPUP_METHOD_LIST.vue'
import TOPUP_FORM from '../pages/TOPUP_FORM.vue'
import TOPUP_REVIEW from '../pages/TOPUP_REVIEW.vue'
import TOPUP_SUCCESS from '../pages/TOPUP_SUCCESS.vue'

const routes = [
  { path: '/', name: 'HOME', component: HOME },
  { path: '/accounts', name: 'ACCOUNTS_DASHBOARD', component: ACCOUNTS_DASHBOARD },
  { path: '/account-detail', name: 'ACCOUNT_DETAIL', component: ACCOUNT_DETAIL },
  { path: '/payments', name: 'PAYMENTS_LIST', component: PAYMENTS_LIST },
  { path: '/beneficiary', name: 'BENEFICIARY_DETAIL', component: BENEFICIARY_DETAIL },
  { path: '/transfer', name: 'TRANSFER_FORM', component: TRANSFER_FORM },
  { path: '/transfer-review', name: 'TRANSFER_REVIEW', component: TRANSFER_REVIEW },
  { path: '/transfer-success', name: 'TRANSFER_SUCCESS', component: TRANSFER_SUCCESS },
  { path: '/exchange', name: 'EXCHANGE_DASHBOARD', component: EXCHANGE_DASHBOARD },
  { path: '/exchange-form', name: 'EXCHANGE_FORM', component: EXCHANGE_FORM },
  { path: '/exchange-review', name: 'EXCHANGE_REVIEW', component: EXCHANGE_REVIEW },
  { path: '/exchange-success', name: 'EXCHANGE_SUCCESS', component: EXCHANGE_SUCCESS },
  { path: '/cards', name: 'CARDS_LIST', component: CARDS_LIST },
  { path: '/card-detail', name: 'CARD_DETAIL', component: CARD_DETAIL },
  { path: '/card-freeze', name: 'CARD_FREEZE_FORM', component: CARD_FREEZE_FORM },
  { path: '/card-freeze-success', name: 'CARD_FREEZE_SUCCESS', component: CARD_FREEZE_SUCCESS },
  { path: '/card-limits', name: 'CARD_LIMITS_FORM', component: CARD_LIMITS_FORM },
  { path: '/card-limits-success', name: 'CARD_LIMITS_SUCCESS', component: CARD_LIMITS_SUCCESS },
  { path: '/topup', name: 'TOPUP_METHOD_LIST', component: TOPUP_METHOD_LIST },
  { path: '/topup-form', name: 'TOPUP_FORM', component: TOPUP_FORM },
  { path: '/topup-review', name: 'TOPUP_REVIEW', component: TOPUP_REVIEW },
  { path: '/topup-success', name: 'TOPUP_SUCCESS', component: TOPUP_SUCCESS }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  // Ensure FSM state is updated with current page
  if (to.name) {
    signatureStore.setCurrentPageId(to.name)
  }
  next()
})

export default router