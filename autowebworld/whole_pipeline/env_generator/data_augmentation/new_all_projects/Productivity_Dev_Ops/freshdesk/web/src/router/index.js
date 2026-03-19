import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    {
      path: '/',
      name: 'HOME',
      component: () => import('../pages/HOME.vue')
    },
    {
      path: '/tickets',
      name: 'TICKETS_LIST',
      component: () => import('../pages/TICKETS_LIST.vue')
    },
    {
      path: '/tickets/:id',
      name: 'TICKET_DETAIL',
      component: () => import('../pages/TICKET_DETAIL.vue')
    },
    {
      path: '/tickets/new',
      name: 'NEW_TICKET_FORM',
      component: () => import('../pages/NEW_TICKET_FORM.vue')
    },
    {
      path: '/tickets/new/review',
      name: 'NEW_TICKET_REVIEW',
      component: () => import('../pages/NEW_TICKET_REVIEW.vue')
    },
    {
      path: '/tickets/new/success',
      name: 'TICKET_CREATION_SUCCESS',
      component: () => import('../pages/TICKET_CREATION_SUCCESS.vue')
    },
    {
      path: '/tickets/:id/reply/review',
      name: 'REPLY_REVIEW',
      component: () => import('../pages/REPLY_REVIEW.vue')
    },
    {
      path: '/tickets/:id/reply/success',
      name: 'REPLY_SENT_SUCCESS',
      component: () => import('../pages/REPLY_SENT_SUCCESS.vue')
    },
    {
      path: '/tickets/:id/assign',
      name: 'ASSIGN_TICKET',
      component: () => import('../pages/ASSIGN_TICKET.vue')
    },
    {
      path: '/tickets/:id/assign/success',
      name: 'ASSIGN_SUCCESS',
      component: () => import('../pages/ASSIGN_SUCCESS.vue')
    },
    {
      path: '/tickets/:id/merge/select',
      name: 'MERGE_TICKET_SELECT',
      component: () => import('../pages/MERGE_TICKET_SELECT.vue')
    },
    {
      path: '/tickets/:id/merge/confirm',
      name: 'MERGE_TICKET_CONFIRM',
      component: () => import('../pages/MERGE_TICKET_CONFIRM.vue')
    },
    {
      path: '/tickets/:id/merge/success',
      name: 'MERGE_SUCCESS',
      component: () => import('../pages/MERGE_SUCCESS.vue')
    },
    {
      path: '/contacts',
      name: 'CONTACTS_LIST',
      component: () => import('../pages/CONTACTS_LIST.vue')
    },
    {
      path: '/contacts/:id',
      name: 'CONTACT_DETAIL',
      component: () => import('../pages/CONTACT_DETAIL.vue')
    },
    {
      path: '/contacts/new',
      name: 'NEW_CONTACT_FORM',
      component: () => import('../pages/NEW_CONTACT_FORM.vue')
    },
    {
      path: '/contacts/new/review',
      name: 'NEW_CONTACT_REVIEW',
      component: () => import('../pages/NEW_CONTACT_REVIEW.vue')
    },
    {
      path: '/contacts/new/success',
      name: 'CONTACT_CREATION_SUCCESS',
      component: () => import('../pages/CONTACT_CREATION_SUCCESS.vue')
    },
    {
      path: '/dashboard',
      name: 'DASHBOARD',
      component: () => import('../pages/DASHBOARD.vue')
    }
  ]
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  signatureStore.setCurrentPageId(to.name)
  next()
})

export default router