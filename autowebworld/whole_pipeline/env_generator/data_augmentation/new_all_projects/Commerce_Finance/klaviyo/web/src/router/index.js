import { createRouter, createWebHistory } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

// Import pages (Lazy loading recommended for many pages, but direct import is fine for template simplicity)
// Using dynamic imports for cleaner file

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'HOME', component: () => import('../pages/HOME.vue') },
    { path: '/dashboard', name: 'DASHBOARD', component: () => import('../pages/DASHBOARD.vue') },
    
    // Campaigns
    { path: '/campaigns', name: 'CAMPAIGNS_LIST', component: () => import('../pages/CAMPAIGNS_LIST.vue') },
    { path: '/campaigns/create/channel', name: 'CREATE_CAMPAIGN_CHANNEL', component: () => import('../pages/CREATE_CAMPAIGN_CHANNEL.vue') },
    { path: '/campaigns/create/email/basics', name: 'CREATE_CAMPAIGN_BASICS', component: () => import('../pages/CREATE_CAMPAIGN_BASICS.vue') },
    { path: '/campaigns/create/email/recipients', name: 'CREATE_CAMPAIGN_RECIPIENTS', component: () => import('../pages/CREATE_CAMPAIGN_RECIPIENTS.vue') },
    { path: '/campaigns/create/email/content', name: 'CREATE_CAMPAIGN_CONTENT', component: () => import('../pages/CREATE_CAMPAIGN_CONTENT.vue') },
    { path: '/campaigns/create/email/schedule', name: 'CREATE_CAMPAIGN_REVIEW_SCHEDULE', component: () => import('../pages/CREATE_CAMPAIGN_REVIEW_SCHEDULE.vue') },
    { path: '/campaigns/create/email/success', name: 'EMAIL_CAMPAIGN_SCHEDULED_SUCCESS', component: () => import('../pages/EMAIL_CAMPAIGN_SCHEDULED_SUCCESS.vue') },
    
    // SMS Campaigns
    { path: '/campaigns/create/sms/basics', name: 'CREATE_SMS_CAMPAIGN_BASICS', component: () => import('../pages/CREATE_SMS_CAMPAIGN_BASICS.vue') },
    { path: '/campaigns/create/sms/recipients', name: 'CREATE_SMS_CAMPAIGN_RECIPIENTS', component: () => import('../pages/CREATE_SMS_CAMPAIGN_RECIPIENTS.vue') },
    { path: '/campaigns/create/sms/content', name: 'CREATE_SMS_CAMPAIGN_CONTENT', component: () => import('../pages/CREATE_SMS_CAMPAIGN_CONTENT.vue') },
    { path: '/campaigns/create/sms/schedule', name: 'CREATE_SMS_CAMPAIGN_REVIEW_SCHEDULE', component: () => import('../pages/CREATE_SMS_CAMPAIGN_REVIEW_SCHEDULE.vue') },
    { path: '/campaigns/create/sms/success', name: 'SMS_CAMPAIGN_SCHEDULED_SUCCESS', component: () => import('../pages/SMS_CAMPAIGN_SCHEDULED_SUCCESS.vue') },
    
    { path: '/campaigns/:id', name: 'CAMPAIGN_DETAIL', component: () => import('../pages/CAMPAIGN_DETAIL.vue') },

    // Flows
    { path: '/flows', name: 'FLOWS_LIST', component: () => import('../pages/FLOWS_LIST.vue') },
    { path: '/flows/create/trigger', name: 'FLOW_TRIGGER_SETUP', component: () => import('../pages/FLOW_TRIGGER_SETUP.vue') },
    { path: '/flows/create/content', name: 'FLOW_EMAIL_CONTENT', component: () => import('../pages/FLOW_EMAIL_CONTENT.vue') },
    { path: '/flows/create/review', name: 'FLOW_REVIEW_ACTIVATE', component: () => import('../pages/FLOW_REVIEW_ACTIVATE.vue') },
    { path: '/flows/create/success', name: 'FLOW_CREATED_SUCCESS', component: () => import('../pages/FLOW_CREATED_SUCCESS.vue') },
    { path: '/flows/:id', name: 'FLOW_DETAIL', component: () => import('../pages/FLOW_DETAIL.vue') },

    // Lists & Segments
    { path: '/lists', name: 'LISTS_SEGMENTS', component: () => import('../pages/LISTS_SEGMENTS.vue') },
    { path: '/segments/create', name: 'SEGMENT_BUILDER', component: () => import('../pages/SEGMENT_BUILDER.vue') },
    { path: '/segments/create/success', name: 'SEGMENT_CREATED_SUCCESS', component: () => import('../pages/SEGMENT_CREATED_SUCCESS.vue') },

    // Signup Forms
    { path: '/forms', name: 'SIGNUP_FORMS_LIST', component: () => import('../pages/SIGNUP_FORMS_LIST.vue') },
    { path: '/forms/create', name: 'SIGNUP_FORM_BUILDER', component: () => import('../pages/SIGNUP_FORM_BUILDER.vue') },
    { path: '/forms/create/success', name: 'SIGNUP_FORM_PUBLISHED_SUCCESS', component: () => import('../pages/SIGNUP_FORM_PUBLISHED_SUCCESS.vue') }
  ]
})

router.beforeEach((to, from, next) => {
  const signatureStore = useSignatureStore()
  if (to.name) {
    signatureStore.setCurrentPageId(to.name)
  }
  next()
})

export default router