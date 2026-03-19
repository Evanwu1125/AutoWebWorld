import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  
  // --- Campaigns (Email & SMS) ---
  const campaigns = ref([
    { id: 'cmp_1', name: 'Black Friday Sale', type: 'email', status: 'scheduled', revenue: 5000, sent: '2025-11-25', image: '/images/campaigns_cmp_1.jpg' },
    { id: 'cmp_2', name: 'Cyber Monday Blast', type: 'email', status: 'draft', revenue: 0, sent: null, image: '/images/campaigns_cmp_2.jpg' },
    { id: 'cmp_3', name: 'Welcome Series Intro', type: 'email', status: 'sent', revenue: 1200, sent: '2025-10-01', image: '/images/campaigns_cmp_3.jpg' },
    { id: 'cmp_4', name: 'Abandoned Cart Recovery', type: 'sms', status: 'sent', revenue: 3400, sent: '2025-10-05', image: '/images/campaigns_cmp_4.jpg' },
    { id: 'cmp_5', name: 'VIP Exclusive Access', type: 'sms', status: 'scheduled', revenue: 8000, sent: '2025-12-01', image: '/images/campaigns_cmp_5.jpg' },
    { id: 'cmp_6', name: 'Holiday Gift Guide', type: 'email', status: 'draft', revenue: 0, sent: null, image: '/images/campaigns_cmp_6.jpg' },
    { id: 'cmp_7', name: 'Spring Collection Tease', type: 'email', status: 'sent', revenue: 4500, sent: '2025-03-15', image: '/images/campaigns_cmp_7.jpg' },
    { id: 'cmp_8', name: 'Flash Sale Alert', type: 'sms', status: 'sent', revenue: 2100, sent: '2025-06-20', image: '/images/campaigns_cmp_8.jpg' },
    { id: 'cmp_9', name: 'Newsletter October', type: 'email', status: 'sent', revenue: 500, sent: '2025-10-30', image: '/images/campaigns_cmp_9.jpg' },
    { id: 'cmp_10', name: 'Product Launch: Sneakers', type: 'email', status: 'scheduled', revenue: 15000, sent: '2025-11-10', image: '/images/campaigns_cmp_10.jpg' },
    { id: 'cmp_11', name: 'Webinar Invitation', type: 'email', status: 'sent', revenue: 0, sent: '2025-09-12', image: '/images/campaigns_cmp_11.jpg' },
    { id: 'cmp_12', name: 'Feedback Survey', type: 'email', status: 'sent', revenue: 0, sent: '2025-08-05', image: '/images/campaigns_cmp_12.jpg' },
    { id: 'cmp_13', name: 'Win Back Campaign', type: 'email', status: 'draft', revenue: 0, sent: null, image: '/images/campaigns_cmp_13.jpg' },
    { id: 'cmp_14', name: 'Birthday Discount', type: 'sms', status: 'sent', revenue: 900, sent: '2025-01-15', image: '/images/campaigns_cmp_14.jpg' },
    { id: 'cmp_15', name: 'Summer Clearance', type: 'email', status: 'sent', revenue: 6700, sent: '2025-07-20', image: '/images/campaigns_cmp_15.jpg' },
    { id: 'cmp_16', name: 'New Year Resolution', type: 'email', status: 'scheduled', revenue: 2000, sent: '2026-01-01', image: '/images/campaigns_cmp_16.jpg' }
  ])

  // --- Flows (Automations) ---
  const flows = ref([
    { id: 'flw_1', name: 'Welcome Series', status: 'live', trigger: 'List Subscribe', revenue: 12500, image: '/images/flows_flw_1.jpg' },
    { id: 'flw_2', name: 'Abandoned Cart', status: 'live', trigger: 'Checkout Started', revenue: 45000, image: '/images/flows_flw_2.jpg' },
    { id: 'flw_3', name: 'Browse Abandonment', status: 'live', trigger: 'Viewed Product', revenue: 8900, image: '/images/flows_flw_3.jpg' },
    { id: 'flw_4', name: 'Customer Winback', status: 'draft', trigger: 'Placed Order', revenue: 0, image: '/images/flows_flw_4.jpg' },
    { id: 'flw_5', name: 'Post-Purchase Thank You', status: 'live', trigger: 'Placed Order', revenue: 3200, image: '/images/flows_flw_5.jpg' },
    { id: 'flw_6', name: 'Happy Birthday', status: 'live', trigger: 'Date Property', revenue: 5600, image: '/images/flows_flw_6.jpg' },
    { id: 'flw_7', name: 'VIP Recognition', status: 'draft', trigger: 'Segment Join', revenue: 0, image: '/images/flows_flw_7.jpg' },
    { id: 'flw_8', name: 'First Purchase Anniversary', status: 'live', trigger: 'Date Property', revenue: 1500, image: '/images/flows_flw_8.jpg' },
    { id: 'flw_9', name: 'Cross-Sell Electronics', status: 'live', trigger: 'Placed Order', revenue: 7800, image: '/images/flows_flw_9.jpg' },
    { id: 'flw_10', name: 'Sunset Flow (Unengaged)', status: 'live', trigger: 'Segment Join', revenue: 200, image: '/images/flows_flw_10.jpg' },
    { id: 'flw_11', name: 'Back in Stock Alert', status: 'live', trigger: 'Subscribed to Back in Stock', revenue: 9500, image: '/images/flows_flw_11.jpg' },
    { id: 'flw_12', name: 'Price Drop Alert', status: 'draft', trigger: 'Price Drop', revenue: 0, image: '/images/flows_flw_12.jpg' },
    { id: 'flw_13', name: 'Review Request', status: 'live', trigger: 'Fulfilled Order', revenue: 0, image: '/images/flows_flw_13.jpg' },
    { id: 'flw_14', name: 'Loyalty Tier Upgrade', status: 'live', trigger: 'Segment Join', revenue: 4100, image: '/images/flows_flw_14.jpg' },
    { id: 'flw_15', name: 'Shipping Confirmation', status: 'live', trigger: 'Fulfilled Order', revenue: 0, image: '/images/flows_flw_15.jpg' }
  ])

  // --- Lists & Segments ---
  const lists = ref([
    { id: 'list_1', name: 'Newsletter Subscribers', size: 15400, type: 'list', image: '/images/lists_list_1.jpg' },
    { id: 'list_2', name: 'SMS Subscribers', size: 5200, type: 'list', image: '/images/lists_list_2.jpg' },
    { id: 'list_3', name: 'VIP Customers', size: 350, type: 'list', image: '/images/lists_list_3.jpg' },
    { id: 'list_4', name: 'Employee List', size: 45, type: 'list', image: '/images/lists_list_4.jpg' },
    { id: 'list_5', name: 'Contest Entrants', size: 2300, type: 'list', image: '/images/lists_list_5.jpg' },
    { id: 'sms_list_1', name: 'Main SMS List', size: 5000, type: 'list', image: '/images/lists_sms_list_1.jpg' },
    { id: 'sms_list_2', name: 'Black Friday Early Access', size: 1200, type: 'list', image: '/images/lists_sms_list_2.jpg' }
  ])

  const segments = ref([
    { id: 'seg_1', name: 'Engaged (30 Days)', size: 4500, type: 'segment', condition: 'Opened email in last 30 days', image: '/images/segments_seg_1.jpg' },
    { id: 'seg_2', name: 'Engaged (90 Days)', size: 8900, type: 'segment', condition: 'Opened email in last 90 days', image: '/images/segments_seg_2.jpg' },
    { id: 'seg_3', name: 'Unengaged (180 Days)', size: 3200, type: 'segment', condition: 'No open in last 180 days', image: '/images/segments_seg_3.jpg' },
    { id: 'seg_4', name: 'Recent Purchasers', size: 600, type: 'segment', condition: 'Placed Order in last 30 days', image: '/images/segments_seg_4.jpg' },
    { id: 'seg_5', name: 'Big Spenders (> $500)', size: 150, type: 'segment', condition: 'Revenue > 500', image: '/images/segments_seg_5.jpg' },
    { id: 'seg_6', name: 'Potential VIPs', size: 85, type: 'segment', condition: 'Placed Order count > 3', image: '/images/segments_seg_6.jpg' },
    { id: 'seg_7', name: 'Churn Risk', size: 1200, type: 'segment', condition: 'No order in 120 days', image: '/images/segments_seg_7.jpg' },
    { id: 'seg_8', name: 'Local: New York', size: 2400, type: 'segment', condition: 'City = New York', image: '/images/segments_seg_8.jpg' },
    { id: 'seg_9', name: 'Local: California', size: 3100, type: 'segment', condition: 'State = CA', image: '/images/segments_seg_9.jpg' },
    { id: 'seg_10', name: 'Men\'s Interest', size: 4500, type: 'segment', condition: 'Viewed Men\'s Category', image: '/images/segments_seg_10.jpg' },
    { id: 'seg_11', name: 'Women\'s Interest', size: 5600, type: 'segment', condition: 'Viewed Women\'s Category', image: '/images/segments_seg_11.jpg' },
    { id: 'seg_12', name: 'Holiday Shoppers', size: 2100, type: 'segment', condition: 'Purchased in Nov/Dec', image: '/images/segments_seg_12.jpg' },
    { id: 'seg_13', name: 'Email Bounced', size: 120, type: 'segment', condition: 'Bounced > 0', image: '/images/segments_seg_13.jpg' },
    { id: 'seg_14', name: 'Suppressed Profiles', size: 450, type: 'segment', condition: 'Is Suppressed = true', image: '/images/segments_seg_14.jpg' },
    { id: 'seg_15', name: 'GDPR Consent True', size: 2300, type: 'segment', condition: 'Consent = true', image: '/images/segments_seg_15.jpg' }
  ])

  // --- Signup Forms ---
  const signup_forms = ref([
    { id: 'form_1', name: 'Newsletter Popup', type: 'popup', status: 'live', views: 12000, submissions: 450, image: '/images/signup_forms_form_1.jpg' },
    { id: 'form_2', name: 'Footer Embed', type: 'embed', status: 'live', views: 45000, submissions: 120, image: '/images/signup_forms_form_2.jpg' },
    { id: 'form_3', name: 'Exit Intent Offer', type: 'flyout', status: 'live', views: 5600, submissions: 300, image: '/images/signup_forms_form_3.jpg' },
    { id: 'form_4', name: 'Black Friday Teaser', type: 'popup', status: 'draft', views: 0, submissions: 0, image: '/images/signup_forms_form_4.jpg' },
    { id: 'form_5', name: 'SMS Opt-in Mobile', type: 'popup', status: 'live', views: 8900, submissions: 890, image: '/images/signup_forms_form_5.jpg' },
    { id: 'form_6', name: 'Contact Us Page', type: 'embed', status: 'live', views: 2300, submissions: 50, image: '/images/signup_forms_form_6.jpg' },
    { id: 'form_7', name: 'Sidebar Subscribe', type: 'embed', status: 'paused', views: 1200, submissions: 12, image: '/images/signup_forms_form_7.jpg' },
    { id: 'form_8', name: 'Contest Entry Form', type: 'full_page', status: 'live', views: 5000, submissions: 2500, image: '/images/signup_forms_form_8.jpg' },
    { id: 'form_9', name: 'Webinar Registration', type: 'full_page', status: 'live', views: 1200, submissions: 400, image: '/images/signup_forms_form_9.jpg' },
    { id: 'form_10', name: 'Quiz: Find Your Style', type: 'popup', status: 'live', views: 3400, submissions: 1500, image: '/images/signup_forms_form_10.jpg' },
    { id: 'form_11', name: 'Pre-order Interest', type: 'flyout', status: 'draft', views: 0, submissions: 0, image: '/images/signup_forms_form_11.jpg' },
    { id: 'form_12', name: 'Free Shipping Banner', type: 'embed', status: 'live', views: 67000, submissions: 230, image: '/images/signup_forms_form_12.jpg' }
  ])

  // --- Email Templates ---
  const email_templates = ref([
    { id: 'template_1', name: 'Basic Newsletter', image: '/images/email_templates_template_1.jpg' },
    { id: 'template_2', name: 'Product Showcase', image: '/images/email_templates_template_2.jpg' },
    { id: 'template_3', name: 'Welcome Letter', image: '/images/email_templates_template_3.jpg' },
    { id: 'template_4', name: 'Event Invite', image: '/images/email_templates_template_4.jpg' }
  ])

  return {
    campaigns,
    flows,
    lists,
    segments,
    signup_forms,
    email_templates
  }
}, {
  persist: {
    storage: sessionStorage
  }
})