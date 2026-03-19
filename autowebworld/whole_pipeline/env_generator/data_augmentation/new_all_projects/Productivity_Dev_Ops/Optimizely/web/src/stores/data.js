import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // Mock Data - Experiments
  const experiments = ref([
    { id: 'exp_001', name: 'Homepage Hero Redesign', status: 'Running', type: 'A/B Test', visitors: 12500, conversions: 340, created: '2024-01-15', last_modified: '2024-02-01', image: '/images/experiments_exp_001.jpg' },
    { id: 'exp_002', name: 'Checkout Flow Simplified', status: 'Paused', type: 'A/B Test', visitors: 5400, conversions: 120, created: '2024-01-10', last_modified: '2024-01-25', image: '/images/experiments_exp_002.jpg' },
    { id: 'exp_003', name: 'Pricing Tier Reordering', status: 'Draft', type: 'Multivariate', visitors: 0, conversions: 0, created: '2024-02-05', last_modified: '2024-02-05', image: '/images/experiments_exp_003.jpg' },
    { id: 'exp_004', name: 'Mobile Navigation Bar', status: 'Running', type: 'A/B Test', visitors: 45000, conversions: 890, created: '2023-12-01', last_modified: '2024-02-02', image: '/images/experiments_exp_004.jpg' },
    { id: 'exp_005', name: 'Product Page Gallery', status: 'Archived', type: 'A/B Test', visitors: 15000, conversions: 450, created: '2023-11-15', last_modified: '2023-12-30', image: '/images/experiments_exp_005.jpg' },
    { id: 'exp_006', name: 'Cart Upsell Modal', status: 'Running', type: 'Personalization', visitors: 8000, conversions: 210, created: '2024-01-20', last_modified: '2024-02-03', image: '/images/experiments_exp_006.jpg' },
    { id: 'exp_007', name: 'SignUp Form V2', status: 'Draft', type: 'A/B Test', visitors: 0, conversions: 0, created: '2024-02-06', last_modified: '2024-02-06', image: '/images/experiments_exp_007.jpg' },
    { id: 'exp_008', name: 'Black Friday Banner', status: 'Archived', type: 'A/B Test', visitors: 95000, conversions: 3200, created: '2023-11-01', last_modified: '2023-12-01', image: '/images/experiments_exp_008.jpg' },
    { id: 'exp_009', name: 'Search Results Algorithm', status: 'Running', type: 'A/B Test', visitors: 32000, conversions: 670, created: '2024-01-05', last_modified: '2024-02-04', image: '/images/experiments_exp_009.jpg' },
    { id: 'exp_010', name: 'Recommended Products', status: 'Running', type: 'Personalization', visitors: 18000, conversions: 560, created: '2024-01-12', last_modified: '2024-02-02', image: '/images/experiments_exp_010.jpg' },
    { id: 'exp_011', name: 'Footer Links Optimization', status: 'Paused', type: 'A/B Test', visitors: 2000, conversions: 15, created: '2023-10-10', last_modified: '2023-11-15', image: '/images/experiments_exp_011.jpg' },
    { id: 'exp_012', name: 'Video vs Image Hero', status: 'Running', type: 'A/B Test', visitors: 11000, conversions: 290, created: '2024-01-28', last_modified: '2024-02-05', image: '/images/experiments_exp_012.jpg' },
    { id: 'exp_013', name: 'Free Trial CTA Color', status: 'Draft', type: 'A/B Test', visitors: 0, conversions: 0, created: '2024-02-04', last_modified: '2024-02-04', image: '/images/experiments_exp_013.jpg' },
    { id: 'exp_014', name: 'Exit Intent Popup', status: 'Running', type: 'Personalization', visitors: 6500, conversions: 180, created: '2024-01-25', last_modified: '2024-02-01', image: '/images/experiments_exp_014.jpg' },
    { id: 'exp_015', name: 'Customer Testimonials Layout', status: 'Running', type: 'A/B Test', visitors: 9000, conversions: 220, created: '2024-01-18', last_modified: '2024-02-03', image: '/images/experiments_exp_015.jpg' },
    { id: 'exp_016', name: 'Trust Badges Placement', status: 'Paused', type: 'A/B Test', visitors: 3000, conversions: 80, created: '2023-09-01', last_modified: '2023-10-01', image: '/images/experiments_exp_016.jpg' },
    { id: 'exp_017', name: 'Category Menu Sort', status: 'Running', type: 'Multivariate', visitors: 25000, conversions: 600, created: '2024-01-08', last_modified: '2024-02-04', image: '/images/experiments_exp_017.jpg' },
    { id: 'exp_018', name: 'Live Chat Prompt', status: 'Draft', type: 'Personalization', visitors: 0, conversions: 0, created: '2024-02-07', last_modified: '2024-02-07', image: '/images/experiments_exp_018.jpg' }
  ])

  // Mock Data - Audiences
  const audiences = ref([
    { id: 'aud_001', name: 'All Visitors', size: 150000, description: 'Everyone who visits the site', type: 'Default', last_modified: '2023-01-01', image: '/images/audiences_aud_001.jpg' },
    { id: 'aud_002', name: 'Mobile Users', size: 85000, description: 'Visitors on mobile devices', type: 'Device', last_modified: '2023-06-15', image: '/images/audiences_aud_002.jpg' },
    { id: 'aud_003', name: 'Desktop Users', size: 65000, description: 'Visitors on desktop devices', type: 'Device', last_modified: '2023-06-15', image: '/images/audiences_aud_003.jpg' },
    { id: 'aud_004', name: 'Returning Visitors', size: 45000, description: 'Visitors with > 1 session', type: 'Behavior', last_modified: '2023-08-20', image: '/images/audiences_aud_004.jpg' },
    { id: 'aud_005', name: 'New Visitors', size: 105000, description: 'First time visitors', type: 'Behavior', last_modified: '2023-08-20', image: '/images/audiences_aud_005.jpg' },
    { id: 'aud_006', name: 'High Value (>$100)', size: 5000, description: 'Purchased more than $100', type: 'Revenue', last_modified: '2024-01-10', image: '/images/audiences_aud_006.jpg' },
    { id: 'aud_007', name: 'Cart Abandoners', size: 12000, description: 'Added to cart but did not buy', type: 'Behavior', last_modified: '2024-01-15', image: '/images/audiences_aud_007.jpg' },
    { id: 'aud_008', name: 'US Visitors', size: 60000, description: 'Geo location is United States', type: 'Geo', last_modified: '2023-05-01', image: '/images/audiences_aud_008.jpg' },
    { id: 'aud_009', name: 'EU Visitors', size: 40000, description: 'Geo location is Europe', type: 'Geo', last_modified: '2023-05-01', image: '/images/audiences_aud_009.jpg' },
    { id: 'aud_010', name: 'iOS Users', size: 50000, description: 'Device OS is iOS', type: 'Tech', last_modified: '2023-07-01', image: '/images/audiences_aud_010.jpg' },
    { id: 'aud_011', name: 'Android Users', size: 35000, description: 'Device OS is Android', type: 'Tech', last_modified: '2023-07-01', image: '/images/audiences_aud_011.jpg' },
    { id: 'aud_012', name: 'Social Media Referrals', size: 15000, description: 'Source is FB, Twitter, Insta', type: 'Source', last_modified: '2023-09-10', image: '/images/audiences_aud_012.jpg' },
    { id: 'aud_013', name: 'Email Campaign A', size: 8000, description: 'Source is Newsletter Jan', type: 'Source', last_modified: '2024-01-05', image: '/images/audiences_aud_013.jpg' },
    { id: 'aud_014', name: 'Chrome Users', size: 90000, description: 'Browser is Chrome', type: 'Tech', last_modified: '2023-04-20', image: '/images/audiences_aud_014.jpg' },
    { id: 'aud_015', name: 'Loyal Customers (VIP)', size: 1000, description: '> 5 Purchases lifetime', type: 'Revenue', last_modified: '2024-01-25', image: '/images/audiences_aud_015.jpg' }
  ])

  // Mock Data - Feature Flags
  const feature_flags = ref([
    { id: 'flag_001', name: 'Dark Mode', key: 'dark_mode_v1', status: 'Active', rollout: 100, created: '2023-11-01', image: '/images/feature_flags_flag_001.jpg' },
    { id: 'flag_002', name: 'New Checkout', key: 'checkout_v2', status: 'Active', rollout: 50, created: '2024-01-15', image: '/images/feature_flags_flag_002.jpg' },
    { id: 'flag_003', name: 'Beta Dashboard', key: 'dashboard_beta', status: 'Active', rollout: 10, created: '2024-02-01', image: '/images/feature_flags_flag_003.jpg' },
    { id: 'flag_004', name: 'AI Recommendations', key: 'ai_recs', status: 'Off', rollout: 0, created: '2024-02-05', image: '/images/feature_flags_flag_004.jpg' },
    { id: 'flag_005', name: 'SSO Login', key: 'sso_auth', status: 'Active', rollout: 100, created: '2023-10-20', image: '/images/feature_flags_flag_005.jpg' },
    { id: 'flag_006', name: 'GraphQL API', key: 'graphql_endpoint', status: 'Active', rollout: 25, created: '2023-12-10', image: '/images/feature_flags_flag_006.jpg' },
    { id: 'flag_007', name: 'React Migration', key: 'react_migration', status: 'Off', rollout: 0, created: '2024-01-01', image: '/images/feature_flags_flag_007.jpg' },
    { id: 'flag_008', name: 'WebP Images', key: 'webp_support', status: 'Active', rollout: 100, created: '2023-09-15', image: '/images/feature_flags_flag_008.jpg' },
    { id: 'flag_009', name: 'New Footer', key: 'footer_redesign', status: 'Active', rollout: 100, created: '2023-11-30', image: '/images/feature_flags_flag_009.jpg' },
    { id: 'flag_010', name: 'Chat Widget', key: 'intercom_chat', status: 'Active', rollout: 50, created: '2024-01-20', image: '/images/feature_flags_flag_010.jpg' },
    { id: 'flag_011', name: 'Search Autocomplete', key: 'search_auto', status: 'Active', rollout: 75, created: '2024-01-10', image: '/images/feature_flags_flag_011.jpg' },
    { id: 'flag_012', name: 'Push Notifications', key: 'push_notifs', status: 'Off', rollout: 0, created: '2024-02-02', image: '/images/feature_flags_flag_012.jpg' },
    { id: 'flag_013', name: 'Mobile App Banner', key: 'app_banner', status: 'Active', rollout: 100, created: '2023-12-05', image: '/images/feature_flags_flag_013.jpg' },
    { id: 'flag_014', name: 'GDPR Consent', key: 'gdpr_v2', status: 'Active', rollout: 100, created: '2023-08-01', image: '/images/feature_flags_flag_014.jpg' },
    { id: 'flag_015', name: 'New Pricing Table', key: 'pricing_2024', status: 'Off', rollout: 0, created: '2024-02-06', image: '/images/feature_flags_flag_015.jpg' }
  ])
  
  // Mock Data - Activity (Combined for Dashboard)
  const recent_activity = ref([
    { id: 'act_001', type: 'Experiment', action: 'started', item_name: 'Homepage Hero Redesign', time: '2 hours ago', user: 'Alice', image: '/images/recent_activity_act_001.jpg' },
    { id: 'act_002', type: 'Flag', action: 'toggled on', item_name: 'Beta Dashboard', time: '4 hours ago', user: 'Bob', image: '/images/recent_activity_act_002.jpg' },
    { id: 'act_003', type: 'Audience', action: 'created', item_name: 'High Value (>$100)', time: '1 day ago', user: 'Charlie', image: '/images/recent_activity_act_003.jpg' },
    { id: 'act_004', type: 'Experiment', action: 'paused', item_name: 'Checkout Flow Simplified', time: '2 days ago', user: 'Alice', image: '/images/recent_activity_act_004.jpg' },
    { id: 'act_005', type: 'Flag', action: 'rollout updated', item_name: 'New Checkout', time: '2 days ago', user: 'David', image: '/images/recent_activity_act_005.jpg' },
    { id: 'act_006', type: 'Experiment', action: 'archived', item_name: 'Black Friday Banner', time: '1 week ago', user: 'Bob', image: '/images/recent_activity_act_006.jpg' },
    { id: 'act_007', type: 'Experiment', action: 'created', item_name: 'Pricing Tier Reordering', time: '1 week ago', user: 'Eve', image: '/images/recent_activity_act_007.jpg' },
    { id: 'act_008', type: 'Account', action: 'billing updated', item_name: 'Credit Card', time: '2 weeks ago', user: 'Admin', image: '/images/recent_activity_act_008.jpg' }
  ])

  return {
    experiments,
    audiences,
    feature_flags,
    recent_activity
  }
}, {
  persist: {
    storage: sessionStorage
  }
})