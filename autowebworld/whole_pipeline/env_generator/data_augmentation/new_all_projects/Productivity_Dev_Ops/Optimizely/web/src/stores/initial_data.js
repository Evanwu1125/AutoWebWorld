import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // Mock Data - Experiments
  const experiments = ref([
    { id: 'exp_001', name: 'Homepage Hero Redesign', status: 'Running', type: 'A/B Test', visitors: 12500, conversions: 340, created: '2024-01-15', last_modified: '2024-02-01', image: '/images/WebDesign.jpg' },
    { id: 'exp_002', name: 'Checkout Flow Simplified', status: 'Paused', type: 'A/B Test', visitors: 5400, conversions: 120, created: '2024-01-10', last_modified: '2024-01-25', image: '/images/Checkout.jpg' },
    { id: 'exp_003', name: 'Pricing Tier Reordering', status: 'Draft', type: 'Multivariate', visitors: 0, conversions: 0, created: '2024-02-05', last_modified: '2024-02-05', image: '/images/Pricing.jpg' },
    { id: 'exp_004', name: 'Mobile Navigation Bar', status: 'Running', type: 'A/B Test', visitors: 45000, conversions: 890, created: '2023-12-01', last_modified: '2024-02-02', image: '/images/MobileNavigation.jpg' },
    { id: 'exp_005', name: 'Product Page Gallery', status: 'Archived', type: 'A/B Test', visitors: 15000, conversions: 450, created: '2023-11-15', last_modified: '2023-12-30', image: '/images/Shoes.jpg' },
    { id: 'exp_006', name: 'Cart Upsell Modal', status: 'Running', type: 'Personalization', visitors: 8000, conversions: 210, created: '2024-01-20', last_modified: '2024-02-03', image: '/images/ShoppingCart.jpg' },
    { id: 'exp_007', name: 'SignUp Form V2', status: 'Draft', type: 'A/B Test', visitors: 0, conversions: 0, created: '2024-02-06', last_modified: '2024-02-06', image: '/images/Signup.jpg' },
    { id: 'exp_008', name: 'Black Friday Banner', status: 'Archived', type: 'A/B Test', visitors: 95000, conversions: 3200, created: '2023-11-01', last_modified: '2023-12-01', image: '/images/BlackFriday.jpg' },
    { id: 'exp_009', name: 'Search Results Algorithm', status: 'Running', type: 'A/B Test', visitors: 32000, conversions: 670, created: '2024-01-05', last_modified: '2024-02-04', image: '/images/SearchResults.jpg' },
    { id: 'exp_010', name: 'Recommended Products', status: 'Running', type: 'Personalization', visitors: 18000, conversions: 560, created: '2024-01-12', last_modified: '2024-02-02', image: '/images/RecommendationEngine.jpg' },
    { id: 'exp_011', name: 'Footer Links Optimization', status: 'Paused', type: 'A/B Test', visitors: 2000, conversions: 15, created: '2023-10-10', last_modified: '2023-11-15', image: '/images/WebsiteFooter.jpg' },
    { id: 'exp_012', name: 'Video vs Image Hero', status: 'Running', type: 'A/B Test', visitors: 11000, conversions: 290, created: '2024-01-28', last_modified: '2024-02-05', image: '/images/VideoProduction.jpg' },
    { id: 'exp_013', name: 'Free Trial CTA Color', status: 'Draft', type: 'A/B Test', visitors: 0, conversions: 0, created: '2024-02-04', last_modified: '2024-02-04', image: '/images/CalltoAction.jpg' },
    { id: 'exp_014', name: 'Exit Intent Popup', status: 'Running', type: 'Personalization', visitors: 6500, conversions: 180, created: '2024-01-25', last_modified: '2024-02-01', image: '/images/PopupDesign.jpg' },
    { id: 'exp_015', name: 'Customer Testimonials Layout', status: 'Running', type: 'A/B Test', visitors: 9000, conversions: 220, created: '2024-01-18', last_modified: '2024-02-03', image: '/images/Testimonials.jpg' },
    { id: 'exp_016', name: 'Trust Badges Placement', status: 'Paused', type: 'A/B Test', visitors: 3000, conversions: 80, created: '2023-09-01', last_modified: '2023-10-01', image: '/images/SecurityBadges.jpg' },
    { id: 'exp_017', name: 'Category Menu Sort', status: 'Running', type: 'Multivariate', visitors: 25000, conversions: 600, created: '2024-01-08', last_modified: '2024-02-04', image: '/images/MenuNavigation.jpg' },
    { id: 'exp_018', name: 'Live Chat Prompt', status: 'Draft', type: 'Personalization', visitors: 0, conversions: 0, created: '2024-02-07', last_modified: '2024-02-07', image: '/images/LiveChat.jpg' }
  ])

  // Mock Data - Audiences
  const audiences = ref([
    { id: 'aud_001', name: 'All Visitors', size: 150000, description: 'Everyone who visits the site', type: 'Default', last_modified: '2023-01-01', image: '/images/Crowd.jpg' },
    { id: 'aud_002', name: 'Mobile Users', size: 85000, description: 'Visitors on mobile devices', type: 'Device', last_modified: '2023-06-15', image: '/images/MobileUsers.jpg' },
    { id: 'aud_003', name: 'Desktop Users', size: 65000, description: 'Visitors on desktop devices', type: 'Device', last_modified: '2023-06-15', image: '/images/Desktop.jpg' },
    { id: 'aud_004', name: 'Returning Visitors', size: 45000, description: 'Visitors with > 1 session', type: 'Behavior', last_modified: '2023-08-20', image: '/images/ReturningVisitors.jpg' },
    { id: 'aud_005', name: 'New Visitors', size: 105000, description: 'First time visitors', type: 'Behavior', last_modified: '2023-08-20', image: '/images/NewVisitors.jpg' },
    { id: 'aud_006', name: 'High Value (>$100)', size: 5000, description: 'Purchased more than $100', type: 'Revenue', last_modified: '2024-01-10', image: '/images/Shopping.jpg' },
    { id: 'aud_007', name: 'Cart Abandoners', size: 12000, description: 'Added to cart but did not buy', type: 'Behavior', last_modified: '2024-01-15', image: '/images/Shopping.jpg' },
    { id: 'aud_008', name: 'US Visitors', size: 60000, description: 'Geo location is United States', type: 'Geo', last_modified: '2023-05-01', image: '/images/USA.jpg' },
    { id: 'aud_009', name: 'EU Visitors', size: 40000, description: 'Geo location is Europe', type: 'Geo', last_modified: '2023-05-01', image: '/images/Europe.jpg' },
    { id: 'aud_010', name: 'iOS Users', size: 50000, description: 'Device OS is iOS', type: 'Tech', last_modified: '2023-07-01', image: '/images/iOS.jpg' },
    { id: 'aud_011', name: 'Android Users', size: 35000, description: 'Device OS is Android', type: 'Tech', last_modified: '2023-07-01', image: '/images/Android.jpg' },
    { id: 'aud_012', name: 'Social Media Referrals', size: 15000, description: 'Source is FB, Twitter, Insta', type: 'Source', last_modified: '2023-09-10', image: '/images/SocialMedia.jpg' },
    { id: 'aud_013', name: 'Email Campaign A', size: 8000, description: 'Source is Newsletter Jan', type: 'Source', last_modified: '2024-01-05', image: '/images/Email.jpg' },
    { id: 'aud_014', name: 'Chrome Users', size: 90000, description: 'Browser is Chrome', type: 'Tech', last_modified: '2023-04-20', image: '/images/Chrome.jpg' },
    { id: 'aud_015', name: 'Loyal Customers (VIP)', size: 1000, description: '> 5 Purchases lifetime', type: 'Revenue', last_modified: '2024-01-25', image: '/images/VIPCustomers.jpg' }
  ])

  // Mock Data - Feature Flags
  const feature_flags = ref([
    { id: 'flag_001', name: 'Dark Mode', key: 'dark_mode_v1', status: 'Active', rollout: 100, created: '2023-11-01', image: '/images/DarkMode.jpg' },
    { id: 'flag_002', name: 'New Checkout', key: 'checkout_v2', status: 'Active', rollout: 50, created: '2024-01-15', image: '/images/Checkout.jpg' },
    { id: 'flag_003', name: 'Beta Dashboard', key: 'dashboard_beta', status: 'Active', rollout: 10, created: '2024-02-01', image: '/images/Analytics.jpg' },
    { id: 'flag_004', name: 'AI Recommendations', key: 'ai_recs', status: 'Off', rollout: 0, created: '2024-02-05', image: '/images/ArtificialIntelligence.jpg' },
    { id: 'flag_005', name: 'SSO Login', key: 'sso_auth', status: 'Active', rollout: 100, created: '2023-10-20', image: '/images/LoginSecurity.jpg' },
    { id: 'flag_006', name: 'GraphQL API', key: 'graphql_endpoint', status: 'Active', rollout: 25, created: '2023-12-10', image: '/images/Programming.jpg' },
    { id: 'flag_007', name: 'React Migration', key: 'react_migration', status: 'Off', rollout: 0, created: '2024-01-01', image: '/images/React.jpg' },
    { id: 'flag_008', name: 'WebP Images', key: 'webp_support', status: 'Active', rollout: 100, created: '2023-09-15', image: '/images/image-file-format.jpg' },
    { id: 'flag_009', name: 'New Footer', key: 'footer_redesign', status: 'Active', rollout: 100, created: '2023-11-30', image: '/images/FooterDesign.jpg' },
    { id: 'flag_010', name: 'Chat Widget', key: 'intercom_chat', status: 'Active', rollout: 50, created: '2024-01-20', image: '/images/CustomerSupport.jpg' },
    { id: 'flag_011', name: 'Search Autocomplete', key: 'search_auto', status: 'Active', rollout: 75, created: '2024-01-10', image: '/images/Search.jpg' },
    { id: 'flag_012', name: 'Push Notifications', key: 'push_notifs', status: 'Off', rollout: 0, created: '2024-02-02', image: '/images/Notifications.jpg' },
    { id: 'flag_013', name: 'Mobile App Banner', key: 'app_banner', status: 'Active', rollout: 100, created: '2023-12-05', image: '/images/MobileApp.jpg' },
    { id: 'flag_014', name: 'GDPR Consent', key: 'gdpr_v2', status: 'Active', rollout: 100, created: '2023-08-01', image: '/images/GDPR.jpg' },
    { id: 'flag_015', name: 'New Pricing Table', key: 'pricing_2024', status: 'Off', rollout: 0, created: '2024-02-06', image: '/images/Pricing.jpg' }
  ])
  
  // Mock Data - Activity (Combined for Dashboard)
  const recent_activity = ref([
    { id: 'act_001', type: 'Experiment', action: 'started', item_name: 'Homepage Hero Redesign', time: '2 hours ago', user: 'Alice', image: '/images/User.jpg' },
    { id: 'act_002', type: 'Flag', action: 'toggled on', item_name: 'Beta Dashboard', time: '4 hours ago', user: 'Bob', image: '/images/Dashboard.jpg' },
    { id: 'act_003', type: 'Audience', action: 'created', item_name: 'High Value (>$100)', time: '1 day ago', user: 'Charlie', image: '/images/Smile.jpg' },
    { id: 'act_004', type: 'Experiment', action: 'paused', item_name: 'Checkout Flow Simplified', time: '2 days ago', user: 'Alice', image: '/images/User.jpg' },
    { id: 'act_005', type: 'Flag', action: 'rollout updated', item_name: 'New Checkout', time: '2 days ago', user: 'David', image: '/images/Checkout.jpg' },
    { id: 'act_006', type: 'Experiment', action: 'archived', item_name: 'Black Friday Banner', time: '1 week ago', user: 'Bob', image: '/images/Dashboard.jpg' },
    { id: 'act_007', type: 'Experiment', action: 'created', item_name: 'Pricing Tier Reordering', time: '1 week ago', user: 'Eve', image: '/images/Hair.jpg' },
    { id: 'act_008', type: 'Account', action: 'billing updated', item_name: 'Credit Card', time: '2 weeks ago', user: 'Admin', image: '/images/Admin.jpg' }
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