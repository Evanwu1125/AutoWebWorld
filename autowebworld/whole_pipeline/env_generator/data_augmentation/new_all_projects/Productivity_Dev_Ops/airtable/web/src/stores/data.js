import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // --- Mock Data Generators ---
  
  const bases = ref([
    { id: 'base_001', name: 'Product Roadmap', color: 'blue', icon: 'grid', starred: true, activity: 90, last_viewed: '2025-10-25', image: '/images/bases_base_001.jpg' },
    { id: 'base_002', name: 'Content Calendar', color: 'red', icon: 'calendar', starred: false, activity: 75, last_viewed: '2025-10-24', image: '/images/bases_base_002.jpg' },
    { id: 'base_003', name: 'Sales CRM', color: 'green', icon: 'users', starred: true, activity: 95, last_viewed: '2025-10-26', image: '/images/bases_base_003.jpg' },
    { id: 'base_004', name: 'Event Planning', color: 'purple', icon: 'calendar', starred: false, activity: 40, last_viewed: '2025-10-20', image: '/images/bases_base_004.jpg' },
    { id: 'base_005', name: 'User Research', color: 'yellow', icon: 'clipboard', starred: true, activity: 85, last_viewed: '2025-10-25', image: '/images/bases_base_005.jpg' },
    { id: 'base_006', name: 'Bug Tracker', color: 'red', icon: 'bug', starred: false, activity: 60, last_viewed: '2025-10-22', image: '/images/bases_base_006.jpg' },
    { id: 'base_007', name: 'Marketing Campaign', color: 'pink', icon: 'megaphone', starred: false, activity: 50, last_viewed: '2025-10-18', image: '/images/bases_base_007.jpg' },
    { id: 'base_008', name: 'Hiring Pipeline', color: 'teal', icon: 'briefcase', starred: true, activity: 70, last_viewed: '2025-10-23', image: '/images/bases_base_008.jpg' },
    { id: 'base_009', name: 'Inventory Management', color: 'orange', icon: 'box', starred: false, activity: 30, last_viewed: '2025-10-15', image: '/images/bases_base_009.jpg' },
    { id: 'base_010', name: 'Legal Contracts', color: 'gray', icon: 'file-text', starred: false, activity: 20, last_viewed: '2025-10-10', image: '/images/bases_base_010.jpg' },
    { id: 'base_011', name: 'Design Assets', color: 'blue', icon: 'image', starred: true, activity: 88, last_viewed: '2025-10-25', image: '/images/bases_base_011.jpg' },
    { id: 'base_012', name: 'Employee Directory', color: 'green', icon: 'users', starred: false, activity: 10, last_viewed: '2025-09-30', image: '/images/bases_base_012.jpg' },
    { id: 'base_013', name: 'Financial Projections', color: 'green', icon: 'dollar-sign', starred: false, activity: 45, last_viewed: '2025-10-19', image: '/images/bases_base_013.jpg' },
    { id: 'base_014', name: 'Client Feedback', color: 'yellow', icon: 'message-circle', starred: false, activity: 65, last_viewed: '2025-10-21', image: '/images/bases_base_014.jpg' },
    { id: 'base_015', name: 'Vendor List', color: 'purple', icon: 'truck', starred: false, activity: 15, last_viewed: '2025-10-05', image: '/images/bases_base_015.jpg' }
  ])

  const tables = ref([
    { id: 'tbl_001', base_id: 'base_001', name: 'Tasks' },
    { id: 'tbl_002', base_id: 'base_001', name: 'Sprints' },
    { id: 'tbl_003', base_id: 'base_001', name: 'Resources' },
    { id: 'tbl_004', base_id: 'base_002', name: 'Blog Posts' },
    { id: 'tbl_005', base_id: 'base_002', name: 'Social Media' },
    { id: 'tbl_006', base_id: 'base_003', name: 'Leads' },
    { id: 'tbl_007', base_id: 'base_003', name: 'Deals' }
  ])

  const records = ref([
    { id: 'rec_001', table_id: 'tbl_001', title: 'Design Homepage', status: 'In progress', due_date: '2025-11-01', priority: 'High', assigned_to: 'Alice', image: '/images/records_rec_001.jpg' },
    { id: 'rec_002', table_id: 'tbl_001', title: 'Implement Login', status: 'Done', due_date: '2025-10-20', priority: 'High', assigned_to: 'Bob', image: '/images/records_rec_002.jpg' },
    { id: 'rec_003', table_id: 'tbl_001', title: 'Fix Navigation Bug', status: 'To do', due_date: '2025-11-05', priority: 'Medium', assigned_to: 'Charlie', image: '/images/records_rec_003.jpg' },
    { id: 'rec_004', table_id: 'tbl_001', title: 'Write Documentation', status: 'In progress', due_date: '2025-11-10', priority: 'Low', assigned_to: 'Alice', image: '/images/records_rec_004.jpg' },
    { id: 'rec_005', table_id: 'tbl_001', title: 'Database Schema Design', status: 'Done', due_date: '2025-10-15', priority: 'High', assigned_to: 'Bob', image: '/images/records_rec_005.jpg' },
    { id: 'rec_006', table_id: 'tbl_001', title: 'User Testing Round 1', status: 'To do', due_date: '2025-11-15', priority: 'Medium', assigned_to: 'Diana', image: '/images/records_rec_006.jpg' },
    { id: 'rec_007', table_id: 'tbl_001', title: 'Optimize Images', status: 'In progress', due_date: '2025-11-02', priority: 'Low', assigned_to: 'Eve', image: '/images/records_rec_007.jpg' },
    { id: 'rec_008', table_id: 'tbl_001', title: 'Setup CI/CD Pipeline', status: 'Done', due_date: '2025-10-01', priority: 'High', assigned_to: 'Frank', image: '/images/records_rec_008.jpg' },
    { id: 'rec_009', table_id: 'tbl_001', title: 'Client Meeting Preparation', status: 'To do', due_date: '2025-11-08', priority: 'High', assigned_to: 'Alice', image: '/images/records_rec_009.jpg' },
    { id: 'rec_010', table_id: 'tbl_001', title: 'Update Dependencies', status: 'In progress', due_date: '2025-11-03', priority: 'Medium', assigned_to: 'Bob', image: '/images/records_rec_010.jpg' },
    { id: 'rec_011', table_id: 'tbl_001', title: 'Refactor Auth Module', status: 'To do', due_date: '2025-11-20', priority: 'High', assigned_to: 'Charlie', image: '/images/records_rec_011.jpg' },
    { id: 'rec_012', table_id: 'tbl_001', title: 'Create Marketing Assets', status: 'Done', due_date: '2025-10-25', priority: 'Medium', assigned_to: 'Diana', image: '/images/records_rec_012.jpg' },
    { id: 'rec_013', table_id: 'tbl_001', title: 'Review PRs', status: 'In progress', due_date: '2025-11-01', priority: 'High', assigned_to: 'Frank', image: '/images/records_rec_013.jpg' },
    { id: 'rec_014', table_id: 'tbl_001', title: 'Accessibility Audit', status: 'To do', due_date: '2025-11-25', priority: 'Medium', assigned_to: 'Eve', image: '/images/records_rec_014.jpg' },
    { id: 'rec_015', table_id: 'tbl_001', title: 'Analytics Integration', status: 'In progress', due_date: '2025-11-04', priority: 'High', assigned_to: 'Bob', image: '/images/records_rec_015.jpg' },
    { id: 'rec_016', table_id: 'tbl_002', title: 'Sprint 1 Planning', status: 'Done', due_date: '2025-09-01', priority: 'High', assigned_to: 'Alice', image: '/images/records_rec_016.jpg' },
    { id: 'rec_017', table_id: 'tbl_002', title: 'Sprint 1 Review', status: 'Done', due_date: '2025-09-14', priority: 'High', assigned_to: 'Alice', image: '/images/records_rec_017.jpg' },
    { id: 'rec_018', table_id: 'tbl_002', title: 'Sprint 2 Planning', status: 'Done', due_date: '2025-09-15', priority: 'High', assigned_to: 'Alice', image: '/images/records_rec_018.jpg' },
    { id: 'rec_019', table_id: 'tbl_002', title: 'Sprint 2 Retro', status: 'Done', due_date: '2025-09-29', priority: 'High', assigned_to: 'Alice', image: '/images/records_rec_019.jpg' },
    { id: 'rec_020', table_id: 'tbl_002', title: 'Sprint 3 Planning', status: 'In progress', due_date: '2025-10-30', priority: 'High', assigned_to: 'Alice', image: '/images/records_rec_020.jpg' }
  ])

  const automations = ref([
    { id: 'auto_001', base_id: 'base_001', name: 'Email on New Task', trigger: 'when-record-created', action: 'send-email', active: true, image: '/images/automations_auto_001.jpg' },
    { id: 'auto_002', base_id: 'base_001', name: 'Update Status on Due', trigger: 'at-scheduled-time', action: 'update-record', active: true, image: '/images/automations_auto_002.jpg' },
    { id: 'auto_003', base_id: 'base_001', name: 'Notify Manager', trigger: 'when-record-updated', action: 'send-email', active: false, image: '/images/automations_auto_003.jpg' },
    { id: 'auto_004', base_id: 'base_002', name: 'Publish Post', trigger: 'at-scheduled-time', action: 'update-record', active: true, image: '/images/automations_auto_004.jpg' },
    { id: 'auto_005', base_id: 'base_003', name: 'Welcome Email', trigger: 'when-record-created', action: 'send-email', active: true, image: '/images/automations_auto_005.jpg' },
    { id: 'auto_006', base_id: 'base_003', name: 'Follow Up Reminder', trigger: 'when-record-updated', action: 'create-record', active: true, image: '/images/automations_auto_006.jpg' },
    { id: 'auto_007', base_id: 'base_004', name: 'Confirm RSVP', trigger: 'when-record-created', action: 'send-email', active: false, image: '/images/automations_auto_007.jpg' },
    { id: 'auto_008', base_id: 'base_005', name: 'Log Feedback', trigger: 'when-record-created', action: 'create-record', active: true, image: '/images/automations_auto_008.jpg' },
    { id: 'auto_009', base_id: 'base_001', name: 'Archive Old Tasks', trigger: 'at-scheduled-time', action: 'update-record', active: false, image: '/images/automations_auto_009.jpg' },
    { id: 'auto_010', base_id: 'base_002', name: 'Tweet New Post', trigger: 'when-record-updated', action: 'create-record', active: true, image: '/images/automations_auto_010.jpg' }
  ])

  return {
    bases,
    tables,
    records,
    automations
  }
}, {
  persist: {
    storage: sessionStorage
  }
})