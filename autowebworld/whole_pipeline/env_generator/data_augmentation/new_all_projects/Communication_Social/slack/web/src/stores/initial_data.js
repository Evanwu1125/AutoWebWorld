import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // Users (need at least 15 for good listing)
  const users = ref([
    { id: 'user_01', name: 'Sarah Connor', title: 'Product Manager', status: 'available', avatar: '/images/User.jpg' },
    { id: 'user_02', name: 'John Smith', title: 'Senior Developer', status: 'busy', avatar: '/images/Developer.jpg' },
    { id: 'user_03', name: 'Emily Chen', title: 'UX Designer', status: 'away', avatar: '/images/UXDesigner.jpg' },
    { id: 'user_04', name: 'Michael Brown', title: 'DevOps Engineer', status: 'available', avatar: '/images/DevOps.jpg' },
    { id: 'user_05', name: 'Jessica Davis', title: 'QA Lead', status: 'in_meeting', avatar: '/images/JessicaDavis.jpg' },
    { id: 'user_06', name: 'David Wilson', title: 'Frontend Dev', status: 'available', avatar: '/images/DavidWilson.jpg' },
    { id: 'user_07', name: 'Amanda Martinez', title: 'Backend Dev', status: 'busy', avatar: '/images/Backend.jpg' },
    { id: 'user_08', name: 'James Taylor', title: 'Engineering Manager', status: 'away', avatar: '/images/JamesTaylor.jpg' },
    { id: 'user_09', name: 'Robert Anderson', title: 'Product Designer', status: 'available', avatar: '/images/ProductDesigner.jpg' },
    { id: 'user_10', name: 'Jennifer Thomas', title: 'Marketing Lead', status: 'available', avatar: '/images/JenniferThomas.jpg' },
    { id: 'user_11', name: 'William Jackson', title: 'Sales Manager', status: 'busy', avatar: '/images/WilliamJackson.jpg' },
    { id: 'user_12', name: 'Elizabeth White', title: 'HR Specialist', status: 'available', avatar: '/images/HR.jpg' },
    { id: 'user_13', name: 'Joseph Harris', title: 'Data Scientist', status: 'away', avatar: '/images/DataScientist.jpg' },
    { id: 'user_14', name: 'Thomas Martin', title: 'CEO', status: 'busy', avatar: '/images/CEO.jpg' },
    { id: 'user_15', name: 'Charles Thompson', title: 'CTO', status: 'available', avatar: '/images/CharlesThompson.jpg' }
  ])

  // Workspaces
  const workspaces = ref([
    { id: 'ws_01', name: 'Acme Corp', icon: '/images/Workspace.jpg' },
    { id: 'ws_02', name: 'Project Beta', icon: '/images/ProjectBeta.jpg' },
    { id: 'ws_03', name: 'Community Group', icon: '/images/Community.jpg' }
  ])

  // Channels (20 items for scrolling/filtering)
  // Fields for filters: unread (bool), activity (number 0-100), name (string)
  const channels = ref([
    { id: 'ch_01', name: 'general', unread: true, activity: 95, description: 'Company-wide announcements and general discussion', private: false },
    { id: 'ch_02', name: 'random', unread: false, activity: 80, description: 'Non-work banter and water cooler talk', private: false },
    { id: 'ch_03', name: 'engineering', unread: true, activity: 90, description: 'Engineering team discussions', private: false },
    { id: 'ch_04', name: 'design', unread: false, activity: 75, description: 'Design team sync', private: false },
    { id: 'ch_05', name: 'marketing', unread: true, activity: 60, description: 'Marketing campaigns and strategies', private: false },
    { id: 'ch_06', name: 'sales', unread: false, activity: 85, description: 'Sales leads and wins', private: false },
    { id: 'ch_07', name: 'product', unread: true, activity: 88, description: 'Product roadmap and feature planning', private: false },
    { id: 'ch_08', name: 'operations', unread: false, activity: 40, description: 'Office ops and logistics', private: true },
    { id: 'ch_09', name: 'finance', unread: false, activity: 30, description: 'Budgeting and expenses', private: true },
    { id: 'ch_10', name: 'hr-announcements', unread: true, activity: 20, description: 'HR updates', private: false },
    { id: 'ch_11', name: 'frontend-guild', unread: false, activity: 70, description: 'Frontend best practices', private: false },
    { id: 'ch_12', name: 'backend-guild', unread: false, activity: 65, description: 'Backend architecture', private: false },
    { id: 'ch_13', name: 'data-science', unread: true, activity: 50, description: 'ML models and data analysis', private: false },
    { id: 'ch_14', name: 'support', unread: true, activity: 98, description: 'Customer support tickets', private: false },
    { id: 'ch_15', name: 'feedback', unread: false, activity: 55, description: 'Internal feedback', private: false },
    { id: 'ch_16', name: 'incidents', unread: false, activity: 10, description: 'Incident response', private: true },
    { id: 'ch_17', name: 'releases', unread: true, activity: 45, description: 'Release notifications', private: false },
    { id: 'ch_18', name: 'social', unread: false, activity: 82, description: 'Social events planning', private: false },
    { id: 'ch_19', name: 'music', unread: false, activity: 60, description: 'Music sharing', private: false },
    { id: 'ch_20', name: 'gaming', unread: true, activity: 78, description: 'Video games discussion', private: false }
  ])

  // DMs (mapped to users, also need filters properties)
  // Fields for filters: unread (bool), activity (number 0-100)
  const dms = ref(users.value.map((user, index) => ({
    id: `dm_${user.id}`,
    user_id: user.id,
    user_name: user.name,
    user_avatar: user.avatar,
    user_status: user.status,
    unread: index % 3 === 0, // Every 3rd is unread
    activity: Math.floor(Math.random() * 100),
    last_message: 'Hey, do you have a minute?'
  })))

  // Messages (Generic pool to populate channel/dm details)
  const messages = ref([
    { id: 'msg_01', sender_id: 'user_01', text: 'Has anyone seen the latest specs?', time: '10:00 AM', reactions: ['👍'] },
    { id: 'msg_02', sender_id: 'user_02', text: 'Yes, I just pushed them to the repo.', time: '10:05 AM', reactions: ['🚀'] },
    { id: 'msg_03', sender_id: 'user_03', text: 'Great work team!', time: '10:10 AM', reactions: ['❤️'] },
    { id: 'msg_04', sender_id: 'user_04', text: 'Deploying to staging now.', time: '10:15 AM', reactions: [] },
    { id: 'msg_05', sender_id: 'user_01', text: 'Let me know when it is live.', time: '10:16 AM', reactions: ['👀'] }
  ])

  return {
    users,
    workspaces,
    channels,
    dms,
    messages
  }
}, {
  persist: {
    storage: sessionStorage
  }
})