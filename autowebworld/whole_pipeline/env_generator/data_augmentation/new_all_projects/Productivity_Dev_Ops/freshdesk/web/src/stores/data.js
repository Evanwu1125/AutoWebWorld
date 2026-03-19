import { defineStore } from 'pinia'

export const useDataStore = defineStore('data', {
  state: () => ({
    agents: [
      { id: 'agent1', name: 'Sarah Connor', avatar: '/images/Agent.jpg', role: 'Support Lead' },
      { id: 'agent2', name: 'John Doe', avatar: '/images/Agent.jpg', role: 'Agent' },
      { id: 'agent3', name: 'Emily Chen', avatar: '/images/Emily.jpg', role: 'Specialist' },
      { id: 'agent4', name: 'Michael Scott', avatar: '/images/Manager.jpg', role: 'Manager' }
    ],
    contacts: [
      { id: 'c1', name: 'Alice Anderson', email: 'alice@example.com', segment: 'VIP', avatar: '/images/Contact.jpg', phone: '555-0101' },
      { id: 'c2', name: 'Bob Brown', email: 'bob@example.com', segment: 'Standard', avatar: '/images/photo1765352349.jpg', phone: '555-0102' },
      { id: 'c3', name: 'Charlie Davis', email: 'charlie@tech.com', segment: 'VIP', avatar: '/images/Charlie.jpg', phone: '555-0103' },
      { id: 'c4', name: 'Diana Evans', email: 'diana@corp.org', segment: 'Standard', avatar: '/images/Contact.jpg', phone: '555-0104' },
      { id: 'c5', name: 'Evan Foster', email: 'evan@web.net', segment: 'Standard', avatar: '/images/Contact.jpg', phone: '555-0105' },
      { id: 'c6', name: 'Fiona Green', email: 'fiona@green.io', segment: 'VIP', avatar: '/images/FionaGreen.jpg', phone: '555-0106' },
      { id: 'c7', name: 'George Harris', email: 'george@mail.com', segment: 'Standard', avatar: '/images/GeorgeHarris.jpg', phone: '555-0107' },
      { id: 'c8', name: 'Hannah Ian', email: 'hannah@ian.co', segment: 'VIP', avatar: '/images/HannahIan.jpg', phone: '555-0108' },
      { id: 'c9', name: 'Ian Jones', email: 'ian@jones.inc', segment: 'Standard', avatar: '/images/Contact.jpg', phone: '555-0109' },
      { id: 'c10', name: 'Julia King', email: 'julia@king.ltd', segment: 'VIP', avatar: '/images/JuliaKing.jpg', phone: '555-0110' },
      { id: 'c11', name: 'Kevin Lee', email: 'kevin@lee.group', segment: 'Standard', avatar: '/images/KevinLee.jpg', phone: '555-0111' },
      { id: 'c12', name: 'Laura Miller', email: 'laura@miller.biz', segment: 'VIP', avatar: '/images/Contact.jpg', phone: '555-0112' },
      { id: 'c13', name: 'Mike Nelson', email: 'mike@nelson.info', segment: 'Standard', avatar: '/images/MikeNelson.jpg', phone: '555-0113' },
      { id: 'c14', name: 'Nina Oliver', email: 'nina@oliver.org', segment: 'VIP', avatar: '/images/Contact.jpg', phone: '555-0114' },
      { id: 'c15', name: 'Oscar Perry', email: 'oscar@perry.net', segment: 'Standard', avatar: '/images/OscarPerry.jpg', phone: '555-0115' }
    ],
    tickets: [
      { id: 't1', subject: 'Login issue on mobile app', description: 'Cannot login with valid credentials on iOS 15.', status: 'Open', priority: 'High', group: 'Support', requester_id: 'c1', assignee_id: 'agent1', created_at: '2023-10-25T10:00:00Z', image: '/images/tickets_t1.jpg' },
      { id: 't2', subject: 'Billing discrepancy', description: 'Charged twice for the subscription.', status: 'Pending', priority: 'High', group: 'Billing', requester_id: 'c2', assignee_id: 'agent2', created_at: '2023-10-24T14:30:00Z', image: '/images/tickets_t2.jpg' },
      { id: 't3', subject: 'Feature request: Dark mode', description: 'Please add dark mode to the dashboard.', status: 'Open', priority: 'Low', group: 'Support', requester_id: 'c3', assignee_id: null, created_at: '2023-10-26T09:15:00Z', image: '/images/tickets_t3.jpg' },
      { id: 't4', subject: 'Integration failure with Zapier', description: 'Webhook not triggering properly.', status: 'Open', priority: 'Medium', group: 'Support', requester_id: 'c4', assignee_id: 'agent3', created_at: '2023-10-23T11:45:00Z', image: '/images/tickets_t4.jpg' },
      { id: 't5', subject: 'Sales inquiry for Enterprise plan', description: 'Need pricing for 500+ seats.', status: 'Open', priority: 'Medium', group: 'Sales', requester_id: 'c5', assignee_id: 'agent4', created_at: '2023-10-27T08:00:00Z', image: '/images/tickets_t5.jpg' },
      { id: 't6', subject: 'Password reset email not received', description: 'User checked spam folder, nothing there.', status: 'Resolved', priority: 'High', group: 'Support', requester_id: 'c6', assignee_id: 'agent1', created_at: '2023-10-20T16:20:00Z', image: '/images/tickets_t6.jpg' },
      { id: 't7', subject: 'Update credit card info', description: 'Card expired, need link to update.', status: 'Pending', priority: 'Medium', group: 'Billing', requester_id: 'c7', assignee_id: 'agent2', created_at: '2023-10-25T13:10:00Z', image: '/images/tickets_t7.jpg' },
      { id: 't8', subject: 'API Rate limit exceeded', description: 'Getting 429 errors on creating contacts.', status: 'Open', priority: 'High', group: 'Support', requester_id: 'c8', assignee_id: 'agent3', created_at: '2023-10-27T10:05:00Z', image: '/images/tickets_t8.jpg' },
      { id: 't9', subject: 'Account deletion request', description: 'Please remove all my data.', status: 'Open', priority: 'Low', group: 'Support', requester_id: 'c9', assignee_id: null, created_at: '2023-10-22T09:45:00Z', image: '/images/tickets_t9.jpg' },
      { id: 't10', subject: 'Demo request for new features', description: 'Interested in the AI bot.', status: 'Open', priority: 'Medium', group: 'Sales', requester_id: 'c10', assignee_id: 'agent4', created_at: '2023-10-27T11:30:00Z', image: '/images/tickets_t10.jpg' },
      { id: 't11', subject: 'Slow loading times on reports', description: 'Reports page takes > 10s to load.', status: 'Open', priority: 'Medium', group: 'Support', requester_id: 'c11', assignee_id: 'agent1', created_at: '2023-10-26T15:00:00Z', image: '/images/tickets_t11.jpg' },
      { id: 't12', subject: 'Incorrect invoice amount', description: 'Tax calculation seems wrong.', status: 'Pending', priority: 'High', group: 'Billing', requester_id: 'c12', assignee_id: 'agent2', created_at: '2023-10-24T12:00:00Z', image: '/images/tickets_t12.jpg' },
      { id: 't13', subject: 'How to export contacts?', description: 'Cannot find the export button.', status: 'Resolved', priority: 'Low', group: 'Support', requester_id: 'c13', assignee_id: 'agent3', created_at: '2023-10-21T14:45:00Z', image: '/images/tickets_t13.jpg' },
      { id: 't14', subject: 'Partnership opportunity', description: 'Would like to discuss integration.', status: 'Open', priority: 'Medium', group: 'Sales', requester_id: 'c14', assignee_id: 'agent4', created_at: '2023-10-27T09:00:00Z', image: '/images/tickets_t14.jpg' },
      { id: 't15', subject: 'Email template broken', description: 'HTML layout issues in Outlook.', status: 'Open', priority: 'Medium', group: 'Support', requester_id: 'c15', assignee_id: 'agent1', created_at: '2023-10-25T11:20:00Z', image: '/images/tickets_t15.jpg' },
      { id: 't16', subject: 'SSO Configuration help', description: 'Okta integration failing.', status: 'Open', priority: 'High', group: 'Support', requester_id: 'c1', assignee_id: 'agent3', created_at: '2023-10-26T10:30:00Z', image: '/images/tickets_t16.jpg' },
      { id: 't17', subject: 'Upgrade account limits', description: 'Need more API calls.', status: 'Pending', priority: 'Medium', group: 'Sales', requester_id: 'c2', assignee_id: 'agent4', created_at: '2023-10-27T13:45:00Z', image: '/images/tickets_t17.jpg' },
      { id: 't18', subject: 'Mobile app crash on startup', description: 'Android 12, latest version.', status: 'Open', priority: 'High', group: 'Support', requester_id: 'c3', assignee_id: 'agent1', created_at: '2023-10-27T07:15:00Z', image: '/images/tickets_t18.jpg' },
      { id: 't19', subject: 'Refund request', description: 'Service not as described.', status: 'Open', priority: 'High', group: 'Billing', requester_id: 'c4', assignee_id: 'agent2', created_at: '2023-10-23T16:00:00Z', image: '/images/tickets_t19.jpg' },
      { id: 't20', subject: 'Typo in documentation', description: 'Page 5, section 2.', status: 'Resolved', priority: 'Low', group: 'Support', requester_id: 'c5', assignee_id: null, created_at: '2023-10-20T10:00:00Z', image: '/images/tickets_t20.jpg' }
    ]
  }),
  actions: {
    addTicket(ticket) {
      this.tickets.unshift(ticket)
    },
    addContact(contact) {
      this.contacts.unshift(contact)
    },
    getTicketById(id) {
      return this.tickets.find(t => t.id === id)
    },
    getContactById(id) {
      return this.contacts.find(c => c.id === id)
    }
  },
  persist: {
    storage: sessionStorage
  }
})