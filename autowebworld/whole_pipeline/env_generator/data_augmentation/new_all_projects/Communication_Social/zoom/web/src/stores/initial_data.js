import { defineStore } from 'pinia';

export const useDataStore = defineStore('data', {
  state: () => ({
    meetings: [
      { id: 'm1', topic: 'Q4 Roadmap Review', date: '2025-10-22', time: '10:00', duration: 60, host: 'Sarah Johnson', image: '/images/Meetings.jpg' },
      { id: 'm2', topic: 'Design Sprint Kickoff', date: '2025-10-23', time: '14:00', duration: 120, host: 'Mike Chen', image: '/images/DesignSprint.jpg' },
      { id: 'm3', topic: 'Weekly Standup', date: '2025-10-24', time: '09:30', duration: 15, host: 'Team Alpha', image: '/images/WeeklyStandup.jpg' },
      { id: 'm4', topic: 'Client Presentation: Acme Corp', date: '2025-10-25', time: '11:00', duration: 45, host: 'David Smith', image: '/images/ClientPresentation.jpg' },
      { id: 'm5', topic: 'Engineering All-Hands', date: '2025-10-28', time: '16:00', duration: 60, host: 'CTO Office', image: '/images/Engineering.jpg' },
      { id: 'm6', topic: 'Product Sync', date: '2025-10-29', time: '13:00', duration: 30, host: 'Jane Doe', image: '/images/ProductSync.jpg' },
      { id: 'm7', topic: 'Marketing Strategy', date: '2025-10-30', time: '10:30', duration: 90, host: 'Marketing Team', image: '/images/Marketing.jpg' },
      { id: 'm8', topic: 'HR Benefits Workshop', date: '2025-11-01', time: '15:00', duration: 60, host: 'HR Dept', image: '/images/HRBenefits.jpg' },
      { id: 'm9', topic: 'Project Retrospective', date: '2025-11-02', time: '11:00', duration: 45, host: 'Scrum Master', image: '/images/Retrospective.jpg' },
      { id: 'm10', topic: 'Budget Planning 2026', date: '2025-11-05', time: '09:00', duration: 120, host: 'Finance Team', image: '/images/BudgetPlanning.jpg' },
      { id: 'm11', topic: 'Sales Pipeline Review', date: '2025-11-06', time: '14:30', duration: 60, host: 'VP Sales', image: '/images/Sales.jpg' },
      { id: 'm12', topic: 'UX Research Share-out', date: '2025-11-07', time: '13:00', duration: 60, host: 'UX Team', image: '/images/UXResearch.jpg' },
      { id: 'm13', topic: 'Security Training', date: '2025-11-10', time: '10:00', duration: 30, host: 'InfoSec', image: '/images/Security.jpg' },
      { id: 'm14', topic: 'Leadership Sync', date: '2025-11-12', time: '08:00', duration: 60, host: 'CEO', image: '/images/Leadership.jpg' },
      { id: 'm15', topic: 'Holiday Party Planning', date: '2025-11-15', time: '16:00', duration: 45, host: 'Culture Committee', image: '/images/HolidayParty.jpg' },
      { id: 'm16', topic: 'Customer Feedback Session', date: '2025-11-18', time: '11:00', duration: 60, host: 'Support Lead', image: '/images/CustomerFeedback.jpg' },
      { id: 'm17', topic: 'Tech Talk: Vue 3', date: '2025-11-20', time: '12:00', duration: 60, host: 'Frontend Guild', image: '/images/Vue3.jpg' },
      { id: 'm18', topic: 'Onboarding New Hires', date: '2025-11-22', time: '09:00', duration: 180, host: 'People Ops', image: '/images/Onboarding.jpg' },
    ],
    meeting_templates: [
      { id: 'none', name: 'No Template' },
      { id: 'recurring', name: 'Recurring Meeting' },
      { id: 'webinar', name: 'Webinar' }
    ]
  }),
  persist: {
    storage: sessionStorage,
  }
});