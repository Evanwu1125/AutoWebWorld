import { defineStore } from 'pinia';

export const useDataStore = defineStore('data', {
  state: () => ({
    meetings: [
      { id: 'm1', topic: 'Q4 Roadmap Review', date: '2025-10-22', time: '10:00', duration: 60, host: 'Sarah Johnson', image: '/images/meetings_m1.jpg' },
      { id: 'm2', topic: 'Design Sprint Kickoff', date: '2025-10-23', time: '14:00', duration: 120, host: 'Mike Chen', image: '/images/meetings_m2.jpg' },
      { id: 'm3', topic: 'Weekly Standup', date: '2025-10-24', time: '09:30', duration: 15, host: 'Team Alpha', image: '/images/meetings_m3.jpg' },
      { id: 'm4', topic: 'Client Presentation: Acme Corp', date: '2025-10-25', time: '11:00', duration: 45, host: 'David Smith', image: '/images/meetings_m4.jpg' },
      { id: 'm5', topic: 'Engineering All-Hands', date: '2025-10-28', time: '16:00', duration: 60, host: 'CTO Office', image: '/images/meetings_m5.jpg' },
      { id: 'm6', topic: 'Product Sync', date: '2025-10-29', time: '13:00', duration: 30, host: 'Jane Doe', image: '/images/meetings_m6.jpg' },
      { id: 'm7', topic: 'Marketing Strategy', date: '2025-10-30', time: '10:30', duration: 90, host: 'Marketing Team', image: '/images/meetings_m7.jpg' },
      { id: 'm8', topic: 'HR Benefits Workshop', date: '2025-11-01', time: '15:00', duration: 60, host: 'HR Dept', image: '/images/meetings_m8.jpg' },
      { id: 'm9', topic: 'Project Retrospective', date: '2025-11-02', time: '11:00', duration: 45, host: 'Scrum Master', image: '/images/meetings_m9.jpg' },
      { id: 'm10', topic: 'Budget Planning 2026', date: '2025-11-05', time: '09:00', duration: 120, host: 'Finance Team', image: '/images/meetings_m10.jpg' },
      { id: 'm11', topic: 'Sales Pipeline Review', date: '2025-11-06', time: '14:30', duration: 60, host: 'VP Sales', image: '/images/meetings_m11.jpg' },
      { id: 'm12', topic: 'UX Research Share-out', date: '2025-11-07', time: '13:00', duration: 60, host: 'UX Team', image: '/images/meetings_m12.jpg' },
      { id: 'm13', topic: 'Security Training', date: '2025-11-10', time: '10:00', duration: 30, host: 'InfoSec', image: '/images/meetings_m13.jpg' },
      { id: 'm14', topic: 'Leadership Sync', date: '2025-11-12', time: '08:00', duration: 60, host: 'CEO', image: '/images/meetings_m14.jpg' },
      { id: 'm15', topic: 'Holiday Party Planning', date: '2025-11-15', time: '16:00', duration: 45, host: 'Culture Committee', image: '/images/meetings_m15.jpg' },
      { id: 'm16', topic: 'Customer Feedback Session', date: '2025-11-18', time: '11:00', duration: 60, host: 'Support Lead', image: '/images/meetings_m16.jpg' },
      { id: 'm17', topic: 'Tech Talk: Vue 3', date: '2025-11-20', time: '12:00', duration: 60, host: 'Frontend Guild', image: '/images/meetings_m17.jpg' },
      { id: 'm18', topic: 'Onboarding New Hires', date: '2025-11-22', time: '09:00', duration: 180, host: 'People Ops', image: '/images/meetings_m18.jpg' },
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