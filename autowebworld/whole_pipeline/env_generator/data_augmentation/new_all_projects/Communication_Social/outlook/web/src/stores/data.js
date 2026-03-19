import { defineStore } from 'pinia';

// Mock Data Store
export const useDataStore = defineStore('data', {
  state: () => ({
    emails: [
        { id: '101', sender: 'Alice Johnson', subject: 'Project Update: Q4 Goals', preview: 'Hi Team, here are the updated goals for Q4...', time: '10:30 AM', date: '2025-12-12T10:30:00', read: false, hasAttachment: true, size: 500, image: '/images/emails_101.jpg' },
        { id: '102', sender: 'Bob Smith', subject: 'Lunch Plans?', preview: 'Are you free for lunch today around 12:30?', time: '9:15 AM', date: '2025-12-12T09:15:00', read: true, hasAttachment: false, size: 10, image: '/images/emails_102.jpg' },
        { id: '103', sender: 'HR Department', subject: 'Annual Benefits Enrollment', preview: 'It is time to select your benefits for next year...', time: 'Yesterday', date: '2025-12-11T14:00:00', read: false, hasAttachment: true, size: 1200, image: '/images/emails_103.jpg' },
        { id: '104', sender: 'IT Support', subject: 'Scheduled Maintenance', preview: 'System will be down for maintenance this weekend...', time: 'Yesterday', date: '2025-12-11T11:00:00', read: true, hasAttachment: false, size: 20, image: '/images/emails_104.jpg' },
        { id: '105', sender: 'Sarah Connor', subject: 'Meeting Notes', preview: 'Attached are the notes from our discussion.', time: 'Mon', date: '2025-12-08T16:45:00', read: true, hasAttachment: true, size: 350, image: '/images/emails_105.jpg' },
        { id: '106', sender: 'Marketing Team', subject: 'New Campaign Launch', preview: 'The new campaign goes live next week!', time: 'Mon', date: '2025-12-08T09:00:00', read: true, hasAttachment: false, size: 45, image: '/images/emails_106.jpg' },
        { id: '107', sender: 'Netflix', subject: 'New Arrival: Tech Thriller', preview: 'Check out the latest movies added this week.', time: 'Sun', date: '2025-12-07T20:00:00', read: true, hasAttachment: false, size: 15, image: '/images/emails_107.jpg' },
        { id: '108', sender: 'Amazon', subject: 'Your package has been delivered', preview: 'Your order #12345 was left at front door.', time: 'Sat', date: '2025-12-06T15:30:00', read: false, hasAttachment: false, size: 12, image: '/images/emails_108.jpg' },
        { id: '109', sender: 'LinkedIn', subject: 'You appeared in 5 searches', preview: 'See who is looking at your profile.', time: 'Fri', date: '2025-12-05T08:00:00', read: true, hasAttachment: false, size: 18, image: '/images/emails_109.jpg' },
        { id: '110', sender: 'Mom', subject: 'Family Dinner', preview: 'Are you coming over this Sunday?', time: 'Thu', date: '2025-12-04T19:00:00', read: true, hasAttachment: false, size: 5, image: '/images/emails_110.jpg' },
        { id: '111', sender: 'Gym', subject: 'Membership Renewal', preview: 'Your membership expires in 30 days.', time: 'Wed', date: '2025-12-03T10:00:00', read: true, hasAttachment: true, size: 150, image: '/images/emails_111.jpg' },
        { id: '112', sender: 'Newsletter', subject: 'Weekly Tech Digest', preview: 'Top stories in tech this week...', time: 'Tue', date: '2025-12-02T07:30:00', read: false, hasAttachment: false, size: 60, image: '/images/emails_112.jpg' },
        { id: '113', sender: 'Boss', subject: 'Urgent: Client Issue', preview: 'Please call me as soon as you see this.', time: 'Mon', date: '2025-12-01T21:00:00', read: true, hasAttachment: false, size: 8, image: '/images/emails_113.jpg' },
        { id: '114', sender: 'Travel Agency', subject: 'Your Flight Itinerary', preview: 'Flight UA123 to New York is confirmed.', time: '11/30', date: '2025-11-30T14:20:00', read: true, hasAttachment: true, size: 800, image: '/images/emails_114.jpg' },
        { id: '115', sender: 'Bank', subject: 'Statement Available', preview: 'Your November statement is ready to view.', time: '11/29', date: '2025-11-29T06:00:00', read: true, hasAttachment: true, size: 250, image: '/images/emails_115.jpg' }
    ],
    sentEmails: [
        { id: '201', recipient: 'Alice Johnson', subject: 'Re: Project Update', preview: 'Thanks for the update, looks good.', time: '11:00 AM', date: '2025-12-12T11:00:00', hasAttachment: false },
        { id: '202', recipient: 'Bob Smith', subject: 'Re: Lunch Plans?', preview: 'Sure, 12:30 works for me.', time: '9:20 AM', date: '2025-12-12T09:20:00', hasAttachment: false },
        { id: '203', recipient: 'Team', subject: 'Weekly Report', preview: 'Attached is the weekly status report.', time: 'Yesterday', date: '2025-12-11T17:00:00', hasAttachment: true },
        // ... more sent items
    ],
    draftEmails: [
        { id: '301', subject: 'Budget Proposal', preview: 'Dear Finance Team, Attached is the draft budget...', time: '12:00 PM', date: '2025-12-12T12:00:00' },
        { id: '302', subject: '(No subject)', preview: 'Hey, are we still on for...', time: 'Yesterday', date: '2025-12-11T10:00:00' }
    ],
    trashEmails: [
        { id: '401', sender: 'Spam Bot', subject: 'You won a prize!', preview: 'Click here to claim...', time: '10:00 AM', date: '2025-12-12T10:00:00' },
        { id: '402', sender: 'Old Service', subject: 'Goodbye', preview: 'Service discontinued.', time: 'Yesterday', date: '2025-12-11T09:00:00' }
    ],
    contacts: [
        { id: 'c1', name: 'Alice Johnson', email: 'alice.johnson@example.com', jobTitle: 'Project Manager', company: 'Tech Corp', department: 'Engineering', location: 'New York, NY', phone: '+1 555-0101' },
        { id: 'c2', name: 'Bob Smith', email: 'bob.smith@example.com', jobTitle: 'Developer', company: 'Tech Corp', department: 'Engineering', location: 'San Francisco, CA', phone: '+1 555-0102' },
        { id: 'c3', name: 'Charlie Davis', email: 'charlie.davis@example.com', jobTitle: 'Designer', company: 'Design Studio', department: 'Creative', location: 'Austin, TX', phone: '+1 555-0103' },
        { id: 'c4', name: 'Diana Evans', email: 'diana.evans@example.com', jobTitle: 'VP of Sales', company: 'Global Sales Inc', department: 'Sales', location: 'Chicago, IL', phone: '+1 555-0104' },
        { id: 'c5', name: 'Ethan Hunt', email: 'ethan.hunt@example.com', jobTitle: 'Security Analyst', company: 'SecureNet', department: 'IT', location: 'Washington, DC', phone: '+1 555-0105' },
        { id: 'c6', name: 'Fiona Gallagher', email: 'fiona.g@example.com', jobTitle: 'HR Manager', company: 'People First', department: 'HR', location: 'Boston, MA', phone: '+1 555-0106' },
        { id: 'c7', name: 'George Martin', email: 'george.m@example.com', jobTitle: 'Writer', company: 'Freelance', department: 'Content', location: 'Santa Fe, NM', phone: '+1 555-0107' },
        { id: 'c8', name: 'Hannah Lee', email: 'hannah.lee@example.com', jobTitle: 'Data Scientist', company: 'DataViz', department: 'Analytics', location: 'Seattle, WA', phone: '+1 555-0108' },
        { id: 'c9', name: 'Ian Wright', email: 'ian.wright@example.com', jobTitle: 'Architect', company: 'BuildIt', department: 'Operations', location: 'Denver, CO', phone: '+1 555-0109' },
        { id: 'c10', name: 'Julia Roberts', email: 'julia.r@example.com', jobTitle: 'Director', company: 'Movie Magic', department: 'Production', location: 'Los Angeles, CA', phone: '+1 555-0110' }
    ]
  }),
  persist: {
    storage: sessionStorage,
  },
});