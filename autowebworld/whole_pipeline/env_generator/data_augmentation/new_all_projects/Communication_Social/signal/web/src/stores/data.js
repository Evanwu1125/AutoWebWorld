import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  
  // 1. Contacts (20 items)
  const contacts = ref([
    { id: 'c1', name: 'Alice Smith', phone: '+15550101', avatar: '/images/Contacts.jpg', is_blocked: false, is_favorite: true },
    { id: 'c2', name: 'Bob Johnson', phone: '+15550102', avatar: '/images/BobJohnson.jpg', is_blocked: false, is_favorite: false },
    { id: 'c3', name: 'Charlie Brown', phone: '+15550103', avatar: '/images/User.jpg', is_blocked: true, is_favorite: false },
    { id: 'c4', name: 'Diana Prince', phone: '+15550104', avatar: '/images/DianaPrince.jpg', is_blocked: false, is_favorite: true },
    { id: 'c5', name: 'Evan Wright', phone: '+15550105', avatar: '/images/User.jpg', is_blocked: false, is_favorite: false },
    { id: 'c6', name: 'Fiona Gallagher', phone: '+15550106', avatar: '/images/FionaGallagher.jpg', is_blocked: false, is_favorite: true },
    { id: 'c7', name: 'George Martin', phone: '+15550107', avatar: '/images/GeorgeMartin.jpg', is_blocked: true, is_favorite: false },
    { id: 'c8', name: 'Hannah Abbott', phone: '+15550108', avatar: '/images/User.jpg', is_blocked: false, is_favorite: false },
    { id: 'c9', name: 'Ian Malcolm', phone: '+15550109', avatar: '/images/IanMalcolm.jpg', is_blocked: false, is_favorite: false },
    { id: 'c10', name: 'Julia Roberts', phone: '+15550110', avatar: '/images/JuliaRoberts.jpg', is_blocked: false, is_favorite: true },
    { id: 'c11', name: 'Kevin Bacon', phone: '+15550111', avatar: '/images/KevinBacon.jpg', is_blocked: false, is_favorite: false },
    { id: 'c12', name: 'Laura Croft', phone: '+15550112', avatar: '/images/LauraCroft.jpg', is_blocked: false, is_favorite: false },
    { id: 'c13', name: 'Mike Ross', phone: '+15550113', avatar: '/images/MikeRoss.jpg', is_blocked: false, is_favorite: false },
    { id: 'c14', name: 'Nina Simone', phone: '+15550114', avatar: '/images/NinaSimone.jpg', is_blocked: false, is_favorite: true },
    { id: 'c15', name: 'Oscar Wilde', phone: '+15550115', avatar: '/images/OscarWilde.jpg', is_blocked: true, is_favorite: false },
    { id: 'c16', name: 'Paul Atreides', phone: '+15550116', avatar: '/images/PaulAtreides.jpg', is_blocked: false, is_favorite: false },
    { id: 'c17', name: 'Quincy Jones', phone: '+15550117', avatar: '/images/QuincyJones.jpg', is_blocked: false, is_favorite: false },
    { id: 'c18', name: 'Rachel Green', phone: '+15550118', avatar: '/images/RachelGreen.jpg', is_blocked: false, is_favorite: true },
    { id: 'c19', name: 'Steve Rogers', phone: '+15550119', avatar: '/images/SteveRogers.jpg', is_blocked: false, is_favorite: false },
    { id: 'c20', name: 'Tony Stark', phone: '+15550120', avatar: '/images/TonyStark.jpg', is_blocked: false, is_favorite: true },
  ])

  // 2. Chats (Active conversations - mapped to contacts)
  const chats = ref([
    { id: 'chat1', contact_id: 'c1', last_message: 'See you tomorrow!', timestamp: '10:30 AM', unread: 2, pinned: true, muted: false },
    { id: 'chat2', contact_id: 'c2', last_message: 'Can you send the file?', timestamp: 'Yesterday', unread: 0, pinned: true, muted: true },
    { id: 'chat4', contact_id: 'c4', last_message: 'Thanks!', timestamp: 'Yesterday', unread: 1, pinned: false, muted: false },
    { id: 'chat5', contact_id: 'c5', last_message: 'Meeting at 5?', timestamp: 'Mon', unread: 0, pinned: false, muted: false },
    { id: 'chat6', contact_id: 'c6', last_message: 'Call me later.', timestamp: 'Sun', unread: 5, pinned: false, muted: false },
    { id: 'chat8', contact_id: 'c8', last_message: 'Did you get the email?', timestamp: 'Oct 20', unread: 0, pinned: false, muted: true },
    { id: 'chat10', contact_id: 'c10', last_message: 'Happy Birthday!', timestamp: 'Oct 15', unread: 0, pinned: false, muted: false },
    { id: 'chat11', contact_id: 'c11', last_message: 'Ok', timestamp: 'Oct 12', unread: 0, pinned: false, muted: false },
    { id: 'chat12', contact_id: 'c12', last_message: 'Where are you?', timestamp: 'Oct 10', unread: 1, pinned: false, muted: false },
    { id: 'chat14', contact_id: 'c14', last_message: 'Sound good.', timestamp: 'Oct 05', unread: 0, pinned: false, muted: false },
    { id: 'chat16', contact_id: 'c16', last_message: 'On my way', timestamp: 'Sep 28', unread: 0, pinned: false, muted: false },
    { id: 'chat17', contact_id: 'c17', last_message: 'Sure thing', timestamp: 'Sep 25', unread: 0, pinned: false, muted: false },
    { id: 'chat18', contact_id: 'c18', last_message: 'Lunch?', timestamp: 'Sep 20', unread: 0, pinned: false, muted: false },
    { id: 'chat19', contact_id: 'c19', last_message: 'Check this out', timestamp: 'Sep 18', unread: 0, pinned: false, muted: false },
    { id: 'chat20', contact_id: 'c20', last_message: 'Done.', timestamp: 'Sep 15', unread: 0, pinned: false, muted: false },
  ])

  // 3. Groups (15 items)
  const groups = ref([
    { id: 'g1', name: 'Family Group', member_count: 5, avatar: '/images/Family.jpg', last_message: 'Mom: Dinner at 6', timestamp: '11:00 AM', muted: false },
    { id: 'g2', name: 'Work Team', member_count: 12, avatar: '/images/WorkTeam.jpg', last_message: 'Boss: Project update?', timestamp: '9:00 AM', muted: true },
    { id: 'g3', name: 'Weekend Trip', member_count: 4, avatar: '/images/WeekendTrip.jpg', last_message: 'Alice: Packed already!', timestamp: 'Yesterday', muted: false },
    { id: 'g4', name: 'Book Club', member_count: 8, avatar: '/images/BookClub.jpg', last_message: 'Sarah: Next book is...', timestamp: 'Tue', muted: false },
    { id: 'g5', name: 'Gym Buddies', member_count: 3, avatar: '/images/Gym.jpg', last_message: 'Mike: Leg day!', timestamp: 'Mon', muted: true },
    { id: 'g6', name: 'College Friends', member_count: 20, avatar: '/images/Reunion.jpg', last_message: 'John: Reunion details', timestamp: 'Sun', muted: false },
    { id: 'g7', name: 'Gaming Squad', member_count: 6, avatar: '/images/Gaming.jpg', last_message: 'Online now', timestamp: 'Oct 22', muted: false },
    { id: 'g8', name: 'Project Alpha', member_count: 5, avatar: '/images/ProjectAlpha.jpg', last_message: 'Deadline extended', timestamp: 'Oct 20', muted: false },
    { id: 'g9', name: 'Neighbors', member_count: 45, avatar: '/images/Neighbors.jpg', last_message: 'Package at front desk', timestamp: 'Oct 18', muted: true },
    { id: 'g10', name: 'Crypto Talk', member_count: 150, avatar: '/images/Crypto.jpg', last_message: 'To the moon!', timestamp: 'Oct 15', muted: true },
    { id: 'g11', name: 'Developers', member_count: 30, avatar: '/images/Developers.jpg', last_message: 'New release out', timestamp: 'Oct 12', muted: false },
    { id: 'g12', name: 'Design Team', member_count: 8, avatar: '/images/Design.jpg', last_message: 'Review mockups', timestamp: 'Oct 10', muted: false },
    { id: 'g13', name: 'Marketing', member_count: 10, avatar: '/images/Marketing.jpg', last_message: 'Campaign results', timestamp: 'Oct 08', muted: false },
    { id: 'g14', name: 'Sales', member_count: 15, avatar: '/images/Sales.jpg', last_message: 'Q4 targets', timestamp: 'Oct 05', muted: false },
    { id: 'g15', name: 'HR Updates', member_count: 50, avatar: '/images/HR.jpg', last_message: 'Holiday schedule', timestamp: 'Oct 01', muted: true },
  ])

  // 4. Calls (20 items)
  const calls = ref([
    { id: 'call1', contact_id: 'c1', type: 'incoming', status: 'missed', timestamp: '10:45 AM', duration: '0s' },
    { id: 'call2', contact_id: 'c2', type: 'outgoing', status: 'connected', timestamp: 'Yesterday', duration: '5m 23s' },
    { id: 'call3', contact_id: 'c3', type: 'incoming', status: 'connected', timestamp: 'Yesterday', duration: '2m 10s' },
    { id: 'call4', contact_id: 'c4', type: 'video', status: 'missed', timestamp: 'Mon', duration: '0s' },
    { id: 'call5', contact_id: 'c1', type: 'outgoing', status: 'connected', timestamp: 'Sun', duration: '15m 00s' },
    { id: 'call6', contact_id: 'c5', type: 'incoming', status: 'missed', timestamp: 'Sat', duration: '0s' },
    { id: 'call7', contact_id: 'c6', type: 'video', status: 'connected', timestamp: 'Fri', duration: '30m 12s' },
    { id: 'call8', contact_id: 'c2', type: 'outgoing', status: 'missed', timestamp: 'Thu', duration: '0s' },
    { id: 'call9', contact_id: 'c7', type: 'incoming', status: 'connected', timestamp: 'Wed', duration: '1m 05s' },
    { id: 'call10', contact_id: 'c8', type: 'outgoing', status: 'connected', timestamp: 'Tue', duration: '4m 30s' },
    { id: 'call11', contact_id: 'c9', type: 'incoming', status: 'missed', timestamp: 'Oct 20', duration: '0s' },
    { id: 'call12', contact_id: 'c10', type: 'video', status: 'connected', timestamp: 'Oct 18', duration: '10m 15s' },
    { id: 'call13', contact_id: 'c1', type: 'outgoing', status: 'missed', timestamp: 'Oct 15', duration: '0s' },
    { id: 'call14', contact_id: 'c11', type: 'incoming', status: 'connected', timestamp: 'Oct 12', duration: '8m 45s' },
    { id: 'call15', contact_id: 'c12', type: 'outgoing', status: 'connected', timestamp: 'Oct 10', duration: '2m 22s' },
    { id: 'call16', contact_id: 'c13', type: 'incoming', status: 'missed', timestamp: 'Oct 08', duration: '0s' },
    { id: 'call17', contact_id: 'c14', type: 'video', status: 'connected', timestamp: 'Oct 05', duration: '45m 00s' },
    { id: 'call18', contact_id: 'c15', type: 'outgoing', status: 'connected', timestamp: 'Oct 02', duration: '3m 10s' },
    { id: 'call19', contact_id: 'c16', type: 'incoming', status: 'missed', timestamp: 'Sep 30', duration: '0s' },
    { id: 'call20', contact_id: 'c17', type: 'outgoing', status: 'connected', timestamp: 'Sep 28', duration: '6m 50s' },
  ])

  // 5. Messages (Mock messages for chat thread)
  const messages = ref({
    'chat1': [
      { id: 'm1', text: 'Hey Alice!', sender: 'me', time: '10:00 AM' },
      { id: 'm2', text: 'Hi! How are you?', sender: 'them', time: '10:05 AM' },
      { id: 'm3', text: 'Good, you?', sender: 'me', time: '10:06 AM' },
      { id: 'm4', text: 'Great! See you tomorrow!', sender: 'them', time: '10:30 AM' },
    ],
    'chat2': [
      { id: 'm5', text: 'File sent.', sender: 'me', time: 'Yesterday' },
      { id: 'm6', text: 'Can you send the file?', sender: 'them', time: 'Yesterday' },
    ]
  })

  return {
    contacts,
    chats,
    groups,
    calls,
    messages
  }
}, {
  persist: {
    storage: sessionStorage
  }
})