import { defineStore } from 'pinia'

export const useDataStore = defineStore('data', {
  state: () => ({
    notebooks: [
      { id: 'nb_1', name: 'Personal Journal', size: 120, shared: false, created_at: '2023-01-01', image: '/images/notebooks_nb_1.jpg' },
      { id: 'nb_2', name: 'Work Projects', size: 450, shared: true, created_at: '2023-01-15', image: '/images/notebooks_nb_2.jpg' },
      { id: 'nb_3', name: 'Travel Plans 2024', size: 85, shared: true, created_at: '2023-02-10', image: '/images/notebooks_nb_3.jpg' },
      { id: 'nb_4', name: 'Recipes', size: 210, shared: false, created_at: '2023-03-05', image: '/images/notebooks_nb_4.jpg' },
      { id: 'nb_5', name: 'Learning Vue.js', size: 300, shared: false, created_at: '2023-03-20', image: '/images/notebooks_nb_5.jpg' },
      { id: 'nb_6', name: 'Home Renovation', size: 150, shared: true, created_at: '2023-04-01', image: '/images/notebooks_nb_6.jpg' },
      { id: 'nb_7', name: 'Fitness Tracker', size: 60, shared: false, created_at: '2023-04-15', image: '/images/notebooks_nb_7.jpg' },
      { id: 'nb_8', name: 'Reading List', size: 40, shared: false, created_at: '2023-05-01', image: '/images/notebooks_nb_8.jpg' },
      { id: 'nb_9', name: 'Financial Planning', size: 90, shared: true, created_at: '2023-05-20', image: '/images/notebooks_nb_9.jpg' },
      { id: 'nb_10', name: 'Gardening', size: 110, shared: false, created_at: '2023-06-01', image: '/images/notebooks_nb_10.jpg' },
      { id: 'nb_11', name: 'Music Theory', size: 130, shared: false, created_at: '2023-06-15', image: '/images/notebooks_nb_11.jpg' },
      { id: 'nb_12', name: 'Photography Ideas', size: 200, shared: true, created_at: '2023-07-01', image: '/images/notebooks_nb_12.jpg' },
      { id: 'nb_13', name: 'Game Development', size: 350, shared: true, created_at: '2023-07-20', image: '/images/notebooks_nb_13.jpg' },
      { id: 'nb_14', name: 'Wedding Planning', size: 280, shared: true, created_at: '2023-08-01', image: '/images/notebooks_nb_14.jpg' },
      { id: 'nb_15', name: 'Sketchbook', size: 180, shared: false, created_at: '2023-08-15', image: '/images/notebooks_nb_15.jpg' }
    ],
    sections: [
      // Personal Journal
      { id: 'sec_1_1', notebook_id: 'nb_1', name: 'January', activity: 80, pinned: true, created_at: '2023-01-01', image: '/images/sections_sec_1_1.jpg' },
      { id: 'sec_1_2', notebook_id: 'nb_1', name: 'February', activity: 60, pinned: false, created_at: '2023-02-01', image: '/images/sections_sec_1_2.jpg' },
      { id: 'sec_1_3', notebook_id: 'nb_1', name: 'March', activity: 90, pinned: true, created_at: '2023-03-01', image: '/images/sections_sec_1_3.jpg' },
      // Work Projects
      { id: 'sec_2_1', notebook_id: 'nb_2', name: 'Q1 Goals', activity: 95, pinned: true, created_at: '2023-01-15', image: '/images/sections_sec_2_1.jpg' },
      { id: 'sec_2_2', notebook_id: 'nb_2', name: 'Meeting Notes', activity: 100, pinned: true, created_at: '2023-01-20', image: '/images/sections_sec_2_2.jpg' },
      { id: 'sec_2_3', notebook_id: 'nb_2', name: 'Project Alpha', activity: 85, pinned: false, created_at: '2023-02-01', image: '/images/sections_sec_2_3.jpg' },
      // Travel Plans
      { id: 'sec_3_1', notebook_id: 'nb_3', name: 'Itinerary', activity: 70, pinned: true, created_at: '2023-02-10', image: '/images/sections_sec_3_1.jpg' },
      { id: 'sec_3_2', notebook_id: 'nb_3', name: 'Accommodations', activity: 50, pinned: false, created_at: '2023-02-12', image: '/images/sections_sec_3_2.jpg' },
      // Recipes
      { id: 'sec_4_1', notebook_id: 'nb_4', name: 'Desserts', activity: 88, pinned: true, created_at: '2023-03-05', image: '/images/sections_sec_4_1.jpg' },
      { id: 'sec_4_2', notebook_id: 'nb_4', name: 'Italian', activity: 75, pinned: false, created_at: '2023-03-10', image: '/images/sections_sec_4_2.jpg' },
      // Learning Vue
      { id: 'sec_5_1', notebook_id: 'nb_5', name: 'Components', activity: 92, pinned: true, created_at: '2023-03-20', image: '/images/sections_sec_5_1.jpg' },
      { id: 'sec_5_2', notebook_id: 'nb_5', name: 'Pinia', activity: 80, pinned: false, created_at: '2023-04-01', image: '/images/sections_sec_5_2.jpg' },
      // Home Renovation
      { id: 'sec_6_1', notebook_id: 'nb_6', name: 'Kitchen', activity: 60, pinned: false, created_at: '2023-04-05', image: '/images/sections_sec_6_1.jpg' },
      // Fitness
      { id: 'sec_7_1', notebook_id: 'nb_7', name: 'Cardio Logs', activity: 70, pinned: false, created_at: '2023-04-15', image: '/images/sections_sec_7_1.jpg' },
      // Add more sections to reach 20+
      { id: 'sec_8_1', notebook_id: 'nb_8', name: 'Sci-Fi', activity: 40, pinned: false, created_at: '2023-05-01', image: '/images/sections_sec_8_1.jpg' },
      { id: 'sec_9_1', notebook_id: 'nb_9', name: 'Stocks', activity: 85, pinned: true, created_at: '2023-05-20', image: '/images/sections_sec_9_1.jpg' },
      { id: 'sec_10_1', notebook_id: 'nb_10', name: 'Vegetables', activity: 50, pinned: false, created_at: '2023-06-01', image: '/images/sections_sec_10_1.jpg' },
      { id: 'sec_11_1', notebook_id: 'nb_11', name: 'Scales', activity: 30, pinned: false, created_at: '2023-06-15', image: '/images/sections_sec_11_1.jpg' },
      { id: 'sec_12_1', notebook_id: 'nb_12', name: 'Portraits', activity: 65, pinned: false, created_at: '2023-07-01', image: '/images/sections_sec_12_1.jpg' },
      { id: 'sec_13_1', notebook_id: 'nb_13', name: 'Unity Tips', activity: 88, pinned: false, created_at: '2023-07-20', image: '/images/sections_sec_13_1.jpg' },
      { id: 'sec_14_1', notebook_id: 'nb_14', name: 'Guest List', activity: 90, pinned: true, created_at: '2023-08-01', image: '/images/sections_sec_14_1.jpg' },
      { id: 'sec_15_1', notebook_id: 'nb_15', name: 'Doodles', activity: 45, pinned: false, created_at: '2023-08-15', image: '/images/sections_sec_15_1.jpg' }
    ],
    pages: [
      // sec_1_1 January
      { id: 'pg_1', section_id: 'sec_1_1', title: 'New Year Resolutions', body: '1. Gym\n2. Read more', length: 50, favorite: true, created_at: '2023-01-01', image: '/images/pages_pg_1.jpg' },
      { id: 'pg_2', section_id: 'sec_1_1', title: 'Daily Log Jan 5', body: 'Went to the park.', length: 20, favorite: false, created_at: '2023-01-05', image: '/images/pages_pg_2.jpg' },
      // sec_2_1 Q1 Goals
      { id: 'pg_3', section_id: 'sec_2_1', title: 'Revenue Targets', body: 'Q1 Target: $500k', length: 100, favorite: true, created_at: '2023-01-15', image: '/images/pages_pg_3.jpg' },
      { id: 'pg_4', section_id: 'sec_2_1', title: 'Team Hiring', body: 'Need 2 devs.', length: 30, favorite: false, created_at: '2023-01-18', image: '/images/pages_pg_4.jpg' },
      // sec_4_1 Desserts
      { id: 'pg_5', section_id: 'sec_4_1', title: 'Cheesecake', body: 'Ingredients: Cheese, Sugar...', length: 200, favorite: true, created_at: '2023-03-05', image: '/images/pages_pg_5.jpg' },
      { id: 'pg_6', section_id: 'sec_4_1', title: 'Brownies', body: 'Best chocolate brownies ever.', length: 150, favorite: false, created_at: '2023-03-07', image: '/images/pages_pg_6.jpg' },
      // More pages
      { id: 'pg_7', section_id: 'sec_5_1', title: 'Setup Guide', body: 'npm install vue', length: 80, favorite: true, created_at: '2023-03-20', image: '/images/pages_pg_7.jpg' },
      { id: 'pg_8', section_id: 'sec_3_1', title: 'Flight Details', body: 'Flight UA999', length: 40, favorite: false, created_at: '2023-02-10', image: '/images/pages_pg_8.jpg' },
      { id: 'pg_9', section_id: 'sec_1_3', title: 'Spring Cleaning', body: 'Garage, Attic', length: 60, favorite: false, created_at: '2023-03-01', image: '/images/pages_pg_9.jpg' },
      { id: 'pg_10', section_id: 'sec_2_2', title: 'Weekly Sync', body: 'Discuss roadmap', length: 90, favorite: true, created_at: '2023-01-20', image: '/images/pages_pg_10.jpg' },
      { id: 'pg_11', section_id: 'sec_6_1', title: 'Cabinet Colors', body: 'Navy Blue or White?', length: 25, favorite: false, created_at: '2023-04-05', image: '/images/pages_pg_11.jpg' },
      { id: 'pg_12', section_id: 'sec_7_1', title: '5k Run Time', body: '25:30', length: 10, favorite: true, created_at: '2023-04-15', image: '/images/pages_pg_12.jpg' },
      { id: 'pg_13', section_id: 'sec_13_1', title: 'Physics Engine', body: 'Check friction settings', length: 120, favorite: false, created_at: '2023-07-20', image: '/images/pages_pg_13.jpg' },
      { id: 'pg_14', section_id: 'sec_14_1', title: 'RSVP List', body: 'John, Doe, Smith...', length: 300, favorite: true, created_at: '2023-08-01', image: '/images/pages_pg_14.jpg' },
      { id: 'pg_15', section_id: 'sec_8_1', title: 'Dune Review', body: 'Great book.', length: 50, favorite: false, created_at: '2023-05-01', image: '/images/pages_pg_15.jpg' }
    ],
    quick_notes: [
      { id: 'qn_1', title: 'Grocery List', body: 'Milk, Eggs, Bread', created_at: '2023-09-01', image: '/images/Milk.jpg' },
      { id: 'qn_2', title: 'Phone Number', body: '555-0199', created_at: '2023-09-02', image: '/images/Phone.jpg' },
      { id: 'qn_3', title: 'Idea for App', body: 'AI Note Taker', created_at: '2023-09-03', image: '/images/AI.jpg' },
      { id: 'qn_4', title: 'Meeting ID', body: '998-221-000', created_at: '2023-09-04', image: '/images/Zoom.jpg' },
      { id: 'qn_5', title: 'Book Rec', body: 'Atomic Habits', created_at: '2023-09-05', image: '/images/AtomicHabits.jpg' },
      { id: 'qn_6', title: 'Wifi Password', body: 'Secret123', created_at: '2023-09-06', image: '/images/Wifi.jpg' },
      { id: 'qn_7', title: 'Flight Confirmation', body: 'XYZ123', created_at: '2023-09-07', image: '/images/FlightTicket.jpg' },
      { id: 'qn_8', title: 'Hex Code', body: '#7719AA', created_at: '2023-09-08', image: '/images/Color.jpg' },
      { id: 'qn_9', title: 'Reminder', body: 'Call Mom', created_at: '2023-09-09', image: '/images/PhoneCall.jpg' },
      { id: 'qn_10', title: 'Quote', body: 'Stay hungry.', created_at: '2023-09-10', image: '/images/Quote.jpg' },
      { id: 'qn_11', title: 'Address', body: '123 Main St', created_at: '2023-09-11', image: '/images/Location.jpg' },
      { id: 'qn_12', title: 'Workout', body: 'Pushups 3x10', created_at: '2023-09-12', image: '/images/Workout.jpg' },
      { id: 'qn_13', title: 'Song', body: 'Bohemian Rhapsody', created_at: '2023-09-13', image: '/images/Music.jpg' },
      { id: 'qn_14', title: 'Size', body: 'W: 100, H: 200', created_at: '2023-09-14', image: '/images/Ruler.jpg' },
      { id: 'qn_15', title: 'Date', body: 'Anniversary Oct 5', created_at: '2023-09-15', image: '/images/Anniversary.jpg' }
    ]
  }),
  persist: {
    storage: sessionStorage
  }
})