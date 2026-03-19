import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // --- MOCK DATA GENERATION ---
  
  // Topics
  const topics = ref([
    { id: 'topic_tech', name: 'Technology', followers: 125000, activity: 95, image: '/images/Technology.jpg' },
    { id: 'topic_science', name: 'Science', followers: 89000, activity: 88, image: '/images/Science.jpg' },
    { id: 'topic_movies', name: 'Movies', followers: 210000, activity: 92, image: '/images/Movies.jpg' },
    { id: 'topic_health', name: 'Health', followers: 75000, activity: 70, image: '/images/Health.jpg' },
    { id: 'topic_history', name: 'History', followers: 64000, activity: 65, image: '/images/History.jpg' },
    { id: 'topic_finance', name: 'Finance', followers: 98000, activity: 85, image: '/images/Finance.jpg' },
    { id: 'topic_travel', name: 'Travel', followers: 150000, activity: 90, image: '/images/Travel.jpg' },
    { id: 'topic_cooking', name: 'Cooking', followers: 110000, activity: 80, image: '/images/Cooking.jpg' },
    { id: 'topic_music', name: 'Music', followers: 180000, activity: 89, image: '/images/Music.jpg' },
    { id: 'topic_art', name: 'Art', followers: 55000, activity: 60, image: '/images/Art.jpg' },
    { id: 'topic_books', name: 'Books', followers: 70000, activity: 68, image: '/images/Books.jpg' },
    { id: 'topic_psychology', name: 'Psychology', followers: 95000, activity: 82, image: '/images/Psychology.jpg' },
    { id: 'topic_design', name: 'Design', followers: 45000, activity: 55, image: '/images/Design.jpg' },
    { id: 'topic_gaming', name: 'Gaming', followers: 250000, activity: 98, image: '/images/Gaming.jpg' },
    { id: 'topic_sports', name: 'Sports', followers: 200000, activity: 96, image: '/images/Sports.jpg' }
  ])

  // Questions
  const questions = ref([
    { id: 'q_001', topic_id: 'topic_tech', title: 'What is the future of AI?', details: 'With GPT-4 and beyond...', upvotes: 1200, time: 2, image: '/images/AI.jpg', answered: true },
    { id: 'q_002', topic_id: 'topic_science', title: 'Why is the sky blue?', details: 'Can someone explain Rayleigh scattering?', upvotes: 850, time: 24, image: '/images/BlueSky.jpg', answered: true },
    { id: 'q_003', topic_id: 'topic_movies', title: 'What is the best movie of 2024?', details: 'Looking for recommendations.', upvotes: 500, time: 5, image: '/images/Movies.jpg', answered: false },
    { id: 'q_004', topic_id: 'topic_cooking', title: 'How to make perfect pasta?', details: 'Al dente tips required.', upvotes: 300, time: 48, image: '/images/Pasta.jpg', answered: true },
    { id: 'q_005', topic_id: 'topic_travel', title: 'Best places to visit in Japan?', details: 'Planning a trip for cherry blossom season.', upvotes: 2100, time: 1, image: '/images/Japan.jpg', answered: true },
    { id: 'q_006', topic_id: 'topic_finance', title: 'Stocks vs Real Estate?', details: 'Which is a better long term investment?', upvotes: 900, time: 72, image: '/images/RealEstate.jpg', answered: true },
    { id: 'q_007', topic_id: 'topic_tech', title: 'Is VR dead?', details: 'Apple Vision Pro hype vs reality.', upvotes: 450, time: 10, image: '/images/VR.jpg', answered: false },
    { id: 'q_008', topic_id: 'topic_health', title: 'Benefits of intermittent fasting?', details: 'Does it really work for weight loss?', upvotes: 670, time: 3, image: '/images/IntermittentFasting.jpg', answered: true },
    { id: 'q_009', topic_id: 'topic_history', title: 'Who was the most influential Roman Emperor?', details: 'Augustus or Trajan?', upvotes: 1500, time: 120, image: '/images/RomanEmperor.jpg', answered: true },
    { id: 'q_010', topic_id: 'topic_gaming', title: 'PS5 Pro worth it?', details: 'Considering an upgrade.', upvotes: 2000, time: 0, image: '/images/Console.jpg', answered: false },
    { id: 'q_011', topic_id: 'topic_music', title: 'Why is vinyl making a comeback?', details: 'Is it sound quality or nostalgia?', upvotes: 340, time: 15, image: '/images/Vinyl.jpg', answered: true },
    { id: 'q_012', topic_id: 'topic_books', title: 'Must read sci-fi books?', details: 'Apart from Dune and Foundation.', upvotes: 890, time: 30, image: '/images/ScienceFiction.jpg', answered: true },
    { id: 'q_013', topic_id: 'topic_psychology', title: 'How to overcome procrastination?', details: 'Serious answers only.', upvotes: 5000, time: 6, image: '/images/procrastination.jpg', answered: true },
    { id: 'q_014', topic_id: 'topic_design', title: 'Minimalism vs Maximalism?', details: 'Current trends in 2025.', upvotes: 230, time: 9, image: '/images/Design.jpg', answered: false },
    { id: 'q_015', topic_id: 'topic_sports', title: 'Who will win the next World Cup?', details: 'Predictions?', upvotes: 1100, time: 4, image: '/images/WorldCup.jpg', answered: true },
    { id: 'q_016', topic_id: 'topic_tech', title: 'Best coding language for beginners?', details: 'Python or JS?', upvotes: 3000, time: 100, image: '/images/Coding.jpg', answered: true },
    { id: 'q_017', topic_id: 'topic_travel', title: 'Tips for solo travel?', details: 'Safety and social tips.', upvotes: 780, time: 8, image: '/images/Travel.jpg', answered: true }
  ])

  // Answers
  const answers = ref([
    { id: 'a_001', question_id: 'q_001', author: 'Jane Doe', body: 'AI is tools, not replacement...', image: '/images/User.jpg', upvotes: 50 },
    { id: 'a_002', question_id: 'q_001', author: 'John Smith', body: 'It will change everything.', image: '/images/AI.jpg', upvotes: 20 },
    { id: 'a_003', question_id: 'q_002', author: 'Albert E.', body: 'Scattering of light by particles.', image: '/images/LightScattering.jpg', upvotes: 100 },
    // More answers can be generated dynamically or added here if needed
  ])

  // Notifications
  const notifications = ref([
    { id: 'n_001', content: 'Someone upvoted your answer.', time: 1, read: false, image: '/images/Notifications.jpg' },
    { id: 'n_002', content: 'New answer on "Future of AI".', time: 5, read: false, image: '/images/Comment.jpg' },
    { id: 'n_003', content: 'Welcome to Quora App!', time: 100, read: true, image: '/images/Quora.jpg' },
    { id: 'n_004', content: 'Trend alert: Technology.', time: 2, read: false, image: '/images/Technology.jpg' },
    { id: 'n_005', content: 'John followed you.', time: 24, read: true, image: '/images/User.jpg' },
    { id: 'n_006', content: 'Question similar to yours.', time: 48, read: true, image: '/images/Question.jpg' },
    { id: 'n_007', content: 'Weekly digest available.', time: 10, read: false, image: '/images/Mail.jpg' },
    { id: 'n_008', content: 'Security alert: New login.', time: 1, read: false, image: '/images/Security.jpg' }
  ])
  
  // Bookmarks
  const bookmarks = ref([
    { id: 'b_001', question_id: 'q_005', added_at: 2 },
    { id: 'b_002', question_id: 'q_013', added_at: 10 },
    { id: 'b_003', question_id: 'q_001', added_at: 1 }
  ])

  return {
    topics,
    questions,
    answers,
    notifications,
    bookmarks
  }
}, {
  persist: {
    storage: sessionStorage
  }
})