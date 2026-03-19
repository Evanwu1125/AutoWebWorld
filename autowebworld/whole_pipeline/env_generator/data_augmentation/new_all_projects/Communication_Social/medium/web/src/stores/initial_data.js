import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // --- Mock Data Generation ---

  // Users
  const users = ref([
    { id: 'user_1', name: 'Alex Chen', bio: 'Software Engineer & Writer', avatar: '/images/User.jpg', location: 'San Francisco, CA', is_member: true },
    { id: 'user_2', name: 'Sarah Jones', bio: 'Digital Nomad | Travel', avatar: '/images/Travel.jpg', location: 'New York, NY', is_member: false },
    { id: 'user_3', name: 'David Miller', bio: 'Tech enthusiast', avatar: '/images/Tech.jpg', location: 'London, UK', is_member: true },
    { id: 'user_4', name: 'Emily Davis', bio: 'UX Designer', avatar: '/images/UXDesigner.jpg', location: 'Berlin, DE', is_member: false },
    { id: 'user_5', name: 'Michael Wilson', bio: 'Data Scientist', avatar: '/images/DataScientist.jpg', location: 'Toronto, CA', is_member: true },
    { id: 'user_6', name: 'Jessica Taylor', bio: 'Product Manager', avatar: '/images/ProductManager.jpg', location: 'Austin, TX', is_member: false },
    { id: 'user_7', name: 'Christopher Anderson', bio: 'Startup Founder', avatar: '/images/Startup.jpg', location: 'Seattle, WA', is_member: true },
    { id: 'user_8', name: 'Amanda Thomas', bio: 'Content Creator', avatar: '/images/ContentCreator.jpg', location: 'Los Angeles, CA', is_member: false },
  ])

  // Publications
  const publications = ref([
    { id: 'pub_1', name: 'Towards Data Science', description: 'Sharing concepts, ideas, and codes.', icon: '/images/DataScience.jpg', member_count: 500000, size_category: 'large', is_featured: true },
    { id: 'pub_2', name: 'The Startup', description: 'Get smarter at building your thing.', icon: '/images/Startup.jpg', member_count: 700000, size_category: 'large', is_featured: true },
    { id: 'pub_3', name: 'UX Collective', description: 'Curated stories on UX, Visual & Product Design.', icon: '/images/UXDesign.jpg', member_count: 400000, size_category: 'medium', is_featured: true },
    { id: 'pub_4', name: 'Better Humans', description: 'Explore your potential.', icon: '/images/BetterHumans.jpg', member_count: 300000, size_category: 'medium', is_featured: false },
    { id: 'pub_5', name: 'Personal Growth', description: 'Sharing our ideas and experiences.', icon: '/images/PersonalGrowth.jpg', member_count: 200000, size_category: 'medium', is_featured: false },
    { id: 'pub_6', name: 'The/Code', description: 'A publication for developers.', icon: '/images/Code.jpg', member_count: 100000, size_category: 'small', is_featured: false },
    { id: 'pub_7', name: 'Writing Cooperative', description: 'Helping each other write better.', icon: '/images/Writing.jpg', member_count: 150000, size_category: 'small', is_featured: true },
    { id: 'pub_8', name: 'JavaScript in Plain English', description: 'New JavaScript features.', icon: '/images/JavaScript.jpg', member_count: 80000, size_category: 'small', is_featured: false },
    { id: 'pub_9', name: 'Level Up Coding', description: 'Coding tutorials and news.', icon: '/images/Coding.jpg', member_count: 120000, size_category: 'medium', is_featured: true },
    { id: 'pub_10', name: 'History of Yesterday', description: 'History tailored for you.', icon: '/images/History.jpg', member_count: 50000, size_category: 'small', is_featured: false },
  ])

  // Posts
  const posts = ref([
    { id: 'post_1', title: 'The Future of AI', subtitle: 'What to expect in the next decade', author_id: 'user_1', content: 'Artificial Intelligence is evolving rapidly...', image: '/images/AI.jpg', claps: 1200, responses: 45, published_date: '2023-10-01', tag: 'technology', length_minutes: 5 },
    { id: 'post_2', title: 'Minimalist Living', subtitle: 'How to declutter your life', author_id: 'user_2', content: 'Minimalism is not just about owning less...', image: '/images/Minimalism.jpg', claps: 850, responses: 30, published_date: '2023-10-02', tag: 'culture', length_minutes: 3 },
    { id: 'post_3', title: 'Mastering Vue 3', subtitle: 'A comprehensive guide', author_id: 'user_3', content: 'Vue 3 introduces the Composition API...', image: '/images/Vue.jpg', claps: 2000, responses: 100, published_date: '2023-09-28', tag: 'technology', length_minutes: 10 },
    { id: 'post_4', title: 'Remote Work Tips', subtitle: 'Staying productive at home', author_id: 'user_4', content: 'Working from home can be challenging...', image: '/images/RemoteWork.jpg', claps: 500, responses: 20, published_date: '2023-10-05', tag: 'productivity', length_minutes: 4 },
    { id: 'post_5', title: 'Understanding UX', subtitle: 'User experience basics', author_id: 'user_4', content: 'UX is about how a user feels...', image: '/images/UXDesign.jpg', claps: 1500, responses: 60, published_date: '2023-10-03', tag: 'technology', length_minutes: 6 },
    { id: 'post_6', title: 'Healthy Eating', subtitle: 'Simple recipes for busy people', author_id: 'user_5', content: 'Eating healthy doesn\'t have to be hard...', image: '/images/HealthyEating.jpg', claps: 900, responses: 35, published_date: '2023-10-04', tag: 'culture', length_minutes: 7 },
    { id: 'post_7', title: 'Investing 101', subtitle: 'Start building wealth today', author_id: 'user_7', content: 'The best time to plant a tree was 20 years ago...', image: '/images/Investing.jpg', claps: 3000, responses: 150, published_date: '2023-09-25', tag: 'productivity', length_minutes: 8 },
    { id: 'post_8', title: 'Travel on a Budget', subtitle: 'See the world for less', author_id: 'user_2', content: 'You don\'t need to be rich to travel...', image: '/images/Travel.jpg', claps: 1100, responses: 55, published_date: '2023-10-06', tag: 'culture', length_minutes: 5 },
    { id: 'post_9', title: 'The Art of Writing', subtitle: 'Improve your storytelling', author_id: 'user_1', content: 'Writing is a craft that takes practice...', image: '/images/Writing.jpg', claps: 1800, responses: 80, published_date: '2023-09-30', tag: 'culture', length_minutes: 6 },
    { id: 'post_10', title: 'Javascript ES2024', subtitle: 'New features explained', author_id: 'user_3', content: 'Let\'s look at the new features in JS...', image: '/images/Javascript.jpg', claps: 2500, responses: 120, published_date: '2023-10-07', tag: 'technology', length_minutes: 9 },
    { id: 'post_11', title: 'Meditation Guide', subtitle: 'Find your inner peace', author_id: 'user_6', content: 'Meditation helps reduce stress...', image: '/images/Meditation.jpg', claps: 700, responses: 25, published_date: '2023-10-08', tag: 'culture', length_minutes: 4 },
    { id: 'post_12', title: 'Product Management', subtitle: 'Leading without authority', author_id: 'user_6', content: 'PMs sit at the intersection of tech, ux, and business...', image: '/images/ProductManagement.jpg', claps: 1300, responses: 50, published_date: '2023-10-09', tag: 'technology', length_minutes: 7 },
    { id: 'post_13', title: 'Startup Metrics', subtitle: 'What to measure', author_id: 'user_7', content: 'CAC, LTV, Churn...', image: '/images/StartupMetrics.jpg', claps: 1600, responses: 70, published_date: '2023-10-10', tag: 'productivity', length_minutes: 6 },
    { id: 'post_14', title: 'Photography Basics', subtitle: 'Take better photos', author_id: 'user_8', content: 'Lighting is key...', image: '/images/Photography.jpg', claps: 950, responses: 40, published_date: '2023-10-11', tag: 'culture', length_minutes: 5 },
    { id: 'post_15', title: 'Video Editing', subtitle: 'Tools and techniques', author_id: 'user_8', content: 'Choosing the right software...', image: '/images/VideoEditing.jpg', claps: 800, responses: 30, published_date: '2023-10-12', tag: 'technology', length_minutes: 8 },
  ])

  // Drafts (for current user)
  const drafts = ref([
    { id: 'draft_1', title: 'My First Draft', subtitle: 'Work in progress', body: 'Starting a new journey...', tag: 'technology', status: 'draft', updated_at: '2023-10-20', length_minutes: 2, published: false },
    { id: 'draft_2', title: 'Untitled Story', subtitle: '', body: 'Some ideas...', tag: null, status: 'draft', updated_at: '2023-10-18', length_minutes: 1, published: false },
    { id: 'draft_3', title: 'Vue State Management', subtitle: 'Pinia vs Vuex', body: 'Comparing the two...', tag: 'technology', status: 'draft', updated_at: '2023-10-15', length_minutes: 5, published: false },
    { id: 'draft_4', title: 'Published Story 1', subtitle: 'Already out there', body: 'Content...', tag: 'culture', status: 'published', updated_at: '2023-09-10', length_minutes: 4, published: true },
    { id: 'draft_5', title: 'Published Story 2', subtitle: 'Another one', body: 'Content...', tag: 'productivity', status: 'published', updated_at: '2023-08-20', length_minutes: 6, published: true },
  ])

  function getPostById(id) {
    return posts.value.find(p => p.id === id)
  }

  function getUserById(id) {
    return users.value.find(u => u.id === id)
  }

  function getPublicationById(id) {
    return publications.value.find(p => p.id === id)
  }

  return {
    users,
    publications,
    posts,
    drafts,
    getPostById,
    getUserById,
    getPublicationById
  }
}, {
  persist: {
    storage: sessionStorage
  }
})