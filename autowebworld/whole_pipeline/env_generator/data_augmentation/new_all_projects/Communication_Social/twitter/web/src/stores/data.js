import { defineStore } from 'pinia';
import { ref } from 'vue';

export const useDataStore = defineStore('data', () => {
  // --- Users (10+ items) ---
  const users = ref([
    {
      id: 'user_me',
      handle: '@myself',
      name: 'My Profile',
      avatar: '/images/UserAvatar.jpg',
      bio: 'Just a developer building cool things.',
      location: 'San Francisco, CA',
      verified: true,
      following_count: 142,
      followers_count: 853,
      joined_date: 'September 2018'
    },
    {
      id: 'u1',
      handle: '@elonmusk',
      name: 'Elon Musk',
      avatar: '/images/ElonMusk.jpg',
      bio: '',
      location: '',
      verified: true,
      following_count: 504,
      followers_count: 167000000,
      joined_date: 'June 2009'
    },
    {
      id: 'u2',
      handle: '@NASA',
      name: 'NASA',
      avatar: '/images/NASA.jpg',
      bio: 'Exploring the universe and our home planet.',
      location: 'Washington, DC',
      verified: true,
      following_count: 180,
      followers_count: 75000000,
      joined_date: 'December 2007'
    },
    {
      id: 'u3',
      handle: '@TechCrunch',
      name: 'TechCrunch',
      avatar: '/images/user-techcrunch-logo.jpg',
      bio: 'Breaking technology news and analysis.',
      location: 'San Francisco, CA',
      verified: true,
      following_count: 890,
      followers_count: 10200000,
      joined_date: 'August 2008'
    },
    {
      id: 'u4',
      handle: '@nature_org',
      name: 'The Nature Conservancy',
      avatar: '/images/NatureConservancy.jpg',
      bio: 'Conserving the lands and waters on which all life depends.',
      location: 'Worldwide',
      verified: true,
      following_count: 2400,
      followers_count: 1500000,
      joined_date: 'March 2009'
    },
    {
      id: 'u5',
      handle: '@NBA',
      name: 'NBA',
      avatar: '/images/NBA.jpg',
      bio: 'The official Twitter handle of the NBA.',
      location: 'New York, NY',
      verified: true,
      following_count: 1800,
      followers_count: 45000000,
      joined_date: 'February 2009'
    },
    {
      id: 'u6',
      handle: '@GordonRamsay',
      name: 'Gordon Ramsay',
      avatar: '/images/GordonRamsay.jpg',
      bio: 'Chef, Restaurateur, Writer, TV Personality.',
      location: 'London / Los Angeles',
      verified: true,
      following_count: 3200,
      followers_count: 7800000,
      joined_date: 'February 2010'
    },
    {
      id: 'u7',
      handle: '@JaneDoe_Art',
      name: 'Jane Doe Art',
      avatar: '/images/Art.jpg',
      bio: 'Digital Artist & Illustrator. Commissions Open.',
      location: 'Portland, OR',
      verified: false,
      following_count: 450,
      followers_count: 1200,
      joined_date: 'January 2021'
    },
    {
      id: 'u8',
      handle: '@TravelWithTom',
      name: 'Tom Explorer',
      avatar: '/images/Travel.jpg',
      bio: 'Traveling the world one photo at a time.',
      location: 'Nomad',
      verified: false,
      following_count: 120,
      followers_count: 3400,
      joined_date: 'May 2019'
    },
    {
      id: 'u9',
      handle: '@CodeWizard',
      name: 'Alex Dev',
      avatar: '/images/Developer.jpg',
      bio: 'Full Stack Developer | Vue.js Enthusiast',
      location: 'Austin, TX',
      verified: false,
      following_count: 300,
      followers_count: 890,
      joined_date: 'July 2020'
    },
    {
      id: 'u10',
      handle: '@FoodieHeaven',
      name: 'Sarah Eats',
      avatar: '/images/Food.jpg',
      bio: 'Loving food and sharing recipes.',
      location: 'Chicago, IL',
      verified: false,
      following_count: 600,
      followers_count: 5600,
      joined_date: 'November 2017'
    }
  ]);

  // --- Tweets (20+ items) ---
  const tweets = ref([
    {
      id: 't1',
      author_id: 'u1',
      content: 'Mars is looking great tonight!',
      timestamp: '2h',
      likes: 45000,
      retweets: 12000,
      replies: 3400,
      views: '1.2M',
      has_media: true,
      media_url: '/images/Mars.jpg'
    },
    {
      id: 't2',
      author_id: 'u2',
      content: 'New images from the James Webb Space Telescope reveal stunning details of the nebula.',
      timestamp: '4h',
      likes: 89000,
      retweets: 23000,
      replies: 1500,
      views: '2.5M',
      has_media: true,
      media_url: '/images/Nebula.jpg'
    },
    {
      id: 't3',
      author_id: 'u3',
      content: 'Breaking: New AI model outperforms humans in coding tasks.',
      timestamp: '30m',
      likes: 1200,
      retweets: 450,
      replies: 120,
      views: '50K',
      has_media: false,
      media_url: null
    },
    {
      id: 't4',
      author_id: 'u4',
      content: 'Protecting our oceans is crucial for the future of our planet. 🌊',
      timestamp: '1d',
      likes: 5600,
      retweets: 1800,
      replies: 230,
      views: '120K',
      has_media: true,
      media_url: '/images/Ocean.jpg'
    },
    {
      id: 't5',
      author_id: 'u5',
      content: 'What a game last night! The buzzer beater was insane! 🏀',
      timestamp: '12h',
      likes: 34000,
      retweets: 8900,
      replies: 1200,
      views: '900K',
      has_media: true,
      media_url: '/images/Basketball.jpg'
    },
    {
      id: 't6',
      author_id: 'u6',
      content: 'This risotto is RAW! #Hellskitchen',
      timestamp: '5h',
      likes: 23000,
      retweets: 4500,
      replies: 890,
      views: '600K',
      has_media: true,
      media_url: '/images/Risotto.jpg'
    },
    {
      id: 't7',
      author_id: 'u7',
      content: 'Just finished this new commission. What do you think?',
      timestamp: '1h',
      likes: 450,
      retweets: 89,
      replies: 45,
      views: '12K',
      has_media: true,
      media_url: '/images/DigitalArt.jpg'
    },
    {
      id: 't8',
      author_id: 'u8',
      content: 'Sunset in Santorini is magical. 🌅',
      timestamp: '3h',
      likes: 1200,
      retweets: 230,
      replies: 56,
      views: '45K',
      has_media: true,
      media_url: '/images/Sunset.jpg'
    },
    {
      id: 't9',
      author_id: 'u9',
      content: 'Debugging for 5 hours straight. Finally fixed it! 🐛❌',
      timestamp: '10m',
      likes: 89,
      retweets: 12,
      replies: 5,
      views: '2K',
      has_media: false,
      media_url: null
    },
    {
      id: 't10',
      author_id: 'u10',
      content: 'Best chocolate cake recipe ever! Check the thread below. 🍫🍰',
      timestamp: '6h',
      likes: 2300,
      retweets: 560,
      replies: 120,
      views: '89K',
      has_media: true,
      media_url: '/images/ChocolateCake.jpg'
    },
    {
      id: 't11',
      author_id: 'user_me',
      content: 'Just deployed my new app! Check it out.',
      timestamp: '2d',
      likes: 56,
      retweets: 12,
      replies: 4,
      views: '1.2K',
      has_media: false,
      media_url: null
    },
    {
      id: 't12',
      author_id: 'u1',
      content: 'Rockets are cool.',
      timestamp: '1d',
      likes: 120000,
      retweets: 34000,
      replies: 5600,
      views: '5M',
      has_media: true,
      media_url: '/images/Rocket.jpg'
    },
    {
      id: 't13',
      author_id: 'u2',
      content: 'The ISS just passed over Europe.',
      timestamp: '8h',
      likes: 4500,
      retweets: 1200,
      replies: 230,
      views: '150K',
      has_media: true,
      media_url: '/images/ISS.jpg'
    },
    {
      id: 't14',
      author_id: 'u8',
      content: 'Hiking the Alps next week! Any tips?',
      timestamp: '2h',
      likes: 340,
      retweets: 23,
      replies: 45,
      views: '12K',
      has_media: true,
      media_url: '/images/Alps.jpg'
    },
    {
      id: 't15',
      author_id: 'u6',
      content: 'Lamb sauce located.',
      timestamp: '1d',
      likes: 56000,
      retweets: 12000,
      replies: 2300,
      views: '1.5M',
      has_media: false,
      media_url: null
    },
    {
      id: 't16',
      author_id: 'u9',
      content: 'Why does CSS center div have to be so hard? 😂',
      timestamp: '45m',
      likes: 230,
      retweets: 45,
      replies: 23,
      views: '5K',
      has_media: false,
      media_url: null
    },
    {
      id: 't17',
      author_id: 'u4',
      content: 'Plant a tree today! 🌳',
      timestamp: '2h',
      likes: 890,
      retweets: 340,
      replies: 56,
      views: '34K',
      has_media: true,
      media_url: '/images/TreePlanting.jpg'
    },
    {
      id: 't18',
      author_id: 'u10',
      content: 'Homemade pasta is worth the effort.',
      timestamp: '5h',
      likes: 1200,
      retweets: 230,
      replies: 89,
      views: '45K',
      has_media: true,
      media_url: '/images/HomemadePasta.jpg'
    },
    {
      id: 't19',
      author_id: 'user_me',
      content: 'Exploring the new city. Love the vibes here.',
      timestamp: '3d',
      likes: 89,
      retweets: 5,
      replies: 2,
      views: '1.5K',
      has_media: true,
      media_url: '/images/CityExploration.jpg'
    },
    {
      id: 't20',
      author_id: 'u3',
      content: 'Review of the latest smartphone dropping soon.',
      timestamp: '6h',
      likes: 2300,
      retweets: 450,
      replies: 120,
      views: '89K',
      has_media: true,
      media_url: '/images/smartphone.jpg'
    },
    {
        id: 't21',
        author_id: 'u1',
        content: 'Thinking about buying a new social media platform.',
        timestamp: '5y',
        likes: 1500000,
        retweets: 500000,
        replies: 200000,
        views: '50M',
        has_media: false,
        media_url: null
    }
  ]);

  // --- Messages / Threads (15+ items) ---
  const threads = ref([
    {
      id: 'th1',
      participant_id: 'u1',
      last_message: 'To the moon! 🚀',
      timestamp: '2h',
      unread: true,
      is_request: false
    },
    {
      id: 'th2',
      participant_id: 'u7',
      last_message: 'Can you send the reference image?',
      timestamp: '5h',
      unread: false,
      is_request: false
    },
    {
      id: 'th3',
      participant_id: 'u9',
      last_message: 'Hey, saw your GitHub repo. Nice work!',
      timestamp: '1d',
      unread: true,
      is_request: true
    },
    {
      id: 'th4',
      participant_id: 'u8',
      last_message: 'Where did you stay in Tokyo?',
      timestamp: '2d',
      unread: false,
      is_request: false
    },
    {
      id: 'th5',
      participant_id: 'u10',
      last_message: 'Thanks for the recipe!',
      timestamp: '3d',
      unread: false,
      is_request: false
    },
    {
      id: 'th6',
      participant_id: 'u6',
      last_message: 'Is the risotto ready yet?',
      timestamp: '1w',
      unread: false,
      is_request: true
    },
    {
      id: 'th7',
      participant_id: 'u2',
      last_message: 'We are hiring astronauts.',
      timestamp: '2w',
      unread: true,
      is_request: false
    },
    {
      id: 'th8',
      participant_id: 'u3',
      last_message: 'Did you see the article?',
      timestamp: '1d',
      unread: false,
      is_request: false
    },
    {
      id: 'th9',
      participant_id: 'u4',
      last_message: 'Donation received. Thank you!',
      timestamp: '3w',
      unread: false,
      is_request: false
    },
    {
      id: 'th10',
      participant_id: 'u5',
      last_message: 'Season tickets are available.',
      timestamp: '1mo',
      unread: false,
      is_request: false
    },
    {
      id: 'th11',
      participant_id: 'u1',
      last_message: 'Another update on Starship.',
      timestamp: '2d',
      unread: false,
      is_request: false
    },
    {
      id: 'th12',
      participant_id: 'u7',
      last_message: 'Sketch is ready for review.',
      timestamp: '3d',
      unread: true,
      is_request: false
    },
    {
      id: 'th13',
      participant_id: 'u9',
      last_message: 'Could use some help with this bug.',
      timestamp: '4h',
      unread: false,
      is_request: false
    },
    {
      id: 'th14',
      participant_id: 'u8',
      last_message: 'Just booked my flight!',
      timestamp: '1h',
      unread: true,
      is_request: false
    },
    {
      id: 'th15',
      participant_id: 'u10',
      last_message: 'Trying that pizza place tonight.',
      timestamp: '30m',
      unread: false,
      is_request: false
    }
  ]);

  const messages = ref([
    { id: 'm1', thread_id: 'th1', sender_id: 'u1', text: 'To the moon! 🚀', timestamp: '2h' },
    { id: 'm2', thread_id: 'th1', sender_id: 'user_me', text: 'Cant wait!', timestamp: '2h 5m' },
    // Add more dummy messages if needed for specific threads
  ]);

  // --- Notifications (15+ items) ---
  const notifications = ref([
    { id: 'n1', type: 'like', user_id: 'u1', tweet_id: 't11', timestamp: '10m', is_read: false },
    { id: 'n2', type: 'retweet', user_id: 'u2', tweet_id: 't11', timestamp: '30m', is_read: false },
    { id: 'n3', type: 'follow', user_id: 'u3', tweet_id: null, timestamp: '1h', is_read: true },
    { id: 'n4', type: 'mention', user_id: 'u9', tweet_id: 't16', timestamp: '2h', is_read: false, text: 'Check this out @myself' },
    { id: 'n5', type: 'like', user_id: 'u4', tweet_id: 't19', timestamp: '3h', is_read: true },
    { id: 'n6', type: 'reply', user_id: 'u7', tweet_id: 't11', timestamp: '4h', is_read: false, text: 'Great job!' },
    { id: 'n7', type: 'like', user_id: 'u8', tweet_id: 't19', timestamp: '5h', is_read: true },
    { id: 'n8', type: 'follow', user_id: 'u10', tweet_id: null, timestamp: '6h', is_read: true },
    { id: 'n9', type: 'mention', user_id: 'u6', tweet_id: 't6', timestamp: '1d', is_read: false, text: '@myself what do you think?' },
    { id: 'n10', type: 'retweet', user_id: 'u5', tweet_id: 't19', timestamp: '1d', is_read: true },
    { id: 'n11', type: 'like', user_id: 'u2', tweet_id: 't11', timestamp: '1d', is_read: true },
    { id: 'n12', type: 'follow', user_id: 'u9', tweet_id: null, timestamp: '2d', is_read: true },
    { id: 'n13', type: 'reply', user_id: 'u1', tweet_id: 't11', timestamp: '2d', is_read: true, text: 'Interesting.' },
    { id: 'n14', type: 'like', user_id: 'u3', tweet_id: 't19', timestamp: '3d', is_read: true },
    { id: 'n15', type: 'mention', user_id: 'u4', tweet_id: 't4', timestamp: '4d', is_read: true, text: 'Thanks @myself for sharing.' }
  ]);

  // --- Trends (10+ items) ---
  const trends = ref([
    { id: 'tr1', name: '#SpaceX', category: 'Science', tweets_count: '150K', is_trending: true },
    { id: 'tr2', name: 'James Webb', category: 'Science', tweets_count: '120K', is_trending: true },
    { id: 'tr3', name: '#NBAPlayoffs', category: 'Sports', tweets_count: '500K', is_trending: true },
    { id: 'tr4', name: 'Gordon Ramsay', category: 'Entertainment', tweets_count: '45K', is_trending: false },
    { id: 'tr5', name: '#VueJS', category: 'Technology', tweets_count: '25K', is_trending: false },
    { id: 'tr6', name: 'Climate Action', category: 'News', tweets_count: '200K', is_trending: true },
    { id: 'tr7', name: '#DigitalArt', category: 'Arts', tweets_count: '80K', is_trending: false },
    { id: 'tr8', name: 'Santorini', category: 'Travel', tweets_count: '30K', is_trending: false },
    { id: 'tr9', name: '#CodingLife', category: 'Technology', tweets_count: '15K', is_trending: false },
    { id: 'tr10', name: 'Chocolate Cake', category: 'Food', tweets_count: '10K', is_trending: false },
    { id: 'tr11', name: '#MondayMotivation', category: 'Lifestyle', tweets_count: '100K', is_trending: true },
    { id: 'tr12', name: 'New iPhone', category: 'Technology', tweets_count: '300K', is_trending: true }
  ]);

  // --- Bookmarks (10+ items) ---
  const bookmarks = ref([
    { id: 'b1', tweet_id: 't2', saved_at: '2025-10-20' },
    { id: 'b2', tweet_id: 't10', saved_at: '2025-10-18' },
    { id: 'b3', tweet_id: 't12', saved_at: '2025-10-15' },
    { id: 'b4', tweet_id: 't1', saved_at: '2025-10-10' },
    { id: 'b5', tweet_id: 't6', saved_at: '2025-10-05' },
    { id: 'b6', tweet_id: 't8', saved_at: '2025-10-01' },
    { id: 'b7', tweet_id: 't3', saved_at: '2025-09-28' },
    { id: 'b8', tweet_id: 't16', saved_at: '2025-09-25' },
    { id: 'b9', tweet_id: 't7', saved_at: '2025-09-20' },
    { id: 'b10', tweet_id: 't14', saved_at: '2025-09-15' }
  ]);

  // Helper getters
  const getUserById = (id) => users.value.find(u => u.id === id);
  const getTweetById = (id) => tweets.value.find(t => t.id === id);
  const getThreadById = (id) => threads.value.find(t => t.id === id);

  return {
    users,
    tweets,
    threads,
    messages,
    notifications,
    trends,
    bookmarks,
    getUserById,
    getTweetById,
    getThreadById
  };
}, {
  persist: {
    storage: sessionStorage,
  }
});