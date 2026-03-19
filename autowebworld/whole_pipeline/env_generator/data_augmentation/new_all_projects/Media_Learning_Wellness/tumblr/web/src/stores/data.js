import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // Mock Blogs (Creators)
  const blogs = ref([
    { id: 'blog_1', name: 'ArtisticSoul', handle: '@artisticsoul', avatar: '/images/Art.jpg', cover: '/images/Art.jpg', description: 'Digital art and vibes.', followers: 12400, following: true },
    { id: 'blog_2', name: 'RetroWave', handle: '@retrowave_official', avatar: '/images/RetroWave.jpg', cover: '/images/RetroWave.jpg', description: 'Nostalgia for a time you never lived.', followers: 8900, following: true },
    { id: 'blog_3', name: 'CodeLife', handle: '@codelife_dev', avatar: '/images/avatar-code-3.jpg', cover: '/images/Coffee.jpg', description: 'Sleeping is for the weak. Coffee is life.', followers: 4500, following: false },
    { id: 'blog_4', name: 'NatureWhispers', handle: '@nature_w', avatar: '/images/Nature.jpg', cover: '/images/Nature.jpg', description: 'Forests, mountains, and the deep blue sea.', followers: 21000, following: true },
    { id: 'blog_5', name: 'UrbanExplorer', handle: '@urban_ex', avatar: '/images/Urban.jpg', cover: '/images/UrbanExploration.jpg', description: 'City lights and late nights.', followers: 3200, following: false },
    { id: 'blog_6', name: 'MinimalistDesign', handle: '@minimal_d', avatar: '/images/MinimalistDesign.jpg', cover: '/images/MinimalistDesign.jpg', description: 'Less is more.', followers: 56000, following: true },
    { id: 'blog_7', name: 'FoodieHeaven', handle: '@foodie_h', avatar: '/images/photo1766058524.jpg', cover: '/images/Food.jpg', description: 'Eating my way through the world.', followers: 1500, following: false },
    { id: 'blog_8', name: 'SciFiGeek', handle: '@scifi_universe', avatar: '/images/SciFi.jpg', cover: '/images/ScienceFiction.jpg', description: 'Space: the final frontier.', followers: 9800, following: true },
    { id: 'blog_9', name: 'VintageFashion', handle: '@vintage_style', avatar: '/images/VintageFashion.jpg', cover: '/images/VintageFashion.jpg', description: 'Thrift store finds and classic looks.', followers: 11200, following: false },
    { id: 'blog_10', name: 'MusicJunkie', handle: '@music_j', avatar: '/images/Music.jpg', cover: '/images/Music.jpg', description: 'Vinyl collector and concert goer.', followers: 6700, following: true },
    { id: 'blog_11', name: 'TravelBug', handle: '@travel_b', avatar: '/images/Travel.jpg', cover: '/images/Travel.jpg', description: 'Wanderlust.', followers: 34000, following: false },
    { id: 'blog_12', name: 'CatMemesDaily', handle: '@cat_memes', avatar: '/images/Cat.jpg', cover: '/images/Cats.jpg', description: 'Just cats being cats.', followers: 100000, following: true },
    { id: 'blog_13', name: 'PhilosophyNow', handle: '@philo_n', avatar: '/images/Philosophy.jpg', cover: '/images/Philosophy.jpg', description: 'Thinking about thinking.', followers: 2300, following: false },
    { id: 'blog_14', name: 'GamerZone', handle: '@gamer_z', avatar: '/images/Gamer.jpg', cover: '/images/Gaming.jpg', description: 'Level up.', followers: 78000, following: true },
    { id: 'blog_15', name: 'DIYProjects', handle: '@diy_p', avatar: '/images/DIY.jpg', cover: '/images/DIY.jpg', description: 'Make it yourself.', followers: 5400, following: false }
  ])

  // Mock Posts
  // Using varied dates for slider testing
  const posts = ref([
    { id: 'post_1', blog_id: 'blog_1', type: 'photo', content: '/images/Cyberpunk.jpg', caption: 'Neon dreams in the digital void.', tags: ['#cyberpunk', '#art'], date: '2025-12-18T10:00:00', notes: 152 },
    { id: 'post_2', blog_id: 'blog_2', type: 'text', title: 'Why the 80s were better', content: 'Everything was neon, music had synths, and the future looked bright.', tags: ['#80s', '#nostalgia'], date: '2025-12-17T14:30:00', notes: 89 },
    { id: 'post_3', blog_id: 'blog_4', type: 'photo', content: '/images/Nature.jpg', caption: 'Morning mist over the mountains.', tags: ['#nature', '#morning'], date: '2025-12-16T08:15:00', notes: 340 },
    { id: 'post_4', blog_id: 'blog_6', type: 'quote', content: '"Simplicity is the ultimate sophistication."', source: '- Leonardo da Vinci', tags: ['#design', '#quotes'], date: '2025-12-15T18:45:00', notes: 1200 },
    { id: 'post_5', blog_id: 'blog_12', type: 'photo', content: '/images/photo1766058522.jpg', caption: 'If I fits, I sits.', tags: ['#cats', '#funny'], date: '2025-12-18T09:00:00', notes: 5600 },
    { id: 'post_6', blog_id: 'blog_8', type: 'text', title: 'Mars Colonization', content: 'We are closer than ever. The technology is almost ready.', tags: ['#space', '#mars'], date: '2025-12-14T20:20:00', notes: 45 },
    { id: 'post_7', blog_id: 'blog_10', type: 'audio', title: 'New Synth Track', content: 'Check out this new track I found!', album_art: '/images/Music.jpg', tags: ['#music', '#synthwave'], date: '2025-12-13T11:11:00', notes: 23 },
    { id: 'post_8', blog_id: 'blog_1', type: 'photo', content: '/images/Fractal.jpg', caption: 'Fractal geometry.', tags: ['#math', '#art'], date: '2025-12-12T15:00:00', notes: 88 },
    { id: 'post_9', blog_id: 'blog_3', type: 'text', title: 'Bug fixing', content: '99 little bugs in the code, 99 little bugs in the code. Take one down, patch it around, 127 little bugs in the code.', tags: ['#coding', '#humor'], date: '2025-12-18T12:00:00', notes: 404 },
    { id: 'post_10', blog_id: 'blog_5', type: 'photo', content: '/images/RainyStreets.jpg', caption: 'Rainy streets.', tags: ['#photography', '#city'], date: '2025-12-11T22:30:00', notes: 210 },
    { id: 'post_11', blog_id: 'blog_7', type: 'photo', content: '/images/Ramen.jpg', caption: 'Homemade ramen.', tags: ['#food', '#cooking'], date: '2025-12-10T19:00:00', notes: 150 },
    { id: 'post_12', blog_id: 'blog_9', type: 'photo', content: '/images/Fashion.jpg', caption: '70s vibes today.', tags: ['#vintage', '#ootd'], date: '2025-12-09T10:00:00', notes: 76 },
    { id: 'post_13', blog_id: 'blog_11', type: 'photo', content: '/images/Travel.jpg', caption: 'Sunset in Santorini.', tags: ['#travel', '#sunset'], date: '2025-12-08T17:45:00', notes: 890 },
    { id: 'post_14', blog_id: 'blog_14', type: 'text', title: 'Game of the Year?', content: 'Elden Ring DLC is absolutely massive.', tags: ['#gaming', '#eldenring'], date: '2025-12-07T13:00:00', notes: 560 },
    { id: 'post_15', blog_id: 'blog_15', type: 'photo', content: '/images/Bookshelf.jpg', caption: 'Built a bookshelf this weekend.', tags: ['#woodworking', '#diy'], date: '2025-12-06T16:20:00', notes: 112 },
    { id: 'post_16', blog_id: 'blog_2', type: 'photo', content: '/images/RetroMusic.jpg', caption: 'Cassette tapes had a soul.', tags: ['#retro', '#music'], date: '2025-12-05T09:30:00', notes: 67 },
    { id: 'post_17', blog_id: 'blog_4', type: 'photo', content: '/images/Nature.jpg', caption: 'Deep in the forest.', tags: ['#nature', '#forest'], date: '2025-12-04T14:15:00', notes: 230 },
    { id: 'post_18', blog_id: 'blog_6', type: 'text', title: 'Design Tip', content: 'White space is not empty space. It is an active design element.', tags: ['#design', '#ux'], date: '2025-12-03T11:00:00', notes: 450 },
    { id: 'post_19', blog_id: 'blog_12', type: 'photo', content: '/images/Cats.jpg', caption: 'Sleeping all day.', tags: ['#cats', '#sleep'], date: '2025-12-02T15:30:00', notes: 3400 },
    { id: 'post_20', blog_id: 'blog_13', type: 'quote', content: '"I think, therefore I am."', source: '- Descartes', tags: ['#philosophy', '#classics'], date: '2025-12-01T20:00:00', notes: 890 },
    { id: 'post_21', blog_id: 'blog_1', type: 'photo', content: '/images/AbstractArt.jpg', caption: 'Abstract expressionism.', tags: ['#art', '#abstract'], date: '2025-11-30T12:00:00', notes: 123 },
    { id: 'post_22', blog_id: 'blog_5', type: 'photo', content: '/images/Subway.jpg', caption: 'Subway station at 3AM.', tags: ['#urban', '#night'], date: '2025-11-29T03:00:00', notes: 180 },
    { id: 'post_23', blog_id: 'blog_10', type: 'text', title: 'Concert Review', content: 'The energy last night was insane.', tags: ['#livemusic', '#review'], date: '2025-11-28T10:00:00', notes: 56 },
    { id: 'post_24', blog_id: 'blog_11', type: 'photo', content: '/images/Kyoto.jpg', caption: 'Kyoto in autumn.', tags: ['#japan', '#travel'], date: '2025-11-27T16:00:00', notes: 670 },
    { id: 'post_25', blog_id: 'blog_8', type: 'photo', content: '/images/ScienceFiction.jpg', caption: 'Concept art for new movie.', tags: ['#scifi', '#art'], date: '2025-11-26T14:00:00', notes: 345 }
  ])

  // Mock Messages
  const messages = ref([
    { id: 'msg_1', recipient_id: 'blog_1', recipient_name: 'ArtisticSoul', avatar: '/images/Art.jpg', last_message: 'Hey, loved your latest piece!', timestamp: '2025-12-18T09:30:00', unread: true },
    { id: 'msg_2', recipient_id: 'blog_3', recipient_name: 'CodeLife', avatar: '/images/avatar-code-3.jpg', last_message: 'Did you fix that bug?', timestamp: '2025-12-17T15:45:00', unread: false },
    { id: 'msg_3', recipient_id: 'blog_5', recipient_name: 'UrbanExplorer', avatar: '/images/Urban.jpg', last_message: 'Let\'s go shoot some photos this weekend.', timestamp: '2025-12-16T20:00:00', unread: true },
    { id: 'msg_4', recipient_id: 'blog_2', recipient_name: 'RetroWave', avatar: '/images/RetroWave.jpg', last_message: 'Where did you get that cassette player?', timestamp: '2025-12-15T11:20:00', unread: false },
    { id: 'msg_5', recipient_id: 'blog_10', recipient_name: 'MusicJunkie', avatar: '/images/Music.jpg', last_message: 'Listen to this track.', timestamp: '2025-12-14T18:10:00', unread: false },
    { id: 'msg_6', recipient_id: 'blog_14', recipient_name: 'GamerZone', avatar: '/images/Gamer.jpg', last_message: '1v1 me bro.', timestamp: '2025-12-13T22:00:00', unread: true },
    { id: 'msg_7', recipient_id: 'blog_4', recipient_name: 'NatureWhispers', avatar: '/images/Nature.jpg', last_message: 'The hike is planned for Saturday.', timestamp: '2025-12-12T08:00:00', unread: false },
    { id: 'msg_8', recipient_id: 'blog_11', recipient_name: 'TravelBug', avatar: '/images/Travel.jpg', last_message: 'Send me the itinerary!', timestamp: '2025-12-11T14:30:00', unread: false },
    { id: 'msg_9', recipient_id: 'blog_9', recipient_name: 'VintageFashion', avatar: '/images/VintageFashion.jpg', last_message: 'Is this authentic 70s?', timestamp: '2025-12-10T16:45:00', unread: true },
    { id: 'msg_10', recipient_id: 'blog_6', recipient_name: 'MinimalistDesign', avatar: '/images/MinimalistDesign.jpg', last_message: 'Less clutter, more focus.', timestamp: '2025-12-09T09:15:00', unread: false }
  ])

  // Mock Message Threads (Details)
  const messageThreads = ref({
    'msg_1': [
      { id: 't1_1', sender: 'me', text: 'Hey, loved your latest piece!', timestamp: '2025-12-18T09:25:00' },
      { id: 't1_2', sender: 'them', text: 'Thank you so much! It means a lot.', timestamp: '2025-12-18T09:30:00' }
    ],
    'msg_2': [
      { id: 't2_1', sender: 'them', text: 'Hey, did you look at the PR?', timestamp: '2025-12-17T10:00:00' },
      { id: 't2_2', sender: 'me', text: 'Not yet, looking now.', timestamp: '2025-12-17T12:00:00' },
      { id: 't2_3', sender: 'them', text: 'Did you fix that bug?', timestamp: '2025-12-17T15:45:00' }
    ]
    // Add others if needed for detail view, generating on fly for others
  })

  // User Profile
  const userProfile = ref({
    id: 'user_current',
    display_name: 'MyAwesomeBlog',
    bio: 'Just another dreamer on the internet.',
    theme_color: 'dark'
  })

  return {
    blogs,
    posts,
    messages,
    messageThreads,
    userProfile
  }
}, {
  persist: {
    storage: sessionStorage
  }
})