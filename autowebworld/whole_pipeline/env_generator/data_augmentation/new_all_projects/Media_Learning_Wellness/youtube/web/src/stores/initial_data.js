import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // Mock Data Generators

  // Videos (20 items)
  const videos = ref([
    { id: 'v1', title: 'Top 10 Travel Destinations 2024', channel: 'Travel Guide', views: '1.2M', date: '2 days ago', duration: 600, image: '/images/Travel.jpg' },
    { id: 'v2', title: 'Learn Vue 3 in 30 Minutes', channel: 'Code Master', views: '500K', date: '1 week ago', duration: 1800, image: '/images/Coding.jpg' },
    { id: 'v3', title: 'Relaxing Jazz Music', channel: 'Chill Vibes', views: '2.5M', date: '1 month ago', duration: 3600, image: '/images/JazzMusic.jpg' },
    { id: 'v4', title: 'Funny Cat Compilation', channel: 'Pet Lovers', views: '10M', date: '3 days ago', duration: 420, image: '/images/FunnyCat.jpg' },
    { id: 'v5', title: 'Ultimate Gaming Setup Tour', channel: 'Tech Review', views: '800K', date: '5 days ago', duration: 900, image: '/images/GamingSetup.jpg' },
    { id: 'v6', title: 'How to Cook Perfect Steak', channel: 'Chef Gordon', views: '3M', date: '2 weeks ago', duration: 720, image: '/images/Cooking.jpg' },
    { id: 'v7', title: 'SpaceX Launch Highlights', channel: 'Space News', views: '5M', date: '1 day ago', duration: 300, image: '/images/RocketLaunch.jpg' },
    { id: 'v8', title: 'iPhone 16 Review', channel: 'Gadget Guru', views: '2M', date: '4 days ago', duration: 1200, image: '/images/iPhoneReview.jpg' },
    { id: 'v9', title: 'Morning Yoga Routine', channel: 'Fitness Life', views: '600K', date: '1 month ago', duration: 1500, image: '/images/Yoga.jpg' },
    { id: 'v10', title: 'Minecraft Survival Ep. 1', channel: 'Gamer Steve', views: '1.5M', date: '1 year ago', duration: 2400, image: '/images/Minecraft.jpg' },
    { id: 'v11', title: 'Street Food in Japan', channel: 'Foodie Travels', views: '4M', date: '3 weeks ago', duration: 900, image: '/images/JapaneseFood.jpg' },
    { id: 'v12', title: 'Advanced CSS Animation', channel: 'Web Dev Tips', views: '100K', date: '2 days ago', duration: 600, image: '/images/CSS.jpg' },
    { id: 'v13', title: 'History of Rome', channel: 'History Buff', views: '900K', date: '6 months ago', duration: 5400, image: '/images/Rome.jpg' },
    { id: 'v14', title: 'DIY Home Decor', channel: 'Crafty Hands', views: '750K', date: '1 week ago', duration: 480, image: '/images/HomeDecor.jpg' },
    { id: 'v15', title: 'Electric Car Comparison', channel: 'Auto Weekly', views: '1.1M', date: '3 days ago', duration: 1600, image: '/images/ElectricCars.jpg' },
    { id: 'v16', title: 'Learn Guitar: Beginner', channel: 'Music School', views: '200K', date: '2 months ago', duration: 1200, image: '/images/Guitar.jpg' },
    { id: 'v17', title: 'Beautiful Nature 4K', channel: 'Earth Views', views: '8M', date: '1 year ago', duration: 600, image: '/images/Nature.jpg' },
    { id: 'v18', title: 'Photography Basics', channel: 'Photo Pro', views: '300K', date: '2 weeks ago', duration: 800, image: '/images/Photography.jpg' },
    { id: 'v19', title: 'Best Sci-Fi Movies 2024', channel: 'Movie Talk', views: '1.3M', date: '1 month ago', duration: 1000, image: '/images/Movies.jpg' },
    { id: 'v20', title: 'Investment Tips for 2025', channel: 'Finance Daily', views: '400K', date: '5 days ago', duration: 900, image: '/images/Finance.jpg' }
  ])

  // Playlists (15 items)
  const playlists = ref([
    { id: 'p1', title: 'Watch Later', count: 12, image: '/images/WatchLater.jpg' },
    { id: 'p2', title: 'Liked Videos', count: 45, image: '/images/LikedVideos.jpg' },
    { id: 'p3', title: 'Coding Tutorials', count: 20, image: '/images/Coding.jpg' },
    { id: 'p4', title: 'Music Mix 2024', count: 50, image: '/images/Music.jpg' },
    { id: 'p5', title: 'Workout Jams', count: 15, image: '/images/Workout.jpg' },
    { id: 'p6', title: 'Travel Vlogs', count: 8, image: '/images/Travel.jpg' },
    { id: 'p7', title: 'Gaming Highlights', count: 30, image: '/images/Gaming.jpg' },
    { id: 'p8', title: 'Cooking Recipes', count: 25, image: '/images/Cooking.jpg' },
    { id: 'p9', title: 'Tech Reviews', count: 18, image: '/images/Tech.jpg' },
    { id: 'p10', title: 'Documentaries', count: 5, image: '/images/Documentaries.jpg' },
    { id: 'p11', title: 'Funny Clips', count: 100, image: '/images/FunnyClips.jpg' },
    { id: 'p12', title: 'Relaxation', count: 10, image: '/images/Relaxation.jpg' },
    { id: 'p13', title: 'Study Music', count: 60, image: '/images/StudyMusic.jpg' },
    { id: 'p14', title: 'My Favorites', count: 99, image: '/images/Favorites.jpg' },
    { id: 'p15', title: 'News Updates', count: 22, image: '/images/News.jpg' }
  ])

  // Channels (15 items)
  const channels = ref([
    { id: 'c1', name: 'Travel Guide', subscribers: '1M', avatar: '/images/Travel.jpg', activity: 90 },
    { id: 'c2', name: 'Code Master', subscribers: '500K', avatar: '/images/CodeMaster.jpg', activity: 80 },
    { id: 'c3', name: 'Chill Vibes', subscribers: '2M', avatar: '/images/ChillVibes.jpg', activity: 60 },
    { id: 'c4', name: 'Pet Lovers', subscribers: '5M', avatar: '/images/Pet.jpg', activity: 95 },
    { id: 'c5', name: 'Tech Review', subscribers: '800K', avatar: '/images/Tech.jpg', activity: 85 },
    { id: 'c6', name: 'Chef Gordon', subscribers: '10M', avatar: '/images/Chef.jpg', activity: 70 },
    { id: 'c7', name: 'Space News', subscribers: '300K', avatar: '/images/SpaceNews.jpg', activity: 50 },
    { id: 'c8', name: 'Gadget Guru', subscribers: '1.2M', avatar: '/images/Gadgets.jpg', activity: 88 },
    { id: 'c9', name: 'Fitness Life', subscribers: '600K', avatar: '/images/Fitness.jpg', activity: 75 },
    { id: 'c10', name: 'Gamer Steve', subscribers: '2.5M', avatar: '/images/Gamer.jpg', activity: 92 },
    { id: 'c11', name: 'Foodie Travels', subscribers: '150K', avatar: '/images/Food.jpg', activity: 40 },
    { id: 'c12', name: 'Web Dev Tips', subscribers: '900K', avatar: '/images/WebDevelopment.jpg', activity: 65 },
    { id: 'c13', name: 'History Buff', subscribers: '200K', avatar: '/images/History.jpg', activity: 30 },
    { id: 'c14', name: 'Crafty Hands', subscribers: '400K', avatar: '/images/Crafts.jpg', activity: 55 },
    { id: 'c15', name: 'Auto Weekly', subscribers: '700K', avatar: '/images/AutoWeekly.jpg', activity: 82 }
  ])

  return {
    videos,
    playlists,
    channels
  }
}, {
  persist: {
    storage: sessionStorage
  }
})