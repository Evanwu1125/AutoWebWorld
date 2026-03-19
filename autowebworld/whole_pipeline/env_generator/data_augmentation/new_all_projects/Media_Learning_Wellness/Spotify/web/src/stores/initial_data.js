import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // Categories (Browse)
  const categories = ref([
    { id: 'cat_pop', name: 'Pop', image: '/images/Music.jpg', color: '#8c1932' },
    { id: 'cat_hiphop', name: 'Hip-Hop', image: '/images/HipHop.jpg', color: '#ba5d07' },
    { id: 'cat_rock', name: 'Rock', image: '/images/Rock.jpg', color: '#e61e32' },
    { id: 'cat_indie', name: 'Indie', image: '/images/Indie.jpg', color: '#608108' },
    { id: 'cat_chill', name: 'Chill', image: '/images/Chill.jpg', color: '#477d95' },
    { id: 'cat_workout', name: 'Workout', image: '/images/Workout.jpg', color: '#777777' },
    { id: 'cat_party', name: 'Party', image: '/images/Party.jpg', color: '#537aa1' },
    { id: 'cat_focus', name: 'Focus', image: '/images/Focus.jpg', color: '#af2896' },
    { id: 'cat_sleep', name: 'Sleep', image: '/images/Sleep.jpg', color: '#1e3264' },
    { id: 'cat_jazz', name: 'Jazz', image: '/images/JazzCat.jpg', color: '#7d4b32' }
  ])

  // Artists
  const artists = ref([
    { id: 'art_1', name: 'The Midnight', image: '/images/Midnight.jpg', followers: '1.2M' },
    { id: 'art_2', name: 'Gunship', image: '/images/Gunship.jpg', followers: '500K' },
    { id: 'art_3', name: 'FM-84', image: '/images/FM84.jpg', followers: '350K' },
    { id: 'art_4', name: 'Timecop1983', image: '/images/artist-timecop.jpg', followers: '450K' },
    { id: 'art_5', name: 'Daft Punk', image: '/images/DaftPunk.jpg', followers: '15M' },
    { id: 'art_6', name: 'The Weeknd', image: '/images/artist-weeknd.jpg', followers: '40M' },
    { id: 'art_7', name: 'Tame Impala', image: '/images/TameImpala.jpg', followers: '8M' },
    { id: 'art_8', name: 'Arctic Monkeys', image: '/images/ArcticMonkeys.jpg', followers: '12M' },
    { id: 'art_9', name: 'Glass Animals', image: '/images/GlassAnimals.jpg', followers: '4M' },
    { id: 'art_10', name: 'Gorillaz', image: '/images/artist-gorillaz.jpg', followers: '9M' }
  ])

  // Albums
  const albums = ref([
    { id: 'alb_1', artist_id: 'art_1', name: 'Endless Summer', year: 2016, image: '/images/album-endless.jpg' },
    { id: 'alb_2', artist_id: 'art_1', name: 'Nocturnal', year: 2017, image: '/images/Nocturnal.jpg' },
    { id: 'alb_3', artist_id: 'art_2', name: 'Dark All Day', year: 2018, image: '/images/DarkAllDay.jpg' },
    { id: 'alb_4', artist_id: 'art_3', name: 'Atlas', year: 2016, image: '/images/Atlas.jpg' },
    { id: 'alb_5', artist_id: 'art_5', name: 'Discovery', year: 2001, image: '/images/Discovery.jpg' },
    { id: 'alb_6', artist_id: 'art_5', name: 'Random Access Memories', year: 2013, image: '/images/RandomAccessMemories.jpg' },
    { id: 'alb_7', artist_id: 'art_6', name: 'After Hours', year: 2020, image: '/images/AfterHours.jpg' },
    { id: 'alb_8', artist_id: 'art_6', name: 'Starboy', year: 2016, image: '/images/album-starboy.jpg' },
    { id: 'alb_9', artist_id: 'art_7', name: 'Currents', year: 2015, image: '/images/Currents.jpg' },
    { id: 'alb_10', artist_id: 'art_8', name: 'AM', year: 2013, image: '/images/AM.jpg' }
  ])

  // Playlists
  const playlists = ref([
    { id: 'pl_1', name: 'Synthwave Essentials', description: 'Retro futuristic sounds.', owner: 'Spotify', image: '/images/Synthwave.jpg', featured: true, category_id: 'cat_pop' },
    { id: 'pl_2', name: 'Night Drive', description: 'Music for late night drives.', owner: 'Spotify', image: '/images/NightDrive.jpg', featured: true, category_id: 'cat_chill' },
    { id: 'pl_3', name: 'Cyberpunk 2077', description: 'Official Soundtrack', owner: 'CDPR', image: '/images/Cyberpunk.jpg', featured: false, category_id: 'cat_rock' },
    { id: 'pl_4', name: 'Neon Lights', description: 'Glow in the dark beats.', owner: 'Spotify', image: '/images/Neon.jpg', featured: true, category_id: 'cat_pop' },
    { id: 'pl_5', name: 'Retro Gaming', description: '8-bit nostalgia.', owner: 'GamerOne', image: '/images/RetroGaming.jpg', featured: false, category_id: 'cat_focus' },
    { id: 'pl_6', name: 'Today\'s Top Hits', description: 'The hottest tracks right now.', owner: 'Spotify', image: '/images/Spotify.jpg', featured: true, category_id: 'cat_pop' },
    { id: 'pl_7', name: 'Rap Caviar', description: 'New music from Drake, Travis Scott...', owner: 'Spotify', image: '/images/Rap.jpg', featured: true, category_id: 'cat_hiphop' },
    { id: 'pl_8', name: 'Rock Classics', description: 'Rock legends & anthems.', owner: 'Spotify', image: '/images/Rock.jpg', featured: false, category_id: 'cat_rock' },
    { id: 'pl_9', name: 'Indie Pop', description: 'The best new indie pop.', owner: 'Spotify', image: '/images/IndiePop.jpg', featured: false, category_id: 'cat_indie' },
    { id: 'pl_10', name: 'Deep Focus', description: 'Keep calm and focus.', owner: 'Spotify', image: '/images/Focus.jpg', featured: true, category_id: 'cat_focus' },
    { id: 'pl_11', name: 'Sleep', description: 'Music to help you sleep.', owner: 'Spotify', image: '/images/Sleep.jpg', featured: false, category_id: 'cat_sleep' },
    { id: 'pl_12', name: 'Workout Beast', description: 'Get pumped.', owner: 'GymRat', image: '/images/Workout.jpg', featured: false, category_id: 'cat_workout' },
    { id: 'pl_13', name: 'Party Anthems', description: 'All the hits.', owner: 'PartyBoii', image: '/images/Party.jpg', featured: true, category_id: 'cat_party' },
    { id: 'pl_14', name: 'Jazz Vibes', description: 'Smooth jazz for relaxing.', owner: 'JazzCat', image: '/images/Jazz.jpg', featured: false, category_id: 'cat_jazz' },
    { id: 'pl_15', name: 'Discover Weekly', description: 'Your weekly mixtape.', owner: 'Spotify', image: '/images/Discover.jpg', featured: false, category_id: 'cat_pop' }
  ])

  // Tracks
  const tracks = ref([
    { id: 'tr_1', name: 'Sunset', artist: 'The Midnight', album: 'Endless Summer', duration: '5:26', image: '/images/TheMidnight.jpg', playlist_id: 'pl_1' },
    { id: 'tr_2', name: 'Vampires', artist: 'The Midnight', album: 'Nocturnal', duration: '5:17', image: '/images/TheMidnight.jpg', playlist_id: 'pl_1' },
    { id: 'tr_3', name: 'Tech Noir', artist: 'Gunship', album: 'Gunship', duration: '4:57', image: '/images/Gunship.jpg', playlist_id: 'pl_1' },
    { id: 'tr_4', name: 'Running in the Night', artist: 'FM-84', album: 'Atlas', duration: '4:30', image: '/images/FM84.jpg', playlist_id: 'pl_2' },
    { id: 'tr_5', name: 'On the Run', artist: 'Timecop1983', album: 'Reflections', duration: '4:15', image: '/images/tr-5.jpg', playlist_id: 'pl_2' },
    { id: 'tr_6', name: 'One More Time', artist: 'Daft Punk', album: 'Discovery', duration: '5:20', image: '/images/DaftPunk.jpg', playlist_id: 'pl_6' },
    { id: 'tr_7', name: 'Get Lucky', artist: 'Daft Punk', album: 'RAM', duration: '6:09', image: '/images/GetLucky.jpg', playlist_id: 'pl_6' },
    { id: 'tr_8', name: 'Blinding Lights', artist: 'The Weeknd', album: 'After Hours', duration: '3:20', image: '/images/tr-8.jpg', playlist_id: 'pl_6' },
    { id: 'tr_9', name: 'Starboy', artist: 'The Weeknd', album: 'Starboy', duration: '3:50', image: '/images/tr-9.jpg', playlist_id: 'pl_7' },
    { id: 'tr_10', name: 'The Less I Know The Better', artist: 'Tame Impala', album: 'Currents', duration: '3:36', image: '/images/TameImpala.jpg', playlist_id: 'pl_9' },
    { id: 'tr_11', name: 'Do I Wanna Know?', artist: 'Arctic Monkeys', album: 'AM', duration: '4:32', image: '/images/ArcticMonkeys.jpg', playlist_id: 'pl_8' },
    { id: 'tr_12', name: 'Heat Waves', artist: 'Glass Animals', album: 'Dreamland', duration: '3:58', image: '/images/HeatWaves.jpg', playlist_id: 'pl_9' },
    { id: 'tr_13', name: 'Feel Good Inc.', artist: 'Gorillaz', album: 'Demon Days', duration: '3:41', image: '/images/tr-13.jpg', playlist_id: 'pl_8' },
    { id: 'tr_14', name: 'Harder, Better, Faster, Stronger', artist: 'Daft Punk', album: 'Discovery', duration: '3:44', image: '/images/DaftPunk.jpg', playlist_id: 'pl_12' },
    { id: 'tr_15', name: 'Save Your Tears', artist: 'The Weeknd', album: 'After Hours', duration: '3:35', image: '/images/tr-15.jpg', playlist_id: 'pl_6' }
  ])

  // Payment Methods
  const payment_methods = ref([
    { id: 'pm_1', type: 'Visa', last4: '4242', expiry: '12/25', name: 'Personal Card', is_active: true },
    { id: 'pm_2', type: 'Mastercard', last4: '8888', expiry: '01/24', name: 'Business Card', is_active: true },
    { id: 'pm_3', type: 'PayPal', last4: 'user@email.com', expiry: 'N/A', name: 'PayPal Account', is_active: false },
    { id: 'pm_4', type: 'Visa', last4: '1111', expiry: '05/26', name: 'Backup Card', is_active: false }
  ])

  return {
    categories,
    artists,
    albums,
    playlists,
    tracks,
    payment_methods
  }
}, {
  persist: {
    storage: sessionStorage
  }
})