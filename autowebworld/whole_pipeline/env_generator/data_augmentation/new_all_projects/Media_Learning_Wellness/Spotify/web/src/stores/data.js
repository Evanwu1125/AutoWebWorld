import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // Categories (Browse)
  const categories = ref([
    { id: 'cat_pop', name: 'Pop', image: '/images/categories_cat_pop.jpg', color: '#8c1932' },
    { id: 'cat_hiphop', name: 'Hip-Hop', image: '/images/categories_cat_hiphop.jpg', color: '#ba5d07' },
    { id: 'cat_rock', name: 'Rock', image: '/images/categories_cat_rock.jpg', color: '#e61e32' },
    { id: 'cat_indie', name: 'Indie', image: '/images/categories_cat_indie.jpg', color: '#608108' },
    { id: 'cat_chill', name: 'Chill', image: '/images/categories_cat_chill.jpg', color: '#477d95' },
    { id: 'cat_workout', name: 'Workout', image: '/images/categories_cat_workout.jpg', color: '#777777' },
    { id: 'cat_party', name: 'Party', image: '/images/categories_cat_party.jpg', color: '#537aa1' },
    { id: 'cat_focus', name: 'Focus', image: '/images/categories_cat_focus.jpg', color: '#af2896' },
    { id: 'cat_sleep', name: 'Sleep', image: '/images/categories_cat_sleep.jpg', color: '#1e3264' },
    { id: 'cat_jazz', name: 'Jazz', image: '/images/categories_cat_jazz.jpg', color: '#7d4b32' }
  ])

  // Artists
  const artists = ref([
    { id: 'art_1', name: 'The Midnight', image: '/images/artists_art_1.jpg', followers: '1.2M' },
    { id: 'art_2', name: 'Gunship', image: '/images/artists_art_2.jpg', followers: '500K' },
    { id: 'art_3', name: 'FM-84', image: '/images/artists_art_3.jpg', followers: '350K' },
    { id: 'art_4', name: 'Timecop1983', image: '/images/artists_art_4.jpg', followers: '450K' },
    { id: 'art_5', name: 'Daft Punk', image: '/images/artists_art_5.jpg', followers: '15M' },
    { id: 'art_6', name: 'The Weeknd', image: '/images/artists_art_6.jpg', followers: '40M' },
    { id: 'art_7', name: 'Tame Impala', image: '/images/artists_art_7.jpg', followers: '8M' },
    { id: 'art_8', name: 'Arctic Monkeys', image: '/images/artists_art_8.jpg', followers: '12M' },
    { id: 'art_9', name: 'Glass Animals', image: '/images/artists_art_9.jpg', followers: '4M' },
    { id: 'art_10', name: 'Gorillaz', image: '/images/artists_art_10.jpg', followers: '9M' }
  ])

  // Albums
  const albums = ref([
    { id: 'alb_1', artist_id: 'art_1', name: 'Endless Summer', year: 2016, image: '/images/albums_alb_1.jpg' },
    { id: 'alb_2', artist_id: 'art_1', name: 'Nocturnal', year: 2017, image: '/images/albums_alb_2.jpg' },
    { id: 'alb_3', artist_id: 'art_2', name: 'Dark All Day', year: 2018, image: '/images/albums_alb_3.jpg' },
    { id: 'alb_4', artist_id: 'art_3', name: 'Atlas', year: 2016, image: '/images/albums_alb_4.jpg' },
    { id: 'alb_5', artist_id: 'art_5', name: 'Discovery', year: 2001, image: '/images/albums_alb_5.jpg' },
    { id: 'alb_6', artist_id: 'art_5', name: 'Random Access Memories', year: 2013, image: '/images/albums_alb_6.jpg' },
    { id: 'alb_7', artist_id: 'art_6', name: 'After Hours', year: 2020, image: '/images/albums_alb_7.jpg' },
    { id: 'alb_8', artist_id: 'art_6', name: 'Starboy', year: 2016, image: '/images/albums_alb_8.jpg' },
    { id: 'alb_9', artist_id: 'art_7', name: 'Currents', year: 2015, image: '/images/albums_alb_9.jpg' },
    { id: 'alb_10', artist_id: 'art_8', name: 'AM', year: 2013, image: '/images/albums_alb_10.jpg' }
  ])

  // Playlists
  const playlists = ref([
    { id: 'pl_1', name: 'Synthwave Essentials', description: 'Retro futuristic sounds.', owner: 'Spotify', image: '/images/playlists_pl_1.jpg', featured: true, category_id: 'cat_pop', downloaded: true },
    { id: 'pl_2', name: 'Night Drive', description: 'Music for late night drives.', owner: 'Spotify', image: '/images/playlists_pl_2.jpg', featured: true, category_id: 'cat_chill', downloaded: true },
    { id: 'pl_3', name: 'Cyberpunk 2077', description: 'Official Soundtrack', owner: 'CDPR', image: '/images/playlists_pl_3.jpg', featured: false, category_id: 'cat_rock', downloaded: false },
    { id: 'pl_4', name: 'Neon Lights', description: 'Glow in the dark beats.', owner: 'Spotify', image: '/images/playlists_pl_4.jpg', featured: true, category_id: 'cat_pop', downloaded: true },
    { id: 'pl_5', name: 'Retro Gaming', description: '8-bit nostalgia.', owner: 'GamerOne', image: '/images/playlists_pl_5.jpg', featured: false, category_id: 'cat_focus', downloaded: false },
    { id: 'pl_6', name: 'Today\'s Top Hits', description: 'The hottest tracks right now.', owner: 'Spotify', image: '/images/playlists_pl_6.jpg', featured: true, category_id: 'cat_pop', downloaded: true },
    { id: 'pl_7', name: 'Rap Caviar', description: 'New music from Drake, Travis Scott...', owner: 'Spotify', image: '/images/playlists_pl_7.jpg', featured: true, category_id: 'cat_hiphop', downloaded: true },
    { id: 'pl_8', name: 'Rock Classics', description: 'Rock legends & anthems.', owner: 'Spotify', image: '/images/playlists_pl_8.jpg', featured: false, category_id: 'cat_rock', downloaded: false },
    { id: 'pl_9', name: 'Indie Pop', description: 'The best new indie pop.', owner: 'Spotify', image: '/images/playlists_pl_9.jpg', featured: false, category_id: 'cat_indie', downloaded: false },
    { id: 'pl_10', name: 'Deep Focus', description: 'Keep calm and focus.', owner: 'Spotify', image: '/images/playlists_pl_10.jpg', featured: true, category_id: 'cat_focus', downloaded: true },
    { id: 'pl_11', name: 'Sleep', description: 'Music to help you sleep.', owner: 'Spotify', image: '/images/playlists_pl_11.jpg', featured: false, category_id: 'cat_sleep', downloaded: true },
    { id: 'pl_12', name: 'Workout Beast', description: 'Get pumped.', owner: 'GymRat', image: '/images/playlists_pl_12.jpg', featured: false, category_id: 'cat_workout', downloaded: false },
    { id: 'pl_13', name: 'Party Anthems', description: 'All the hits.', owner: 'PartyBoii', image: '/images/playlists_pl_13.jpg', featured: true, category_id: 'cat_party', downloaded: true },
    { id: 'pl_14', name: 'Jazz Vibes', description: 'Smooth jazz for relaxing.', owner: 'JazzCat', image: '/images/playlists_pl_14.jpg', featured: false, category_id: 'cat_jazz', downloaded: false },
    { id: 'pl_15', name: 'Discover Weekly', description: 'Your weekly mixtape.', owner: 'Spotify', image: '/images/playlists_pl_15.jpg', featured: false, category_id: 'cat_pop', downloaded: true }
  ])

  // Tracks
  const tracks = ref([
    { id: 'tr_1', name: 'Sunset', artist: 'The Midnight', album: 'Endless Summer', duration: '5:26', image: '/images/tracks_tr_1.jpg', playlist_id: 'pl_1' },
    { id: 'tr_2', name: 'Vampires', artist: 'The Midnight', album: 'Nocturnal', duration: '5:17', image: '/images/tracks_tr_2.jpg', playlist_id: 'pl_1' },
    { id: 'tr_3', name: 'Tech Noir', artist: 'Gunship', album: 'Gunship', duration: '4:57', image: '/images/tracks_tr_3.jpg', playlist_id: 'pl_1' },
    { id: 'tr_4', name: 'Running in the Night', artist: 'FM-84', album: 'Atlas', duration: '4:30', image: '/images/tracks_tr_4.jpg', playlist_id: 'pl_2' },
    { id: 'tr_5', name: 'On the Run', artist: 'Timecop1983', album: 'Reflections', duration: '4:15', image: '/images/tracks_tr_5.jpg', playlist_id: 'pl_2' },
    { id: 'tr_6', name: 'One More Time', artist: 'Daft Punk', album: 'Discovery', duration: '5:20', image: '/images/tracks_tr_6.jpg', playlist_id: 'pl_6' },
    { id: 'tr_7', name: 'Get Lucky', artist: 'Daft Punk', album: 'RAM', duration: '6:09', image: '/images/tracks_tr_7.jpg', playlist_id: 'pl_6' },
    { id: 'tr_8', name: 'Blinding Lights', artist: 'The Weeknd', album: 'After Hours', duration: '3:20', image: '/images/tracks_tr_8.jpg', playlist_id: 'pl_6' },
    { id: 'tr_9', name: 'Starboy', artist: 'The Weeknd', album: 'Starboy', duration: '3:50', image: '/images/tracks_tr_9.jpg', playlist_id: 'pl_7' },
    { id: 'tr_10', name: 'The Less I Know The Better', artist: 'Tame Impala', album: 'Currents', duration: '3:36', image: '/images/tracks_tr_10.jpg', playlist_id: 'pl_9' },
    { id: 'tr_11', name: 'Do I Wanna Know?', artist: 'Arctic Monkeys', album: 'AM', duration: '4:32', image: '/images/tracks_tr_11.jpg', playlist_id: 'pl_8' },
    { id: 'tr_12', name: 'Heat Waves', artist: 'Glass Animals', album: 'Dreamland', duration: '3:58', image: '/images/tracks_tr_12.jpg', playlist_id: 'pl_9' },
    { id: 'tr_13', name: 'Feel Good Inc.', artist: 'Gorillaz', album: 'Demon Days', duration: '3:41', image: '/images/tracks_tr_13.jpg', playlist_id: 'pl_8' },
    { id: 'tr_14', name: 'Harder, Better, Faster, Stronger', artist: 'Daft Punk', album: 'Discovery', duration: '3:44', image: '/images/tracks_tr_14.jpg', playlist_id: 'pl_12' },
    { id: 'tr_15', name: 'Save Your Tears', artist: 'The Weeknd', album: 'After Hours', duration: '3:35', image: '/images/tracks_tr_15.jpg', playlist_id: 'pl_6' }
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