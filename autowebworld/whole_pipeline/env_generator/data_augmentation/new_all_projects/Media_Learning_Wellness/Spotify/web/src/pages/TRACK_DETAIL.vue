<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-playlist" @click="handleBackPlaylist" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
         <span>Back to Playlist</span>
      </div>
    </aside>

    <main class="flex-1 flex flex-col relative bg-gradient-to-b from-[#535353] to-[#121212] overflow-hidden justify-center items-center">
      <div class="max-w-4xl w-full p-8 flex flex-col md:flex-row items-center md:items-end gap-8">
        <!-- Album Art -->
        <div class="w-64 h-64 md:w-80 md:h-80 shadow-2xl relative group">
           <img :src="track?.image" class="w-full h-full object-cover shadow-lg rounded-md" />
        </div>

        <!-- Info -->
        <div class="flex-1 text-center md:text-left">
           <h2 class="text-sm font-bold uppercase tracking-widest mb-2 text-white/80">Song</h2>
           <h1 class="text-5xl md:text-7xl font-bold mb-6 tracking-tight">{{ track?.name }}</h1>
           <div class="flex flex-col md:flex-row items-center md:items-center gap-2 text-lg font-medium text-white/90">
              <div 
                id="track-artist-link"
                class="hover:underline cursor-pointer flex items-center gap-2"
                @click="handleOpenArtist"
              >
                 <img src="/images/Artist.jpg" class="w-6 h-6 rounded-full" /> <!-- Mock artist img -->
                 {{ track?.artist }}
              </div>
              <span class="hidden md:inline">•</span>
              <div 
                id="track-album-link" 
                class="hover:underline cursor-pointer text-[#B3B3B3] hover:text-white"
                @click="handleOpenAlbum"
              >
                 {{ track?.album }}
              </div>
              <span class="hidden md:inline">•</span>
              <span class="text-[#B3B3B3]">{{ track?.duration }}</span>
           </div>
        </div>
      </div>

      <!-- Controls -->
      <div class="w-full max-w-4xl px-8 mt-8 flex items-center justify-center md:justify-start gap-8">
         <button class="w-16 h-16 bg-[#1DB954] rounded-full flex items-center justify-center hover:scale-105 hover:bg-[#1ed760] transition-transform shadow-lg">
            <svg class="w-8 h-8 text-black fill-current ml-1" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
         </button>

         <button 
           id="track-like-button" 
           @click="handleToggleLike" 
           class="text-[#B3B3B3] hover:text-white transition-colors transform hover:scale-110"
           :class="{'text-[#1DB954]': isLiked}"
         >
            <svg v-if="isLiked" class="w-10 h-10 fill-current" viewBox="0 0 24 24"><path d="M12 21.35l-1.45-1.32C5.4 15.36 2 12.28 2 8.5 2 5.42 4.42 3 7.5 3c1.74 0 3.41.81 4.5 2.09C13.09 3.81 14.76 3 16.5 3 19.58 3 22 5.42 22 8.5c0 3.78-3.4 6.86-8.55 11.54L12 21.35z"/></svg>
            <svg v-else class="w-10 h-10" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"/></svg>
         </button>

         <button class="text-[#B3B3B3] hover:text-white transition-colors">
            <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 24 24"><path d="M12 8c1.1 0 2-.9 2-2s-.9-2-2-2-2 .9-2 2 .9 2 2 2zm0 2c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zm0 6c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"/></svg>
         </button>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import { useRouter, useRoute } from 'vue-router'

export default {
  name: 'TRACK_DETAIL',
  setup() {
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const router = useRouter()
    const route = useRoute()

    const trackId = route.params.id || store.selected_track_id
    const track = computed(() => dataStore.tracks.find(t => t.id === trackId))
    const isLiked = computed(() => store.track_liked === true)

    const handleBackPlaylist = async () => {
      // Logic to go back to correct previous page is simplified here to PLAYLIST_DETAIL as per FSM primary flow
      // Or could check history
      store.setCurrentPageId('PLAYLIST_DETAIL')
      await router.push({ name: 'PLAYLIST_DETAIL', params: { id: track.value?.playlist_id } })
    }

    const handleToggleLike = () => {
      store.track_liked = true
    }

    const handleOpenArtist = async () => {
      store.selected_artist_id = 'art_1' // Mock artist ID
      store.setCurrentPageId('ARTIST_DETAIL')
      await router.push({ name: 'ARTIST_DETAIL', params: { id: 'art_1' } })
    }

    const handleOpenAlbum = async () => {
      store.selected_album_id = 'alb_1' // Mock album ID
      store.setCurrentPageId('ALBUM_DETAIL')
      await router.push({ name: 'ALBUM_DETAIL', params: { id: 'alb_1' } })
    }

    return {
      track,
      isLiked,
      handleBackPlaylist,
      handleToggleLike,
      handleOpenArtist,
      handleOpenAlbum
    }
  }
}
</script>