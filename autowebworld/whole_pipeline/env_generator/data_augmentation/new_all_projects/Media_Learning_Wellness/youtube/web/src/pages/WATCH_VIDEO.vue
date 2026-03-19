<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col">
    <!-- Navbar -->
    <nav class="sticky top-0 z-50 bg-[#0F0F0F]/95 backdrop-blur border-b border-gray-800 px-4 h-14 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <div id="logo-home" @click="goHome" class="flex items-center gap-1 cursor-pointer">
          <div class="bg-red-600 text-white rounded-lg p-1">
            <svg class="w-6 h-6 fill-current" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/></svg>
          </div>
          <span class="text-xl font-bold tracking-tight">YouTube</span>
        </div>
      </div>
      
      <!-- Back Navigation Actions (Contextual) -->
      <div class="flex gap-4 text-sm font-medium">
        <button 
          id="back-to-results" 
          @click="goBackResults"
          class="flex items-center gap-2 hover:text-gray-300 transition-colors"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          Back to Results
        </button>
        <button 
          id="back-to-trending" 
          @click="goBackTrending"
          class="flex items-center gap-2 hover:text-gray-300 transition-colors"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"></path></svg>
          Back to Trending
        </button>
      </div>
    </nav>

    <main class="flex-1 max-w-[1800px] mx-auto w-full p-4 lg:p-6 grid grid-cols-1 lg:grid-cols-3 gap-6">
      <!-- Left Column: Player & Info -->
      <div class="lg:col-span-2">
        <!-- Video Player Simulation -->
        <div class="w-full aspect-video bg-black rounded-xl overflow-hidden relative group shadow-2xl mb-4">
          <img :src="video?.image" class="w-full h-full object-cover opacity-80" />
          
          <!-- Play Button Overlay -->
          <div class="absolute inset-0 flex items-center justify-center">
            <div class="w-16 h-16 bg-red-600 rounded-full flex items-center justify-center cursor-pointer hover:scale-110 transition-transform shadow-lg">
              <svg class="w-8 h-8 text-white ml-1" fill="currentColor" viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
            </div>
          </div>

          <!-- Controls Bar (Visual only) -->
          <div class="absolute bottom-0 left-0 right-0 h-12 bg-gradient-to-t from-black/80 to-transparent flex items-center px-4 gap-4">
             <div class="text-white text-xs">0:00 / {{ formatDuration(video?.duration || 0) }}</div>
             <div class="flex-1 h-1 bg-gray-600 rounded-full"><div class="w-1/3 h-full bg-red-600 rounded-full relative"><div class="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-red-600 rounded-full transform scale-0 group-hover:scale-100 transition-transform"></div></div></div>
          </div>
        </div>

        <!-- Video Info -->
        <h1 class="text-xl md:text-2xl font-bold mb-2">{{ video?.title }}</h1>
        
        <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-gray-800">
          <!-- Channel Info -->
          <div class="flex items-center gap-3">
             <div class="w-10 h-10 rounded-full bg-gray-600 overflow-hidden">
                <!-- Avatar placeholder -->
                <svg class="w-full h-full text-gray-400 p-1" fill="currentColor" viewBox="0 0 24 24"><path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/></svg>
             </div>
             <div>
               <h3 class="font-bold hover:text-white cursor-pointer">{{ video?.channel }}</h3>
               <p class="text-xs text-gray-400">1.2M subscribers</p>
             </div>
             
             <!-- Subscribe Button -->
             <button 
               id="subscribe-button" 
               @click="goSubscribe"
               class="ml-4 bg-white text-black px-4 py-2 rounded-full font-medium text-sm hover:bg-gray-200 transition-colors"
             >
               Subscribe
             </button>
          </div>

          <!-- Actions Bar -->
          <div class="flex items-center gap-2">
            <!-- Like Button Area -->
            <div class="flex items-center bg-[#272727] rounded-full overflow-hidden">
               <button 
                 id="like-button" 
                 @click="toggleLike"
                 class="flex items-center gap-2 px-4 py-2 hover:bg-[#3F3F3F] transition-colors border-r border-gray-700"
                 :class="{'text-blue-400': store.is_liked}"
               >
                 <svg class="w-5 h-5" :fill="store.is_liked ? 'currentColor' : 'none'" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"></path></svg>
                 <span>{{ store.is_liked ? 'Liked' : 'Like' }}</span>
               </button>
               <button class="px-4 py-2 hover:bg-[#3F3F3F] transition-colors">
                 <svg class="w-5 h-5 transform rotate-180" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"></path></svg>
               </button>
            </div>
            
            <button class="flex items-center gap-2 bg-[#272727] px-4 py-2 rounded-full hover:bg-[#3F3F3F] transition-colors">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"></path></svg>
              Share
            </button>
          </div>
        </div>

        <!-- Like Success Banner (Visible only after liking) -->
        <div 
          v-if="store.is_liked"
          id="like-confirm-banner"
          @click="goLikeSuccess"
          class="mt-4 bg-green-900/30 border border-green-500/50 rounded-xl p-4 flex items-center justify-between cursor-pointer hover:bg-green-900/50 transition-colors"
        >
          <div class="flex items-center gap-3">
            <div class="bg-green-500 p-1 rounded-full text-black">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <span class="font-medium text-green-400">Added to Liked Videos</span>
          </div>
          <div class="flex items-center text-sm text-green-400 font-bold">
            VIEW LIST <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
          </div>
        </div>

        <!-- Description Box -->
        <div class="mt-4 bg-[#272727] rounded-xl p-4 text-sm cursor-pointer hover:bg-[#333] transition-colors">
          <div class="font-bold mb-2">{{ video?.views }} views • {{ video?.date }}</div>
          <p class="text-gray-200">
            This is a comprehensive description of the video content. It includes details about the topic, links to resources, and other relevant information provided by the creator.
          </p>
          <div class="mt-2 text-gray-400 font-medium">Show more</div>
        </div>

        <!-- Comments Section -->
        <div class="mt-8">
          <div class="flex items-center gap-6 mb-6">
            <h3 class="text-xl font-bold">Comments</h3>
            <div class="flex items-center gap-2 text-sm font-medium text-gray-400 cursor-pointer hover:text-white">
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h7"></path></svg>
              Sort by
            </div>
          </div>

          <!-- Add Comment -->
          <div class="flex gap-4 mb-8">
            <div class="w-10 h-10 rounded-full bg-purple-600 flex items-center justify-center text-sm font-bold flex-shrink-0">U</div>
            <div class="flex-1">
              <input 
                id="comment-input"
                v-model="commentText"
                @input="handleCommentInput"
                type="text" 
                placeholder="Add a comment..."
                class="w-full bg-transparent border-b border-gray-700 pb-2 focus:border-white focus:outline-none transition-colors"
              >
              <div class="flex justify-end gap-3 mt-3" v-if="commentText">
                <button @click="commentText = ''" class="px-4 py-2 rounded-full hover:bg-white/10 font-medium text-sm">Cancel</button>
                <button 
                  id="comment-submit-button"
                  @click="submitComment"
                  class="bg-[#3EA6FF] text-black px-4 py-2 rounded-full font-medium text-sm hover:bg-blue-400 transition-colors"
                  :disabled="!commentText"
                >
                  Comment
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Right Column: Recommendations -->
      <div class="hidden lg:block space-y-3">
        <div v-for="i in 10" :key="i" class="flex gap-2 cursor-pointer group">
          <div class="w-40 aspect-video bg-gray-800 rounded-lg overflow-hidden relative">
            <div class="absolute bottom-1 right-1 bg-black/80 text-white text-[10px] px-1 rounded">10:35</div>
          </div>
          <div class="flex-1">
            <h4 class="font-bold text-sm line-clamp-2 mb-1 group-hover:text-blue-400">Recommended Video Title That Is Quite Long</h4>
            <div class="text-xs text-gray-400">Channel Name</div>
            <div class="text-xs text-gray-400">50K views • 2 days ago</div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'WATCH_VIDEO',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const commentText = ref('')

    // Get current video
    const video = computed(() => {
      return dataStore.videos.find(v => v.id === store.selected_video_id) || dataStore.videos[0]
    })

    const formatDuration = (seconds) => {
      const m = Math.floor(seconds / 60)
      const s = seconds % 60
      return `${m}:${s.toString().padStart(2, '0')}`
    }

    // Actions
    const goHome = () => {
      store.currentPageId = 'HOME'
      router.push({ name: 'HOME' })
    }

    const goBackResults = () => {
      store.currentPageId = 'SEARCH_RESULTS'
      router.push({ name: 'SEARCH_RESULTS' })
    }

    const goBackTrending = () => {
      store.currentPageId = 'TRENDING'
      router.push({ name: 'TRENDING' })
    }

    const toggleLike = () => {
      store.is_liked = true
    }

    const goLikeSuccess = () => {
      if (store.is_liked) {
        store.currentPageId = 'WATCH_LIKE_SUCCESS'
        router.push({ name: 'WATCH_LIKE_SUCCESS' })
      }
    }

    const goSubscribe = () => {
      store.currentPageId = 'CHANNEL_SUBSCRIBE_CONFIRM'
      router.push({ name: 'CHANNEL_SUBSCRIBE_CONFIRM' })
    }

    const handleCommentInput = () => {
      if (commentText.value.length > 0) {
        store.comment_text_entered = 'typed'
      }
    }

    const submitComment = () => {
      store.currentPageId = 'WATCH_COMMENT_SUCCESS'
      router.push({ name: 'WATCH_COMMENT_SUCCESS' })
    }

    return {
      store,
      video,
      commentText,
      formatDuration,
      goHome,
      goBackResults,
      goBackTrending,
      toggleLike,
      goLikeSuccess,
      goSubscribe,
      handleCommentInput,
      submitComment
    }
  }
}
</script>