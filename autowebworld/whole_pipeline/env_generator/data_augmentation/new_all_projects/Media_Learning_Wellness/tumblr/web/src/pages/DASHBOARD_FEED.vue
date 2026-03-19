<template>
  <div class="min-h-screen bg-slate-900 text-white pb-20 md:pl-20 lg:pl-64">
    <!-- Permission Interceptor -->
    <PermissionModal />

    <!-- Sidebar Navigation (Desktop) -->
    <aside class="hidden md:flex flex-col fixed left-0 top-0 h-full w-20 lg:w-64 border-r border-slate-800 bg-slate-900 z-40">
      <div class="p-4 lg:p-6 text-2xl font-bold lg:text-3xl tracking-tighter mb-4">
        <span class="hidden lg:inline">tumblr</span>
        <span class="lg:hidden">t</span>
      </div>
      
      <nav class="flex-1 space-y-2 px-2 lg:px-4">
        <button id="nav-home" @click="goHome" class="flex items-center gap-4 w-full p-3 rounded-full hover:bg-slate-800 transition-colors text-slate-200">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>
          <span class="hidden lg:inline font-bold">Home</span>
        </button>
        <button id="nav-explore" @click="goExplore" class="flex items-center gap-4 w-full p-3 rounded-full hover:bg-slate-800 transition-colors text-slate-200">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" /></svg>
          <span class="hidden lg:inline font-bold">Explore</span>
        </button>
        <button id="nav-messages" @click="goMessages" class="flex items-center gap-4 w-full p-3 rounded-full hover:bg-slate-800 transition-colors text-slate-200">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg>
          <span class="hidden lg:inline font-bold">Messages</span>
        </button>
        <button id="nav-account-settings" @click="goSettings" class="flex items-center gap-4 w-full p-3 rounded-full hover:bg-slate-800 transition-colors text-slate-200">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg>
          <span class="hidden lg:inline font-bold">Account</span>
        </button>
      </nav>

      <div class="p-4">
        <button id="create-post-text" @click="goCompose" class="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 rounded-full transition-all shadow-lg shadow-blue-500/20 flex items-center justify-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>
          <span class="hidden lg:inline">Create</span>
        </button>
      </div>
    </aside>

    <!-- Main Feed Area -->
    <main class="max-w-[600px] mx-auto pt-6 px-4">
      
      <!-- Search & Filter Header -->
      <div class="mb-8 space-y-4">
        <!-- Search Bar -->
        <div class="relative">
          <input 
            id="feed-search-input"
            type="text"
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            placeholder="Search your dashboard..."
            class="w-full bg-slate-800 border-none rounded-full py-3 px-12 text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
          />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 absolute left-4 top-3.5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
        </div>

        <!-- Filters Row -->
        <div class="flex flex-wrap gap-4 items-center bg-slate-800/50 p-4 rounded-xl backdrop-blur-sm border border-slate-700/50">
          <!-- Checkbox Filter -->
          <label class="flex items-center gap-2 cursor-pointer select-none">
            <input 
              id="filter-following-checkbox"
              type="checkbox"
              v-model="filterFollowing"
              class="w-5 h-5 rounded border-slate-500 text-blue-500 focus:ring-blue-500 bg-slate-700"
            />
            <span class="text-sm font-medium text-slate-300">Following Only</span>
          </label>

          <!-- Sort Dropdown -->
          <div class="relative">
            <button 
              id="sort-dropdown"
              @click="sortOpen = !sortOpen"
              class="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white"
            >
              Sort by: <span class="text-blue-400">{{ currentSort === 'newest' ? 'Newest' : 'Oldest' }}</span>
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
            </button>
            
            <div v-if="sortOpen" class="absolute top-full left-0 mt-2 w-32 bg-slate-800 rounded-lg shadow-xl border border-slate-700 overflow-hidden z-20">
              <div id="sort-option-newest" @click="setSort('newest')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Newest</div>
              <div id="sort-option-oldest" @click="setSort('oldest')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Oldest</div>
            </div>
          </div>

          <!-- Slider Filter -->
          <div class="flex-1 min-w-[200px] flex items-center gap-3">
            <span class="text-xs text-slate-400">Notes &gt; {{ filterNotes }}</span>
            <input 
              id="filter-time-slider"
              type="range"
              min="0"
              max="1000"
              step="10"
              v-model="filterNotes"
              class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-500"
            />
          </div>
        </div>
      </div>

      <!-- Feed List -->
      <div id="feed-list-container" class="space-y-6">
        <!-- Empty State -->
        <div v-if="filteredPosts.length === 0" class="text-center py-20 text-slate-500">
          <p class="text-xl">No posts found.</p>
          <p class="text-sm">Try adjusting your filters.</p>
        </div>

        <!-- Post Cards -->
        <div 
          v-for="post in filteredPosts" 
          :key="post.id"
          :id="`post-${post.id}`"
          :class="[
            'bg-slate-800 rounded-lg overflow-hidden border border-slate-700 transition-all hover:border-slate-600',
            hasSearched && post.id === matchedPostId ? 'post-card-matched ring-2 ring-blue-500' : 'post-card-visible',
            filtersApplied ? 'post-card-filtered' : ''
          ]"
          @click="openPost(post.id)"
        >
          <!-- Post Header -->
          <div class="p-4 flex items-center gap-3 border-b border-slate-700/50">
            <img :src="getBlog(post.blog_id)?.avatar" class="w-10 h-10 rounded-lg object-cover bg-slate-700" alt="Avatar" />
            <div>
              <div class="font-bold text-white hover:underline cursor-pointer">{{ getBlog(post.blog_id)?.name }}</div>
              <div class="text-xs text-slate-400">{{ formatDate(post.date) }}</div>
            </div>
          </div>

          <!-- Post Content based on Type -->
          <div class="cursor-pointer">
            <!-- Photo Post -->
            <div v-if="post.type === 'photo'">
              <img :src="post.content" class="w-full h-auto object-cover max-h-[600px]" alt="Post image" />
              <div v-if="post.caption" class="p-4 text-slate-200" v-html="post.caption"></div>
            </div>

            <!-- Text Post -->
            <div v-else-if="post.type === 'text'" class="p-6">
              <h3 v-if="post.title" class="text-2xl font-bold mb-3 font-serif">{{ post.title }}</h3>
              <p class="text-slate-200 leading-relaxed whitespace-pre-line">{{ post.content }}</p>
            </div>

            <!-- Quote Post -->
            <div v-else-if="post.type === 'quote'" class="p-8 bg-serif font-serif">
              <blockquote class="text-2xl md:text-3xl italic text-white leading-tight mb-4">
                {{ post.content }}
              </blockquote>
              <cite class="block text-right text-slate-400 not-italic">— {{ post.source }}</cite>
            </div>

            <!-- Audio Post (Simplified UI) -->
            <div v-else-if="post.type === 'audio'" class="p-4 bg-slate-900 flex items-center gap-4">
               <div class="w-20 h-20 bg-slate-700 rounded flex items-center justify-center">🎵</div>
               <div>
                 <div class="font-bold text-lg">{{ post.title }}</div>
                 <div class="text-slate-400">{{ post.content }}</div>
               </div>
            </div>
          </div>

          <!-- Post Footer / Tags / Notes -->
          <div class="p-4 pt-2">
            <div class="flex flex-wrap gap-2 mb-3">
              <span v-for="tag in post.tags" :key="tag" class="text-slate-400 hover:underline text-sm cursor-pointer">{{ tag }}</span>
            </div>
            
            <div class="flex justify-between items-center text-slate-400 text-sm border-t border-slate-700/50 pt-3 mt-2">
              <div class="font-bold text-white">{{ post.notes }} notes</div>
              <div class="flex gap-4">
                 <button class="hover:text-pink-500 transition-colors">
                   <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>
                 </button>
                 <button class="hover:text-green-500 transition-colors">
                   <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
                 </button>
              </div>
            </div>
          </div>
        </div>
      </div>

    </main>

    <!-- Mobile Bottom Navigation -->
    <nav class="md:hidden fixed bottom-0 left-0 w-full bg-slate-900 border-t border-slate-800 flex justify-around p-3 z-50">
      <button id="nav-home" @click="goHome" class="p-2 text-white"><svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg></button>
      <button id="nav-explore" @click="goExplore" class="p-2 text-slate-400 hover:text-white"><svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg></button>
      <button id="create-post-text" @click="goCompose" class="p-3 bg-blue-500 rounded-full text-white -mt-8 shadow-lg border-4 border-slate-900"><svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" /></svg></button>
      <button id="nav-messages" @click="goMessages" class="p-2 text-slate-400 hover:text-white"><svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" /></svg></button>
      <button id="nav-account-settings" @click="goSettings" class="p-2 text-slate-400 hover:text-white"><svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg></button>
    </nav>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import PermissionModal from '../components/PermissionModal.vue'

export default {
  name: 'DASHBOARD_FEED',
  components: {
    PermissionModal
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // Filter & Sort State
    const searchQuery = ref('')
    const filterFollowing = ref(false)
    const filterNotes = ref(0)
    const currentSort = ref('newest')
    const sortOpen = ref(false)

    // FSM State Mappers
    const hasSearched = computed(() => store.dashboard_feed_has_searched)
    const matchedPostId = computed(() => store.matched_post_id)
    const filtersApplied = computed(() => store.dashboard_feed_filters_applied)

    // Helper to get blog info
    const getBlog = (id) => dataStore.blogs.find(b => b.id === id)

    // Date formatting
    const formatDate = (isoString) => {
      const date = new Date(isoString)
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    }

    // Filter Logic
    const filteredPosts = computed(() => {
      let result = [...dataStore.posts]

      // 1. Filter by Following
      if (filterFollowing.value) {
        // Find blogs I follow
        const followedBlogIds = dataStore.blogs.filter(b => b.following).map(b => b.id)
        result = result.filter(p => followedBlogIds.includes(p.blog_id))
      }

      // 2. Filter by Notes Count (Slider)
      if (filterNotes.value > 0) {
        result = result.filter(p => p.notes > filterNotes.value)
      }

      // 3. Search Query
      if (searchQuery.value) {
        const query = searchQuery.value.toLowerCase()
        result = result.filter(p => 
          (p.title && p.title.toLowerCase().includes(query)) ||
          (p.content && p.content.toLowerCase().includes(query)) ||
          (p.tags && p.tags.some(t => t.toLowerCase().includes(query)))
        )
      }

      // 4. Sort
      result.sort((a, b) => {
        const dateA = new Date(a.date)
        const dateB = new Date(b.date)
        return currentSort.value === 'newest' ? dateB - dateA : dateA - dateB
      })

      return result
    })

    // Actions
    const handleSearch = () => {
      store.dashboard_feed_has_searched = true
      // Matched post ID logic for FSM - grab first result if any
      if (filteredPosts.value.length > 0) {
        store.matched_post_id = filteredPosts.value[0].id
      }
    }

    const setSort = (type) => {
      currentSort.value = type
      sortOpen.value = false
      store.dashboard_feed_filters_applied = true
    }

    // Watchers for filter changes to update FSM state
    const updateFilterState = () => {
      store.dashboard_feed_filters_applied = true
    }

    // Navigation
    const goHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }
    const goExplore = async () => {
      store.currentPageId = 'EXPLORE'
      await router.push({ name: 'EXPLORE' })
    }
    const goMessages = async () => {
      store.currentPageId = 'MESSAGES_INBOX'
      await router.push({ name: 'MESSAGES_INBOX' })
    }
    const goSettings = async () => {
      store.currentPageId = 'ACCOUNT_SETTINGS'
      await router.push({ name: 'ACCOUNT_SETTINGS' })
    }
    const goCompose = async () => {
      store.currentPageId = 'COMPOSE_TEXT_POST'
      await router.push({ name: 'COMPOSE_TEXT_POST' })
    }

    const openPost = async (id) => {
      // Logic from FSM: 
      // If searched -> OPEN_MATCHED_POST
      // If filtered -> OPEN_FILTERED_POST
      // Else -> OPEN_ANY_POST
      
      // We set the state and navigate
      store.selected_post_id = id
      
      // Clear flags as per effects
      if (hasSearched.value) store.dashboard_feed_has_searched = null
      if (filtersApplied.value) store.dashboard_feed_filters_applied = null

      store.currentPageId = 'POST_DETAIL'
      await router.push({ name: 'POST_DETAIL', params: { id } })
    }

    return {
      store,
      searchQuery,
      filterFollowing,
      filterNotes,
      currentSort,
      sortOpen,
      filteredPosts,
      hasSearched,
      matchedPostId,
      filtersApplied,
      handleSearch,
      setSort,
      getBlog,
      formatDate,
      goHome,
      goExplore,
      goMessages,
      goSettings,
      goCompose,
      openPost
    }
  },
  watch: {
    filterFollowing() { this.store.dashboard_feed_filters_applied = true },
    filterNotes() { this.store.dashboard_feed_filters_applied = true }
  }
}
</script>