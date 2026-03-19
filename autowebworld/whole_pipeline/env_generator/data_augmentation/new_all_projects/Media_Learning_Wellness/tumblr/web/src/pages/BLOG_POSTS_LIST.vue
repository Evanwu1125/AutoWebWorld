<template>
  <div v-if="blog" class="min-h-screen bg-slate-900 text-white pb-20">
    <!-- Header -->
    <header class="sticky top-0 z-30 bg-slate-900/90 backdrop-blur-md border-b border-slate-800 p-4 flex items-center gap-4">
      <button 
        id="blog-posts-back-overview" 
        @click="goBackOverview"
        class="p-2 hover:bg-slate-800 rounded-full transition-colors"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
      </button>
      
      <div class="flex items-center gap-3">
        <img :src="blog.avatar" class="w-8 h-8 rounded-full" />
        <h1 class="font-bold text-lg">{{ blog.name }}'s Posts</h1>
      </div>
    </header>

    <!-- Filters -->
    <div class="max-w-[600px] mx-auto p-4 space-y-4">
      <!-- Search -->
      <div class="relative">
        <input 
          id="blog-posts-search-input"
          type="text"
          v-model="searchQuery"
          @keypress.enter="handleSearch"
          placeholder="Search this blog..."
          class="w-full bg-slate-800 border-none rounded-full py-3 px-12 text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
        />
        <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 absolute left-4 top-3.5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
      </div>

      <!-- Controls -->
      <div class="flex flex-wrap gap-4 items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
        <!-- Checkbox -->
        <label class="flex items-center gap-2 cursor-pointer select-none">
          <input 
            id="filter-text-posts-checkbox"
            type="checkbox"
            v-model="filterTextOnly"
            class="w-5 h-5 rounded border-slate-500 text-blue-500 focus:ring-blue-500 bg-slate-700"
          />
          <span class="text-sm font-medium text-slate-300">Text posts only</span>
        </label>

        <!-- Sort -->
        <div class="relative">
            <button 
              id="blog-posts-sort-dropdown"
              @click="sortOpen = !sortOpen"
              class="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white"
            >
              Sort: <span class="text-blue-400">{{ currentSort === 'newest' ? 'Newest' : 'Oldest' }}</span>
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
            </button>
            
            <div v-if="sortOpen" class="absolute top-full left-0 mt-2 w-32 bg-slate-800 rounded-lg shadow-xl border border-slate-700 overflow-hidden z-20">
              <div id="blog-posts-sort-option-newest" @click="setSort('newest')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Newest</div>
              <div id="blog-posts-sort-option-oldest" @click="setSort('oldest')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Oldest</div>
            </div>
          </div>

          <!-- Slider (Date/Time Filter Mock) -->
          <div class="flex-1 min-w-[200px] flex items-center gap-3">
            <span class="text-xs text-slate-400">Time range</span>
            <input 
              id="filter-date-slider"
              type="range"
              min="0"
              max="100"
              step="10"
              v-model="filterDateRange"
              class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-green-500"
            />
          </div>
      </div>
    </div>

    <!-- Posts List -->
    <div id="blog-posts-list-container" class="max-w-[600px] mx-auto px-4 space-y-6">
       <!-- Empty State -->
       <div v-if="filteredPosts.length === 0" class="text-center py-20 text-slate-500">
          <p class="text-xl">No posts found.</p>
        </div>

       <div 
          v-for="post in filteredPosts" 
          :key="post.id"
          :id="`blog-post-${post.id}`"
          :class="[
            'bg-slate-800 rounded-lg overflow-hidden border border-slate-700 transition-all hover:border-slate-600 cursor-pointer',
            hasSearched && post.id === matchedPostId ? 'post-card-matched ring-2 ring-blue-500' : 'post-card-visible',
            filtersApplied ? 'post-card-filtered' : ''
          ]"
          @click="openPost(post.id)"
        >
          <!-- Simplified Post View -->
          <div v-if="post.type === 'photo'">
             <img :src="post.content" class="w-full h-auto object-cover max-h-[500px]" />
          </div>
          <div class="p-6">
            <h3 v-if="post.title" class="font-bold text-xl mb-2">{{ post.title }}</h3>
            <p v-if="post.type === 'text'" class="text-slate-300 line-clamp-3">{{ post.content }}</p>
            <blockquote v-if="post.type === 'quote'" class="italic text-lg text-slate-200 border-l-4 border-slate-600 pl-4 mb-2">{{ post.content }}</blockquote>
            
            <div class="mt-4 flex justify-between items-center text-xs text-slate-500">
              <span>{{ formatDate(post.date) }}</span>
              <span class="font-bold text-slate-400">{{ post.notes }} notes</span>
            </div>
          </div>
        </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'BLOG_POSTS_LIST',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const blogId = computed(() => route.params.id || store.selected_blog_id)
    const blog = computed(() => dataStore.blogs.find(b => b.id === blogId.value))

    const searchQuery = ref('')
    const filterTextOnly = ref(false)
    const filterDateRange = ref(100) // Default to max (all time)
    const currentSort = ref('newest')
    const sortOpen = ref(false)

    // FSM State Mappers
    const hasSearched = computed(() => store.blog_posts_list_has_searched)
    const matchedPostId = computed(() => store.matched_post_id)
    const filtersApplied = computed(() => store.blog_posts_list_filters_applied)

    const filteredPosts = computed(() => {
      let result = dataStore.posts.filter(p => p.blog_id === blogId.value)

      if (filterTextOnly.value) {
        result = result.filter(p => p.type === 'text')
      }

      // Mock date slider: if < 100, filter out some older posts
      if (filterDateRange.value < 100) {
        result = result.slice(0, Math.max(1, Math.floor(result.length * (filterDateRange.value / 100))))
      }

      if (searchQuery.value) {
        const query = searchQuery.value.toLowerCase()
        result = result.filter(p => 
          (p.title && p.title.toLowerCase().includes(query)) ||
          (p.content && p.content.toLowerCase().includes(query))
        )
      }

      result.sort((a, b) => {
        const dateA = new Date(a.date)
        const dateB = new Date(b.date)
        return currentSort.value === 'newest' ? dateB - dateA : dateA - dateB
      })

      return result
    })

    const formatDate = (isoString) => {
      return new Date(isoString).toLocaleDateString()
    }

    const handleSearch = () => {
      store.blog_posts_list_has_searched = true
      if (filteredPosts.value.length > 0) {
        store.matched_post_id = filteredPosts.value[0].id
      }
    }

    const setSort = (type) => {
      currentSort.value = type
      sortOpen.value = false
      store.blog_posts_list_filters_applied = true
    }

    const goBackOverview = async () => {
      store.currentPageId = 'BLOG_OVERVIEW'
      await router.push({ name: 'BLOG_OVERVIEW', params: { id: blogId.value } })
    }

    const openPost = async (id) => {
      store.selected_post_id = id
      
      if (hasSearched.value) store.blog_posts_list_has_searched = null
      if (filtersApplied.value) store.blog_posts_list_filters_applied = null

      store.currentPageId = 'POST_DETAIL'
      await router.push({ name: 'POST_DETAIL', params: { id } })
    }

    onMounted(() => {
      if (!blogId.value) router.push({ name: 'EXPLORE' })
    })

    return {
      blog,
      searchQuery,
      filterTextOnly,
      filterDateRange,
      currentSort,
      sortOpen,
      filteredPosts,
      hasSearched,
      matchedPostId,
      filtersApplied,
      handleSearch,
      setSort,
      formatDate,
      goBackOverview,
      openPost
    }
  },
  watch: {
    filterTextOnly() { this.store.blog_posts_list_filters_applied = true },
    filterDateRange() { this.store.blog_posts_list_filters_applied = true }
  }
}
</script>