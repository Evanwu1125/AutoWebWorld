<template>
  <div class="min-h-screen bg-slate-900 text-white pb-20 md:pl-20 lg:pl-64">
    <!-- Sidebar Navigation (Same as Dashboard for consistency) -->
    <aside class="hidden md:flex flex-col fixed left-0 top-0 h-full w-20 lg:w-64 border-r border-slate-800 bg-slate-900 z-40">
      <div class="p-4 lg:p-6 text-2xl font-bold lg:text-3xl tracking-tighter mb-4">
        <span class="hidden lg:inline">tumblr</span>
        <span class="lg:hidden">t</span>
      </div>
      
      <nav class="flex-1 space-y-2 px-2 lg:px-4">
        <button id="explore-back-dashboard" @click="goDashboard" class="flex items-center gap-4 w-full p-3 rounded-full hover:bg-slate-800 transition-colors text-slate-200">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>
          <span class="hidden lg:inline font-bold">Dashboard</span>
        </button>
        <!-- Active State -->
        <button class="flex items-center gap-4 w-full p-3 rounded-full bg-slate-800 text-white font-bold">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" /></svg>
          <span class="hidden lg:inline">Explore</span>
        </button>
      </nav>
    </aside>

    <!-- Main Content -->
    <main class="max-w-6xl mx-auto pt-6 px-4">
      <h1 class="text-3xl font-bold mb-6 tracking-tight">Explore</h1>

      <!-- Search & Filters -->
      <div class="mb-8 space-y-4">
        <!-- Search -->
        <div class="relative max-w-2xl">
          <input 
            id="explore-search-input"
            type="text"
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            placeholder="Search blogs..."
            class="w-full bg-slate-800 border-none rounded-full py-3 px-12 text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
          />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 absolute left-4 top-3.5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
        </div>

        <!-- Filters Row -->
        <div class="flex flex-wrap gap-6 items-center bg-slate-800/50 p-4 rounded-xl backdrop-blur-sm border border-slate-700/50">
           <!-- Checkbox -->
           <label class="flex items-center gap-2 cursor-pointer select-none">
            <input 
              id="filter-recommended-checkbox"
              type="checkbox"
              v-model="filterRecommended"
              class="w-5 h-5 rounded border-slate-500 text-blue-500 focus:ring-blue-500 bg-slate-700"
            />
            <span class="text-sm font-medium text-slate-300">Recommended for you</span>
          </label>

          <!-- Sort -->
           <div class="relative">
            <button 
              id="explore-sort-dropdown"
              @click="sortOpen = !sortOpen"
              class="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white"
            >
              Sort: <span class="text-blue-400 capitalize">{{ currentSort }}</span>
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
            </button>
            
            <div v-if="sortOpen" class="absolute top-full left-0 mt-2 w-32 bg-slate-800 rounded-lg shadow-xl border border-slate-700 overflow-hidden z-20">
              <div id="explore-sort-option-trending" @click="setSort('trending')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Trending</div>
              <div id="explore-sort-option-recent" @click="setSort('recent')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Recent</div>
            </div>
          </div>

          <!-- Slider (Popularity) -->
          <div class="flex-1 min-w-[200px] flex items-center gap-3">
            <span class="text-xs text-slate-400">Popularity &gt; {{ filterPopularity }}k</span>
            <input 
              id="filter-popularity-slider"
              type="range"
              min="0"
              max="100"
              step="1"
              v-model="filterPopularity"
              class="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-500"
            />
          </div>
        </div>
      </div>

      <!-- Masonry Grid of Blogs -->
      <div id="explore-grid-container" class="min-h-[500px]">
        <div id="explore-grid" class="columns-1 sm:columns-2 lg:columns-3 xl:columns-4 gap-4 space-y-4">
          
          <div 
            v-for="blog in filteredBlogs" 
            :key="blog.id"
            :class="[
              'break-inside-avoid bg-slate-800 rounded-xl overflow-hidden hover:transform hover:scale-[1.02] transition-all duration-200 cursor-pointer relative group',
              hasSearched && blog.id === matchedBlogId ? 'blog-card-matched ring-2 ring-blue-500' : 'blog-card-visible',
              filtersApplied ? 'blog-card-filtered' : ''
            ]"
            @click="openBlog(blog.id)"
          >
             <!-- Cover Image -->
             <div class="h-24 bg-slate-700 relative">
               <img :src="blog.cover" class="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity" alt="Cover" />
             </div>
             
             <!-- Avatar & Info -->
             <div class="px-4 pb-4 -mt-8 relative z-10">
               <img :src="blog.avatar" class="w-16 h-16 rounded-lg border-4 border-slate-800 shadow-md object-cover bg-slate-700" alt="Avatar" />
               
               <div class="mt-2">
                 <h3 class="font-bold text-lg text-white leading-tight">{{ blog.name }}</h3>
                 <p class="text-slate-400 text-sm mb-2">{{ blog.handle }}</p>
                 <p class="text-slate-300 text-sm leading-relaxed mb-3 line-clamp-3">{{ blog.description }}</p>
                 
                 <div class="flex items-center justify-between text-xs text-slate-500 font-medium border-t border-slate-700/50 pt-2">
                   <span>{{ (blog.followers / 1000).toFixed(1) }}k followers</span>
                   <button v-if="!blog.following" class="text-blue-400 hover:text-blue-300 uppercase tracking-wider font-bold">Follow</button>
                   <span v-else class="text-slate-600 uppercase tracking-wider font-bold">Following</span>
                 </div>
               </div>
             </div>
          </div>

        </div>
        
        <!-- Empty State -->
        <div v-if="filteredBlogs.length === 0" class="text-center py-20 text-slate-500">
          <p class="text-xl">No blogs found.</p>
        </div>
      </div>

    </main>

    <!-- Mobile Navigation -->
    <nav class="md:hidden fixed bottom-0 left-0 w-full bg-slate-900 border-t border-slate-800 flex justify-around p-3 z-50">
      <button id="explore-back-dashboard" @click="goDashboard" class="p-2 text-slate-400 hover:text-white"><svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg></button>
      <button class="p-2 text-white"><svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg></button>
    </nav>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'EXPLORE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterRecommended = ref(false)
    const filterPopularity = ref(0)
    const currentSort = ref('trending')
    const sortOpen = ref(false)

    // FSM State Mappers
    const hasSearched = computed(() => store.explore_has_searched)
    const matchedBlogId = computed(() => store.matched_blog_id)
    const filtersApplied = computed(() => store.explore_filters_applied)

    const filteredBlogs = computed(() => {
      let result = [...dataStore.blogs]

      if (filterRecommended.value) {
        // Mock recommendation logic - just shuffle or filter some random ones
        result = result.filter((_, i) => i % 2 === 0)
      }

      if (filterPopularity.value > 0) {
        // Assume follower count > slider * 1000
        result = result.filter(b => b.followers > filterPopularity.value * 1000)
      }

      if (searchQuery.value) {
        const query = searchQuery.value.toLowerCase()
        result = result.filter(b => 
          b.name.toLowerCase().includes(query) || 
          b.handle.toLowerCase().includes(query) ||
          b.description.toLowerCase().includes(query)
        )
      }

      if (currentSort.value === 'trending') {
        result.sort((a, b) => b.followers - a.followers)
      } else {
        // Mock recent by ID reverse
        result.sort((a, b) => b.id.localeCompare(a.id))
      }

      return result
    })

    const handleSearch = () => {
      store.explore_has_searched = true
      if (filteredBlogs.value.length > 0) {
        store.matched_blog_id = filteredBlogs.value[0].id
      }
    }

    const setSort = (type) => {
      currentSort.value = type
      sortOpen.value = false
      store.explore_filters_applied = true
    }

    const goDashboard = async () => {
      store.currentPageId = 'DASHBOARD_FEED'
      await router.push({ name: 'DASHBOARD_FEED' })
    }

    const openBlog = async (id) => {
      store.selected_blog_id = id
      
      // Clear flags
      if (hasSearched.value) store.explore_has_searched = null
      if (filtersApplied.value) store.explore_filters_applied = null

      store.currentPageId = 'BLOG_OVERVIEW'
      await router.push({ name: 'BLOG_OVERVIEW', params: { id } })
    }

    return {
      store,
      searchQuery,
      filterRecommended,
      filterPopularity,
      currentSort,
      sortOpen,
      filteredBlogs,
      hasSearched,
      matchedBlogId,
      filtersApplied,
      handleSearch,
      setSort,
      goDashboard,
      openBlog
    }
  },
  watch: {
    filterRecommended() { this.store.explore_filters_applied = true },
    filterPopularity() { this.store.explore_filters_applied = true }
  }
}
</script>