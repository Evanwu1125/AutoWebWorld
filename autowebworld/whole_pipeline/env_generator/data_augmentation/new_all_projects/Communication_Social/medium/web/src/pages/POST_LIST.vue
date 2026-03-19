<template>
  <div class="min-h-screen bg-white pb-20">
    <!-- Top Bar with Search & Filter -->
    <header class="sticky top-0 z-30 bg-white border-b border-gray-200">
      <div class="max-w-3xl mx-auto px-4 h-16 flex items-center justify-between gap-4">
        <div class="flex items-center gap-4 flex-1">
          <button id="nav-home" @click="handleBackHome" class="text-gray-400 hover:text-gray-900 transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
            </svg>
          </button>
          <div class="relative flex-1 max-w-md">
            <span class="absolute inset-y-0 left-0 flex items-center pl-3 text-gray-400">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </span>
            <input 
              id="post-list-search-input"
              type="text"
              v-model="searchQuery"
              @keyup.enter="handleSearch"
              placeholder="Search stories..."
              class="w-full bg-gray-50 border-none rounded-full py-2 pl-10 pr-4 text-sm focus:ring-1 focus:ring-gray-300 font-sans"
            />
          </div>
        </div>
        
        <!-- Filters -->
        <div class="flex items-center gap-4">
          <!-- Sort Dropdown -->
          <div class="relative">
             <div id="post-list-sort-dropdown" class="flex items-center gap-1 text-sm text-gray-600 cursor-pointer font-sans hover:text-black" @click="toggleSortMenu">
               <span>Sort by</span>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </div>
             <div v-if="sortMenuOpen" class="absolute right-0 mt-2 w-40 bg-white border border-gray-100 rounded-lg shadow-xl py-1 z-50">
                <div id="post-list-sort-option-latest-desc" class="px-4 py-2 text-sm hover:bg-gray-50 cursor-pointer font-sans" @click="handleSort('latest')">Latest</div>
                <div id="post-list-sort-option-claps" class="px-4 py-2 text-sm hover:bg-gray-50 cursor-pointer font-sans" @click="handleSort('claps')">Most Claps</div>
                <div id="post-list-sort-option-responses" class="px-4 py-2 text-sm hover:bg-gray-50 cursor-pointer font-sans" @click="handleSort('responses')">Most Responses</div>
             </div>
          </div>
        </div>
      </div>
      
      <!-- Secondary Filters Row -->
      <div class="max-w-3xl mx-auto px-4 py-3 flex items-center gap-6 overflow-x-auto">
         <label class="flex items-center gap-2 cursor-pointer">
            <input type="checkbox" id="post-list-filter-tag-checkbox" v-model="filterTechOnly" @change="applyFilters" class="rounded text-green-600 focus:ring-green-500 border-gray-300" />
            <span class="text-sm text-gray-600 font-sans">Technology Only</span>
         </label>
         
         <div class="flex items-center gap-3 flex-1">
            <span class="text-sm text-gray-600 font-sans whitespace-nowrap">Min Length: {{ filterMinLength }} min</span>
            <input 
              id="post-list-filter-length-slider" 
              type="range" 
              min="0" 
              max="20" 
              step="1" 
              v-model.number="filterMinLength"
              @input="applyFilters"
              class="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
            />
         </div>
      </div>
    </header>

    <!-- List Content -->
    <main class="max-w-3xl mx-auto px-4 py-8" id="post-list-container" @drag.end="handleScrollDrag">
      <div id="post-list" class="space-y-12">
         <div v-for="post in filteredPosts" :key="post.id" :class="`data-id-${post.id}`">
           <article class="flex gap-6 group cursor-pointer border-b border-gray-100 pb-10">
             <div class="flex-1">
                <div class="flex items-center gap-2 mb-2">
                   <img :src="getUser(post.author_id).avatar" class="w-6 h-6 rounded-full object-cover" />
                   <span class="text-sm font-medium font-sans">{{ getUser(post.author_id).name }}</span>
                   <span class="text-sm text-gray-500 font-sans">in {{ post.tag || 'General' }}</span>
                </div>
                
                <h2 
                  :class="{
                    'text-2xl font-bold font-serif mb-2 group-hover:underline decoration-2 post-filtered post-matched': true,
                    'post-matched': isMatched(post.id),
                    'post-filtered': hasFilters,
                    'post-visible': !isMatched(post.id) && !hasFilters
                  }"
                  @click="handleOpenPost(post.id)"
                >
                  {{ post.title }}
                </h2>
                <p class="text-gray-500 font-serif mb-3 line-clamp-3 text-base leading-relaxed">{{ post.content }}</p>
                
                <div class="flex items-center justify-between text-xs text-gray-500 font-sans mt-4">
                   <div class="flex items-center gap-4">
                      <span>{{ post.length_minutes }} min read</span>
                      <span v-if="post.claps > 0">👏 {{ post.claps }}</span>
                   </div>
                   <div>
                      <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400 hover:text-gray-800" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                      </svg>
                   </div>
                </div>
             </div>
             <div class="w-40 h-28 flex-shrink-0 hidden sm:block">
                <img :src="post.image" class="w-full h-full object-cover rounded" :alt="post.title" />
             </div>
           </article>
         </div>
         
         <!-- Empty State -->
         <div v-if="filteredPosts.length === 0" class="text-center py-20">
            <div class="text-6xl mb-4">🕵️‍♀️</div>
            <h3 class="text-xl font-bold font-sans text-gray-900 mb-2">No stories found</h3>
            <p class="text-gray-500 font-serif">Try adjusting your search or filters to find what you're looking for.</p>
         </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import _ from 'lodash-es'

export default {
  name: 'POST_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterTechOnly = ref(false)
    const filterMinLength = ref(0)
    const sortOption = ref('latest')
    const sortMenuOpen = ref(false)

    // Sync with store state if needed (e.g. if coming back)
    if (signatureStore.post_list_has_searched && signatureStore.post_list_matched_post_id) {
       // Restore search state conceptually, though FSM usually resets on entry
    }

    const getUser = (id) => dataStore.getUserById(id)

    const filteredPosts = computed(() => {
      let result = [...dataStore.posts]

      // Search
      if (signatureStore.post_list_has_searched && signatureStore.post_list_matched_post_id) {
         // In FSM, searching sets "matched_post_id".
         // We should filter to show this matched post prominent, or just filter by query text
         if (searchQuery.value) {
            const q = searchQuery.value.toLowerCase()
            result = result.filter(p => p.title.toLowerCase().includes(q) || p.content.toLowerCase().includes(q))
         }
      }

      // Filters
      if (filterTechOnly.value) {
        result = result.filter(p => p.tag && p.tag.toLowerCase() === 'technology')
      }
      
      if (filterMinLength.value > 0) {
        result = result.filter(p => p.length_minutes >= filterMinLength.value)
      }

      // Sort
      if (sortOption.value === 'latest') {
         result.sort((a, b) => new Date(b.published_date) - new Date(a.published_date))
      } else if (sortOption.value === 'claps') {
         result.sort((a, b) => b.claps - a.claps)
      } else if (sortOption.value === 'responses') {
         result.sort((a, b) => b.responses - a.responses)
      }

      return result
    })
    
    const hasFilters = computed(() => filterTechOnly.value || filterMinLength.value > 0 || sortOption.value !== 'latest')
    
    const isMatched = (id) => {
       return signatureStore.post_list_matched_post_id === id
    }

    const handleSearch = () => {
      // Simulate finding a match
      if (searchQuery.value) {
         const match = filteredPosts.value[0]
         if (match) {
            signatureStore.post_list_matched_post_id = match.id
            signatureStore.post_list_has_searched = true
         }
      }
    }

    const applyFilters = () => {
      signatureStore.post_list_filters_applied = true
    }
    
    const toggleSortMenu = () => {
       sortMenuOpen.value = !sortMenuOpen.value
    }

    const handleSort = (option) => {
       sortOption.value = option
       signatureStore.post_list_filters_applied = true
       sortMenuOpen.value = false
    }

    const handleOpenPost = async (id) => {
      signatureStore.post_list_selected_post_id = id
      signatureStore.post_list_matched_post_id = null
      signatureStore.post_list_has_searched = null
      signatureStore.post_list_filters_applied = null
      signatureStore.setCurrentPageId('POST_DETAIL')
      await router.push({ name: 'POST_DETAIL', params: { id } })
    }

    const handleBackHome = async () => {
      signatureStore.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }
    
    const handleScrollDrag = () => {
       // FSM: ACT_POST_LIST_SCROLL_INTO_VIEW effect
       // Just mock setting an anchor
       if (filteredPosts.value.length > 3) {
          signatureStore.post_list_viewport_anchor_id = filteredPosts.value[3].id
       }
    }

    return {
      searchQuery,
      filterTechOnly,
      filterMinLength,
      sortMenuOpen,
      filteredPosts,
      getUser,
      hasFilters,
      isMatched,
      handleSearch,
      applyFilters,
      toggleSortMenu,
      handleSort,
      handleOpenPost,
      handleBackHome,
      handleScrollDrag
    }
  }
}
</script>