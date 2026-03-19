<template>
  <div class="min-h-screen bg-[#F1F2F2]">
    <nav class="bg-white shadow-sm sticky top-0 z-50">
      <div class="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <button id="back-home" @click="goHome" class="text-gray-500 hover:text-gray-700">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          </button>
          <h1 class="text-[#B92B27] text-2xl font-bold font-serif">Topics</h1>
        </div>
        
        <div class="flex-1 max-w-lg mx-4">
           <input 
              id="topics-search-input"
              v-model="searchQuery"
              @keyup.enter="handleSearch"
              type="text" 
              placeholder="Search Topics" 
              class="w-full bg-white border border-gray-300 hover:border-blue-500 px-4 py-1.5 rounded-full text-sm focus:outline-none focus:border-blue-500 transition-colors"
            />
        </div>
      </div>
    </nav>

    <main class="max-w-5xl mx-auto px-4 py-8 grid grid-cols-1 md:grid-cols-4 gap-6">
      <!-- Filters Sidebar -->
      <div class="md:col-span-1 space-y-6">
        <div class="bg-white p-4 rounded-lg shadow-sm">
          <h3 class="font-bold text-gray-800 mb-4">Filter Topics</h3>
          
          <div class="space-y-6">
            <!-- Checkbox -->
            <div class="flex items-center gap-3 cursor-pointer" id="topics-filter-followed-checkbox" @click="toggleFollowedFilter">
              <div :class="['w-5 h-5 border rounded flex items-center justify-center transition-all', isFollowedFilterActive ? 'bg-blue-600 border-blue-600' : 'border-gray-300 bg-white']">
                <svg v-if="isFollowedFilterActive" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
              </div>
              <span class="text-sm text-gray-700">Followed Only</span>
            </div>

            <!-- Slider (Activity) -->
            <div class="space-y-2">
              <div class="flex justify-between text-xs font-medium text-gray-500">
                <span>Min Activity Score</span>
                <span class="text-blue-600">{{ activityFilterValue }}</span>
              </div>
              <input 
                id="topics-activity-slider"
                type="range" 
                v-model.number="activityFilterValue" 
                :min="minActivity" 
                :max="maxActivity" 
                step="1"
                class="w-full h-1.5 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                @input="applyActivityFilter"
              />
            </div>

            <!-- Sort -->
            <div class="relative">
              <label class="text-xs font-medium text-gray-500 mb-1.5 block">Sort Order</label>
              <div id="topics-sort-dropdown" class="w-full border border-gray-200 rounded-lg px-3 py-2 text-sm bg-gray-50 cursor-pointer flex justify-between items-center hover:bg-white hover:border-blue-300 transition-all" @click="toggleSortDropdown">
                <span class="text-gray-700">{{ currentSortLabel }}</span>
                <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
              </div>
              
              <div v-if="isSortDropdownOpen" class="absolute top-full left-0 w-full bg-white border border-gray-100 shadow-xl rounded-lg mt-2 z-20 overflow-hidden">
                <div id="topics-sort-popular-desc" @click="selectSort('popular')" class="px-4 py-2.5 hover:bg-blue-50 text-sm cursor-pointer text-gray-700 transition-colors">Most Popular</div>
                <div id="topics-sort-alpha-inc" @click="selectSort('alphabetical')" class="px-4 py-2.5 hover:bg-blue-50 text-sm cursor-pointer text-gray-700 transition-colors">A-Z</div>
                <div id="topics-sort-most-active-desc" @click="selectSort('most_active')" class="px-4 py-2.5 hover:bg-blue-50 text-sm cursor-pointer text-gray-700 transition-colors">Most Active</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Topics Grid -->
      <div class="md:col-span-3" id="topics-list">
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <div 
            v-for="topic in displayedTopics" 
            :key="topic.id" 
            :class="[
              'bg-white rounded-xl overflow-hidden border border-gray-200 shadow-sm hover:shadow-md transition-all duration-300 group cursor-pointer h-full flex flex-col',
              `data-id-${topic.id}`,
              isFiltered ? 'topic-row-filtered' : '',
              hasSearched && topic.id === matchedId ? 'topic-row-matched' : '',
              !isFiltered && !hasSearched ? 'topic-row-visible' : ''
            ]"
            @click="openTopic(topic)"
          >
            <!-- Topic Image -->
            <div class="h-32 bg-gray-100 relative overflow-hidden">
              <img :src="topic.image" class="w-full h-full object-cover transform group-hover:scale-110 transition-transform duration-700" :alt="topic.name" />
              <div class="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
              <h3 class="absolute bottom-3 left-4 text-white font-bold text-lg shadow-sm">{{ topic.name }}</h3>
            </div>
            
            <!-- Topic Stats -->
            <div class="p-4 flex flex-col flex-1 justify-between">
              <div class="flex justify-between items-center text-xs text-gray-500 mb-3">
                 <span class="bg-blue-50 text-blue-700 px-2 py-1 rounded-full font-medium">{{ (topic.followers / 1000).toFixed(1) }}k Followers</span>
                 <span class="flex items-center gap-1">
                   <svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20"><path d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z"></path></svg>
                   Act: {{ topic.activity }}
                 </span>
              </div>
              <button class="w-full mt-2 border border-[#B92B27] text-[#B92B27] hover:bg-[#B92B27] hover:text-white rounded-full py-1.5 text-sm font-semibold transition-all">
                View Topic
              </button>
            </div>
          </div>
        </div>

        <!-- Empty State -->
        <div v-if="displayedTopics.length === 0" class="col-span-3 text-center py-16">
          <div class="bg-gray-50 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-4">
             <svg class="w-10 h-10 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
          </div>
          <h3 class="text-gray-900 font-bold text-lg">No topics found</h3>
          <p class="text-gray-500">Try adjusting filters or search query.</p>
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
  name: 'TOPIC_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const isFollowedFilterActive = ref(false)
    const activityFilterValue = ref(0)
    const isSortDropdownOpen = ref(false)
    const currentSort = ref(null)

    // Computed Stats
    const minActivity = computed(() => {
      return 1
    })
    const maxActivity = computed(() => {
      if (!dataStore.topics.length) return 100
      return Math.max(...dataStore.topics.map(t => t.activity))
    })

    // Filter Logic
    const displayedTopics = computed(() => {
      let result = [...dataStore.topics]

      // Search
      if (signatureStore.topics_has_searched && signatureStore.matched_topic_id) {
         if (searchQuery.value) {
            result = result.filter(t => t.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
         }
      }

      // Filter
      if (signatureStore.topics_filters_applied) {
        if (isFollowedFilterActive.value) {
           // Simulate "followed" logic (e.g., activity > 90 implies heavily followed for mock)
           // Or just generic filter logic for demo
           result = result.filter(t => t.followers > 100000)
        }
        // Slider: Value > Filter Value
        result = result.filter(t => t.activity > activityFilterValue.value)
      }

      // Sort
      if (currentSort.value) {
        if (currentSort.value === 'popular') {
          result.sort((a, b) => b.followers - a.followers)
        } else if (currentSort.value === 'alphabetical') {
          result.sort((a, b) => a.name.localeCompare(b.name))
        } else if (currentSort.value === 'most_active') {
          result.sort((a, b) => b.activity - a.activity) // Sort by activity score
        }
      }

      return result
    })

    const isFiltered = computed(() => signatureStore.topics_filters_applied === true)
    const hasSearched = computed(() => signatureStore.topics_has_searched === true)
    const matchedId = computed(() => signatureStore.matched_topic_id)
    
    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default'
      const labels = {
        'popular': 'Most Popular',
        'alphabetical': 'A-Z',
        'recent': 'Recently Added'
      }
      return labels[currentSort.value] || 'Sort By'
    })

    // Actions
    function goHome() {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    function toggleFollowedFilter() {
      isFollowedFilterActive.value = !isFollowedFilterActive.value
      signatureStore.topics_filters_applied = true
    }

    function applyActivityFilter() {
      signatureStore.topics_filters_applied = true
    }

    function toggleSortDropdown() {
      isSortDropdownOpen.value = !isSortDropdownOpen.value
    }

    function selectSort(type) {
      currentSort.value = type
      isSortDropdownOpen.value = false
      signatureStore.topics_filters_applied = true
    }

    function handleSearch() {
       if (!searchQuery.value.trim()) return
       const match = dataStore.topics.find(t => t.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
       const matchId = match ? match.id : null
       
       signatureStore.matched_topic_id = matchId
       signatureStore.topics_has_searched = true
    }

    async function openTopic(topic) {
      signatureStore.selected_topic_id = topic.id
      signatureStore.clearTopicFilters()
      signatureStore.setCurrentPageId('TOPIC_DETAIL')
      await router.push({ name: 'TOPIC_DETAIL', params: { id: topic.id } })
    }

    return {
      searchQuery,
      displayedTopics,
      isFollowedFilterActive,
      activityFilterValue,
      minActivity,
      maxActivity,
      isSortDropdownOpen,
      currentSortLabel,
      isFiltered,
      hasSearched,
      matchedId,
      goHome,
      toggleFollowedFilter,
      applyActivityFilter,
      toggleSortDropdown,
      selectSort,
      handleSearch,
      openTopic
    }
  }
}
</script>