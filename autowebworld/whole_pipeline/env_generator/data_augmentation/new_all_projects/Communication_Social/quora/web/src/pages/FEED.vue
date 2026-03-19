<template>
  <div class="min-h-screen bg-[#F1F2F2]">
    <!-- Navbar (Simplified for subpages, but usually reused) -->
    <nav class="bg-white shadow-sm sticky top-0 z-50">
      <div class="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
        <h1 class="text-[#B92B27] text-2xl font-bold font-serif cursor-pointer" id="logo-home" @click="goHome">Quora</h1>
        <div class="flex-1 max-w-lg mx-4">
          <div class="relative">
            <input 
              id="feed-search-input"
              v-model="searchQuery"
              @keyup.enter="handleSearch"
              type="text" 
              placeholder="Search Quora" 
              class="w-full bg-white border border-gray-300 hover:border-blue-500 px-4 py-1.5 rounded-sm text-sm focus:outline-none focus:border-blue-500 transition-colors"
            />
            <svg class="w-4 h-4 text-gray-400 absolute right-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
          </div>
        </div>
        <button id="ask-question-button" @click="goAsk" class="bg-[#B92B27] text-white px-4 py-1.5 rounded-full text-sm font-medium hover:bg-[#a02521] transition-colors">
          Add Question
        </button>
      </div>
    </nav>

    <main class="max-w-5xl mx-auto px-4 py-6 grid grid-cols-1 md:grid-cols-12 gap-6">
      <!-- Left Sidebar: Filters -->
      <div class="hidden md:block md:col-span-3 space-y-6">
        <div class="bg-white p-4 rounded shadow-sm">
          <h3 class="font-bold text-gray-700 mb-3 text-sm uppercase tracking-wide border-b pb-2">Feed Filters</h3>
          
          <div class="space-y-4">
            <!-- Checkbox Filter -->
            <div class="flex items-center gap-2 cursor-pointer hover:bg-gray-50 p-1 rounded" id="filter-answered-checkbox" @click="toggleAnsweredFilter">
              <div :class="['w-4 h-4 border rounded flex items-center justify-center transition-colors', isAnsweredFilterActive ? 'bg-blue-600 border-blue-600' : 'border-gray-300 bg-white']">
                <svg v-if="isAnsweredFilterActive" class="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
              </div>
              <span class="text-sm text-gray-600 select-none">Show Answered Only</span>
            </div>

            <!-- Slider Filter (Time) -->
            <div class="space-y-2">
              <div class="flex justify-between text-xs text-gray-500">
                <span>Filter by Time (Hours)</span>
                <span>{{ timeFilterValue }}h</span>
              </div>
              <input 
                id="filter-time-slider"
                type="range" 
                v-model.number="timeFilterValue" 
                :min="minTime" 
                :max="maxTime" 
                step="1"
                class="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                @input="applyTimeFilter"
              />
              <div class="flex justify-between text-[10px] text-gray-400">
                <span>{{ minTime }}h</span>
                <span>{{ maxTime }}h</span>
              </div>
            </div>

            <!-- Sort Dropdown -->
            <div class="relative">
              <label class="text-xs text-gray-500 mb-1 block">Sort By</label>
              <div id="feed-sort-dropdown" class="w-full border border-gray-300 rounded px-3 py-2 text-sm bg-white cursor-pointer flex justify-between items-center" @click="toggleSortDropdown">
                <span>{{ currentSortLabel }}</span>
                <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
              </div>
              
              <!-- Dropdown Options -->
              <div v-if="isSortDropdownOpen" class="absolute top-full left-0 w-full bg-white border border-gray-200 shadow-lg rounded mt-1 z-10">
                <div id="feed-sort-newest-inc" @click="selectSort('newest')" class="px-4 py-2 hover:bg-gray-50 text-sm cursor-pointer text-gray-700">Newest</div>
                <div id="feed-sort-top" @click="selectSort('top')" class="px-4 py-2 hover:bg-gray-50 text-sm cursor-pointer text-gray-700">Top Rated</div>
                <div id="feed-sort-unanswered" @click="selectSort('unanswered')" class="px-4 py-2 hover:bg-gray-50 text-sm cursor-pointer text-gray-700">Unanswered</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Center: Feed List -->
      <div class="md:col-span-9 space-y-4" id="feed-list">
        
        <!-- List Header -->
        <div class="flex items-center justify-between mb-2">
          <h2 class="text-lg font-bold text-gray-800 flex items-center gap-2">
            <span class="bg-red-600 text-white p-1 rounded-sm text-xs">Feed</span>
            Top Questions for You
          </h2>
          <span class="text-xs text-gray-500">{{ displayedQuestions.length }} results</span>
        </div>

        <!-- Questions List -->
        <div 
          v-for="question in displayedQuestions" 
          :key="question.id" 
          :class="[
            'bg-white p-4 rounded border border-gray-200 shadow-sm hover:shadow-md transition-shadow',
            `data-id-${question.id}`,
            isFiltered ? 'question-row-filtered' : '',
            hasSearched && question.id === matchedId ? 'question-row-matched' : '',
            !isFiltered && !hasSearched ? 'question-row-visible' : ''
          ]"
          @click="openQuestion(question)"
        >
          <div class="flex gap-4">
             <!-- Thumbnail -->
             <div class="flex-shrink-0 w-24 h-16 md:w-32 md:h-24 bg-gray-100 rounded overflow-hidden">
               <img :src="question.image" class="w-full h-full object-cover transform hover:scale-105 transition-transform duration-500" alt="Question" />
             </div>
             
             <!-- Content -->
             <div class="flex-1">
               <h3 class="text-lg font-bold text-gray-900 mb-1 font-serif hover:underline cursor-pointer">{{ question.title }}</h3>
               <p class="text-sm text-gray-600 line-clamp-2 mb-2">{{ question.details }}</p>
               
               <!-- Metadata -->
               <div class="flex items-center gap-4 text-xs text-gray-500">
                 <span class="flex items-center gap-1">
                   <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                   {{ question.time }}h ago
                 </span>
                 <span class="flex items-center gap-1">
                   <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path></svg>
                   {{ question.upvotes }} Upvotes
                 </span>
                 <span v-if="question.answered" class="text-green-600 font-medium bg-green-50 px-2 py-0.5 rounded-full">Answered</span>
               </div>
             </div>
          </div>
        </div>

        <!-- Empty State -->
        <div v-if="displayedQuestions.length === 0" class="text-center py-12 bg-white rounded border border-gray-200">
           <img src="/images/NoQuestions.jpg" class="w-32 h-32 mx-auto mb-4 opacity-50" />
           <p class="text-gray-500">No questions found matching your criteria.</p>
        </div>

      </div>
    </main>

    <!-- Permission Modal (Global Component handles visibility logic via store) -->
    <PermissionModal />
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import PermissionModal from '../components/PermissionModal.vue'

export default {
  name: 'FEED',
  components: { PermissionModal },
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    // Local state for UI controls
    const searchQuery = ref('')
    const isAnsweredFilterActive = ref(false)
    const timeFilterValue = ref(0)
    const isSortDropdownOpen = ref(false)
    const currentSort = ref(null)

    // Compute min/max for slider
    const minTime = computed(() => {
      if (!dataStore.questions.length) return 0
      return Math.min(...dataStore.questions.map(q => q.time))
    })
    const maxTime = computed(() => {
      if (!dataStore.questions.length) return 100
      return Math.max(...dataStore.questions.map(q => q.time))
    })

    // Computed filtered questions
    const displayedQuestions = computed(() => {
      let result = [...dataStore.questions]

      // Search Logic
      if (signatureStore.feed_has_searched && signatureStore.matched_question_id) {
        // If searched, prioritize/filter matched ID or show search results logic
        // FSM implies simple filtering or highlighting. 
        // Here we'll filter by ID if it's a specific match, OR filter by query if we stored query.
        // ACT_FEED_SEARCH effects: set matched_question_id (ref to ITEM_ANY -> usually found item)
        // Actually the FSM search parameters say "question_id: {ITEM_ANY}", meaning the search result is an ID.
        // So we should show that ID first or only.
        // Let's implement robust text search first, then map to ID.
        if (searchQuery.value) {
            result = result.filter(q => q.title.toLowerCase().includes(searchQuery.value.toLowerCase()))
        }
      }

      // Filter Logic
      if (signatureStore.feed_filters_applied) {
        if (isAnsweredFilterActive.value) {
          result = result.filter(q => q.answered)
        }
        // Time filter: Show items NEWER than value (time < value) or OLDER?
        // Usually slider 0 -> all. Slider max -> few.
        // Let's assume time is "hours ago". Slider sets MAX hours ago to show (recentness).
        // Or slider sets MIN hours (at least X hours old).
        // Standard UI: "Past X hours". 
        // Let's do: Show questions where time <= timeFilterValue
        // Wait, FSM instruction: "slider at 0 shows all items, slider at 500 shows items > 500"
        // FSM Guide says: "ALWAYS use 'greater than' comparison (item.value > filterValue)"
        // So: time > timeFilterValue. 
        // If time is "hours ago", larger time = older.
        // So filtering for time > 0 shows everything (assuming time > 0).
        // If I increase slider to 10, shows items > 10 hours old (older posts).
        result = result.filter(q => q.time >= timeFilterValue.value)
      }

      // Sort Logic
      if (currentSort.value) {
        if (currentSort.value === 'newest') {
          result.sort((a, b) => a.time - b.time) // Smaller time = newer
        } else if (currentSort.value === 'top') {
          result.sort((a, b) => b.upvotes - a.upvotes)
        } else if (currentSort.value === 'unanswered') {
          // Sort unanswered first
          result.sort((a, b) => (a.answered === b.answered ? 0 : a.answered ? 1 : -1))
        }
      }

      return result
    })

    // State helpers
    const isFiltered = computed(() => signatureStore.feed_filters_applied === true)
    const hasSearched = computed(() => signatureStore.feed_has_searched === true)
    const matchedId = computed(() => signatureStore.matched_question_id)

    // Actions
    function goHome() {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    function goAsk() {
      signatureStore.setCurrentPageId('ASK_QUESTION_FORM')
      router.push({ name: 'ASK_QUESTION_FORM' })
    }

    // Filter Handlers
    function toggleAnsweredFilter() {
      isAnsweredFilterActive.value = !isAnsweredFilterActive.value
      signatureStore.feed_filters_applied = true
    }

    function applyTimeFilter() {
      signatureStore.feed_filters_applied = true
    }

    function toggleSortDropdown() {
      isSortDropdownOpen.value = !isSortDropdownOpen.value
    }

    function selectSort(sortType) {
      currentSort.value = sortType
      isSortDropdownOpen.value = false
      signatureStore.feed_filters_applied = true
    }
    
    // Search Handler
    function handleSearch() {
      if (!searchQuery.value.trim()) return
      
      // Find matches
      const match = dataStore.questions.find(q => q.title.toLowerCase().includes(searchQuery.value.toLowerCase()))
      const matchId = match ? match.id : null
      
      // Update store effects
      signatureStore.matched_question_id = matchId
      signatureStore.feed_has_searched = true
    }

    // Navigation Handlers
    async function openQuestion(question) {
      signatureStore.selected_question_id = question.id
      
      // Clear filters/search state if needed or dictated by FSM effects (which usually happen on transition)
      // FSM Effect: clear feed_filters_applied, clear feed_has_searched
      if (isFiltered.value) signatureStore.feed_filters_applied = null
      if (hasSearched.value) signatureStore.feed_has_searched = null
      
      signatureStore.setCurrentPageId('QUESTION_DETAIL')
      await router.push({ name: 'QUESTION_DETAIL', params: { id: question.id } })
    }

    const currentSortLabel = computed(() => {
      const labels = {
        'newest': 'Newest',
        'top': 'Top Rated',
        'unanswered': 'Unanswered'
      }
      return labels[currentSort.value] || 'Relevant'
    })

    return {
      searchQuery,
      displayedQuestions,
      isAnsweredFilterActive,
      timeFilterValue,
      minTime,
      maxTime,
      isSortDropdownOpen,
      currentSortLabel,
      isFiltered,
      hasSearched,
      matchedId,
      goHome,
      goAsk,
      handleSearch,
      toggleAnsweredFilter,
      applyTimeFilter,
      toggleSortDropdown,
      selectSort,
      openQuestion
    }
  }
}
</script>