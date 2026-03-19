<template>
  <div class="min-h-screen bg-[#F1F2F2]">
    <nav class="bg-white shadow-sm sticky top-0 z-50">
      <div class="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <button id="bookmarks-back-home" @click="goHome" class="text-gray-500 hover:text-gray-700 p-2 rounded-full hover:bg-gray-100 transition-colors">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          </button>
          <h1 class="text-[#B92B27] text-xl font-bold font-serif">Bookmarks</h1>
        </div>
      </div>
    </nav>

    <main class="max-w-3xl mx-auto px-4 py-8 grid grid-cols-1 md:grid-cols-3 gap-6">
      
      <!-- Filters -->
      <div class="space-y-6">
        <div class="bg-white p-4 rounded shadow-sm">
          <h3 class="font-bold text-gray-700 mb-3 text-sm uppercase tracking-wide">Filters</h3>
          
          <div class="space-y-4">
            <div class="flex items-center gap-2 cursor-pointer" id="bookmarks-filter-questions-checkbox" @click="toggleQuestionsFilter">
              <div :class="['w-4 h-4 border rounded flex items-center justify-center transition-colors', isQuestionsFilterActive ? 'bg-blue-600 border-blue-600' : 'border-gray-300 bg-white']">
                <svg v-if="isQuestionsFilterActive" class="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
              </div>
              <span class="text-sm text-gray-600">Questions Only</span>
            </div>

            <div class="space-y-2">
              <div class="flex justify-between text-xs text-gray-500">
                <span>Added Time</span>
                <span>{{ timeFilterValue }}h</span>
              </div>
              <input 
                id="bookmarks-time-slider"
                type="range" 
                v-model.number="timeFilterValue" 
                :min="0" 
                :max="100" 
                step="1"
                class="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                @input="applyTimeFilter"
              />
            </div>

            <div class="relative">
              <label class="text-xs text-gray-500 mb-1 block">Sort By</label>
              <div id="bookmarks-sort-dropdown" class="w-full border border-gray-300 rounded px-3 py-2 text-sm bg-white cursor-pointer flex justify-between items-center" @click="toggleSortDropdown">
                <span>{{ currentSortLabel }}</span>
                <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
              </div>
              
              <div v-if="isSortDropdownOpen" class="absolute top-full left-0 w-full bg-white border border-gray-200 shadow-lg rounded mt-1 z-10">
                <div id="bookmarks-sort-recent" @click="selectSort('recent')" class="px-4 py-2 hover:bg-gray-50 text-sm cursor-pointer">Recently Added</div>
                <div id="bookmarks-sort-top" @click="selectSort('top')" class="px-4 py-2 hover:bg-gray-50 text-sm cursor-pointer">Top Rated</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- List -->
      <div class="md:col-span-2 space-y-4">
        <div 
          v-for="item in displayedBookmarks" 
          :key="item.id"
          class="bg-white p-5 rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow"
        >
          <div class="flex gap-4">
             <div class="w-20 h-20 bg-gray-100 rounded overflow-hidden flex-shrink-0">
               <img :src="item.question_image || '/images/photo1765097829.jpg'" class="w-full h-full object-cover" />
             </div>
             <div>
               <h3 class="font-bold text-gray-900 mb-1 hover:text-blue-600 cursor-pointer font-serif text-lg">{{ item.question_title }}</h3>
               <p class="text-sm text-gray-500 mb-2">Bookmarked {{ item.added_at }}h ago</p>
               <div class="text-xs text-blue-600 font-medium bg-blue-50 inline-block px-2 py-0.5 rounded">Question</div>
             </div>
          </div>
        </div>

        <div v-if="displayedBookmarks.length === 0" class="text-center py-10 bg-white rounded border border-gray-200">
          <p class="text-gray-500">No bookmarks found.</p>
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
  name: 'BOOKMARKS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const isQuestionsFilterActive = ref(false)
    const timeFilterValue = ref(0)
    const isSortDropdownOpen = ref(false)
    const currentSort = ref(null)

    const displayedBookmarks = computed(() => {
      // Join bookmarks with questions to get details
      let result = dataStore.bookmarks.map(b => {
        const q = dataStore.questions.find(q => q.id === b.question_id)
        return {
          ...b,
          question_title: q ? q.title : 'Unknown Question',
          question_image: q ? q.image : null,
          upvotes: q ? q.upvotes : 0
        }
      })

      if (store.bookmarks_filters_applied) {
        if (isQuestionsFilterActive.value) {
          // Mock logic: assume all bookmarks are questions for now
        }
        // Time filter: added_at > value
        result = result.filter(b => b.added_at > timeFilterValue.value)
      }

      // Sort
      if (currentSort.value === 'recent') {
        result.sort((a, b) => a.added_at - b.added_at) // newer = smaller added_at? No, assume added_at is "hours ago", so smaller = newer
        // Wait, if added_at is "hours ago", 1h ago is newer than 10h ago.
        // So ascending sort puts smallest (newest) first.
      } else if (currentSort.value === 'top') {
        result.sort((a, b) => b.upvotes - a.upvotes)
      }

      return result
    })

    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default'
      return currentSort.value === 'recent' ? 'Recently Added' : 'Top Rated'
    })

    function goHome() {
      store.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    function toggleQuestionsFilter() {
      isQuestionsFilterActive.value = !isQuestionsFilterActive.value
      store.bookmarks_filters_applied = true
    }

    function applyTimeFilter() {
      store.bookmarks_filters_applied = true
    }

    function toggleSortDropdown() {
      isSortDropdownOpen.value = !isSortDropdownOpen.value
    }

    function selectSort(type) {
      currentSort.value = type
      isSortDropdownOpen.value = false
      store.bookmarks_filters_applied = true
    }

    return {
      displayedBookmarks,
      isQuestionsFilterActive,
      timeFilterValue,
      isSortDropdownOpen,
      currentSortLabel,
      goHome,
      toggleQuestionsFilter,
      applyTimeFilter,
      toggleSortDropdown,
      selectSort
    }
  }
}
</script>