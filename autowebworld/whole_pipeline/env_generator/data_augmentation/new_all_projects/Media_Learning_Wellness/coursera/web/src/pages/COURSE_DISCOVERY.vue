<template>
  <div class="min-h-screen bg-gray-50">
    <PermissionModal 
      :show="showPermissionModal"
      @allow="grantLocation"
      @deny="showPermissionModal = false"
    />

    <!-- Navigation -->
    <nav class="bg-white shadow-sm sticky top-0 z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex items-center">
            <div id="header-logo-home" class="flex-shrink-0 flex items-center cursor-pointer" @click="goHome">
              <span class="text-2xl font-bold text-blue-700">Coursera</span>
            </div>
            <div class="ml-10 flex items-baseline space-x-4">
               <h1 class="text-xl font-semibold text-gray-800">Explore Courses</h1>
            </div>
          </div>
        </div>
      </div>
    </nav>

    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <!-- Search Bar -->
      <div class="flex justify-center mb-8">
        <div class="w-full max-w-2xl relative">
          <input 
            id="course-search-input"
            type="text" 
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            placeholder="What do you want to learn?" 
            class="w-full px-5 py-3 border border-gray-300 rounded-full shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent text-lg"
          >
          <button 
            @click="handleSearch"
            class="absolute right-2 top-2 bg-blue-700 text-white p-2 rounded-full hover:bg-blue-800"
          >
            <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </button>
        </div>
      </div>

      <div class="flex flex-col lg:flex-row gap-8">
        <!-- Sidebar Filters -->
        <div class="w-full lg:w-64 flex-shrink-0 space-y-6">
          <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
            <h3 class="text-lg font-medium text-gray-900 mb-4">Filters</h3>
            
            <!-- Level Filter -->
            <div class="mb-6">
              <h4 class="text-sm font-semibold text-gray-700 mb-2">Level</h4>
              <div class="flex items-center">
                <input 
                  id="filter-level-beginner-checkbox"
                  type="checkbox" 
                  v-model="filterBeginner"
                  @change="applyFilters"
                  class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                >
                <label for="filter-level-beginner-checkbox" class="ml-2 block text-sm text-gray-700">
                  Beginner
                </label>
              </div>
            </div>

            <!-- Duration Filter (Slider) -->
            <div>
              <h4 class="text-sm font-semibold text-gray-700 mb-2">Duration (Hours)</h4>
              <div class="flex items-center justify-between text-xs text-gray-500 mb-1">
                <span>{{ minDuration }}</span>
                <span>{{ maxDuration }}+</span>
              </div>
              <input 
                id="filter-duration-slider"
                type="range" 
                :min="minDuration" 
                :max="maxDuration"
                step="1"
                v-model="durationFilterValue"
                @input="applyFilters"
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              >
              <div class="text-center text-sm font-medium text-blue-700 mt-2">
                Min: {{ durationFilterValue }} hours
              </div>
            </div>
          </div>
        </div>

        <!-- Main Content -->
        <div class="flex-1">
          <!-- Sort & Count -->
          <div class="flex justify-between items-center mb-6">
            <span class="text-gray-600">{{ filteredCourses.length }} results</span>
            
            <!-- Sort Dropdown -->
            <div class="relative">
              <button 
                id="sort-dropdown"
                @click="toggleSortMenu"
                class="inline-flex justify-center w-full rounded-md border border-gray-300 shadow-sm px-4 py-2 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none"
              >
                Sort by: {{ currentSortLabel }}
                <svg class="ml-2 -mr-1 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              <div v-if="isSortMenuOpen" class="origin-top-right absolute right-0 mt-2 w-40 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-10">
                <div class="py-1" role="menu">
                  <div 
                    id="sort-option-relevance"
                    @click="setSort('relevance', 'Relevance')"
                    class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                  >
                    Relevance
                  </div>
                  <div 
                    id="sort-option-newest"
                    @click="setSort('newest', 'Newest')"
                    class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                  >
                    Newest
                  </div>
                  <div 
                    id="sort-option-highest-rated"
                    @click="setSort('highest-rated', 'Highest Rated')"
                    class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                  >
                    Highest Rated
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Course List -->
          <div id="course-results-list" class="space-y-6">
            <div id="course-results">
            <div 
              v-for="course in filteredCourses" 
              :key="course.id"
              :class="[getCardClass(course), `data-id-${course.id}`]"
              class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden hover:shadow-md transition-shadow cursor-pointer flex flex-col sm:flex-row"
              @click="openCourse(course)"
            >
              <div class="sm:w-48 h-48 sm:h-auto flex-shrink-0">
                <img :src="course.image" :alt="course.title" class="w-full h-full object-cover">
              </div>
              <div class="p-6 flex-1 flex flex-col justify-between">
                <div>
                  <div class="flex items-center text-xs font-semibold tracking-wide uppercase text-blue-600 mb-1">
                    {{ course.university }}
                  </div>
                  <h3 class="text-xl font-bold text-gray-900 mb-2">{{ course.title }}</h3>
                  <p class="text-gray-600 text-sm mb-4 line-clamp-2">{{ course.description }}</p>
                  
                  <div class="flex items-center space-x-4 text-sm text-gray-500">
                    <span class="flex items-center">
                      <span class="text-yellow-400 mr-1">★</span> {{ course.rating }}
                    </span>
                    <span>{{ (course.students / 1000).toFixed(0) }}k students</span>
                    <span>{{ course.level }}</span>
                  </div>
                </div>
                
                <div class="mt-4 flex items-center justify-between">
                   <span class="text-sm font-medium text-gray-900">{{ course.duration }} hours</span>
                   <span v-if="course.price > 0" class="text-blue-700 font-bold">${{ course.price }}</span>
                   <span v-else class="text-green-600 font-bold">Free</span>
                </div>
              </div>
              
              <!-- Hidden data attributes for testing -->
              <!-- <div :class="`data-id-${course.id}`" class="hidden"></div> -->
            </div>
            </div>
            <!-- Empty State -->
            <div v-if="filteredCourses.length === 0" class="text-center py-12">
               <p class="text-gray-500 text-lg">No courses found matching your criteria.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import PermissionModal from '../components/PermissionModal.vue'

export default {
  name: 'COURSE_DISCOVERY',
  components: {
    PermissionModal
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const showPermissionModal = ref(false)
    const searchQuery = ref('')
    const filterBeginner = ref(false)
    const durationFilterValue = ref(0)
    const currentSort = ref(null)
    const currentSortLabel = ref('Default')
    const isSortMenuOpen = ref(false)

    // Calculate slider range
    const minDuration = computed(() => {
      const durations = dataStore.courses.map(c => c.duration)
      return Math.min(...durations)
    })
    
    const maxDuration = computed(() => {
      const durations = dataStore.courses.map(c => c.duration)
      return Math.max(...durations)
    })

    // Init slider value to min (show all)
    durationFilterValue.value = minDuration.value

    // Computed filtered list
    const filteredCourses = computed(() => {
      let result = [...dataStore.courses]

      // Filter: Level (Checkbox)
      if (filterBeginner.value) {
        result = result.filter(c => c.level.toLowerCase() === 'beginner')
      }

      // Filter: Duration (Slider) - Show courses LONGER than value
      // FSM logic: drag slider right increases value
      // Usually "filter by duration" means "max duration" or "min duration".
      // Let's assume user wants courses AT LEAST X hours (slider at 0 shows all)
      if (durationFilterValue.value > minDuration.value) {
        result = result.filter(c => c.duration >= durationFilterValue.value)
      }

      // Search
      if (store.course_list_has_searched && store.matched_course_id) {
        // If exact match found by ID (from action effect), prioritize it
        // But here we implement general search filtering
        // The FSM separates "search" action from "open matched"
        // Here we just filter by text for visual feedback
      }
      
      if (searchQuery.value) {
         const q = searchQuery.value.toLowerCase()
         result = result.filter(c => 
           c.title.toLowerCase().includes(q) || 
           c.university.toLowerCase().includes(q)
         )
      }

      // Sort
      if (currentSort.value === 'newest') {
        // Mock date or use ID reverse
        result.sort((a, b) => b.id.localeCompare(a.id))
      } else if (currentSort.value === 'highest-rated') {
        result.sort((a, b) => b.rating - a.rating)
      }

      return result
    })

    onMounted(() => {
      if (!store.location_permission_granted) {
        showPermissionModal.value = true
      }
    })

    function grantLocation() {
      store.location_permission_granted = true
      showPermissionModal.value = false
    }

    function applyFilters() {
      store.course_list_filters_applied = true
    }

    function handleSearch() {
      store.course_list_has_searched = true
      // Find best match for FSM matched_course_id
      const match = filteredCourses.value.length > 0 ? filteredCourses.value[0] : null
      if (match) {
        store.matched_course_id = match.id
      } else {
        store.matched_course_id = null
      }
    }

    function toggleSortMenu() {
      isSortMenuOpen.value = !isSortMenuOpen.value
    }

    function setSort(value, label) {
      currentSort.value = value
      currentSortLabel.value = label
      store.course_list_sort_applied = true
      isSortMenuOpen.value = false
    }

    function getCardClass(course) {
      // Determine class based on state (matched vs filtered vs visible)
      // This helps test scripts find the right element
      if (store.course_list_has_searched && course.id === store.matched_course_id) {
        return 'course-card-matched'
      }
      if (store.course_list_filters_applied) {
        return 'course-card-filtered'
      }
      return 'course-card-visible'
    }

    async function openCourse(course) {
      store.selected_course_id = course.id
      // Clear search/filter flags as per effects (optional but good for consistency)
      if (store.course_list_has_searched) store.course_list_has_searched = null
      if (store.course_list_filters_applied) store.course_list_filters_applied = null
      
      store.setCurrentPageId('COURSE_DETAIL')
      await router.push({ name: 'COURSE_DETAIL', params: { id: course.id } })
    }

    async function goHome() {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      showPermissionModal,
      searchQuery,
      filterBeginner,
      durationFilterValue,
      minDuration,
      maxDuration,
      filteredCourses,
      currentSortLabel,
      isSortMenuOpen,
      grantLocation,
      applyFilters,
      handleSearch,
      toggleSortMenu,
      setSort,
      getCardClass,
      openCourse,
      goHome
    }
  }
}
</script>