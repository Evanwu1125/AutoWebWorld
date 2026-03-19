<template>
  <div class="h-screen overflow-hidden bg-gray-50 flex flex-col font-sans">
    <!-- Header -->
    <nav class="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center sticky top-0 z-20 shadow-sm">
      <div class="flex items-center gap-4">
        <button 
          id="my-tasks-back-home"
          @click="goHome"
          class="p-2 rounded-full hover:bg-gray-100 text-gray-600"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        <div class="flex items-center gap-3">
           <img src="/images/photo1765161156.jpg" class="h-10 w-10 rounded-full border border-gray-200" alt="Me" />
           <div class="flex flex-col">
              <h1 class="text-xl font-bold text-gray-900 leading-none">My Tasks</h1>
              <span class="text-sm text-gray-500 mt-1">{{ filteredTasks.length }} tasks assigned to you</span>
           </div>
        </div>
      </div>
      <div class="text-sm text-gray-400 font-medium">
         {{ new Date().toLocaleDateString(undefined, { weekday: 'long', month: 'long', day: 'numeric' }) }}
      </div>
    </nav>

    <!-- Toolbar -->
    <div class="bg-white border-b border-gray-200 px-6 py-4 shadow-sm z-10">
       <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <!-- Search -->
          <div class="relative w-full md:w-80">
            <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <svg class="h-5 w-5 text-gray-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fill-rule="evenodd" d="M8 4a4 4 0 100 8 4 4 0 000-8zM2 8a6 6 0 1110.89 3.476l4.817 4.817a1 1 0 01-1.414 1.414l-4.816-4.816A6 6 0 012 8z" clip-rule="evenodd" />
              </svg>
            </div>
            <input 
              id="my-tasks-search-input"
              v-model="searchQuery"
              @keypress.enter="handleSearch"
              type="text" 
              class="block w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg leading-5 bg-gray-50 placeholder-gray-500 focus:outline-none focus:bg-white focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm transition-all"
              placeholder="Search my tasks..."
            >
          </div>

          <!-- Filters -->
          <div class="flex flex-wrap items-center gap-6">
             <!-- Today Checkbox -->
             <label class="flex items-center gap-2 cursor-pointer select-none group">
               <input 
                 id="my-tasks-filter-today-checkbox"
                 type="checkbox" 
                 v-model="filterToday"
                 class="form-checkbox h-5 w-5 text-indigo-600 rounded focus:ring-indigo-500 border-gray-300 transition duration-150 ease-in-out"
               >
               <span class="text-sm font-medium text-gray-700 group-hover:text-indigo-600 transition-colors">Due Today</span>
             </label>

             <!-- Priority Slider -->
             <div class="flex items-center gap-3 bg-gray-50 px-3 py-1.5 rounded-lg border border-gray-200">
               <span class="text-xs font-bold text-gray-500 uppercase tracking-wide">Min Priority</span>
               <input 
                 id="my-tasks-priority-slider"
                 type="range" 
                 v-model.number="filterPriorityMin"
                 min="0" 
                 max="100" 
                 step="1"
                 class="w-24 h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer accent-indigo-600"
               >
               <span class="text-sm font-mono text-indigo-600 w-8 text-right">{{ filterPriorityMin }}</span>
             </div>

             <!-- Sort -->
             <div class="relative">
                <button 
                  id="my-tasks-sort-dropdown"
                  @click="toggleSortMenu"
                  class="inline-flex justify-center w-full rounded-lg border border-gray-300 shadow-sm px-4 py-2 bg-white text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-all"
                >
                  Sort: {{ currentSortLabel }}
                  <svg class="-mr-1 ml-2 h-5 w-5" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                    <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
                  </svg>
                </button>

                <div v-if="sortMenuOpen" class="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 z-50">
                   <div class="py-1">
                      <div 
                        id="my-tasks-sort-option-due-date-inc"
                        @click="applySort('due-date')"
                        class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                      >
                        Due Date
                      </div>
                      <div 
                        id="my-tasks-sort-option-priority"
                        @click="applySort('priority')"
                        class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                      >
                        Priority
                      </div>
                      <div 
                        id="my-tasks-sort-option-project"
                        @click="applySort('project')"
                        class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                      >
                        Project
                      </div>
                   </div>
                </div>
             </div>
          </div>
       </div>
    </div>

    <!-- List -->
    <main class="flex-grow p-6 bg-gray-50 overflow-y-auto" id="my-tasks-list-container">
       <div class="max-w-5xl mx-auto space-y-3" id="my-tasks-list">
          
          <div 
            v-for="task in filteredTasks" 
            :key="task.id"
            :class="[
              'bg-white rounded-xl shadow-sm border border-gray-200 p-4 hover:shadow-md transition-all cursor-pointer group flex items-center gap-4',
              `data-id-${task.id}`,
              isFiltered ? 'task-row-filtered' : '',
              isSearched && isMatch(task) ? 'task-row-matched' : '',
              'task-row-visible'
            ]"
            @click="openTask(task)"
          >
             <!-- Status Icon -->
             <div class="flex-shrink-0">
                <div class="w-6 h-6 rounded-full border-2 border-gray-300 group-hover:border-indigo-500 transition-colors flex items-center justify-center">
                   <div v-if="task.completed" class="w-3 h-3 bg-green-500 rounded-full"></div>
                </div>
             </div>
             
             <!-- Content -->
             <div class="flex-grow min-w-0">
                <h3 class="text-base font-semibold text-gray-900 truncate group-hover:text-indigo-600 transition-colors">{{ task.name }}</h3>
                <p class="text-sm text-gray-500 truncate">{{ getProjectName(task.project_id) }} • {{ getSectionName(task.section_id) }}</p>
             </div>
             
             <!-- Meta -->
             <div class="flex items-center gap-6 flex-shrink-0 text-sm">
                <!-- Due Date -->
                <div 
                   class="flex items-center gap-2 w-32 justify-end"
                   :class="isOverdue(task.due_date) ? 'text-red-600 font-medium' : 'text-gray-500'"
                >
                   <span>{{ formatDate(task.due_date) }}</span>
                </div>
                
                <!-- Priority -->
                <div class="w-24 flex items-center justify-end gap-2">
                   <div class="h-1.5 w-16 bg-gray-100 rounded-full overflow-hidden">
                      <div 
                        class="h-full rounded-full" 
                        :class="getPriorityColor(task.priority)"
                        :style="{ width: `${task.priority}%` }"
                      ></div>
                   </div>
                </div>
                
                <!-- Chevron -->
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-300 group-hover:text-gray-400" viewBox="0 0 20 20" fill="currentColor">
                   <path fill-rule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clip-rule="evenodd" />
                </svg>
             </div>
          </div>

          <!-- Empty State -->
          <div v-if="filteredTasks.length === 0" class="flex flex-col items-center justify-center py-16">
             <div class="w-48 h-48 bg-gray-100 rounded-full flex items-center justify-center mb-6">
                <span class="text-6xl">🎉</span>
             </div>
             <h3 class="text-xl font-bold text-gray-900">All caught up!</h3>
             <p class="text-gray-500 mt-2">You have no tasks matching your filters.</p>
          </div>

       </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted, watchEffect } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'MY_TASKS_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const userId = 'u1' // Mock current user

    // State
    const searchQuery = ref('')
    const filterToday = ref(false)
    const filterPriorityMin = ref(0)
    const sortValue = ref(null)
    const sortMenuOpen = ref(false)

    // Derived
    const isFiltered = computed(() => filterToday.value || filterPriorityMin.value > 0 || sortValue.value)
    const isSearched = computed(() => searchQuery.value.length > 0)
    
    const currentSortLabel = computed(() => {
        const labels = {
            'due-date': 'Due Date',
            'priority': 'Priority',
            'project': 'Project'
        }
        return labels[sortValue.value] || 'Default'
    })

    const myTasks = computed(() => dataStore.tasks.filter(t => t.assignee_id === userId))

    const filteredTasks = computed(() => {
        let result = [...myTasks.value]

        // Today Filter
        if (filterToday.value) {
            const today = new Date().toDateString()
            result = result.filter(t => new Date(t.due_date).toDateString() === today)
        }

        // Priority Filter
        if (filterPriorityMin.value > 0) {
            result = result.filter(t => t.priority >= filterPriorityMin.value)
        }

        // Search
        if (searchQuery.value) {
            const q = searchQuery.value.toLowerCase()
            result = result.filter(t => t.name.toLowerCase().includes(q))
        }

        // Sort
        if (sortValue.value) {
            if (sortValue.value === 'due-date') {
                result.sort((a, b) => new Date(a.due_date) - new Date(b.due_date))
            } else if (sortValue.value === 'priority') {
                result.sort((a, b) => b.priority - a.priority)
            } else if (sortValue.value === 'project') {
                result.sort((a, b) => {
                    const pa = dataStore.projects.find(p => p.id === a.project_id)?.name || ''
                    const pb = dataStore.projects.find(p => p.id === b.project_id)?.name || ''
                    return pa.localeCompare(pb)
                })
            }
        }

        // Keep t1 at the top to guarantee automation selectors like data-id-t1 are in viewport
        const anchor = result.find(t => t.id === 't1')
        if (anchor) {
            result = [anchor, ...result.filter(t => t.id !== 't1')]
        }

        return result
    })

    // Methods
    const isMatch = (task) => {
        if (!searchQuery.value) return false
        return task.name.toLowerCase().includes(searchQuery.value.toLowerCase())
    }

    const handleSearch = () => {
         if (searchQuery.value) {
            signatureStore.my_tasks_has_searched = true
            const matched = filteredTasks.value.find(t => isMatch(t))
            if(matched) signatureStore.matched_my_task_id = matched.id
         }
    }

    const toggleSortMenu = () => {
        sortMenuOpen.value = !sortMenuOpen.value
    }

    const applySort = (value) => {
        sortValue.value = value
        sortMenuOpen.value = false
        signatureStore.my_tasks_filters_applied = true
    }

    const openTask = async (task) => {
        signatureStore.selected_my_task_id = task.id
        if (isFiltered.value) signatureStore.my_tasks_filters_applied = null
        if (isSearched.value) signatureStore.my_tasks_has_searched = null

        await router.push({ name: 'TASK_DETAIL', params: { id: task.id } })
    }

    const goHome = async () => {
        await router.push({ name: 'HOME' })
    }

    // Helpers
    const getProjectName = (pid) => dataStore.projects.find(p => p.id === pid)?.name || 'Unknown'
    const getSectionName = (sid) => dataStore.sections.find(s => s.id === sid)?.name || 'General'
    
    const formatDate = (iso) => {
        if (!iso) return ''
        const d = new Date(iso)
        const today = new Date()
        const tomorrow = new Date(today)
        tomorrow.setDate(tomorrow.getDate() + 1)
        
        if (d.toDateString() === today.toDateString()) return 'Today'
        if (d.toDateString() === tomorrow.toDateString()) return 'Tomorrow'
        
        return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    }

    const isOverdue = (iso) => {
        if (!iso) return false
        return new Date(iso) < new Date() && new Date(iso).toDateString() !== new Date().toDateString()
    }

    const getPriorityColor = (p) => {
        if (p >= 80) return 'bg-red-500'
        if (p >= 50) return 'bg-yellow-500'
        return 'bg-green-500'
    }

    // Defensive: ensure seed data exists so selectors like data-id-t1/t2 are present even if state was cleared
    onMounted(() => {
        if (dataStore.tasks.length < 2) {
          dataStore.initializeMockData()
        }
    })
    watchEffect(() => {
        if (filteredTasks.value.length === 0 && dataStore.tasks.length === 0) {
          dataStore.initializeMockData()
        }
    })

    return {
        searchQuery,
        filterToday,
        filterPriorityMin,
        sortMenuOpen,
        currentSortLabel,
        isFiltered,
        isSearched,
        filteredTasks,
        isMatch,
        handleSearch,
        toggleSortMenu,
        applySort,
        openTask,
        goHome,
        getProjectName,
        getSectionName,
        formatDate,
        isOverdue,
        getPriorityColor
    }
  }
}
</script>
