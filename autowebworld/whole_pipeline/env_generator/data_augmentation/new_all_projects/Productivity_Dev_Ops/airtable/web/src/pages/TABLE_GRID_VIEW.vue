<template>
  <div class="h-screen flex flex-col bg-white overflow-hidden">
    <!-- Toolbar -->
    <div class="h-14 border-b border-gray-200 flex items-center justify-between px-4 bg-white z-20 shadow-sm">
      <div class="flex items-center gap-2">
         <button id="back-base-workspace" @click="goBack" class="mr-2 text-gray-500 hover:text-blue-600">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
           </svg>
         </button>
         <h2 class="font-bold text-gray-800 flex items-center gap-2">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M3 14h18m-9-4v8m-7-8v8m14-8v8M5 21h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v14a2 2 0 002 2z" />
           </svg>
           Grid View
         </h2>
      </div>

      <div class="flex items-center gap-3">
         <!-- Search -->
         <div class="relative">
           <input 
              id="grid-search-input"
              v-model="searchQuery"
              @keyup.enter="handleSearch"
              type="text" 
              placeholder="Find in view" 
              class="pl-8 pr-3 py-1.5 border border-gray-200 rounded-md text-sm focus:outline-none focus:ring-1 focus:ring-blue-500 w-48 transition-all"
           >
           <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-400 absolute left-2 top-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
           </svg>
         </div>
         
         <!-- Filter: Status -->
         <button 
           id="filter-status-checkbox"
           @click="toggleStatusFilter"
           class="px-3 py-1.5 border border-dashed border-gray-300 rounded hover:bg-gray-50 text-sm font-medium text-gray-600 flex items-center gap-2"
           :class="{'bg-blue-50 border-blue-200 text-blue-700': filterStatus}"
         >
           <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
           </svg>
           Filter: Active
         </button>

         <!-- Filter: Priority (Slider) -->
         <div class="flex items-center gap-2 border border-dashed border-gray-300 rounded px-2 py-1">
            <span class="text-xs font-medium text-gray-500">Priority ></span>
            <input 
               id="priority-slider"
               type="range" 
               min="0" 
               max="3" 
               step="1"
               v-model.number="filterPriority"
               @input="handlePriorityChange"
               class="w-20 h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            >
         </div>

         <!-- Sort -->
         <div class="relative">
            <button 
               id="grid-sort-dropdown"
               @click="sortOpen = !sortOpen"
               class="px-3 py-1.5 hover:bg-gray-100 rounded text-sm font-medium text-gray-600 flex items-center gap-1"
            >
               <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16V4m0 0L3 8m4-4l4 4m6 0v12m0 0l4-4m-4 4l-4-4" />
               </svg>
               {{ sortLabel }}
            </button>
            <div v-if="sortOpen" class="absolute top-full right-0 w-40 mt-1 bg-white border border-gray-200 rounded shadow-lg z-20">
               <div id="grid-sort-due-date" @click="setSort('due-date')" class="px-4 py-2 hover:bg-gray-50 text-sm cursor-pointer">Due Date</div>
               <div id="grid-sort-created" @click="setSort('created')" class="px-4 py-2 hover:bg-gray-50 text-sm cursor-pointer">Created Date</div>
               <div id="grid-sort-priority" @click="setSort('priority')" class="px-4 py-2 hover:bg-gray-50 text-sm cursor-pointer">Priority</div>
            </div>
         </div>
      </div>
    </div>

    <!-- Grid Header -->
    <div class="flex bg-gray-100 border-b border-gray-300 text-xs font-bold text-gray-500 uppercase tracking-wider sticky top-0 z-10">
      <div class="w-10 p-2 text-center border-r border-gray-300">#</div>
      <div class="flex-1 p-2 border-r border-gray-300 min-w-[200px]">Name</div>
      <div class="w-32 p-2 border-r border-gray-300">Status</div>
      <div class="w-32 p-2 border-r border-gray-300">Due Date</div>
      <div class="w-24 p-2 border-r border-gray-300">Priority</div>
      <div class="w-40 p-2 border-r border-gray-300">Assignee</div>
      <div class="w-10 p-2 flex items-center justify-center">
         <button id="add-record-button" @click="openCreateRecord" class="w-6 h-6 bg-blue-600 hover:bg-blue-700 text-white rounded flex items-center justify-center shadow-sm transition-colors">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
            </svg>
         </button>
      </div>
    </div>

    <!-- Grid Body -->
    <div id="grid-viewport" class="flex-1 overflow-y-auto bg-white" @scroll="handleScroll">
       <div id="grid-records">
          <div 
             v-for="(record, index) in displayRecords" 
             :key="record.id"
             :class="[
                'flex border-b border-gray-200 hover:bg-blue-50 cursor-pointer transition-colors text-sm text-gray-800',
                'data-id-' + record.id,
                rowClass(record)
             ]"
             @click="openRecord(record)"
          >
             <div class="w-10 p-2 text-center border-r border-gray-100 text-gray-400 text-xs">{{ index + 1 }}</div>
             <div class="flex-1 p-2 border-r border-gray-100 font-medium flex items-center gap-2">
               <img v-if="record.image" :src="record.image" class="w-6 h-6 rounded object-cover" />
               {{ record.title }}
             </div>
             <div class="w-32 p-2 border-r border-gray-100 flex items-center">
                <span :class="statusBadgeClass(record.status)">{{ record.status }}</span>
             </div>
             <div class="w-32 p-2 border-r border-gray-100 text-gray-600">{{ formatDate(record.due_date) }}</div>
             <div class="w-24 p-2 border-r border-gray-100">
                <span class="px-2 py-0.5 rounded text-xs" :class="priorityClass(record.priority)">{{ record.priority }}</span>
             </div>
             <div class="w-40 p-2 border-r border-gray-100 flex items-center gap-2">
                <div class="w-5 h-5 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-xs font-bold">
                   {{ record.assigned_to ? record.assigned_to[0] : '?' }}
                </div>
                {{ record.assigned_to }}
             </div>
             <div class="w-10 p-2"></div>
          </div>

          <!-- Empty State -->
          <div v-if="displayRecords.length === 0" class="p-8 text-center text-gray-500">
             No records found.
          </div>
       </div>
       
       <!-- Spacer for scroll -->
       <div class="h-20"></div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'TABLE_GRID_VIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const filterStatus = ref(false)
    const filterPriority = ref(0) // 0: All, 1: Low, 2: Medium, 3: High
    const sortBy = ref('')
    const sortOpen = ref(false)
    const searchQuery = ref('')

    const sortLabel = computed(() => {
      if (sortBy.value === 'due-date') return 'Due Date'
      if (sortBy.value === 'created') return 'Created'
      if (sortBy.value === 'priority') return 'Priority'
      return 'Sort'
    })

    const displayRecords = computed(() => {
      let result = [...dataStore.records]
      
      // Filter by table
      if (store.selected_table_id) {
        result = result.filter(r => r.table_id === store.selected_table_id)
      }

      // 1. Search
      if (store.table_grid_has_searched && store.matched_record_id) {
        return result.filter(r => r.id === store.matched_record_id)
      }
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(r => r.title.toLowerCase().includes(q) || (r.assigned_to && r.assigned_to.toLowerCase().includes(q)))
      }

      // 2. Filters
      if (filterStatus.value) {
        // "Active" typically means not done
        result = result.filter(r => r.status !== 'Done')
      }
      
      if (filterPriority.value > 0) {
        // Simple mapping: Low=1, Medium=2, High=3
        const pMap = { 'Low': 1, 'Medium': 2, 'High': 3 }
        result = result.filter(r => (pMap[r.priority] || 0) >= filterPriority.value)
      }

      // 3. Sort
      if (sortBy.value === 'due-date') {
        result.sort((a, b) => new Date(a.due_date) - new Date(b.due_date))
      } else if (sortBy.value === 'priority') {
         const pMap = { 'High': 3, 'Medium': 2, 'Low': 1 }
         result.sort((a, b) => (pMap[b.priority] || 0) - (pMap[a.priority] || 0))
      }

      return result
    })

    const rowClass = (record) => {
      if (store.table_grid_has_searched && record.id === store.matched_record_id) {
        return 'record-row-matched bg-yellow-50'
      }
      if (store.table_grid_filters_applied) {
        return 'record-row-filtered'
      }
      return 'record-row-visible'
    }

    const statusBadgeClass = (status) => {
      if (status === 'Done') return 'bg-green-100 text-green-700 px-2 py-0.5 rounded-full text-xs font-semibold'
      if (status === 'In progress') return 'bg-blue-100 text-blue-700 px-2 py-0.5 rounded-full text-xs font-semibold'
      if (status === 'To do') return 'bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full text-xs font-semibold'
      return 'bg-gray-100 text-gray-700 px-2 py-0.5 rounded-full text-xs'
    }

    const priorityClass = (priority) => {
      if (priority === 'High') return 'bg-red-100 text-red-700'
      if (priority === 'Medium') return 'bg-yellow-100 text-yellow-700'
      return 'bg-green-100 text-green-700'
    }

    const formatDate = (d) => d // Simplified

    // Actions
    const goBack = async () => {
      store.setCurrentPageId('BASE_WORKSPACE')
      await router.push({ name: 'BASE_WORKSPACE' })
    }

    const toggleStatusFilter = () => {
      filterStatus.value = !filterStatus.value
      store.table_grid_filters_applied = true
    }

    const handlePriorityChange = () => {
      store.table_grid_filters_applied = true
    }

    const setSort = (type) => {
      sortBy.value = type
      sortOpen.value = false
      store.table_grid_filters_applied = true
    }

    const handleSearch = () => {
      const match = displayRecords.value.find(r => r.title.toLowerCase().includes(searchQuery.value.toLowerCase()))
      if (match) {
        store.matched_record_id = match.id
        store.table_grid_has_searched = true
      }
    }

    const openCreateRecord = async () => {
      store.setCurrentPageId('RECORD_CREATE_FORM')
      await router.push({ name: 'RECORD_CREATE_FORM' })
    }

    const openRecord = async (record) => {
      store.selected_record_id = record.id
      store.table_grid_filters_applied = false
      store.table_grid_has_searched = false
      
      store.setCurrentPageId('RECORD_DETAIL')
      await router.push({ name: 'RECORD_DETAIL' })
    }
    
    const handleScroll = () => {
       // ACT_GRID_SCROLL_RECORD_INTO_VIEW
    }

    return {
      searchQuery,
      filterStatus,
      filterPriority,
      sortOpen,
      sortLabel,
      displayRecords,
      
      goBack,
      toggleStatusFilter,
      handlePriorityChange,
      setSort,
      handleSearch,
      openCreateRecord,
      openRecord,
      handleScroll,
      rowClass,
      statusBadgeClass,
      priorityClass,
      formatDate
    }
  }
}
</script>