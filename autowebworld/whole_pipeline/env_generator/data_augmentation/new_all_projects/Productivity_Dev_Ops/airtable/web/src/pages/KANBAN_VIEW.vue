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
           <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
           </svg>
           Kanban View
         </h2>
      </div>

      <div class="flex items-center gap-3">
         <!-- Filter: Status -->
         <button 
           id="kanban-filter-status-checkbox"
           @click="toggleStatusFilter"
           class="px-3 py-1.5 border border-dashed border-gray-300 rounded hover:bg-gray-50 text-sm font-medium text-gray-600 flex items-center gap-2"
           :class="{'bg-blue-50 border-blue-200 text-blue-700': filterStatus}"
         >
           <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
           </svg>
           Hide Done
         </button>
      </div>
    </div>

    <!-- Kanban Board -->
    <div id="kanban-board" class="flex-1 overflow-x-auto overflow-y-hidden bg-gray-100 p-6" @scroll="handleScroll">
       <div class="flex gap-6 h-full min-w-max">
          <!-- Column: To Do -->
          <div class="w-80 flex flex-col h-full bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
             <div class="p-3 border-b border-gray-200 flex justify-between items-center bg-white rounded-t-lg">
                <div class="flex items-center gap-2">
                   <div class="w-3 h-3 rounded-full bg-gray-400"></div>
                   <h3 class="font-bold text-gray-700">To Do</h3>
                </div>
                <span class="text-xs font-bold text-gray-400">{{ getRecordsByStatus('To do').length }}</span>
             </div>
             <div class="flex-1 overflow-y-auto p-3 space-y-3">
                <div 
                   v-for="record in getRecordsByStatus('To do')"
                   :key="record.id"
                   :class="['bg-white p-3 rounded shadow-sm border border-gray-100 hover:shadow-md cursor-pointer group', 'kanban-card-visible', 'data-id-' + record.id]"
                   @click="openRecord(record)"
                >
                   <div v-if="record.image" class="h-32 mb-3 rounded overflow-hidden">
                     <img :src="record.image" class="w-full h-full object-cover transition-transform group-hover:scale-105" />
                   </div>
                   <div class="font-medium text-gray-800 mb-2">{{ record.title }}</div>
                   <div class="flex justify-between items-center">
                      <div class="flex -space-x-2">
                        <div class="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-xs font-bold border-2 border-white">
                           {{ record.assigned_to ? record.assigned_to[0] : '?' }}
                        </div>
                      </div>
                      <span class="text-xs text-gray-500">{{ formatDate(record.due_date) }}</span>
                   </div>
                </div>
             </div>
          </div>

          <!-- Column: In Progress -->
          <div class="w-80 flex flex-col h-full bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
             <div class="p-3 border-b border-gray-200 flex justify-between items-center bg-white rounded-t-lg">
                <div class="flex items-center gap-2">
                   <div class="w-3 h-3 rounded-full bg-blue-400"></div>
                   <h3 class="font-bold text-gray-700">In Progress</h3>
                </div>
                <span class="text-xs font-bold text-gray-400">{{ getRecordsByStatus('In progress').length }}</span>
             </div>
             <div class="flex-1 overflow-y-auto p-3 space-y-3">
                <div 
                   v-for="record in getRecordsByStatus('In progress')"
                   :key="record.id"
                   :class="['bg-white p-3 rounded shadow-sm border border-gray-100 hover:shadow-md cursor-pointer group', 'kanban-card-visible', 'data-id-' + record.id]"
                   @click="openRecord(record)"
                >
                   <div v-if="record.image" class="h-32 mb-3 rounded overflow-hidden">
                     <img :src="record.image" class="w-full h-full object-cover transition-transform group-hover:scale-105" />
                   </div>
                   <div class="font-medium text-gray-800 mb-2">{{ record.title }}</div>
                   <div class="flex justify-between items-center">
                      <div class="flex -space-x-2">
                        <div class="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-xs font-bold border-2 border-white">
                           {{ record.assigned_to ? record.assigned_to[0] : '?' }}
                        </div>
                      </div>
                      <span class="text-xs text-gray-500">{{ formatDate(record.due_date) }}</span>
                   </div>
                </div>
             </div>
          </div>

          <!-- Column: Done -->
          <div v-if="!filterStatus" class="w-80 flex flex-col h-full bg-gray-50 rounded-lg border border-gray-200 shadow-sm">
             <div class="p-3 border-b border-gray-200 flex justify-between items-center bg-white rounded-t-lg">
                <div class="flex items-center gap-2">
                   <div class="w-3 h-3 rounded-full bg-green-400"></div>
                   <h3 class="font-bold text-gray-700">Done</h3>
                </div>
                <span class="text-xs font-bold text-gray-400">{{ getRecordsByStatus('Done').length }}</span>
             </div>
             <div class="flex-1 overflow-y-auto p-3 space-y-3">
                <div 
                   v-for="record in getRecordsByStatus('Done')"
                   :key="record.id"
                   :class="['bg-white p-3 rounded shadow-sm border border-gray-100 hover:shadow-md cursor-pointer group', 'kanban-card-visible', 'data-id-' + record.id]"
                   @click="openRecord(record)"
                >
                   <div v-if="record.image" class="h-32 mb-3 rounded overflow-hidden">
                     <img :src="record.image" class="w-full h-full object-cover transition-transform group-hover:scale-105" />
                   </div>
                   <div class="font-medium text-gray-800 mb-2">{{ record.title }}</div>
                   <div class="flex justify-between items-center">
                      <div class="flex -space-x-2">
                         <div class="w-6 h-6 rounded-full bg-blue-100 text-blue-600 flex items-center justify-center text-xs font-bold border-2 border-white">
                           {{ record.assigned_to ? record.assigned_to[0] : '?' }}
                        </div>
                      </div>
                      <span class="text-xs text-gray-500">{{ formatDate(record.due_date) }}</span>
                   </div>
                </div>
             </div>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'KANBAN_VIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const filterStatus = ref(false)

    const allRecords = computed(() => {
      let result = [...dataStore.records]
      if (store.selected_table_id) {
        result = result.filter(r => r.table_id === store.selected_table_id)
      }
      return result
    })

    const getRecordsByStatus = (status) => {
      return allRecords.value.filter(r => r.status === status)
    }

    const formatDate = (d) => {
       if(!d) return ''
       const date = new Date(d)
       return date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    }

    const goBack = async () => {
      store.setCurrentPageId('BASE_WORKSPACE')
      await router.push({ name: 'BASE_WORKSPACE' })
    }

    const toggleStatusFilter = () => {
      // ACT_KANBAN_FILTER_STATUS_CHECKBOX
      filterStatus.value = !filterStatus.value
      store.kanban_filters_applied = true
    }

    const openRecord = async (record) => {
      // ACT_KANBAN_OPEN_RECORD
      store.selected_record_id = record.id
      store.kanban_viewport_anchor_id = null
      
      store.setCurrentPageId('RECORD_DETAIL')
      await router.push({ name: 'RECORD_DETAIL' })
    }

    const handleScroll = () => {
      // ACT_KANBAN_SCROLL_COLUMN
    }

    return {
      filterStatus,
      getRecordsByStatus,
      formatDate,
      goBack,
      toggleStatusFilter,
      openRecord,
      handleScroll
    }
  }
}
</script>