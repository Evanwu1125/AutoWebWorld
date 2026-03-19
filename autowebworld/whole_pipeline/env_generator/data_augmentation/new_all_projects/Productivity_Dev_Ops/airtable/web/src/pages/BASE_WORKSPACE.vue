<template>
  <div class="h-screen flex flex-col bg-white overflow-hidden">
    <!-- Header -->
    <header class="h-14 bg-blue-600 flex items-center justify-between px-4 text-white shadow-md z-30">
      <div class="flex items-center gap-4">
        <button id="back-bases-dashboard" @click="goBack" class="p-1 hover:bg-blue-700 rounded transition-colors">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <div class="flex items-center gap-2">
          <div class="w-8 h-8 bg-white/20 rounded flex items-center justify-center text-lg">
             ⊞
          </div>
          <h1 class="font-bold text-lg tracking-wide">{{ currentBaseName }}</h1>
        </div>
      </div>
      
      <div class="flex items-center gap-2">
         <button id="nav-automations" @click="goToAutomations" class="px-3 py-1.5 bg-blue-700 hover:bg-blue-800 rounded text-sm font-medium transition-colors flex items-center gap-2">
           <span>⚡</span> Automations
         </button>
         <div class="flex -space-x-2">
           <div class="w-8 h-8 rounded-full bg-yellow-400 border-2 border-blue-600"></div>
           <div class="w-8 h-8 rounded-full bg-green-400 border-2 border-blue-600"></div>
         </div>
         <button class="bg-blue-500 hover:bg-blue-400 px-3 py-1.5 rounded text-sm font-medium ml-2">Share</button>
      </div>
    </header>

    <div class="flex flex-1 overflow-hidden">
      <!-- Sidebar Tables List -->
      <aside class="w-60 bg-gray-50 border-r border-gray-200 flex flex-col">
         <div class="p-3 border-b border-gray-200 flex justify-between items-center">
            <span class="text-xs font-bold text-gray-500 uppercase">Tables</span>
            <button class="text-gray-400 hover:text-blue-600">+</button>
         </div>
         <div class="overflow-y-auto flex-1 p-2 space-y-1">
            <div 
              v-for="table in tables" 
              :key="table.id"
              class="group flex items-center justify-between px-3 py-2 rounded cursor-pointer hover:bg-gray-200"
              :class="{'bg-blue-100 text-blue-700': selectedTableId === table.id}"
            >
               <span class="text-sm font-medium">{{ table.name }}</span>
            </div>
         </div>
      </aside>

      <!-- Main Content Area -->
      <main class="flex-1 flex flex-col bg-white">
        <!-- View Switcher Bar -->
        <div class="h-12 border-b border-gray-200 flex items-center px-4 gap-4 bg-white z-20">
           <div id="table-tabs" class="flex items-center gap-1">
              <div 
                 v-for="table in tables"
                 :key="'tab-' + table.id"
                 class="table-grid px-3 py-1.5 rounded hover:bg-gray-100 cursor-pointer text-sm font-medium text-gray-700 flex items-center gap-2"
                 @click="openGridView(table.id)"
              >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h18M3 14h18m-9-4v8m-7-8v8m14-8v8M5 21h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v14a2 2 0 002 2z" />
                </svg>
                {{ table.name }} Grid
              </div>
           </div>
           
           <div class="h-6 w-px bg-gray-300 mx-2"></div>
           
           <button id="view-switcher-kanban" @click="openKanbanView" class="px-3 py-1.5 rounded hover:bg-gray-100 cursor-pointer text-sm font-medium text-gray-700 flex items-center gap-2">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-purple-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 17V7m0 10a2 2 0 01-2 2H5a2 2 0 01-2-2V7a2 2 0 012-2h2a2 2 0 012 2m0 10a2 2 0 002 2h2a2 2 0 002-2M9 7a2 2 0 012-2h2a2 2 0 012 2m0 10V7m0 10a2 2 0 002 2h2a2 2 0 002-2V7a2 2 0 00-2-2h-2a2 2 0 00-2 2" />
             </svg>
             Kanban
           </button>
        </div>

        <!-- View Content Placeholder -->
        <div class="flex-1 flex flex-col items-center justify-center bg-gray-50 text-gray-500">
           <div class="w-16 h-16 mb-4 text-gray-300">
             <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
             </svg>
           </div>
           <p>Select a view above to start working</p>
        </div>
      </main>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'BASE_WORKSPACE',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const currentBaseName = computed(() => {
      const base = dataStore.bases.find(b => b.id === store.selected_base_id)
      return base ? base.name : 'Untitled Base'
    })

    // Filter tables for current base
    const tables = computed(() => {
      // For mock purposes, if no tables found for base, return some defaults or filter properly
      // Assuming dataStore.tables has base_id field
      return dataStore.tables.filter(t => t.base_id === store.selected_base_id)
    })

    const selectedTableId = computed(() => store.selected_table_id)

    const goBack = async () => {
      store.setCurrentPageId('BASES_DASHBOARD')
      await router.push({ name: 'BASES_DASHBOARD' })
    }

    const goToAutomations = async () => {
      store.setCurrentPageId('AUTOMATIONS_DASHBOARD')
      await router.push({ name: 'AUTOMATIONS_DASHBOARD' })
    }

    const openGridView = async (tableId) => {
      store.selected_table_id = tableId
      store.setCurrentPageId('TABLE_GRID_VIEW')
      await router.push({ name: 'TABLE_GRID_VIEW' })
    }

    const openKanbanView = async () => {
      // Default to first table if none selected
      if (!store.selected_table_id && tables.value.length > 0) {
        store.selected_table_id = tables.value[0].id
      }
      store.setCurrentPageId('KANBAN_VIEW')
      await router.push({ name: 'KANBAN_VIEW' })
    }

    return {
      currentBaseName,
      tables,
      selectedTableId,
      goBack,
      goToAutomations,
      openGridView,
      openKanbanView
    }
  }
}
</script>