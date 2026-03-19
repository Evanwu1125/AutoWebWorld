<template>
  <div class="h-screen flex flex-col bg-gray-50 overflow-hidden">
    <!-- Header -->
    <header class="h-14 bg-white border-b border-gray-200 flex items-center justify-between px-6 shadow-sm z-30">
      <div class="flex items-center gap-4">
        <button id="back-base-workspace" @click="goBack" class="p-1 hover:bg-gray-100 rounded transition-colors text-gray-500">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <h1 class="font-bold text-lg text-gray-900 flex items-center gap-2">
           <span>⚡</span> Automations
        </h1>
      </div>
      
      <button 
        id="create-automation-button" 
        @click="createAutomation" 
        class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md shadow-sm transition-all flex items-center gap-2"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
        </svg>
        New Automation
      </button>
    </header>

    <div class="flex flex-1 overflow-hidden">
       <!-- Sidebar / Filters -->
       <aside class="w-64 bg-white border-r border-gray-200 p-6 hidden md:block">
          <div class="mb-6">
             <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4">View</h3>
             <label class="flex items-center gap-3 cursor-pointer group">
               <div id="filter-active-checkbox" 
                    class="w-5 h-5 border-2 border-gray-300 rounded transition-colors flex items-center justify-center group-hover:border-blue-400"
                    :class="{'bg-blue-600 border-blue-600': filterActive}"
                    @click="toggleActiveFilter">
                 <svg v-if="filterActive" xmlns="http://www.w3.org/2000/svg" class="h-3.5 w-3.5 text-white" viewBox="0 0 20 20" fill="currentColor">
                   <path fill-rule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clip-rule="evenodd" />
                 </svg>
               </div>
               <span class="text-sm font-medium text-gray-700 group-hover:text-gray-900">Active Only</span>
             </label>
          </div>
       </aside>

       <!-- Main List -->
       <main id="automations-list" class="flex-1 p-8 overflow-y-auto" @scroll="handleScroll">
          <div class="max-w-3xl mx-auto space-y-4">
             <div 
               v-for="auto in displayAutomations" 
               :key="auto.id"
               class="bg-white rounded-lg border border-gray-200 p-4 shadow-sm hover:shadow-md transition-shadow flex items-center gap-4 cursor-pointer"
               @click="selectAutomation(auto)"
             >
                <!-- Icon -->
                <div class="w-12 h-12 rounded-lg bg-blue-50 flex items-center justify-center text-2xl flex-shrink-0">
                   <span v-if="auto.trigger === 'when-record-created'">📝</span>
                   <span v-else-if="auto.trigger === 'when-record-updated'">🔄</span>
                   <span v-else-if="auto.trigger === 'at-scheduled-time'">⏰</span>
                   <span v-else>⚡</span>
                </div>

                <div class="flex-1">
                   <h3 class="font-bold text-gray-900 text-lg">{{ auto.name }}</h3>
                   <div class="text-sm text-gray-500 flex items-center gap-2 mt-1">
                      <span class="font-medium bg-gray-100 px-2 py-0.5 rounded text-xs">{{ formatTrigger(auto.trigger) }}</span>
                      <span>→</span>
                      <span class="font-medium bg-gray-100 px-2 py-0.5 rounded text-xs">{{ formatAction(auto.action) }}</span>
                   </div>
                </div>

                <!-- Toggle Switch (Visual) -->
                <div class="w-10 h-6 rounded-full relative transition-colors" :class="auto.active ? 'bg-green-500' : 'bg-gray-300'">
                   <div class="absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform" :class="{'translate-x-4': auto.active}"></div>
                </div>
             </div>

             <div v-if="displayAutomations.length === 0" class="text-center py-12 text-gray-500">
                No automations found. Create one to get started!
             </div>
          </div>
          
          <div class="h-20"></div>
       </main>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'AUTOMATIONS_DASHBOARD',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const filterActive = ref(false)

    const displayAutomations = computed(() => {
      let result = [...dataStore.automations]
      // Filter by base
      // Assuming automations have base_id
      if (store.selected_base_id) {
        result = result.filter(a => a.base_id === store.selected_base_id)
      }
      
      if (filterActive.value) {
        result = result.filter(a => a.active)
      }
      return result
    })

    const formatTrigger = (t) => {
      if (t === 'when-record-created') return 'When record created'
      if (t === 'when-record-updated') return 'When record updated'
      if (t === 'at-scheduled-time') return 'At scheduled time'
      return t
    }

    const formatAction = (a) => {
      if (a === 'send-email') return 'Send email'
      if (a === 'update-record') return 'Update record'
      if (a === 'create-record') return 'Create record'
      return a
    }

    const goBack = async () => {
      store.setCurrentPageId('BASE_WORKSPACE')
      await router.push({ name: 'BASE_WORKSPACE' })
    }

    const createAutomation = async () => {
      store.setCurrentPageId('AUTOMATION_CREATE_TRIGGER')
      await router.push({ name: 'AUTOMATION_CREATE_TRIGGER' })
    }

    const toggleActiveFilter = () => {
      filterActive.value = !filterActive.value
      store.automations_filters_applied = true
    }
    
    const selectAutomation = (auto) => {
      store.automations_viewport_anchor_id = auto.id
    }

    const handleScroll = () => {
      // ACT_AUTOMATIONS_SCROLL
    }

    return {
      filterActive,
      displayAutomations,
      formatTrigger,
      formatAction,
      goBack,
      createAutomation,
      toggleActiveFilter,
      selectAutomation,
      handleScroll
    }
  }
}
</script>