<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between">
        <h1 class="text-xl font-bold text-slate-800">Calls</h1>
        <button id="calls-back-chats" @click="goBackChats" class="p-2 text-slate-500 hover:text-blue-600">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
            </svg>
        </button>
      </div>
    </header>

    <div class="bg-white border-b border-slate-100 p-4 sticky top-0 z-10">
      <div class="max-w-2xl mx-auto space-y-3">
        <div class="relative">
          <input 
            id="calls-search-input"
            type="text" 
            placeholder="Search calls..." 
            v-model="searchQuery"
            @keyup.enter="performSearch"
            class="w-full pl-10 pr-4 py-2 bg-slate-100 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow"
          />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400 absolute left-3 top-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        <div class="flex items-center gap-2">
           <div 
             id="filter-missed-calls-checkbox" 
             @click="toggleMissed"
             :class="['px-3 py-1 rounded-full text-sm font-medium cursor-pointer transition-colors select-none', filters.missed ? 'bg-red-100 text-red-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200']"
           >
             Missed
           </div>

           <div class="relative ml-auto">
             <button id="calls-sort-dropdown" @click="showSort = !showSort" class="flex items-center text-sm font-medium text-slate-600 hover:text-blue-600">
               <span>Sort</span>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>
             <div v-if="showSort" class="absolute right-0 mt-2 w-32 bg-white rounded-lg shadow-xl border border-slate-100 py-1 z-50">
               <div id="calls-sort-option-recent-desc" @click="setSort('recent')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Recent</div>
               <div id="calls-sort-option-name-inc" @click="setSort('name')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Name</div>
               <div id="calls-sort-option-incoming" @click="setSort('incoming')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Incoming</div>
               <div id="calls-sort-option-outgoing" @click="setSort('outgoing')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Outgoing</div>
             </div>
           </div>
        </div>
      </div>
    </div>

    <div id="call-history-list-container" class="flex-1 overflow-y-auto bg-white">
      <div class="max-w-2xl mx-auto divide-y divide-slate-100" id="call-history-list">
        <div 
          v-for="call in displayedCalls" 
          :key="call.id"
          :class="['p-4 hover:bg-slate-50 cursor-pointer transition-colors flex items-center space-x-4', getItemClass(call.id)]"
          @click="openCall(call)"
        >
            <div class="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center">
                 <svg v-if="call.type === 'video'" xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                 </svg>
                 <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                 </svg>
            </div>
            
            <div class="flex-1 min-w-0">
                <div class="flex items-center justify-between mb-1">
                    <h3 class="text-base font-semibold text-slate-900 truncate">{{ getContact(call.contact_id).name }}</h3>
                    <span class="text-xs text-slate-500 whitespace-nowrap">{{ call.timestamp }}</span>
                </div>
                <div class="flex items-center space-x-2 text-sm text-slate-500">
                     <svg v-if="call.type === 'incoming'" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                     </svg>
                     <svg v-if="call.type === 'outgoing'" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18" />
                     </svg>
                     <span :class="{'text-red-500 font-medium': call.status === 'missed'}">
                        {{ call.status === 'missed' ? 'Missed Call' : call.duration }}
                     </span>
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
  name: 'CALL_HISTORY',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const showSort = ref(false)
    const filters = ref({
        missed: false
    })
    const sortBy = ref('recent')

    const getContact = (contactId) => {
        return dataStore.contacts.find(c => c.id === contactId) || { name: 'Unknown' }
    }

    const displayedCalls = computed(() => {
        let result = dataStore.calls || []

        // Search
        if (store.call_history_has_searched && store.matched_call_id) {
             result = result.filter(c => getContact(c.contact_id).name.toLowerCase().includes(searchQuery.value.toLowerCase()))
        } else if (searchQuery.value) {
             result = result.filter(c => getContact(c.contact_id).name.toLowerCase().includes(searchQuery.value.toLowerCase()))
        }

        // Filters
        if (filters.value.missed) result = result.filter(c => c.status === 'missed')

        // Sort
        if (sortBy.value === 'recent') {
            // Recent sort: timestamp field exists but is string format ("10:45 AM", "Yesterday")
            // Cannot reliably sort, keep original order (already roughly sorted by recency)
        } else if (sortBy.value === 'name') {
            result.sort((a, b) => getContact(a.contact_id).name.localeCompare(getContact(b.contact_id).name))
        } else if (sortBy.value === 'incoming') {
            result.sort((a, b) => (a.type === 'incoming' ? -1 : 1))
        } else if (sortBy.value === 'outgoing') {
            result.sort((a, b) => (a.type === 'outgoing' ? -1 : 1))
        }

        return result
    })

    const getItemClass = (id) => {
        if (store.call_history_has_searched && store.matched_call_id === id) return `call-row-matched data-id-${id}`
        if (store.call_history_filters_applied) return `call-row-filtered data-id-${id}`
        return `call-row-visible data-id-${id}`
    }

    const performSearch = () => {
        store.call_history_has_searched = true
        if (displayedCalls.value.length > 0) {
            store.matched_call_id = displayedCalls.value[0].id
        }
    }

    const toggleMissed = () => {
        filters.value.missed = !filters.value.missed
        store.call_history_filters_applied = true
    }

    const setSort = (type) => {
        sortBy.value = type
        showSort.value = false
        store.call_history_filters_applied = true
    }

    const openCall = async (call) => {
        store.selected_call_id = call.id
        store.call_history_filters_applied = null
        store.call_history_has_searched = null
        store.currentPageId = 'START_CALL_SETUP'
        await router.push({ name: 'START_CALL_SETUP' })
    }

    const goBackChats = async () => {
        store.currentPageId = 'CHATS_LIST'
        await router.push({ name: 'CHATS_LIST' })
    }

    return {
        searchQuery,
        showSort,
        filters,
        displayedCalls,
        getContact,
        getItemClass,
        performSearch,
        toggleMissed,
        setSort,
        openCall,
        goBackChats
    }
  }
}
</script>