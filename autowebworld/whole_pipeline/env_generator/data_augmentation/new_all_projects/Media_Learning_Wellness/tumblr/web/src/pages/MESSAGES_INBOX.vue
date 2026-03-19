<template>
  <div class="min-h-screen bg-slate-900 text-white flex flex-col md:flex-row">
    <!-- Sidebar Navigation (Reusable) -->
    <aside class="hidden md:flex flex-col fixed left-0 top-0 h-full w-20 lg:w-64 border-r border-slate-800 bg-slate-900 z-40">
      <div class="p-4 lg:p-6 text-2xl font-bold lg:text-3xl tracking-tighter mb-4">
        <span class="hidden lg:inline">tumblr</span>
        <span class="lg:hidden">t</span>
      </div>
      
      <nav class="flex-1 space-y-2 px-2 lg:px-4">
        <button id="messages-back-dashboard" @click="goDashboard" class="flex items-center gap-4 w-full p-3 rounded-full hover:bg-slate-800 transition-colors text-slate-200">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" /></svg>
           <span class="hidden lg:inline font-bold">Dashboard</span>
        </button>
      </nav>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 md:ml-20 lg:ml-64 max-w-4xl mx-auto w-full pt-6 px-4">
      <div class="flex justify-between items-center mb-6">
        <h1 class="text-3xl font-bold tracking-tight">Messages</h1>
        <button 
          id="messages-compose-button" 
          @click="goCompose"
          class="bg-blue-500 hover:bg-blue-600 text-white p-3 rounded-full shadow-lg transition-transform hover:scale-105"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" /></svg>
        </button>
      </div>

      <!-- Search & Filters -->
      <div class="mb-6 space-y-4">
        <!-- Search -->
        <div class="relative">
          <input 
            id="messages-search-input"
            type="text"
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            placeholder="Search messages..."
            class="w-full bg-slate-800 border-none rounded-full py-3 px-12 text-white placeholder-slate-400 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
          />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 absolute left-4 top-3.5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
        </div>

        <!-- Filters Row -->
        <div class="flex gap-4 items-center bg-slate-800/50 p-4 rounded-xl border border-slate-700/50">
          <!-- Checkbox -->
          <label class="flex items-center gap-2 cursor-pointer select-none">
            <input 
              id="filter-unread-checkbox"
              type="checkbox"
              v-model="filterUnread"
              class="w-5 h-5 rounded border-slate-500 text-blue-500 focus:ring-blue-500 bg-slate-700"
            />
            <span class="text-sm font-medium text-slate-300">Unread only</span>
          </label>

          <!-- Sort -->
          <div class="relative">
            <button 
              id="messages-sort-dropdown"
              @click="sortOpen = !sortOpen"
              class="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white"
            >
              Sort: <span class="text-blue-400">{{ currentSort === 'newest' ? 'Newest' : 'Oldest' }}</span>
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
            </button>
            
            <div v-if="sortOpen" class="absolute top-full left-0 mt-2 w-32 bg-slate-800 rounded-lg shadow-xl border border-slate-700 overflow-hidden z-20">
              <div id="messages-sort-option-newest" @click="setSort('newest')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Newest</div>
              <div id="messages-sort-option-oldest" @click="setSort('oldest')" class="px-4 py-2 hover:bg-slate-700 cursor-pointer text-sm">Oldest</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Messages List -->
      <div id="messages-list-container" class="space-y-2 pb-20">
         <div v-if="filteredMessages.length === 0" class="text-center py-20 text-slate-500">
           <p>No messages.</p>
         </div>

         <div 
           id="messages-list"
           v-for="msg in filteredMessages" 
           :key="msg.id"
           :class="[
             'bg-slate-800 hover:bg-slate-750 p-4 rounded-xl cursor-pointer flex items-center gap-4 transition-colors border border-transparent',
             hasSearched && msg.id === matchedMessageId ? 'message-row-matched border-blue-500' : 'message-row-visible',
             filtersApplied ? 'message-row-filtered' : ''
           ]"
           @click="openThread(msg.id)"
         >
           <img :src="msg.avatar" class="w-12 h-12 rounded-full object-cover" />
           
           <div class="flex-1 min-w-0">
             <div class="flex justify-between items-baseline mb-1">
               <h3 class="font-bold text-white truncate">{{ msg.recipient_name }}</h3>
               <span class="text-xs text-slate-400">{{ formatDate(msg.timestamp) }}</span>
             </div>
             <p :class="['text-sm truncate', msg.unread ? 'text-white font-bold' : 'text-slate-400']">
               {{ msg.last_message }}
             </p>
           </div>
           
           <div v-if="msg.unread" class="w-3 h-3 bg-blue-500 rounded-full"></div>
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
  name: 'MESSAGES_INBOX',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterUnread = ref(false)
    const currentSort = ref('newest')
    const sortOpen = ref(false)

    // FSM State Mappers
    const hasSearched = computed(() => store.messages_inbox_has_searched)
    const matchedMessageId = computed(() => store.matched_message_id)
    const filtersApplied = computed(() => store.messages_inbox_filters_applied)

    const filteredMessages = computed(() => {
      let result = [...dataStore.messages]

      if (filterUnread.value) {
        result = result.filter(m => m.unread)
      }

      if (searchQuery.value) {
        const query = searchQuery.value.toLowerCase()
        result = result.filter(m => 
          m.recipient_name.toLowerCase().includes(query) ||
          m.last_message.toLowerCase().includes(query)
        )
      }

      result.sort((a, b) => {
        const dateA = new Date(a.timestamp)
        const dateB = new Date(b.timestamp)
        return currentSort.value === 'newest' ? dateB - dateA : dateA - dateB
      })

      return result
    })

    const formatDate = (isoString) => {
      const date = new Date(isoString)
      // If today, show time, else date
      if (date.toDateString() === new Date().toDateString()) {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
      return date.toLocaleDateString()
    }

    const handleSearch = () => {
      store.messages_inbox_has_searched = true
      if (filteredMessages.value.length > 0) {
        store.matched_message_id = filteredMessages.value[0].id
      }
    }

    const setSort = (type) => {
      currentSort.value = type
      sortOpen.value = false
      store.messages_inbox_filters_applied = true
    }

    const goDashboard = async () => {
      store.currentPageId = 'DASHBOARD_FEED'
      await router.push({ name: 'DASHBOARD_FEED' })
    }

    const goCompose = async () => {
      store.currentPageId = 'MESSAGE_COMPOSE'
      await router.push({ name: 'MESSAGE_COMPOSE' })
    }

    const openThread = async (id) => {
      store.selected_message_id = id
      
      if (hasSearched.value) store.messages_inbox_has_searched = null
      if (filtersApplied.value) store.messages_inbox_filters_applied = null

      store.currentPageId = 'MESSAGE_THREAD'
      await router.push({ name: 'MESSAGE_THREAD', params: { id } })
    }

    return {
      store,
      searchQuery,
      filterUnread,
      currentSort,
      sortOpen,
      filteredMessages,
      hasSearched,
      matchedMessageId,
      filtersApplied,
      formatDate,
      handleSearch,
      setSort,
      goDashboard,
      goCompose,
      openThread
    }
  },
  watch: {
    filterUnread() { this.store.messages_inbox_filters_applied = true }
  }
}
</script>