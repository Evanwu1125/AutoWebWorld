<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between">
        <h1 class="text-xl font-bold text-slate-800">Chats</h1>
        <div class="flex items-center space-x-3">
            <button id="nav-calls" @click="goToCalls" class="p-2 text-slate-500 hover:text-blue-600">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                </svg>
            </button>
            <button id="back-home" @click="goBackHome" class="p-2 text-slate-500 hover:text-blue-600">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                </svg>
            </button>
        </div>
      </div>
    </header>

    <!-- Search & Filter Bar -->
    <div class="bg-white border-b border-slate-100 p-4 sticky top-0 z-10">
      <div class="max-w-2xl mx-auto space-y-3">
        <!-- Search -->
        <div class="relative">
          <input 
            id="chat-search-input"
            type="text" 
            placeholder="Search chats..." 
            v-model="searchQuery"
            @keyup.enter="performSearch"
            class="w-full pl-10 pr-4 py-2 bg-slate-100 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow"
          />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400 absolute left-3 top-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        <!-- Filters -->
        <div class="flex flex-wrap items-center gap-2">
           <!-- Unread Filter -->
           <div 
             id="filter-unread-checkbox" 
             @click="toggleUnread"
             :class="['px-3 py-1 rounded-full text-sm font-medium cursor-pointer transition-colors select-none', filters.unread ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200']"
           >
             Unread
           </div>
           <!-- Muted Filter -->
           <div 
             id="filter-muted-checkbox"
             @click="toggleMuted"
             :class="['px-3 py-1 rounded-full text-sm font-medium cursor-pointer transition-colors select-none', filters.muted ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200']"
           >
             Muted
           </div>
           <!-- Pinned Filter -->
           <div 
             id="filter-pinned-checkbox"
             @click="togglePinned"
             :class="['px-3 py-1 rounded-full text-sm font-medium cursor-pointer transition-colors select-none', filters.pinned ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200']"
           >
             Pinned
           </div>

           <!-- Sort Dropdown -->
           <div class="relative ml-auto">
             <button id="sort-dropdown" @click="showSort = !showSort" class="flex items-center text-sm font-medium text-slate-600 hover:text-blue-600">
               <span>Sort</span>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>
             <div v-if="showSort" class="absolute right-0 mt-2 w-32 bg-white rounded-lg shadow-xl border border-slate-100 py-1 z-50">
               <div id="sort-option-recent-desc" @click="setSort('recent')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Recent</div>
               <div id="sort-option-unread-desc" @click="setSort('unread')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Unread</div>
               <div id="sort-option-name-inc" @click="setSort('name')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Name</div>
             </div>
           </div>
        </div>
      </div>
    </div>

    <!-- Chat List -->
    <div id="chat-list-container" class="flex-1 overflow-y-auto bg-white">
      <div class="max-w-2xl mx-auto divide-y divide-slate-100" id="chat-list">
        <div 
          v-for="chat in displayedChats" 
          :key="chat.id"
          :class="['p-4 hover:bg-slate-50 cursor-pointer transition-colors flex items-center space-x-4', getItemClass(chat.id)]"
          @click="openChat(chat)"
        >
            <!-- Avatar -->
            <div class="relative">
                <img :src="getContact(chat.contact_id).avatar" alt="Avatar" class="w-12 h-12 rounded-full object-cover border border-slate-200" />
                <div v-if="chat.unread > 0" class="absolute -top-1 -right-1 bg-blue-600 text-white text-xs font-bold w-5 h-5 flex items-center justify-center rounded-full border-2 border-white">
                    {{ chat.unread }}
                </div>
            </div>

            <!-- Content -->
            <div class="flex-1 min-w-0">
                <div class="flex items-center justify-between mb-1">
                    <h3 class="text-base font-semibold text-slate-900 truncate">{{ getContact(chat.contact_id).name }}</h3>
                    <span class="text-xs text-slate-500 whitespace-nowrap">{{ chat.timestamp }}</span>
                </div>
                <div class="flex items-center justify-between">
                    <p class="text-sm text-slate-500 truncate pr-4">{{ chat.last_message }}</p>
                    <div class="flex items-center space-x-2">
                        <svg v-if="chat.pinned" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-slate-400" viewBox="0 0 20 20" fill="currentColor">
                           <path d="M5 4a2 2 0 012-2h6a2 2 0 012 2v14l-5-2.5L5 18V4z" />
                        </svg>
                        <svg v-if="chat.muted" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" clip-rule="evenodd" />
                           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
                        </svg>
                    </div>
                </div>
            </div>
        </div>
      </div>
    </div>

    <!-- FAB for New Chat -->
    <div class="fixed bottom-6 right-6 z-30">
        <button 
          id="new-chat-button" 
          @click="goToNewChat"
          class="w-14 h-14 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg flex items-center justify-center transition-transform hover:scale-105"
        >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
            </svg>
        </button>
    </div>

  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CHATS_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const showSort = ref(false)
    const filters = ref({
        unread: false,
        muted: false,
        pinned: false
    })
    const sortBy = ref('recent')

    const getContact = (contactId) => {
        return dataStore.contacts.find(c => c.id === contactId) || { name: 'Unknown', avatar: '/images/Avatar.jpg' }
    }

    const displayedChats = computed(() => {
        let result = dataStore.chats || []

        // Search
        if (store.chats_list_has_searched && store.matched_chat_id) {
            // If searched in FSM flow, we might just return the matched one or filter list
            // But here we implement real search logic
             result = result.filter(c => {
                const contact = getContact(c.contact_id)
                return contact.name.toLowerCase().includes(searchQuery.value.toLowerCase()) || 
                       c.last_message.toLowerCase().includes(searchQuery.value.toLowerCase())
            })
        } else if (searchQuery.value) {
             result = result.filter(c => {
                const contact = getContact(c.contact_id)
                return contact.name.toLowerCase().includes(searchQuery.value.toLowerCase()) || 
                       c.last_message.toLowerCase().includes(searchQuery.value.toLowerCase())
            })
        }

        // Filters
        if (filters.value.unread) result = result.filter(c => c.unread > 0)
        if (filters.value.muted) result = result.filter(c => c.muted)
        if (filters.value.pinned) result = result.filter(c => c.pinned)

        // Sort
        if (sortBy.value === 'recent') {
            // Recent sort: timestamp field exists but is string format ("10:30 AM", "Yesterday")
            // Cannot reliably sort, keep original order (already roughly sorted by recency)
        } else if (sortBy.value === 'unread') {
            result.sort((a, b) => b.unread - a.unread)
        } else if (sortBy.value === 'name') {
            result.sort((a, b) => getContact(a.contact_id).name.localeCompare(getContact(b.contact_id).name))
        }

        return result
    })

    const getItemClass = (id) => {
        // Return classes based on search state for FSM selectors
        if (store.chats_list_has_searched && store.matched_chat_id === id) return `chat-row-matched data-id-${id}`
        if (store.chats_list_filters_applied) return `chat-row-filtered data-id-${id}`
        return `chat-row-visible data-id-${id}`
    }

    const performSearch = () => {
        store.chats_list_has_searched = true
        // In FSM, we set matched_chat_id to the first result's ID usually
        if (displayedChats.value.length > 0) {
            store.matched_chat_id = displayedChats.value[0].id
        }
    }

    const toggleUnread = () => {
        filters.value.unread = !filters.value.unread
        store.chats_list_filters_applied = true
    }

    const toggleMuted = () => {
        filters.value.muted = !filters.value.muted
        store.chats_list_filters_applied = true
    }

    const togglePinned = () => {
        filters.value.pinned = !filters.value.pinned
        store.chats_list_filters_applied = true
    }

    const setSort = (type) => {
        sortBy.value = type
        showSort.value = false
        store.chats_list_filters_applied = true
    }

    const openChat = async (chat) => {
        store.selected_chat_id = chat.id
        store.chats_list_filters_applied = null // clear flags
        store.chats_list_has_searched = null
        store.currentPageId = 'CHAT_THREAD'
        await router.push({ name: 'CHAT_THREAD', params: { id: chat.id } })
    }

    const goToNewChat = async () => {
        store.currentPageId = 'NEW_CHAT_CHOOSE_CONTACT'
        await router.push({ name: 'NEW_CHAT_CHOOSE_CONTACT' })
    }

    const goBackHome = async () => {
        store.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    const goToCalls = async () => {
        store.currentPageId = 'CALL_HISTORY'
        await router.push({ name: 'CALL_HISTORY' })
    }

    return {
        searchQuery,
        showSort,
        filters,
        displayedChats,
        getContact,
        getItemClass,
        performSearch,
        toggleUnread,
        toggleMuted,
        togglePinned,
        setSort,
        openChat,
        goToNewChat,
        goBackHome,
        goToCalls
    }
  }
}
</script>