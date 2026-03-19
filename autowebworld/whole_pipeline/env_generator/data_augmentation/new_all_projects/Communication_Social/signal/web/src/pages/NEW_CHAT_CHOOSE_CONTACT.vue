<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between">
        <h1 class="text-xl font-bold text-slate-800">New Chat</h1>
        <button id="back-chats" @click="goBackChats" class="p-2 text-slate-500 hover:text-blue-600">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
            </svg>
        </button>
      </div>
    </header>

    <div class="bg-white border-b border-slate-100 p-4 sticky top-0 z-10">
      <div class="max-w-2xl mx-auto">
        <div class="relative">
          <input 
            id="contact-search-input"
            type="text" 
            placeholder="Search for a contact..." 
            v-model="searchQuery"
            @keyup.enter="performSearch"
            class="w-full pl-10 pr-4 py-3 bg-slate-100 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 transition-shadow"
          />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-400 absolute left-3 top-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>
    </div>

    <div id="contact-list-container" class="flex-1 overflow-y-auto bg-white">
      <div class="max-w-2xl mx-auto divide-y divide-slate-100" id="contact-list">
        <div 
          v-for="contact in displayedContacts" 
          :key="contact.id"
          :class="['p-4 hover:bg-slate-50 cursor-pointer transition-colors flex items-center space-x-4', getItemClass(contact.id)]"
          @click="selectContact(contact)"
        >
            <img :src="contact.avatar" alt="Avatar" class="w-12 h-12 rounded-full object-cover border border-slate-200" />
            
            <div class="flex-1 min-w-0">
                <h3 class="text-base font-semibold text-slate-900 truncate">{{ contact.name }}</h3>
                <p class="text-sm text-slate-500">{{ contact.phone }}</p>
            </div>
            
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-slate-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7" />
            </svg>
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
  name: 'NEW_CHAT_CHOOSE_CONTACT',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')

    const displayedContacts = computed(() => {
        let result = dataStore.contacts || []
        
        // Search
        if (store.new_chat_has_searched && store.matched_contact_id) {
             result = result.filter(c => c.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
        } else if (searchQuery.value) {
             result = result.filter(c => c.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
        }

        return result
    })

    const getItemClass = (id) => {
        if (store.new_chat_has_searched && store.matched_contact_id === id) return `contact-row-matched data-id-${id}`
        return `contact-row-visible data-id-${id}`
    }

    const performSearch = () => {
        store.new_chat_has_searched = true
        if (displayedContacts.value.length > 0) {
            store.matched_contact_id = displayedContacts.value[0].id
        }
    }

    const selectContact = async (contact) => {
        store.selected_contact_id = contact.id
        store.new_chat_has_searched = null
        store.currentPageId = 'NEW_CHAT_COMPOSE'
        await router.push({ name: 'NEW_CHAT_COMPOSE' })
    }

    const goBackChats = async () => {
        store.currentPageId = 'CHATS_LIST'
        await router.push({ name: 'CHATS_LIST' })
    }

    return {
        searchQuery,
        displayedContacts,
        getItemClass,
        performSearch,
        selectContact,
        goBackChats
    }
  }
}
</script>