<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between">
        <h1 class="text-xl font-bold text-slate-800">Contacts</h1>
        <button id="contacts-back-home" @click="goBackHome" class="p-2 text-slate-500 hover:text-blue-600">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
        </button>
      </div>
    </header>

    <!-- Search & Filter -->
    <div class="bg-white border-b border-slate-100 p-4 sticky top-0 z-10">
      <div class="max-w-2xl mx-auto space-y-3">
        <div class="relative">
          <input 
            id="contacts-search-input"
            type="text" 
            placeholder="Search contacts..." 
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
             id="filter-blocked-checkbox" 
             @click="toggleBlocked"
             :class="['px-3 py-1 rounded-full text-sm font-medium cursor-pointer transition-colors select-none', filters.blocked ? 'bg-red-100 text-red-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200']"
           >
             Blocked
           </div>
           <div 
             id="filter-favorites-checkbox"
             @click="toggleFavorites"
             :class="['px-3 py-1 rounded-full text-sm font-medium cursor-pointer transition-colors select-none', filters.favorites ? 'bg-yellow-100 text-yellow-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200']"
           >
             Favorites
           </div>

           <div class="relative ml-auto">
             <button id="contacts-sort-dropdown" @click="showSort = !showSort" class="flex items-center text-sm font-medium text-slate-600 hover:text-blue-600">
               <span>Sort</span>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>
             <div v-if="showSort" class="absolute right-0 mt-2 w-32 bg-white rounded-lg shadow-xl border border-slate-100 py-1 z-50">
               <div id="contacts-sort-option-name-inc" @click="setSort('name')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Name</div>
               <div id="contacts-sort-option-blocked" @click="setSort('blocked')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Blocked</div>
             </div>
           </div>
        </div>
      </div>
    </div>

    <!-- Contact List -->
    <div id="contacts-list-container" class="flex-1 overflow-y-auto bg-white">
      <div class="max-w-2xl mx-auto divide-y divide-slate-100" id="contacts-list">
        <div 
          v-for="contact in displayedContacts" 
          :key="contact.id"
          :class="['p-4 hover:bg-slate-50 cursor-pointer transition-colors flex items-center space-x-4', getItemClass(contact.id)]"
          @click="openContact(contact)"
        >
            <img :src="contact.avatar" alt="Avatar" class="w-12 h-12 rounded-full object-cover border border-slate-200" />
            
            <div class="flex-1 min-w-0">
                <h3 class="text-base font-semibold text-slate-900 truncate">{{ contact.name }}</h3>
                <p class="text-sm text-slate-500">{{ contact.phone }}</p>
            </div>
            
            <div class="flex items-center space-x-2">
                <svg v-if="contact.is_favorite" xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-yellow-400 fill-current" viewBox="0 0 20 20">
                   <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
                <span v-if="contact.is_blocked" class="px-2 py-0.5 bg-red-100 text-red-600 text-xs rounded-full font-medium">Blocked</span>
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
  name: 'CONTACTS_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const showSort = ref(false)
    const filters = ref({
        blocked: false,
        favorites: false
    })
    const sortBy = ref('name')

    const displayedContacts = computed(() => {
        let result = dataStore.contacts || []

        // Search
        if (store.contacts_list_has_searched && store.matched_contact_id) {
             result = result.filter(c => c.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
        } else if (searchQuery.value) {
             result = result.filter(c => c.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
        }

        // Filters
        if (filters.value.blocked) result = result.filter(c => c.is_blocked)
        if (filters.value.favorites) result = result.filter(c => c.is_favorite)

        // Sort
        if (sortBy.value === 'name') {
            result.sort((a, b) => a.name.localeCompare(b.name))
        } else if (sortBy.value === 'blocked') {
            result.sort((a, b) => (a.is_blocked === b.is_blocked) ? 0 : a.is_blocked ? -1 : 1)
        }
        // 'recent' not implemented in mock data but placeholder exists

        return result
    })

    const getItemClass = (id) => {
        if (store.contacts_list_has_searched && store.matched_contact_id === id) return `contact-row-matched data-id-${id}`
        if (store.contacts_list_filters_applied) return `contact-row-filtered data-id-${id}`
        return `contact-row-visible data-id-${id}`
    }

    const performSearch = () => {
        store.contacts_list_has_searched = true
        if (displayedContacts.value.length > 0) {
            store.matched_contact_id = displayedContacts.value[0].id
        }
    }

    const toggleBlocked = () => {
        filters.value.blocked = !filters.value.blocked
        store.contacts_list_filters_applied = true
    }

    const toggleFavorites = () => {
        filters.value.favorites = !filters.value.favorites
        store.contacts_list_filters_applied = true
    }

    const setSort = (type) => {
        sortBy.value = type
        showSort.value = false
        store.contacts_list_filters_applied = true
    }

    const openContact = async (contact) => {
        store.selected_contact_id = contact.id
        store.contacts_list_filters_applied = null
        store.contacts_list_has_searched = null
        store.currentPageId = 'CONTACT_DETAIL'
        await router.push({ name: 'CONTACT_DETAIL' })
    }

    const goBackHome = async () => {
        store.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        searchQuery,
        showSort,
        filters,
        displayedContacts,
        getItemClass,
        performSearch,
        toggleBlocked,
        toggleFavorites,
        setSort,
        openContact,
        goBackHome
    }
  }
}
</script>