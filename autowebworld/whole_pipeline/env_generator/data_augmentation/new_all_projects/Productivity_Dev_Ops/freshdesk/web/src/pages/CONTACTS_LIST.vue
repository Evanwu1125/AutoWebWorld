<template>
  <div class="min-h-screen bg-slate-50 flex flex-col font-inter text-slate-900">
    <!-- Header -->
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-slate-900">Contacts</h1>
        <div class="flex space-x-4">
            <button id="btn-new-contact" @click="handleNewContact" class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 px-4 rounded-md shadow-sm transition-colors duration-200 flex items-center">
              <span class="mr-2">＋</span> New Contact
            </button>
            <button id="contacts-back-home" @click="handleBackHome" class="bg-white border border-slate-300 text-slate-700 hover:bg-slate-50 font-medium py-2 px-4 rounded-md shadow-sm transition-colors duration-200">
              Home
            </button>
        </div>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <!-- Toolbar -->
      <div class="bg-white p-4 rounded-lg shadow-sm mb-6 space-y-4 lg:space-y-0 lg:flex lg:items-center lg:justify-between">
         <!-- Search -->
         <div class="flex-1 max-w-lg">
            <div class="relative rounded-md shadow-sm">
              <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
                <span class="text-slate-400">🔍</span>
              </div>
              <input type="text" 
                     id="contacts-search-input"
                     v-model="searchQuery"
                     @keypress.enter="handleSearch"
                     class="focus:ring-blue-500 focus:border-blue-500 block w-full pl-10 sm:text-sm border-slate-300 rounded-md py-2" 
                     placeholder="Search contacts...">
            </div>
         </div>

         <!-- Sort -->
         <div class="relative ml-4">
             <button id="contacts-sort-dropdown" @click="toggleSortDropdown" class="bg-white border border-slate-300 text-slate-700 py-2 px-4 rounded-md shadow-sm text-sm font-medium hover:bg-slate-50 flex items-center">
               Sort by: {{ sortLabel }} <span class="ml-2">▼</span>
             </button>
             <div v-if="sortDropdownOpen" class="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 border border-slate-100 z-50">
                <div id="contacts-sort-name" @click="handleSort('name')" class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer">Name</div>
                <div id="contacts-sort-recent" @click="handleSort('recent')" class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer">Recently Added</div>
             </div>
         </div>
      </div>

      <!-- Filters -->
      <div class="bg-white p-4 rounded-lg shadow-sm mb-6">
         <h3 class="text-sm font-medium text-slate-500 mb-3 uppercase tracking-wider">Segment</h3>
         <div class="flex items-center space-x-4">
           <label class="inline-flex items-center">
             <input type="checkbox" id="filter-segment-vip" v-model="filterVip" @change="applyFilters" class="form-checkbox h-4 w-4 text-blue-600 rounded border-slate-300 focus:ring-blue-500">
             <span class="ml-2 text-sm text-slate-600">VIP</span>
           </label>
         </div>
      </div>

      <!-- Contacts Table -->
      <div class="bg-white shadow overflow-hidden sm:rounded-md" id="contacts-table">
        <ul role="list" class="divide-y divide-slate-200">
          <li v-for="contact in filteredContacts" :key="contact.id" class="hover:bg-slate-50 transition-colors duration-150">
             <div 
                  :class="[
                    'block px-4 py-4 sm:px-6 cursor-pointer',
                    `data-id-${contact.id}`,
                    isMatched(contact) ? 'row-matched' : '',
                    isFilteredFirst(contact) ? 'row-filtered-first' : '',
                    'row-visible'
                  ]"
                  @click="handleOpenContact(contact)"
             >
                <div class="flex items-center">
                  <img class="h-10 w-10 rounded-full" :src="contact.avatar" alt="">
                  <div class="ml-4">
                    <div class="text-sm font-medium text-slate-900">{{ contact.name }}</div>
                    <div class="text-sm text-slate-500">{{ contact.email }}</div>
                  </div>
                  <div class="ml-auto flex items-center">
                     <span v-if="contact.segment === 'VIP'" class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-purple-100 text-purple-800">
                       VIP
                     </span>
                     <span v-else class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-slate-100 text-slate-800">
                       Standard
                     </span>
                  </div>
                </div>
             </div>
          </li>
          <li v-if="filteredContacts.length === 0" class="px-4 py-12 text-center text-slate-500">
             <div class="mx-auto h-12 w-12 text-slate-300 text-4xl mb-4">👥</div>
             <p>No contacts found.</p>
          </li>
        </ul>
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
  name: 'CONTACTS_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterVip = ref(false)
    const sortOption = ref('')
    const sortDropdownOpen = ref(false)

    const filteredContacts = computed(() => {
        let result = [...dataStore.contacts]
        
        if (searchQuery.value) {
            const q = searchQuery.value.toLowerCase()
            result = result.filter(c => 
                c.name.toLowerCase().includes(q) || 
                c.email.toLowerCase().includes(q)
            )
        }

        if (filterVip.value) {
            result = result.filter(c => c.segment === 'VIP')
        }

        if (sortOption.value) {
            if (sortOption.value === 'name') {
                result.sort((a, b) => a.name.localeCompare(b.name))
            } else if (sortOption.value === 'recent') {
                // Mock sort by "recent" (using id as proxy for order)
                result.sort((a, b) => b.id.localeCompare(a.id))
            }
        }

        return result
    })

    const isMatched = (contact) => {
        if (!signatureStore.contacts_list_has_searched) return false
        if (filteredContacts.value.length > 0 && filteredContacts.value[0].id === contact.id) return true
        return false
    }

    const isFilteredFirst = (contact) => {
        if (!signatureStore.contacts_list_filters_applied) return false
        if (filteredContacts.value.length > 0 && filteredContacts.value[0].id === contact.id) return true
        return false
    }

    const handleNewContact = async () => {
        signatureStore.setCurrentPageId('NEW_CONTACT_FORM')
        await router.push({ name: 'NEW_CONTACT_FORM' })
    }

    const handleBackHome = async () => {
        signatureStore.setCurrentPageId('HOME')
        await router.push({ name: 'HOME' })
    }

    const handleSearch = () => {
        signatureStore.contacts_list_has_searched = true
        signatureStore.matched_contact_id = filteredContacts.value.length > 0 ? filteredContacts.value[0].id : null
    }

    const toggleSortDropdown = () => sortDropdownOpen.value = !sortDropdownOpen.value

    const handleSort = (option) => {
        sortOption.value = option
        signatureStore.contacts_list_filters_applied = true
        sortDropdownOpen.value = false
    }

    const applyFilters = () => {
        signatureStore.contacts_list_filters_applied = true
    }

    const handleOpenContact = async (contact) => {
        signatureStore.selected_contact_id = contact.id
        // Reset flags
        if (signatureStore.contacts_list_filters_applied) signatureStore.contacts_list_filters_applied = null
        if (signatureStore.contacts_list_has_searched) signatureStore.contacts_list_has_searched = null
        if (signatureStore.contacts_list_viewport_anchor_id) signatureStore.contacts_list_viewport_anchor_id = null

        signatureStore.setCurrentPageId('CONTACT_DETAIL')
        await router.push({ name: 'CONTACT_DETAIL', params: { id: contact.id } })
    }

    return {
        searchQuery,
        filterVip,
        sortDropdownOpen,
        sortLabel: computed(() => sortOption.value ? sortOption.value.charAt(0).toUpperCase() + sortOption.value.slice(1) : 'Default'),
        filteredContacts,
        handleNewContact,
        handleBackHome,
        handleSearch,
        toggleSortDropdown,
        handleSort,
        applyFilters,
        handleOpenContact,
        isMatched,
        isFilteredFirst
    }
  }
}
</script>