<template>
  <div class="h-screen bg-slate-50 flex flex-col">
    <header class="bg-white shadow-sm z-20">
      <div class="max-w-2xl mx-auto px-4 py-3 flex items-center justify-between">
        <h1 class="text-xl font-bold text-slate-800">Groups</h1>
        <button id="groups-back-home" @click="goBackHome" class="p-2 text-slate-500 hover:text-blue-600">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
        </button>
      </div>
    </header>

    <div class="bg-white border-b border-slate-100 p-4 sticky top-0 z-10">
      <div class="max-w-2xl mx-auto space-y-3">
        <div class="relative">
          <input 
            id="groups-search-input"
            type="text" 
            placeholder="Search groups..." 
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
             id="groups-filter-muted-checkbox" 
             @click="toggleMuted"
             :class="['px-3 py-1 rounded-full text-sm font-medium cursor-pointer transition-colors select-none', filters.muted ? 'bg-blue-100 text-blue-700' : 'bg-slate-100 text-slate-600 hover:bg-slate-200']"
           >
             Muted
           </div>

           <div class="relative ml-auto">
             <button id="groups-sort-dropdown" @click="showSort = !showSort" class="flex items-center text-sm font-medium text-slate-600 hover:text-blue-600">
               <span>Sort</span>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 ml-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>
             <div v-if="showSort" class="absolute right-0 mt-2 w-32 bg-white rounded-lg shadow-xl border border-slate-100 py-1 z-50">
               <div id="groups-sort-option-name-inc" @click="setSort('name')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Name</div>
               <div id="groups-sort-option-members-desc" @click="setSort('member_count')" class="px-4 py-2 hover:bg-slate-50 cursor-pointer text-sm">Members</div>
             </div>
           </div>
        </div>
      </div>
    </div>

    <div id="groups-list-container" class="flex-1 overflow-y-auto bg-white">
      <div class="max-w-2xl mx-auto divide-y divide-slate-100" id="groups-list">
        <div 
          v-for="group in displayedGroups" 
          :key="group.id"
          :class="['p-4 hover:bg-slate-50 cursor-pointer transition-colors flex items-center space-x-4', getItemClass(group.id)]"
          @click="openGroup(group)"
        >
            <img :src="group.avatar" alt="Avatar" class="w-12 h-12 rounded-full object-cover border border-slate-200" />
            
            <div class="flex-1 min-w-0">
                <div class="flex items-center justify-between mb-1">
                    <h3 class="text-base font-semibold text-slate-900 truncate">{{ group.name }}</h3>
                    <span class="text-xs text-slate-500 whitespace-nowrap">{{ group.timestamp }}</span>
                </div>
                <div class="flex items-center justify-between">
                     <p class="text-sm text-slate-500 truncate pr-4">{{ group.last_message }}</p>
                     <div class="flex items-center space-x-2">
                        <span class="text-xs bg-slate-100 px-2 py-0.5 rounded-full text-slate-500">{{ group.member_count }} mem</span>
                        <svg v-if="group.muted" xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" clip-rule="evenodd" />
                           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
                        </svg>
                     </div>
                </div>
            </div>
        </div>
      </div>
    </div>

    <div class="fixed bottom-6 right-6 z-30">
        <button 
          id="create-group-button" 
          @click="goToCreateGroup"
          class="w-14 h-14 bg-blue-600 hover:bg-blue-700 text-white rounded-full shadow-lg flex items-center justify-center transition-transform hover:scale-105"
        >
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
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
  name: 'GROUPS_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const showSort = ref(false)
    const filters = ref({
        muted: false
    })
    const sortBy = ref('name')

    const displayedGroups = computed(() => {
        let result = dataStore.groups || []

        // Search
        if (store.groups_list_has_searched && store.matched_group_id) {
             result = result.filter(g => g.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
        } else if (searchQuery.value) {
             result = result.filter(g => g.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
        }

        // Filters
        if (filters.value.muted) result = result.filter(g => g.muted)

        // Sort
        if (sortBy.value === 'name') {
            result.sort((a, b) => a.name.localeCompare(b.name))
        } else if (sortBy.value === 'member_count') {
            result.sort((a, b) => b.member_count - a.member_count)
        }

        return result
    })

    const getItemClass = (id) => {
        if (store.groups_list_has_searched && store.matched_group_id === id) return `group-row-matched data-id-${id}`
        if (store.groups_list_filters_applied) return `group-row-filtered data-id-${id}`
        return `group-row-visible data-id-${id}`
    }

    const performSearch = () => {
        store.groups_list_has_searched = true
        if (displayedGroups.value.length > 0) {
            store.matched_group_id = displayedGroups.value[0].id
        }
    }

    const toggleMuted = () => {
        filters.value.muted = !filters.value.muted
        store.groups_list_filters_applied = true
    }

    const setSort = (type) => {
        sortBy.value = type
        showSort.value = false
        store.groups_list_filters_applied = true
    }

    const openGroup = async (group) => {
        store.selected_group_id = group.id
        store.groups_list_filters_applied = null
        store.groups_list_has_searched = null
        store.currentPageId = 'GROUP_DETAIL'
        await router.push({ name: 'GROUP_DETAIL' })
    }

    const goToCreateGroup = async () => {
        store.currentPageId = 'GROUP_CREATE_DETAILS'
        await router.push({ name: 'GROUP_CREATE_DETAILS' })
    }

    const goBackHome = async () => {
        store.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    return {
        searchQuery,
        showSort,
        filters,
        displayedGroups,
        getItemClass,
        performSearch,
        toggleMuted,
        setSort,
        openGroup,
        goToCreateGroup,
        goBackHome
    }
  }
}
</script>