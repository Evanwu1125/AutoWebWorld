<template>
  <div class="min-h-screen bg-[#FAFBFC] flex flex-col">
    <!-- Header -->
    <header class="bg-white border-b border-gray-200 px-6 py-4 flex justify-between items-center sticky top-0 z-20">
      <div class="flex items-center space-x-4">
        <button id="members-back-home" @click="goHome" class="text-gray-500 hover:text-blue-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
        </button>
        <h1 class="text-2xl font-bold text-[#172B4D]">Workspace Members</h1>
      </div>
    </header>

    <div class="flex flex-1 container mx-auto px-6 py-8 gap-8">
      <!-- Sidebar Filters -->
      <aside class="w-64 flex-shrink-0 space-y-6">
        <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
          <h3 class="font-bold text-[#172B4D] mb-4 uppercase text-xs tracking-wider">Filters</h3>
          
          <!-- Search -->
          <div class="mb-6">
            <div class="relative">
              <input 
                id="members-search-input"
                v-model="searchQuery"
                @keyup.enter="handleSearch"
                type="text" 
                placeholder="Find a member..."
                class="w-full pl-9 pr-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
              <svg class="w-4 h-4 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"/></svg>
            </div>
          </div>

          <!-- Checkboxes -->
          <div class="space-y-3 mb-6">
            <label class="flex items-center space-x-2 cursor-pointer text-sm text-gray-700 hover:text-blue-600">
              <input type="checkbox" id="filter-admins-checkbox" v-model="filterAdmins" class="form-checkbox text-blue-600 rounded">
              <span>Admins only</span>
            </label>
          </div>

          <!-- Slider: Active Score -->
          <div class="mb-6">
            <label class="block text-sm font-medium text-gray-700 mb-2">
              Min Activity: {{ filterActive }}%
            </label>
            <input 
              id="member-active-slider"
              type="range" 
              v-model.number="filterActive" 
              min="0" 
              max="100"
              step="1"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
            >
          </div>
        </div>
      </aside>

      <!-- Main Content -->
      <div class="flex-1">
        <!-- Toolbar -->
        <div class="flex justify-between items-center mb-6">
          <div class="text-sm text-gray-500">
            Showing <span class="font-bold text-gray-900">{{ filteredMembers.length }}</span> members
          </div>
          
          <!-- Sort Dropdown -->
          <div class="relative group" id="members-sort-dropdown">
            <button class="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-blue-600 focus:outline-none">
              <span>Sort by: {{ sortLabel }}</span>
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </button>
            <div class="absolute right-0 top-full mt-2 w-48 bg-white rounded-md shadow-lg border border-gray-100 py-1 hidden group-hover:block z-10">
              <div id="members-sort-option-name" @click="sortBy = 'name'" class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer">Name (A-Z)</div>
              <div id="members-sort-option-recent" @click="sortBy = 'recent'" class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-50 cursor-pointer">Most Recent</div>
            </div>
          </div>
        </div>

        <!-- Members List -->
        <div id="members-list-container" class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden min-h-[500px]">
          <div id="members-list" class="divide-y divide-gray-100">
            <div 
              v-for="member in filteredMembers" 
              :key="member.id"
              class="group p-4 flex items-center space-x-4 hover:bg-blue-50 transition-colors cursor-pointer"
              :class="{
                'member-row-filtered': hasFilters,
                'member-row-matched': hasSearched && matchesSearch(member),
                'member-row-visible': !hasFilters && !hasSearched
              }"
              @click="openMember(member)"
            >
              <!-- Avatar -->
              <div class="flex-shrink-0 w-12 h-12 rounded-full overflow-hidden bg-gray-200 border border-gray-200">
                <img :src="member.image" alt="member avatar" class="w-full h-full object-cover">
              </div>
              
              <div class="flex-1 min-w-0">
                <div class="flex items-center justify-between mb-1">
                  <h3 class="text-base font-semibold text-gray-900 truncate group-hover:text-blue-600" :class="`data-id-${member.id}`">
                    {{ member.name }}
                  </h3>
                  <span 
                    class="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium uppercase"
                    :class="member.role === 'admin' ? 'bg-purple-100 text-purple-800' : 'bg-gray-100 text-gray-800'"
                  >
                    {{ member.role }}
                  </span>
                </div>
                <p class="text-sm text-gray-600 mb-2">{{ member.email }}</p>
                <div class="flex items-center text-xs text-gray-500">
                  <div class="w-full bg-gray-200 rounded-full h-1.5 w-24 mr-2">
                    <div class="bg-green-500 h-1.5 rounded-full" :style="{ width: member.active + '%' }"></div>
                  </div>
                  <span>Activity: {{ member.active }}%</span>
                </div>
              </div>
            </div>
            
            <!-- Empty State -->
            <div v-if="filteredMembers.length === 0" class="p-12 text-center text-gray-500">
               <img src="/images/NoMembers.jpg" alt="No members found" class="w-32 h-32 mx-auto mb-4 opacity-50">
               <p class="text-lg font-medium">No members found</p>
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
  name: 'WORKSPACE_MEMBERS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const filterAdmins = ref(false)
    const filterActive = ref(0)
    const sortBy = ref(null)

    const sortLabel = computed(() => {
      if (sortBy.value === 'name') return 'Name'
      if (sortBy.value === 'recent') return 'Most Recent'
      return 'Default'
    })

    const filteredMembers = computed(() => {
      let result = dataStore.members

      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(m => m.name.toLowerCase().includes(q) || m.email.toLowerCase().includes(q))
      }

      if (filterAdmins.value) {
        result = result.filter(m => m.role === 'admin')
      }

      if (filterActive.value > 0) {
        result = result.filter(m => m.active >= filterActive.value)
      }

      if (sortBy.value === 'name') {
        result = [...result].sort((a, b) => a.name.localeCompare(b.name))
      } else if (sortBy.value === 'recent') {
        // Mock recent by ID or assume random order is recent
        result = [...result].sort((a, b) => b.active - a.active) // Mock sorting logic
      }

      return result
    })

    const hasFilters = computed(() => filterAdmins.value || filterActive.value > 0 || sortBy.value !== null)
    const hasSearched = computed(() => searchQuery.value.length > 0)
    
    const matchesSearch = (member) => {
      if (!searchQuery.value) return false
      return member.name.toLowerCase().includes(searchQuery.value.toLowerCase())
    }

    const handleSearch = () => {
      signatureStore.members_has_searched = true
      signatureStore.matched_member_id = filteredMembers.value.length > 0 ? filteredMembers.value[0].id : null
    }

    const openMember = async (member) => {
      signatureStore.selected_member_id = member.id
      
      if (hasFilters.value) {
        signatureStore.members_filters_applied = true
      }
      if (hasSearched.value) {
        signatureStore.members_has_searched = true
        signatureStore.matched_member_id = member.id
      }
      if (!hasFilters.value && !hasSearched.value) {
        signatureStore.members_viewport_anchor_id = member.id
      }

      await router.push({ name: 'INVITE_MEMBER_FORM', params: { member_id: member.id } })
    }

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      searchQuery,
      filterAdmins,
      filterActive,
      sortBy,
      sortLabel,
      filteredMembers,
      hasFilters,
      hasSearched,
      matchesSearch,
      handleSearch,
      openMember,
      goHome
    }
  }
}
</script>