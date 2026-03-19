<template>
  <div class="h-screen flex bg-white overflow-hidden">
    <!-- Sidebar -->
    <div class="w-64 bg-[#3F0E40] text-[#CDC7CD] flex flex-col flex-shrink-0 z-30">
      <!-- Header -->
      <div class="h-12 flex items-center px-4 border-b border-[#5d2c5d] hover:bg-[#350d36] transition cursor-pointer font-bold text-white text-lg">
        Acme Corp
        <svg class="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
      </div>

      <!-- Navigation Items -->
      <div class="flex-1 overflow-y-auto custom-scrollbar py-2">
         <!-- Back to Workspace -->
        <div id="back-workspace" @click="handleBackWS" class="px-4 py-1 hover:bg-[#350d36] cursor-pointer flex items-center">
            <span class="mr-2">←</span> Back to Workspaces
        </div>
        
        <!-- Channels Section -->
        <div class="mt-4 px-4 flex items-center justify-between group">
          <h3 class="font-medium text-sm uppercase tracking-wider opacity-80 group-hover:opacity-100">Channels</h3>
        </div>
        
        <!-- Channel Filters -->
        <div class="px-4 py-2 space-y-2 text-xs">
            <div class="flex items-center space-x-2">
                <input type="checkbox" id="filter-unread-checkbox" @change="handleFilterCheckbox" class="rounded text-green-600 focus:ring-green-500 bg-transparent border-gray-500">
                <label for="filter-unread-checkbox">Unread only</label>
            </div>
             <!-- Activity Slider -->
             <div>
                <label class="block mb-1">Activity > {{ activityFilter }}</label>
                <input 
                    type="range" 
                    id="activity-slider" 
                    min="0" 
                    max="100" 
                    v-model="activityFilter" 
                    @input="handleFilterSlider"
                    class="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                >
             </div>
             <!-- Sort Dropdown -->
             <div class="relative">
                <div id="sort-dropdown" @click="toggleSort" class="border border-gray-600 rounded px-2 py-1 cursor-pointer flex justify-between items-center">
                    <span>{{ sortOption || 'Sort by...' }}</span>
                    <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                </div>
                <div v-if="showSort" class="absolute left-0 w-full mt-1 bg-white text-gray-900 rounded shadow-lg z-50">
                    <div id="sort-option-most-active-desc" class="px-2 py-1 hover:bg-gray-100 cursor-pointer" @click="handleFilterSort('most_active')">Most Active</div>
                    <div id="sort-option-alpha-inc" class="px-2 py-1 hover:bg-gray-100 cursor-pointer" @click="handleFilterSort('alphabetical')">Name (A-Z)</div>
                    <div id="sort-option-unread-desc" class="px-2 py-1 hover:bg-gray-100 cursor-pointer" @click="handleFilterSort('unread')">Unread</div>
                </div>
             </div>
        </div>

        <!-- Channel List -->
        <div id="channel-list" class="mt-2" @scroll="handleScroll">
          <div 
            v-for="channel in filteredChannels" 
            :key="channel.id"
            :class="[
                'px-4 py-1 cursor-pointer flex items-center hover:bg-[#350d36]',
                `data-id-${channel.id}`,
                channel.id === signatureStore.matched_channel_id ? 'bg-[#1164A3] text-white channel-matched' : '',
                signatureStore.channel_list_filters_applied ? 'channel-filtered' : 'channel-visible'
            ]"
            @click="handleOpenChannel(channel.id)"
          >
            <span class="opacity-70 mr-2">#</span>
            <span :class="{'font-bold text-white': channel.unread}">{{ channel.name }}</span>
          </div>
        </div>

        <!-- DMs Section Link -->
        <div class="mt-8 px-4 flex items-center justify-between group cursor-pointer hover:text-white" id="nav-dms" @click="handleOpenDMList">
           <h3 class="font-medium text-sm uppercase tracking-wider">Direct Messages</h3>
           <span>→</span>
        </div>
      </div>

      <!-- User Profile Footer -->
      <div id="user-menu-profile" @click="handleOpenProfile" class="p-4 bg-[#350d36] flex items-center cursor-pointer hover:bg-[#2c0b2d]">
        <div class="w-8 h-8 rounded bg-gray-400 mr-2 overflow-hidden">
             <img src="/images/UserProfile.jpg" class="w-full h-full object-cover" />
        </div>
        <div class="text-sm font-bold">You</div>
      </div>
    </div>

    <!-- Main Content Area (Placeholder for List Page) -->
    <div class="flex-1 flex flex-col bg-white">
        <!-- Search Header -->
        <div class="h-12 border-b flex items-center px-4 bg-white shadow-sm z-20">
            <div class="flex-1 relative">
                <input 
                    id="channel-search-input"
                    type="text" 
                    placeholder="Search channels..." 
                    class="w-full bg-gray-100 border-none rounded-md py-1 px-8 text-sm focus:ring-2 focus:ring-blue-500"
                    v-model="searchQuery"
                    @keyup.enter="handleSearch"
                >
                <svg class="w-4 h-4 text-gray-500 absolute left-2 top-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
            </div>
        </div>

        <!-- Welcome/Empty State -->
        <div class="flex-1 flex flex-col items-center justify-center text-gray-500">
            <img src="/images/Welcome.jpg" alt="Welcome" class="w-64 mb-4 opacity-50" />
            <p>Select a channel from the sidebar to start chatting.</p>
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
  name: 'CHANNEL_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const activityFilter = ref(0)
    const sortOption = ref(null)
    const showSort = ref(false)
    const searchQuery = ref('')

    // Computed Channels
    const filteredChannels = computed(() => {
        let list = dataStore.channels
        
        // Filter by activity slider
        if (signatureStore.channel_list_filters_applied) {
            list = list.filter(c => c.activity > activityFilter.value)
        }

        // Filter by checkbox (if mapped in logic, assume simplified here or expanded)
        // Note: FSM only checks filters_applied flag, logic is here in computed

        // Search
        if (searchQuery.value) {
            list = list.filter(c => c.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
        }

        // Sort
        if (sortOption.value === 'alphabetical') {
            list = [...list].sort((a, b) => a.name.localeCompare(b.name))
        } else if (sortOption.value === 'most_active') {
            // Sort by activity score (higher = more active)
             list = [...list].sort((a, b) => b.activity - a.activity)
        } else if (sortOption.value === 'unread') {
            list = [...list].sort((a, b) => (b.unread === a.unread) ? 0 : b.unread ? 1 : -1)
        }

        return list
    })

    function handleFilterCheckbox(e) {
        signatureStore.channel_list_filters_applied = true
        // Logic for unread toggle would affect computed property if implemented fully
    }

    function handleFilterSlider() {
        signatureStore.channel_list_filters_applied = true
    }

    function toggleSort() {
        showSort.value = !showSort.value
    }

    function handleFilterSort(option) {
        sortOption.value = option
        signatureStore.channel_list_filters_applied = true
        showSort.value = false
    }

    function handleSearch() {
        signatureStore.channel_list_has_searched = true
        // If matched, set first match ID
        if (filteredChannels.value.length > 0) {
            signatureStore.matched_channel_id = filteredChannels.value[0].id
        }
    }

    function handleScroll(e) {
        // Mock scroll anchor logic
        const el = e.target
        // logic to find visible element
    }

    async function handleOpenChannel(id) {
        signatureStore.selected_channel_id = id
        // Reset flags based on FSM logic (handled in next page usually or effects)
        signatureStore.channel_list_filters_applied = null 
        signatureStore.currentPageId = 'CHANNEL_DETAIL'
        await router.push({ name: 'CHANNEL_DETAIL', params: { id } })
    }

    async function handleOpenDMList() {
        signatureStore.currentPageId = 'DM_LIST'
        await router.push({ name: 'DM_LIST' })
    }

    async function handleOpenProfile() {
        signatureStore.currentPageId = 'PROFILE_VIEW'
        await router.push({ name: 'PROFILE_VIEW' })
    }

    async function handleBackWS() {
        signatureStore.currentPageId = 'WORKSPACE_OVERVIEW'
        await router.push({ name: 'WORKSPACE_OVERVIEW' })
    }

    return {
        signatureStore,
        dataStore,
        activityFilter,
        sortOption,
        showSort,
        searchQuery,
        filteredChannels,
        handleFilterCheckbox,
        handleFilterSlider,
        toggleSort,
        handleFilterSort,
        handleSearch,
        handleScroll,
        handleOpenChannel,
        handleOpenDMList,
        handleOpenProfile,
        handleBackWS
    }
  }
}
</script>