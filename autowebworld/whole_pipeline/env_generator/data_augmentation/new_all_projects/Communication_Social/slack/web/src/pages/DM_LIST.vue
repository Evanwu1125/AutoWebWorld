<template>
  <div class="h-screen flex bg-white overflow-hidden">
    <!-- Sidebar -->
    <div class="w-64 bg-[#3F0E40] text-[#CDC7CD] flex flex-col flex-shrink-0 z-30">
        <!-- Back to Channels -->
        <div id="back-channels" @click="handleBackChannels" class="h-12 flex items-center px-4 hover:bg-[#350d36] transition cursor-pointer border-b border-[#5d2c5d]">
            <span class="mr-2">←</span> Channels
        </div>

        <!-- Filters -->
        <div class="px-4 py-2 space-y-2 text-xs border-b border-[#5d2c5d]">
             <div class="flex items-center space-x-2">
                <input type="checkbox" id="filter-active-dm-checkbox" @change="handleFilterCheckbox" class="rounded bg-transparent border-gray-500">
                <label>Active only</label>
            </div>
             <div>
                <label class="block mb-1">Activity > {{ activityFilter }}</label>
                <input 
                    type="range" 
                    id="dm-activity-slider" 
                    min="0" 
                    max="100" 
                    v-model="activityFilter" 
                    @input="handleFilterSlider"
                    class="w-full h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer"
                >
             </div>
             <!-- Sort -->
             <div class="relative">
                <div id="dm-sort-dropdown" @click="toggleSort" class="border border-gray-600 rounded px-2 py-1 cursor-pointer flex justify-between items-center">
                    <span>{{ sortOption || 'Sort...' }}</span>
                    <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                </div>
                <div v-if="showSort" class="absolute left-0 w-full mt-1 bg-white text-gray-900 rounded shadow-lg z-50">
                    <div id="dm-sort-option-most-active-desc" class="px-2 py-1 hover:bg-gray-100 cursor-pointer" @click="handleFilterSort('most_active')">Most Active</div>
                    <div id="dm-sort-option-alpha-inc" class="px-2 py-1 hover:bg-gray-100 cursor-pointer" @click="handleFilterSort('alphabetical')">Name (A-Z)</div>
                </div>
             </div>
        </div>

        <!-- DM List -->
        <div id="dm-list" class="flex-1 overflow-y-auto custom-scrollbar mt-2" @scroll="handleScroll">
             <div
                v-for="dm in filteredDMs"
                :key="dm.id"
                :class="[
                    'px-4 py-2 cursor-pointer flex items-center hover:bg-[#350d36]',
                    `data-id-${dm.user_id}`,
                    dm.id === signatureStore.matched_dm_id ? 'bg-[#1164A3] text-white dm-matched' : '',
                    signatureStore.dm_list_filters_applied ? 'dm-filtered' : 'dm-visible'
                ]"
                @click="handleOpenDM(dm.id)"
             >
                <div class="w-8 h-8 rounded bg-gray-500 mr-2 overflow-hidden">
                    <img :src="dm.user_avatar" class="w-full h-full object-cover" />
                </div>
                <div class="flex-1">
                    <div class="flex justify-between items-center">
                        <span :class="{'font-bold text-white': dm.unread}">{{ dm.user_name }}</span>
                        <div :class="{'w-2 h-2 rounded-full': true, 'bg-green-500': dm.user_status === 'available', 'bg-red-500': dm.user_status === 'busy', 'bg-gray-500': dm.user_status === 'away'}"></div>
                    </div>
                </div>
             </div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="flex-1 flex flex-col bg-white">
        <!-- Search Header -->
        <div class="h-12 border-b flex items-center px-4 bg-white shadow-sm z-20">
             <div class="flex-1 relative">
                <input 
                    id="dm-search-input"
                    type="text" 
                    placeholder="Search people..." 
                    class="w-full bg-gray-100 border-none rounded-md py-1 px-8 text-sm focus:ring-2 focus:ring-blue-500"
                    v-model="searchQuery"
                    @keyup.enter="handleSearch"
                >
                <svg class="w-4 h-4 text-gray-500 absolute left-2 top-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
            </div>
        </div>
        <div class="flex-1 flex flex-col items-center justify-center text-gray-500">
             <p>Select a conversation to start chatting.</p>
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
  name: 'DM_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const activityFilter = ref(0)
    const sortOption = ref(null)
    const showSort = ref(false)
    const searchQuery = ref('')

    const filteredDMs = computed(() => {
        let list = dataStore.dms
        
        if (signatureStore.dm_list_filters_applied) {
            list = list.filter(d => d.activity > activityFilter.value)
        }

        if (searchQuery.value) {
            list = list.filter(d => d.user_name.toLowerCase().includes(searchQuery.value.toLowerCase()))
        }

        if (sortOption.value === 'alphabetical') {
            list = [...list].sort((a, b) => a.user_name.localeCompare(b.user_name))
        } else if (sortOption.value === 'most_active') {
             list = [...list].sort((a, b) => b.activity - a.activity)
        }

        return list
    })

    function handleFilterCheckbox() {
        signatureStore.dm_list_filters_applied = true
    }

    function handleFilterSlider() {
        signatureStore.dm_list_filters_applied = true
    }

    function toggleSort() {
        showSort.value = !showSort.value
    }

    function handleFilterSort(option) {
        sortOption.value = option
        signatureStore.dm_list_filters_applied = true
        showSort.value = false
    }

    function handleSearch() {
        signatureStore.dm_list_has_searched = true
        if (filteredDMs.value.length > 0) {
            signatureStore.matched_dm_id = filteredDMs.value[0].id
        }
    }

    function handleScroll() {
        // Mock scroll logic
    }

    async function handleOpenDM(id) {
        signatureStore.selected_dm_id = id
        signatureStore.dm_list_filters_applied = null
        signatureStore.currentPageId = 'DM_DETAIL'
        await router.push({ name: 'DM_DETAIL', params: { id } })
    }

    async function handleBackChannels() {
        signatureStore.currentPageId = 'CHANNEL_LIST'
        await router.push({ name: 'CHANNEL_LIST' })
    }

    return {
        signatureStore,
        dataStore,
        activityFilter,
        sortOption,
        showSort,
        searchQuery,
        filteredDMs,
        handleFilterCheckbox,
        handleFilterSlider,
        toggleSort,
        handleFilterSort,
        handleSearch,
        handleScroll,
        handleOpenDM,
        handleBackChannels
    }
  }
}
</script>