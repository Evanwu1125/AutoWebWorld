<template>
  <div id="subscriptions-shell" class="min-h-screen bg-neutral-900 text-white flex flex-col">
    <!-- Navbar -->
    <nav class="sticky top-0 z-50 bg-[#0F0F0F]/95 backdrop-blur border-b border-gray-800 px-4 h-14 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <div id="logo-home" @click="goHome" class="flex items-center gap-1 cursor-pointer">
          <div class="bg-red-600 text-white rounded-lg p-1">
            <svg class="w-6 h-6 fill-current" viewBox="0 0 24 24"><path d="M19.615 3.184c-3.604-.246-11.631-.245-15.23 0-3.897.266-4.356 2.62-4.385 8.816.029 6.185.484 8.549 4.385 8.816 3.6.245 11.626.246 15.23 0 3.897-.266 4.356-2.62 4.385-8.816-.029-6.185-.484-8.549-4.385-8.816zm-10.615 12.816v-8l8 3.993-8 4.007z"/></svg>
          </div>
          <span class="text-xl font-bold tracking-tight">Subscriptions</span>
        </div>
      </div>
      
      <!-- Search -->
      <div class="flex-1 max-w-xl mx-4">
        <div class="flex w-full group">
           <input 
            id="subscriptions-search-input"
            v-model="searchQuery"
            @keyup.enter="performSearch"
            type="text"
            placeholder="Search channels"
            class="w-full bg-[#121212] border border-gray-700 rounded-full px-4 py-2 text-white focus:border-blue-500 focus:outline-none"
          >
        </div>
      </div>
      
      <div class="w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center text-sm font-bold">U</div>
    </nav>

    <main class="flex-1 max-w-7xl mx-auto w-full p-4 md:p-6">
      <!-- Filters Toolbar -->
      <div class="flex flex-wrap items-center gap-6 mb-8 bg-[#1F1F1F] p-4 rounded-xl border border-gray-800">
        <h2 class="text-lg font-bold mr-2">Manage</h2>
        
        <!-- Notifications Checkbox -->
        <div 
          id="filter-notifications-checkbox" 
          @click="toggleNotificationsFilter"
          class="flex items-center gap-2 cursor-pointer select-none px-3 py-1.5 rounded-lg hover:bg-[#333] transition-colors"
        >
          <div class="w-5 h-5 rounded border border-gray-500 flex items-center justify-center" :class="{'bg-blue-500 border-blue-500': isNotificationsFilter}">
            <svg v-if="isNotificationsFilter" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
          </div>
          <span class="text-sm">Has New Posts</span>
        </div>

        <!-- Activity Slider -->
        <div class="flex items-center gap-3">
          <span class="text-sm text-gray-400">Min Activity: {{ activityFilter }}%</span>
          <input 
            id="filter-activity-slider"
            type="range" 
            min="0" 
            max="100" 
            step="10"
            v-model.number="activityFilter"
            @input="applyFilters"
            class="w-32 h-1 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500"
          >
        </div>

        <!-- Sort -->
        <div class="relative ml-auto">
          <div 
            id="subscriptions-sort-dropdown"
            @click="isSortOpen = !isSortOpen"
            class="flex items-center gap-2 cursor-pointer hover:text-blue-400 text-sm font-medium"
          >
            <span>{{ sortLabel }}</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </div>
          
          <div 
            v-if="isSortOpen"
            class="absolute right-0 mt-2 w-40 bg-[#272727] rounded-lg shadow-xl border border-gray-700 py-1 z-20"
          >
            <div id="subscriptions-sort-option-recent" @click="setSort('recent', 'Most Recent')" class="px-4 py-2 hover:bg-gray-700 cursor-pointer text-sm">Most Recent</div>
            <div id="subscriptions-sort-option-a-z-inc" @click="setSort('a_z', 'A-Z')" class="px-4 py-2 hover:bg-gray-700 cursor-pointer text-sm">A-Z</div>
            <div id="subscriptions-sort-option-z-a-desc" @click="setSort('z_a', 'Z-A')" class="px-4 py-2 hover:bg-gray-700 cursor-pointer text-sm">Z-A</div>
          </div>
        </div>
      </div>

      <!-- Grid Layout -->
      <div 
        id="subscriptions-list" 
        class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6"
      >
        <div 
          v-for="channel in filteredChannels" 
          :key="channel.id"
          class="flex flex-col items-center p-6 bg-[#1F1F1F] rounded-2xl hover:bg-[#272727] transition-all cursor-pointer group hover:-translate-y-1"
          :class="getRowClass(channel)"
          :data-id="channel.id"
          @click="openChannel(channel)"
        >
          <div class="w-24 h-24 rounded-full overflow-hidden mb-4 border-2 border-transparent group-hover:border-red-600 transition-colors">
            <img :src="channel.avatar" :alt="channel.name" class="w-full h-full object-cover">
          </div>
          
          <h3 class="font-bold text-center mb-1 group-hover:text-white text-gray-200">{{ channel.name }}</h3>
          <p class="text-xs text-gray-500 mb-4">{{ channel.subscribers }} subscribers</p>
          
          <div class="w-full flex justify-center">
             <button class="bg-[#272727] hover:bg-[#3F3F3F] text-gray-300 hover:text-white px-4 py-1.5 rounded-full text-xs font-medium border border-gray-700 transition-colors">
               View Channel
             </button>
          </div>
        </div>
      </div>
      
      <!-- Empty State -->
      <div v-if="filteredChannels.length === 0" class="text-center py-20 text-gray-500">
        <p class="text-xl">No channels found.</p>
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
  name: 'SUBSCRIPTIONS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // UI State
    const searchQuery = ref('')
    const isNotificationsFilter = ref(false)
    const activityFilter = ref(0)
    const currentSort = ref(null)
    const sortLabel = ref('Default')
    const isSortOpen = ref(false)

    // Actions
    const performSearch = () => {
      if (!searchQuery.value.trim()) return
      store.subscriptions_has_searched = true
      const match = dataStore.channels.find(c => c.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
      store.matched_channel_id = match ? match.id : null
    }

    const toggleNotificationsFilter = () => {
      isNotificationsFilter.value = !isNotificationsFilter.value
      store.subscriptions_filters_applied = true
    }

    const applyFilters = () => {
      store.subscriptions_filters_applied = true
    }

    const setSort = (value, label) => {
      currentSort.value = value
      sortLabel.value = label
      isSortOpen.value = false
      store.subscriptions_filters_applied = true
    }

    const filteredChannels = computed(() => {
      let result = [...dataStore.channels]

      // Search
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(c => c.name.toLowerCase().includes(q))
      }

      // Activity Filter
      if (activityFilter.value > 0) {
        result = result.filter(c => c.activity >= activityFilter.value)
      }

      // Notification Filter (Mock: even IDs have notifications)
      if (isNotificationsFilter.value) {
        result = result.filter((c, i) => i % 2 === 0)
      }

      // Sort
      if (currentSort.value === 'a_z') {
        result.sort((a, b) => a.name.localeCompare(b.name))
      } else if (currentSort.value === 'z_a') {
        result.sort((a, b) => b.name.localeCompare(a.name))
      }
      // 'recent' kept as default order for mock

      return result
    })

    const getRowClass = (channel) => {
      const classes = [`data-id-${channel.id}`]
      
      if (store.subscriptions_filters_applied) {
        classes.push('channel-row-filtered')
      } else if (store.subscriptions_has_searched && store.matched_channel_id === channel.id) {
        classes.push('channel-row-matched')
      } else {
        classes.push('channel-row-visible')
      }
      
      return classes.join(' ')
    }

    const goHome = () => {
      store.currentPageId = 'HOME'
      router.push({ name: 'HOME' })
    }

    const openChannel = (channel) => {
      store.selected_channel_id = channel.id
      store.subscriptions_viewport_anchor_id = channel.id
      store.subscriptions_filters_applied = null
      store.subscriptions_has_searched = null
      store.currentPageId = 'CHANNEL_PAGE'
      router.push({ name: 'CHANNEL_PAGE', params: { id: channel.id } })
    }

    return {
      store,
      searchQuery,
      isNotificationsFilter,
      activityFilter,
      sortLabel,
      isSortOpen,
      filteredChannels,
      performSearch,
      toggleNotificationsFilter,
      applyFilters,
      setSort,
      getRowClass,
      goHome,
      openChannel
    }
  }
}
</script>
