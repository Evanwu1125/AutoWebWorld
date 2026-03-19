<template>
  <div class="min-h-screen bg-[#F1F2F2]">
    <nav class="bg-white shadow-sm sticky top-0 z-50">
      <div class="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <button id="notif-back-home" @click="goHome" class="text-gray-500 hover:text-gray-700 p-2 rounded-full hover:bg-gray-100 transition-colors">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          </button>
          <h1 class="text-[#B92B27] text-xl font-bold font-serif">Notifications</h1>
        </div>
      </div>
    </nav>

    <main class="max-w-3xl mx-auto px-4 py-8 grid grid-cols-1 md:grid-cols-3 gap-6">
      
      <!-- Filters -->
      <div class="space-y-6">
        <div class="bg-white p-4 rounded shadow-sm">
          <h3 class="font-bold text-gray-700 mb-3 text-sm uppercase tracking-wide">Filters</h3>
          
          <div class="space-y-4">
            <div class="flex items-center gap-2 cursor-pointer" id="notif-filter-unread-checkbox" @click="toggleUnreadFilter">
              <div :class="['w-4 h-4 border rounded flex items-center justify-center transition-colors', isUnreadFilterActive ? 'bg-blue-600 border-blue-600' : 'border-gray-300 bg-white']">
                <svg v-if="isUnreadFilterActive" class="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
              </div>
              <span class="text-sm text-gray-600">Unread Only</span>
            </div>

            <div class="space-y-2">
              <div class="flex justify-between text-xs text-gray-500">
                <span>Filter by Time</span>
                <span>{{ timeFilterValue }}h</span>
              </div>
              <input 
                id="notif-time-slider"
                type="range" 
                v-model.number="timeFilterValue" 
                :min="0" 
                :max="100" 
                step="1"
                class="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                @input="applyTimeFilter"
              />
            </div>

            <div class="relative">
              <label class="text-xs text-gray-500 mb-1 block">Sort By</label>
              <div id="notif-sort-dropdown" class="w-full border border-gray-300 rounded px-3 py-2 text-sm bg-white cursor-pointer flex justify-between items-center" @click="toggleSortDropdown">
                <span>{{ currentSortLabel }}</span>
                <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
              </div>
              
              <div v-if="isSortDropdownOpen" class="absolute top-full left-0 w-full bg-white border border-gray-200 shadow-lg rounded mt-1 z-10">
                <div id="notif-sort-newest" @click="selectSort('newest')" class="px-4 py-2 hover:bg-gray-50 text-sm cursor-pointer">Newest</div>
                <div id="notif-sort-oldest" @click="selectSort('oldest')" class="px-4 py-2 hover:bg-gray-50 text-sm cursor-pointer">Oldest</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- List -->
      <div class="md:col-span-2 space-y-3" id="notif-list">
        <h2 class="font-bold text-gray-800 text-lg mb-2">Recent Activity</h2>
        
        <div 
          v-for="notif in displayedNotifications" 
          :key="notif.id"
          :class="[
            'bg-white p-4 rounded border shadow-sm hover:shadow-md transition-shadow flex gap-4 cursor-pointer',
            `data-id-${notif.id}`,
            notif.read ? 'border-gray-200' : 'border-l-4 border-l-blue-600 border-gray-200'
          ]"
          @click="handleScroll(notif)"
        >
          <img :src="notif.image" class="w-10 h-10 rounded-full object-cover border border-gray-100 flex-shrink-0" />
          <div>
            <p :class="['text-gray-800 text-sm', notif.read ? '' : 'font-bold']">{{ notif.content }}</p>
            <span class="text-xs text-gray-500 mt-1 block">{{ notif.time }}h ago</span>
          </div>
          <div v-if="!notif.read" class="ml-auto w-2 h-2 rounded-full bg-blue-600 mt-2"></div>
        </div>

        <div v-if="displayedNotifications.length === 0" class="text-center py-10 bg-white rounded border border-gray-200">
          <p class="text-gray-500">No notifications.</p>
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
  name: 'NOTIFICATIONS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const isUnreadFilterActive = ref(false)
    const timeFilterValue = ref(0)
    const isSortDropdownOpen = ref(false)
    const currentSort = ref(null)

    const displayedNotifications = computed(() => {
      let result = [...dataStore.notifications]

      if (store.notifications_filters_applied) {
        if (isUnreadFilterActive.value) {
          result = result.filter(n => !n.read)
        }
        // FSM Time Filter logic (slider)
        // time > value
        result = result.filter(n => n.time > timeFilterValue.value)
      }

      // Sort
      if (currentSort.value === 'newest') {
        result.sort((a, b) => a.time - b.time)
      } else if (currentSort.value === 'oldest') {
        result.sort((a, b) => b.time - a.time)
      }

      return result
    })

    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default'
      return currentSort.value === 'newest' ? 'Newest' : 'Oldest'
    })

    function goHome() {
      store.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    function toggleUnreadFilter() {
      isUnreadFilterActive.value = !isUnreadFilterActive.value
      store.notifications_filters_applied = true
    }

    function applyTimeFilter() {
      store.notifications_filters_applied = true
    }

    function toggleSortDropdown() {
      isSortDropdownOpen.value = !isSortDropdownOpen.value
    }

    function selectSort(type) {
      currentSort.value = type
      isSortDropdownOpen.value = false
      store.notifications_filters_applied = true
    }

    function handleScroll(notif) {
      store.notifications_viewport_anchor_id = notif.id
      // FSM action ACT_NOTIF_SCROLL just sets anchor, usually scroll logic happens
      // In web app, we can just scrollIntoView if needed, or just set store state as per FSM
      // This action is "click" -> set anchor.
    }

    return {
      displayedNotifications,
      isUnreadFilterActive,
      timeFilterValue,
      isSortDropdownOpen,
      currentSortLabel,
      goHome,
      toggleUnreadFilter,
      applyTimeFilter,
      toggleSortDropdown,
      selectSort,
      handleScroll
    }
  }
}
</script>