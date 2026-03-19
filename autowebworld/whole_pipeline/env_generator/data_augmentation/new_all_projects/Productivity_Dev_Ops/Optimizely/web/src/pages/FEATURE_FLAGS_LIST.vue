<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div class="flex items-center">
          <button id="logo-home" @click="goHome" class="text-2xl font-bold text-blue-600 mr-8">Optimizely</button>
          <h1 class="text-xl font-semibold text-gray-800">Feature Flags</h1>
        </div>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <!-- Controls -->
      <div class="bg-white p-4 rounded-lg shadow-sm mb-6 flex flex-col md:flex-row md:items-center gap-6">
        
        <div class="flex items-center">
          <input 
            id="feature-flags-filter-active-checkbox" 
            type="checkbox" 
            v-model="filterActive"
            @change="applyFilters"
            class="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          >
          <label for="feature-flags-filter-active-checkbox" class="ml-2 block text-sm text-gray-700">
            Active Only
          </label>
        </div>

        <div class="w-64">
          <label class="block text-xs text-gray-500 mb-1">Min Rollout: {{ rolloutThreshold }}%</label>
          <input 
            id="feature-flags-rollout-slider"
            type="range" 
            v-model="rolloutThreshold"
            @input="applyFilters"
            min="0"
            max="100"
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          >
        </div>

        <div class="relative ml-auto" id="feature-flags-sort-dropdown">
          <button @click="toggleSort" class="bg-white border border-gray-300 text-gray-700 px-4 py-2 rounded-md text-sm font-medium hover:bg-gray-50 flex items-center shadow-sm">
            Sort By
            <svg class="ml-2 h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          <div v-if="sortOpen" class="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg z-50 ring-1 ring-black ring-opacity-5">
            <div class="py-1">
              <div id="feature-flags-sort-option-name" @click="sort('name')" class="cursor-pointer block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Name</div>
              <div id="feature-flags-sort-option-created" @click="sort('created')" class="cursor-pointer block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Created Date</div>
              <div id="feature-flags-sort-option-status" @click="sort('status')" class="cursor-pointer block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Status</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Table -->
      <div id="feature-flags-table" class="bg-white shadow-sm rounded-lg overflow-hidden">
        <ul class="divide-y divide-gray-200">
          <li v-for="flag in filteredFlags" :key="flag.id" class="hover:bg-gray-50">
            <div 
              :class="[
                'px-6 py-4 flex items-center cursor-pointer',
                `data-id-${flag.id}`,
                isFiltered ? 'row-filtered' : 'row-visible'
              ]"
              @click="openFlag(flag)"
            >
              <div class="flex-shrink-0 h-10 w-10 mr-4">
                <img :src="flag.image" class="h-10 w-10 rounded-full object-cover" alt="" />
              </div>
              <div class="flex-1 grid grid-cols-1 md:grid-cols-4 gap-4 items-center">
                <div class="md:col-span-1">
                  <div class="text-sm font-medium text-gray-900">{{ flag.name }}</div>
                  <div class="text-xs text-gray-500 font-mono">{{ flag.key }}</div>
                </div>
                <div class="text-sm text-gray-500">
                  {{ flag.created }}
                </div>
                <div>
                  <div class="flex items-center">
                     <div class="w-full bg-gray-200 rounded-full h-2.5 mr-2 max-w-[100px]">
                       <div class="bg-blue-600 h-2.5 rounded-full" :style="{ width: flag.rollout + '%' }"></div>
                     </div>
                     <span class="text-xs text-gray-600">{{ flag.rollout }}%</span>
                  </div>
                </div>
                <div class="text-right">
                  <span :class="[
                    'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                    flag.status === 'Active' ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                  ]">
                    {{ flag.status }}
                  </span>
                </div>
              </div>
            </div>
          </li>
        </ul>
        <div v-if="filteredFlags.length === 0" class="p-8 text-center text-gray-500">
          No feature flags found.
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
  name: 'FEATURE_FLAGS_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const filterActive = ref(false)
    const rolloutThreshold = ref(0)
    const sortOpen = ref(false)
    const activeSort = ref(null)
    const isFiltered = ref(false)

    function applyFilters() {
      isFiltered.value = true
      signatureStore.feature_flags_list_filters_applied = true
    }

    function toggleSort() {
      sortOpen.value = !sortOpen.value
    }

    function sort(field) {
      activeSort.value = field
      sortOpen.value = false
      applyFilters()
    }

    const filteredFlags = computed(() => {
      let items = [...dataStore.feature_flags]

      if (filterActive.value) {
        items = items.filter(f => f.status === 'Active')
      }

      if (rolloutThreshold.value > 0) {
        items = items.filter(f => f.rollout >= rolloutThreshold.value)
      }

      if (activeSort.value) {
        items.sort((a, b) => {
          if (activeSort.value === 'name') return a.name.localeCompare(b.name)
          if (activeSort.value === 'status') return a.status.localeCompare(b.status)
          if (activeSort.value === 'created') return new Date(b.created) - new Date(a.created)
          return 0
        })
      }

      return items
    })

    function openFlag(flag) {
      if (isFiltered.value) {
        signatureStore.feature_flags_list_filters_applied = true // Confirm effect
      } else {
        signatureStore.feature_flags_list_viewport_anchor_id = flag.id
      }
      
      signatureStore.feature_flags_list_selected_item_id = flag.id
      signatureStore.setCurrentPageId('FEATURE_FLAG_DETAIL')
      router.push({ name: 'FEATURE_FLAG_DETAIL', params: { id: flag.id } })
    }

    function goHome() {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    return {
      filterActive,
      rolloutThreshold,
      sortOpen,
      filteredFlags,
      isFiltered,
      applyFilters,
      toggleSort,
      sort,
      openFlag,
      goHome
    }
  }
}
</script>