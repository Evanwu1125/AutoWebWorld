<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <header class="bg-white text-gray-800 p-4 shadow-sm border-b border-gray-200 flex justify-between items-center z-20">
      <div class="font-bold text-lg flex items-center">
        <button id="calls-back-home" @click="goHome" class="mr-4 hover:bg-gray-100 p-1 rounded">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        Calls
      </div>
    </header>

    <div class="flex-1 flex overflow-hidden">
      <!-- Sidebar Filters -->
      <aside class="w-64 bg-gray-50 border-r border-gray-200 p-4 flex flex-col gap-6 overflow-y-auto">
        <div>
          <h3 class="font-semibold text-gray-700 mb-2">Filters</h3>
          <!-- Checkbox Filter ACT_CALLS_FILTER_CHECKBOX -->
          <div class="flex items-center gap-2 mb-4">
            <input 
              id="filter-missed-checkbox"
              type="checkbox" 
              v-model="missedOnly"
              class="w-4 h-4 text-[#6264A7] rounded focus:ring-[#6264A7]"
            />
            <label for="filter-missed-checkbox" class="text-sm text-gray-600">Missed calls only</label>
          </div>

          <!-- Slider Filter ACT_CALLS_FILTER_SLIDER -->
          <div class="mb-4">
            <label class="text-sm text-gray-600 block mb-1">Duration: > {{ minDuration }} sec</label>
            <input 
              id="call-history-range-slider"
              type="range" 
              min="0" 
              max="600"
              step="60" 
              v-model.number="minDuration"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#6264A7]"
            />
          </div>
        </div>

        <div>
          <h3 class="font-semibold text-gray-700 mb-2">Sort By</h3>
          <!-- Sort Dropdown ACT_CALLS_FILTER_SORT -->
          <div id="calls-sort-dropdown" class="relative">
            <div 
              @click="toggleSort"
              class="w-full border rounded px-3 py-2 text-sm text-gray-700 bg-white cursor-pointer flex justify-between items-center"
            >
              {{ sortBy === 'recent' ? 'Most Recent' : (sortBy === 'duration' ? 'Duration' : 'Select...') }}
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            <div v-if="sortOpen" class="absolute top-full left-0 right-0 mt-1 bg-white border rounded shadow-lg z-10">
              <div id="calls-sort-recent" @click="setSort('recent')" class="px-3 py-2 text-sm hover:bg-gray-100 cursor-pointer">Most Recent</div>
              <div id="calls-sort-duration-desc" @click="setSort('duration')" class="px-3 py-2 text-sm hover:bg-gray-100 cursor-pointer">Duration</div>
            </div>
          </div>
        </div>
      </aside>

      <!-- Main Content -->
      <main class="flex-1 overflow-y-auto bg-white p-6">
        <h2 class="text-2xl font-bold text-gray-800 mb-6">History</h2>
        
        <div class="space-y-2">
          <div 
            v-for="call in filteredCalls" 
            :key="call.id"
            class="flex items-center gap-4 p-4 rounded-lg hover:bg-gray-50 border border-gray-100"
          >
            <img 
               :src="call.image" 
               class="w-10 h-10 rounded-full object-cover" 
               alt="Avatar"
               @error="$event.target.src = 'https://picsum.photos/100/100'"
            />
            <div class="flex-1">
                <div class="font-bold text-gray-900">{{ call.name }}</div>
                <div class="flex items-center gap-2 text-sm text-gray-500">
                    <span :class="{'text-red-500': call.status === 'missed', 'text-green-500': call.type === 'outgoing'}">
                        {{ call.status === 'missed' ? 'Missed' : (call.type === 'incoming' ? 'Incoming' : 'Outgoing') }}
                    </span>
                    <span>&bull;</span>
                    <span>{{ call.time }}</span>
                </div>
            </div>
            <div class="text-sm font-medium text-gray-600">{{ Math.floor(call.duration / 60) }}m {{ call.duration % 60 }}s</div>
            <button class="text-[#6264A7] hover:bg-purple-50 p-2 rounded-full">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                </svg>
            </button>
          </div>
          
          <div v-if="filteredCalls.length === 0" class="flex flex-col items-center justify-center p-12 text-gray-500">
             <p>No calls found.</p>
          </div>
        </div>
      </main>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CALLS_HUB',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const missedOnly = ref(false)
    const minDuration = ref(0)
    const sortBy = ref('')
    const sortOpen = ref(false)

    // Filter Logic
    const filteredCalls = computed(() => {
      let result = dataStore.calls;

      if (missedOnly.value) {
        result = result.filter(c => c.status === 'missed');
      }

      if (minDuration.value > 0) {
        result = result.filter(c => c.duration > minDuration.value);
      }

      if (sortBy.value === 'duration') {
        result = [...result].sort((a, b) => b.duration - a.duration);
      } else if (sortBy.value === 'recent') {
        // Mock recent
        result = result; // Assume already recent
      }

      return result;
    })

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value
    }

    const setSort = (type) => {
      sortBy.value = type;
      sortOpen.value = false;
      store.calls_filters_applied = true;
    }

    const goHome = async () => {
      store.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    }

    return {
      missedOnly,
      minDuration,
      sortBy,
      sortOpen,
      filteredCalls,
      toggleSort,
      setSort,
      goHome,
      store
    }
  },
  watch: {
    missedOnly() {
      this.store.calls_filters_applied = true;
    },
    minDuration() {
      this.store.calls_filters_applied = true;
    }
  }
}
</script>