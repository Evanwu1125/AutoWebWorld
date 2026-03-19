<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm z-10">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <h1 class="text-2xl font-bold text-[#722282]">Select a Therapist</h1>
        <button id="back-visit-types" @click="handleBack" class="text-gray-600 hover:text-gray-900">
          Back
        </button>
      </div>
    </header>

    <main class="flex-1 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <!-- Search & Filter Bar -->
      <div class="bg-white p-4 rounded-lg shadow mb-6 space-y-4 md:space-y-0 md:flex md:items-center md:space-x-4">
        <!-- Search -->
        <div class="flex-1 relative">
           <input
             id="mh-search-input"
             type="text"
             placeholder="Search by name or specialty..."
             v-model="searchQuery"
             @keyup.enter="handleSearch"
             class="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-[#722282] focus:border-[#722282]"
           />
           <div class="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
             <svg class="h-5 w-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
           </div>
        </div>

        <!-- Filter Slider (Experience) -->
        <div class="flex items-center space-x-2 w-full md:w-1/3">
           <label for="experience-slider" class="text-sm font-medium text-gray-700 whitespace-nowrap">
             Min Experience: {{ minExperience }} yrs
           </label>
           <input
             id="experience-slider"
             type="range"
             min="0"
             max="30"
             step="1"
             v-model="minExperience"
             @input="handleFilterChange"
             class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#722282]"
           />
        </div>
      </div>

      <!-- List -->
      <div id="mh-list-container" class="space-y-4">
        <div 
          v-for="therapist in filteredTherapists" 
          :key="therapist.id"
          class="bg-white rounded-lg shadow hover:shadow-md transition-shadow duration-200 overflow-hidden"
          :class="{
             'ring-2 ring-green-500': therapist.id === matchedId,
             'ring-2 ring-blue-500': therapist.id === store.mh_list_viewport_anchor_id
          }"
        >
          <div 
             :id="therapist.id === matchedId ? 'mh-list-item-matched' : (isFiltered ? 'mh-list-item-filtered' : 'mh-list-item-visible')"
             :class="`data-id-${therapist.id} p-6 flex items-start space-x-4 cursor-pointer`"
             @click="handleSelectTherapist(therapist)"
          >
             <img :src="therapist.image" :alt="therapist.name" class="h-20 w-20 rounded-full object-cover border border-gray-200" />
             <div class="flex-1 min-w-0">
               <h3 class="text-lg font-bold text-gray-900 truncate">{{ therapist.name }}</h3>
               <p class="text-[#722282] font-medium">{{ therapist.specialty }}</p>
               <p class="text-sm text-gray-500 mt-1">{{ therapist.experience }} years experience</p>
             </div>
             <div class="self-center">
                <svg class="h-6 w-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
             </div>
          </div>
        </div>

        <div v-if="filteredTherapists.length === 0" class="text-center py-12">
           <p class="text-gray-500">No therapists found matching your criteria.</p>
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
  name: 'MENTAL_HEALTH_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const minExperience = ref(0)
    const matchedId = ref(null)

    const filteredTherapists = computed(() => {
      let result = dataStore.therapists

      // Filter by Experience
      if (minExperience.value > 0) {
        result = result.filter(t => t.experience >= minExperience.value)
      }

      return result
    })

    const isFiltered = computed(() => minExperience.value > 0)

    const handleSearch = () => {
      // ACT_MH_SEARCH
      const match = dataStore.therapists.find(t => 
        t.name.toLowerCase().includes(searchQuery.value.toLowerCase()) || 
        t.specialty.toLowerCase().includes(searchQuery.value.toLowerCase())
      )
      
      if (match) {
        store.matched_therapist_id = match.id
        store.mh_list_has_searched = true
        matchedId.value = match.id
      } else {
        matchedId.value = null
      }
    }

    const handleFilterChange = () => {
      // ACT_MH_FILTER_EXPERIENCE
      store.mh_list_filters_applied = true
    }

    const handleSelectTherapist = async (therapist) => {
      // ACT_MH_OPEN_MATCHED, ACT_MH_OPEN_ANY, ACT_MH_OPEN_FILTERED
      store.selected_therapist_id = therapist.id
      
      store.mh_list_has_searched = false
      store.mh_list_viewport_anchor_id = null
      store.mh_list_filters_applied = false

      store.setCurrentPageId('MENTAL_HEALTH_DETAIL')
      await router.push({ name: 'MENTAL_HEALTH_DETAIL' })
    }

    const handleBack = async () => {
      // ACT_MH_BACK_VT
      store.setCurrentPageId('VISIT_TYPE_SELECTION')
      await router.push({ name: 'VISIT_TYPE_SELECTION' })
    }

    return {
      store,
      searchQuery,
      minExperience,
      matchedId,
      filteredTherapists,
      isFiltered,
      handleSearch,
      handleFilterChange,
      handleSelectTherapist,
      handleBack
    }
  }
}
</script>