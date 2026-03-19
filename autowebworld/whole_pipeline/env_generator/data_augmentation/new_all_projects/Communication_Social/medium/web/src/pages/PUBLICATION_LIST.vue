<template>
  <div class="min-h-screen bg-white pb-20">
    <nav class="border-b border-gray-200 sticky top-0 bg-white z-30">
       <div class="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
          <button id="publication-back-home" @click="handleBackHome" class="font-serif text-2xl font-bold">Medium</button>
          <div class="font-sans font-bold text-lg">Publications</div>
       </div>
    </nav>

    <div class="max-w-5xl mx-auto px-4 py-12" id="publication-list" @drag.end="handleScrollDrag">
       <div class="bg-gray-50 p-6 rounded-xl mb-12 border border-gray-100">
          <h2 class="font-serif font-bold text-2xl mb-6">Discover Publications</h2>
          
          <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
             <!-- Checkbox Filter -->
             <div class="flex flex-col gap-2">
                <label class="font-sans text-xs font-bold uppercase text-gray-500 tracking-wide">Type</label>
                <label class="flex items-center gap-2 cursor-pointer bg-white p-3 rounded border border-gray-200 hover:border-gray-300 transition-colors">
                   <input 
                      type="checkbox" 
                      id="publication-filter-member-checkbox" 
                      v-model="filterMemberOnly"
                      @change="applyFilters"
                      class="rounded text-green-600 focus:ring-green-500 border-gray-300" 
                   />
                   <span class="text-sm font-sans">Member Only</span>
                </label>
             </div>
             
             <!-- Slider Filter -->
             <div class="flex flex-col gap-2">
                <label class="font-sans text-xs font-bold uppercase text-gray-500 tracking-wide">Min Followers ({{ filterMinSize * 1000 }}k+)</label>
                <div class="bg-white p-3 rounded border border-gray-200 flex items-center h-[46px]">
                   <input 
                      id="publication-filter-size-slider" 
                      type="range" 
                      min="0" 
                      max="500" 
                      step="10" 
                      v-model.number="filterMinSize"
                      @input="applyFilters"
                      class="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
                   />
                </div>
             </div>
             
             <!-- Sort Dropdown -->
             <div class="flex flex-col gap-2 relative">
                <label class="font-sans text-xs font-bold uppercase text-gray-500 tracking-wide">Sort Order</label>
                <div 
                   id="publication-sort-dropdown" 
                   @click="toggleSortMenu"
                   class="bg-white p-3 rounded border border-gray-200 cursor-pointer flex justify-between items-center h-[46px] hover:border-gray-300 transition-colors"
                >
                   <span class="text-sm font-sans capitalize">{{ sortOption }}</span>
                   <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                     <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                   </svg>
                </div>
                
                <div v-if="sortMenuOpen" class="absolute top-full left-0 w-full mt-1 bg-white border border-gray-100 shadow-lg rounded-md z-10">
                   <div id="publication-sort-option-featured" @click="handleSort('featured')" class="px-4 py-2 text-sm font-sans hover:bg-gray-50 cursor-pointer">Featured</div>
                   <div id="publication-sort-option-new" @click="handleSort('new')" class="px-4 py-2 text-sm font-sans hover:bg-gray-50 cursor-pointer">Newest</div>
                   <div id="publication-sort-option-name" @click="handleSort('name')" class="px-4 py-2 text-sm font-sans hover:bg-gray-50 cursor-pointer">Name</div>
                </div>
             </div>
          </div>
       </div>

       <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div v-for="pub in filteredPublications" :key="pub.id" :class="`data-id-${pub.id}`">
             <div class="flex gap-4 items-start group cursor-pointer">
                <img :src="pub.icon" class="w-16 h-16 rounded-lg object-cover" />
                <div class="flex-1">
                   <h3 
                      :class="{
                         'text-xl font-bold font-serif mb-1 group-hover:underline decoration-2': true,
                         'publication-filtered': hasFilters,
                         'publication-visible': !hasFilters
                      }"
                      @click="handleOpenPublication(pub.id)"
                   >
                      {{ pub.name }}
                   </h3>
                   <p class="text-gray-500 font-serif text-sm mb-2 line-clamp-2">{{ pub.description }}</p>
                   <div class="flex items-center gap-3 text-xs text-gray-400 font-sans">
                      <span>{{ (pub.member_count / 1000).toFixed(0) }}k Followers</span>
                      <span v-if="pub.is_featured" class="bg-yellow-100 text-yellow-800 px-1.5 py-0.5 rounded">Featured</span>
                   </div>
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
  name: 'PUBLICATION_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    const filterMemberOnly = ref(false)
    const filterMinSize = ref(0)
    const sortOption = ref('featured')
    const sortMenuOpen = ref(false)
    
    const filteredPublications = computed(() => {
       let result = [...dataStore.publications]
       
       if (filterMemberOnly.value) {
          // Mock filter logic, assuming some field or random
          result = result.filter(p => p.member_count > 200000)
       }
       
       if (filterMinSize.value > 0) {
          result = result.filter(p => (p.member_count / 1000) >= filterMinSize.value)
       }
       
       if (sortOption.value === 'featured') {
          result.sort((a, b) => (a.is_featured === b.is_featured) ? 0 : a.is_featured ? -1 : 1)
       } else if (sortOption.value === 'new') {
          // No date field, mock sort by id desc
          result.sort((a, b) => b.id.localeCompare(a.id))
       } else if (sortOption.value === 'name') {
          result.sort((a, b) => a.name.localeCompare(b.name))
       }
       
       return result
    })
    
    const hasFilters = computed(() => filterMemberOnly.value || filterMinSize.value > 0 || sortOption.value !== 'featured')

    const applyFilters = () => {
       signatureStore.publication_filters_applied = true
    }
    
    const toggleSortMenu = () => {
       sortMenuOpen.value = !sortMenuOpen.value
    }
    
    const handleSort = (option) => {
       sortOption.value = option
       signatureStore.publication_filters_applied = true
       sortMenuOpen.value = false
    }
    
    const handleOpenPublication = async (id) => {
       signatureStore.publication_selected_id = id
       signatureStore.publication_filters_applied = null
       signatureStore.setCurrentPageId('PUBLICATION_DETAIL')
       await router.push({ name: 'PUBLICATION_DETAIL', params: { id } })
    }
    
    const handleBackHome = async () => {
       signatureStore.setCurrentPageId('HOME')
       await router.push({ name: 'HOME' })
    }

    const handleScrollDrag = () => {
       if (filteredPublications.value.length > 0) {
          signatureStore.publication_viewport_anchor_id = filteredPublications.value[0].id
       }
    }

    return {
       filterMemberOnly,
       filterMinSize,
       sortOption,
       sortMenuOpen,
       filteredPublications,
       hasFilters,
       applyFilters,
       toggleSortMenu,
       handleSort,
       handleOpenPublication,
       handleBackHome,
       handleScrollDrag
    }
  }
}
</script>