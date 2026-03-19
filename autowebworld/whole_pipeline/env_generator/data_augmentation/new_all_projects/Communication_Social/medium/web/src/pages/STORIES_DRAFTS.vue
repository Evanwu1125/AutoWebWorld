<template>
  <div class="min-h-screen bg-white">
    <!-- Nav -->
    <nav class="border-b border-gray-200">
       <div class="max-w-5xl mx-auto px-4 h-16 flex items-center justify-between">
          <div class="flex items-center gap-4">
             <button id="stories-back-profile" @click="handleBackProfile" class="p-2 hover:bg-gray-100 rounded-full transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                </svg>
             </button>
             <span class="font-serif font-bold text-lg">Your Stories</span>
          </div>
          <div class="flex gap-4">
             <button class="text-sm font-sans px-3 py-1 rounded-full bg-black text-white">Drafts</button>
             <button class="text-sm font-sans px-3 py-1 rounded-full text-gray-500 hover:bg-gray-100">Published</button>
          </div>
       </div>
    </nav>

    <div class="max-w-5xl mx-auto px-4 py-12" id="stories-list" @drag.end="handleScrollDrag">
       <div class="flex flex-col md:flex-row md:items-center justify-between gap-6 mb-12 border-b border-gray-100 pb-6">
          <!-- Filters -->
          <div class="flex items-center gap-6">
             <label class="flex items-center gap-2 cursor-pointer">
                <input 
                   type="checkbox" 
                   id="stories-filter-published-checkbox" 
                   v-model="filterPublished"
                   @change="applyFilters"
                   class="rounded text-green-600 focus:ring-green-500 border-gray-300" 
                />
                <span class="text-sm text-gray-600 font-sans">Published Only</span>
             </label>
             
             <div class="flex items-center gap-3">
                <span class="text-sm text-gray-600 font-sans">Min Length: {{ filterMinLength }} min</span>
                <input 
                   id="stories-filter-length-slider" 
                   type="range" 
                   min="0" 
                   max="10" 
                   step="1" 
                   v-model.number="filterMinLength"
                   @input="applyFilters"
                   class="w-32 h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-green-600"
                />
             </div>
          </div>
          
          <!-- Sort -->
          <div class="relative">
             <button id="stories-sort-dropdown" @click="toggleSortMenu" class="flex items-center gap-1 text-sm font-sans text-gray-600 hover:text-black">
                Sort by {{ sortOption }}
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
                </svg>
             </button>
             
             <div v-if="sortMenuOpen" class="absolute right-0 mt-2 w-40 bg-white border border-gray-100 rounded shadow-lg py-1 z-10">
                <div id="stories-sort-option-newest" @click="handleSort('newest')" class="px-4 py-2 text-sm font-sans hover:bg-gray-50 cursor-pointer">Newest</div>
                <div id="stories-sort-option-oldest" @click="handleSort('oldest')" class="px-4 py-2 text-sm font-sans hover:bg-gray-50 cursor-pointer">Oldest</div>
                <div id="stories-sort-option-drafts" @click="handleSort('drafts')" class="px-4 py-2 text-sm font-sans hover:bg-gray-50 cursor-pointer">Drafts First</div>
             </div>
          </div>
       </div>
       
       <div class="space-y-8">
          <div v-for="draft in filteredDrafts" :key="draft.id" :class="`data-id-${draft.id}`">
             <div class="group cursor-pointer py-6 border-b border-gray-100 last:border-0">
                <h3 
                   :class="{
                      'text-xl font-bold font-serif mb-2 group-hover:underline': true,
                      'story-filtered': hasFilters,
                      'story-visible': !hasFilters
                   }"
                   @click="handleOpenDraft(draft.id)"
                >
                   {{ draft.title || 'Untitled Story' }}
                </h3>
                <p class="text-gray-500 font-serif text-sm mb-3">{{ draft.subtitle || draft.body.substring(0, 60) + '...' }}</p>
                
                <div class="flex items-center gap-4 text-xs text-gray-400 font-sans">
                   <span v-if="draft.status === 'published'" class="text-green-600 font-medium">Published</span>
                   <span v-else>Draft</span>
                   <span>Last edited {{ formatDate(draft.updated_at) }}</span>
                   <span>{{ draft.length_minutes }} min read</span>
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
  name: 'STORIES_DRAFTS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    const filterPublished = ref(false)
    const filterMinLength = ref(0)
    const sortOption = ref('newest')
    const sortMenuOpen = ref(false)
    
    const filteredDrafts = computed(() => {
       let result = [...dataStore.drafts]
       
       if (filterPublished.value) {
          result = result.filter(d => d.published)
       }
       
       if (filterMinLength.value > 0) {
          result = result.filter(d => d.length_minutes >= filterMinLength.value)
       }
       
       if (sortOption.value === 'newest') {
          result.sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
       } else if (sortOption.value === 'oldest') {
          result.sort((a, b) => new Date(a.updated_at) - new Date(b.updated_at))
       } else if (sortOption.value === 'drafts') {
          result.sort((a, b) => (a.published === b.published) ? 0 : a.published ? 1 : -1)
       }
       
       return result
    })
    
    const hasFilters = computed(() => filterPublished.value || filterMinLength.value > 0 || sortOption.value !== 'newest')

    const formatDate = (dateStr) => {
       return new Date(dateStr).toLocaleDateString()
    }
    
    const applyFilters = () => {
       signatureStore.stories_filters_applied = true
    }
    
    const toggleSortMenu = () => {
       sortMenuOpen.value = !sortMenuOpen.value
    }
    
    const handleSort = (option) => {
       sortOption.value = option
       signatureStore.stories_filters_applied = true
       sortMenuOpen.value = false
    }
    
    const handleOpenDraft = async (id) => {
       signatureStore.stories_selected_draft_id = id
       signatureStore.setCurrentPageId('NEW_STORY_EDITOR')
       await router.push({ name: 'NEW_STORY_EDITOR' })
    }
    
    const handleBackProfile = async () => {
       signatureStore.setCurrentPageId('PROFILE_OVERVIEW')
       await router.push({ name: 'PROFILE_OVERVIEW' })
    }

    const handleScrollDrag = () => {
       if (filteredDrafts.value.length > 0) {
          signatureStore.stories_viewport_anchor_id = filteredDrafts.value[0].id
       }
    }

    return {
       filterPublished,
       filterMinLength,
       sortOption,
       sortMenuOpen,
       filteredDrafts,
       hasFilters,
       formatDate,
       applyFilters,
       toggleSortMenu,
       handleSort,
       handleOpenDraft,
       handleBackProfile,
       handleScrollDrag
    }
  }
}
</script>