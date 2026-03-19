<template>
  <div class="min-h-screen bg-gray-50 pb-20">
    <!-- Header -->
    <div class="bg-white px-4 py-4 shadow-sm sticky top-0 z-20 flex items-center justify-between">
      <button 
        id="back-home-topup" 
        @click="goHome"
        class="p-2 -ml-2 rounded-full hover:bg-gray-100 text-gray-600"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Add Money</h1>
      <div class="w-10"></div>
    </div>

    <!-- Filters -->
    <div class="bg-white px-4 py-4 mb-4 border-b border-gray-100">
      <div class="flex items-center gap-3">
        <!-- Linked Only Filter -->
        <button 
          id="filter-linked-only"
          @click="toggleLinkedOnly"
          :class="['px-3 py-1.5 rounded-full text-sm font-medium border transition-colors flex items-center gap-1', linkedOnly ? 'bg-blue-100 text-blue-700 border-blue-200' : 'bg-gray-50 text-gray-600 border-gray-200']"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"></path></svg>
          Linked Methods
        </button>
      </div>
    </div>

    <!-- Methods List -->
    <div id="topup-methods-list" class="px-4 space-y-3">
      <div 
        v-for="method in filteredMethods" 
        :key="method.id"
        :class="['bg-white rounded-2xl p-4 shadow-sm border border-transparent hover:border-blue-100 transition-all cursor-pointer flex items-center justify-between', filtersApplied ? 'method-row-filtered' : 'method-row-visible']"
        :data-id="method.id"
        @click="openMethod(method.id)"
      >
        <div class="flex items-center gap-4">
          <div class="w-12 h-12 rounded-xl overflow-hidden bg-gray-50 p-2 border border-gray-100">
             <img :src="method.image" class="w-full h-full object-contain" />
          </div>
          <div>
            <div class="font-bold text-gray-900">{{ method.name }}</div>
            <div class="text-sm text-gray-500">{{ method.type }} • {{ method.processingTime }}</div>
          </div>
        </div>
        <div class="text-gray-400">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path></svg>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredMethods.length === 0" class="text-center py-10">
        <p class="text-gray-500 font-medium">No methods found</p>
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
  name: 'TOPUP_METHOD_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const linkedOnly = ref(false)
    const filtersApplied = ref(false)

    const filteredMethods = computed(() => {
      let result = [...dataStore.topupMethods]
      if (linkedOnly.value) {
        result = result.filter(m => m.isLinked)
      }
      return result
    })

    const goHome = () => {
      signatureStore.setCurrentPageId('HOME')
      router.push({ name: 'HOME' })
    }

    const toggleLinkedOnly = () => {
      linkedOnly.value = !linkedOnly.value
      filtersApplied.value = true
      signatureStore.topup_filters_applied = true
    }

    const openMethod = (id) => {
      signatureStore.topup_selected_method_id = id
      
      if (filtersApplied.value) {
        signatureStore.topup_filters_applied = null
      }
      signatureStore.topup_viewport_anchor_id = null

      signatureStore.setCurrentPageId('TOPUP_FORM')
      router.push({ name: 'TOPUP_FORM' })
    }

    return {
      linkedOnly,
      filteredMethods,
      filtersApplied,
      goHome,
      toggleLinkedOnly,
      openMethod
    }
  }
}
</script>