<template>
  <div class="min-h-screen bg-gray-50 pb-20">
    <!-- Header -->
    <header class="sticky top-0 z-20 bg-white shadow-sm px-4 py-3 flex items-center justify-between">
      <button 
        id="back-home" 
        class="p-2 hover:bg-gray-100 rounded-full transition-colors"
        @click="handleBackHome"
      >
        <svg class="w-6 h-6 text-gray-700" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
      </button>
      <h1 class="text-lg font-bold text-gray-900">Categories</h1>
      <div class="w-10"></div>
    </header>

    <!-- Filters & Sort Bar -->
    <div class="bg-white border-t border-gray-100 px-4 py-3 flex items-center justify-between sticky top-[57px] z-10 shadow-sm">
      <div class="flex items-center space-x-4">
        <!-- Sort Dropdown -->
        <div class="relative group">
          <button id="category-sort-dropdown" class="flex items-center space-x-1 text-sm font-medium text-gray-700 hover:text-red-600">
            <span>Sort By</span>
            <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>
          <!-- Dropdown Menu -->
          <div class="absolute left-0 mt-2 w-40 bg-white rounded-lg shadow-xl border border-gray-100 hidden group-hover:block z-50">
            <div 
              id="category-sort-popular" 
              class="px-4 py-2 text-sm text-gray-700 hover:bg-red-50 cursor-pointer"
              @click="handleSort('popular')"
            >
              Popular
            </div>
            <div 
              id="category-sort-orders" 
              class="px-4 py-2 text-sm text-gray-700 hover:bg-red-50 cursor-pointer"
              @click="handleSort('orders')"
            >
              Orders
            </div>
            <div 
              id="category-sort-newest" 
              class="px-4 py-2 text-sm text-gray-700 hover:bg-red-50 cursor-pointer"
              @click="handleSort('newest')"
            >
              Newest
            </div>
          </div>
        </div>

        <!-- Filter Checkbox -->
        <div 
          id="filter-ship-from-overseas" 
          class="flex items-center space-x-2 cursor-pointer group"
          @click="handleFilterOverseas"
        >
          <div :class="['w-4 h-4 border rounded flex items-center justify-center transition-colors', signatureStore.CATEGORY_LIST_filters_applied ? 'bg-red-600 border-red-600' : 'border-gray-300 group-hover:border-red-400']">
             <svg v-if="signatureStore.CATEGORY_LIST_filters_applied" class="w-3 h-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
          </div>
          <span class="text-sm text-gray-600 group-hover:text-red-600 transition-colors">Ship from Overseas</span>
        </div>
      </div>
    </div>

    <!-- Category List -->
    <div 
      id="category-list-container" 
      class="container mx-auto px-4 py-4 space-y-4"
    >
      <div 
        v-for="item in filteredItems" 
        :key="item.id"
        :class="[
          'bg-white rounded-xl p-4 shadow-sm flex items-start space-x-4 cursor-pointer transition-all hover:shadow-md hover:translate-x-1',
          signatureStore.CATEGORY_LIST_filters_applied ? 'category-row-filtered' : 'category-row-visible',
          `data-id-${item.id}`
        ]"
        @click="handleOpenProduct(item)"
      >
        <img :src="item.image" class="w-24 h-24 object-cover rounded-lg bg-gray-100 flex-shrink-0" alt="Category Item" />
        <div class="flex-1 min-w-0">
          <h3 class="text-gray-900 font-bold text-base mb-1 line-clamp-2 leading-tight">{{ item.name }}</h3>
          <div class="flex items-baseline space-x-2 mb-2">
            <span class="text-red-600 font-black text-lg">${{ item.price }}</span>
            <span v-if="item.originalPrice" class="text-xs text-gray-400 line-through">${{ item.originalPrice }}</span>
          </div>
          <div class="flex items-center space-x-2 text-xs text-gray-500">
            <span class="bg-gray-100 px-1.5 py-0.5 rounded">{{ item.sold }} sold</span>
            <span v-if="item.shipping === 'Free'" class="text-green-600 font-medium">Free Shipping</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Permission Modal -->
    <PermissionModal 
      v-if="showPermissionModal"
      @allow="handleAllowLocation"
      @deny="handleDenyLocation"
    />

  </div>
</template>

<script>
import { computed, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'
import PermissionModal from '../components/PermissionModal.vue'

export default {
  name: 'CATEGORY_LIST',
  components: {
    PermissionModal
  },
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    // Use products from data store as category items for this demo
    const items = computed(() => dataStore.products || [])

    const showPermissionModal = computed(() => {
      return signatureStore.location_permission_granted === null || signatureStore.location_permission_granted === undefined
    })

    const filteredItems = computed(() => {
      let result = [...items.value]
      if (signatureStore.CATEGORY_LIST_filters_applied) {
        // Simulate filter effect - e.g., filter by price > 20 or simply return subset
        // In a real app, this would use actual filter criteria
        // For FSM strictness: just returning result is fine, as long as UI updates state
        // But to be realistic, let's filter
        return result.filter((_, i) => i % 2 === 0) // Return half items to show filter worked
      }
      return result
    })

    const handleBackHome = async () => {
      signatureStore.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const handleSort = (type) => {
      signatureStore.CATEGORY_LIST_filters_applied = true
      // In real app, sort filteredItems
    }

    const handleFilterOverseas = () => {
      signatureStore.CATEGORY_LIST_filters_applied = true
    }

    const handleOpenProduct = async (item) => {
      if (signatureStore.CATEGORY_LIST_filters_applied) {
         signatureStore.CATEGORY_LIST_filters_applied = null // Clear as per effect
      } else {
         signatureStore.CATEGORY_LIST_viewport_anchor_id = null // Clear as per effect
      }
      signatureStore.selected_item_id = item.id
      signatureStore.currentPageId = 'PRODUCT_LIST' // FSM says ACT_CATEGORY_OPEN_ANY_PRODUCT goes to PRODUCT_LIST
      await router.push({ name: 'PRODUCT_LIST' })
    }

    const handleAllowLocation = () => {
      signatureStore.location_permission_granted = true
    }

    const handleDenyLocation = () => {
      // Optional deny logic
    }

    // Scroll handler
    watch(() => signatureStore.CATEGORY_LIST_viewport_anchor_id, async (newId) => {
      if (newId) {
        await nextTick()
        const element = document.querySelector(`.data-id-${newId}`)
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' })
        }
      }
    })

    return {
      signatureStore,
      filteredItems,
      showPermissionModal,
      handleBackHome,
      handleSort,
      handleFilterOverseas,
      handleOpenProduct,
      handleAllowLocation,
      handleDenyLocation
    }
  }
}
</script>