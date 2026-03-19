<template>
  <div class="min-h-screen bg-gray-50 pb-20">
    <!-- Header Search -->
    <header class="sticky top-0 z-30 bg-white shadow-sm">
      <div class="px-4 py-3 flex items-center space-x-3">
        <button 
          id="product-list-back-category" 
          class="p-2 -ml-2 hover:bg-gray-100 rounded-full transition-colors"
          @click="handleBackCategory"
        >
          <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"></path></svg>
        </button>
        <div class="flex-1 relative">
          <input 
            id="search-input"
            type="text" 
            placeholder="Search products..." 
            class="w-full bg-gray-100 border-none rounded-full py-2 pl-10 pr-4 text-sm focus:ring-2 focus:ring-red-500 focus:bg-white transition-all"
            v-model="searchQuery"
            @keyup.enter="handleSearch"
          />
          <svg class="w-5 h-5 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
        </div>
        <button class="p-2 hover:bg-gray-100 rounded-full">
          <svg class="w-6 h-6 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"></path></svg>
        </button>
      </div>
      
      <!-- Filters Bar -->
      <div class="flex items-center justify-between px-4 py-2 border-t border-gray-100 overflow-x-auto no-scrollbar space-x-4">
         <!-- Sort -->
         <div class="relative group flex-shrink-0">
            <button id="sort-dropdown" class="flex items-center space-x-1 text-sm font-medium text-gray-700 hover:text-red-600 whitespace-nowrap">
              <span>Sort</span>
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div class="absolute top-full left-0 mt-2 w-40 bg-white rounded-lg shadow-xl border border-gray-100 hidden group-hover:block z-50">
              <div id="sort-option-orders" class="px-4 py-2 text-sm hover:bg-red-50 cursor-pointer" @click="handleSort('orders')">Orders</div>
              <div id="sort-option-price-low" class="px-4 py-2 text-sm hover:bg-red-50 cursor-pointer" @click="handleSort('price_low_high')">Price Low-High</div>
              <div id="sort-option-price-high" class="px-4 py-2 text-sm hover:bg-red-50 cursor-pointer" @click="handleSort('price_high_low')">Price High-Low</div>
            </div>
         </div>

         <!-- Free Shipping -->
         <div 
           id="filter-free-shipping" 
           class="flex-shrink-0 px-3 py-1 rounded-full border text-xs font-medium cursor-pointer transition-colors whitespace-nowrap"
           :class="signatureStore.PRODUCT_LIST_filters_applied ? 'bg-red-50 border-red-200 text-red-600' : 'bg-gray-100 border-transparent text-gray-600 hover:bg-gray-200'"
           @click="handleFilterFreeShipping"
         >
           Free Shipping
         </div>

         <!-- Price Slider (Compact) -->
         <div class="flex items-center space-x-2 flex-1 min-w-[120px]">
            <span class="text-xs text-gray-500">Price</span>
            <input 
              id="filter-price-slider"
              type="range" 
              class="w-full h-1 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-red-600"
              min="0" max="1000" step="10"
              @input="handlePriceSlider"
            />
         </div>
      </div>
    </header>

    <!-- Product List Grid -->
    <div id="product-list-container" class="container mx-auto px-2 py-2">
      <div id="product-list" class="grid grid-cols-2 gap-2">
        <div 
          v-for="item in filteredItems" 
          :key="item.id"
          :class="[
            'bg-white rounded-lg overflow-hidden shadow-sm hover:shadow-md cursor-pointer flex flex-col h-full',
            getItemClass(item),
            `data-id-${item.id}`
          ]"
          @click="handleOpenItem(item)"
        >
          <div class="aspect-square relative bg-gray-200">
            <img :src="item.image" class="w-full h-full object-cover" alt="Product Image" />
            <!-- Visual badges -->
            <div v-if="item.shipping === 'Free'" class="absolute bottom-0 left-0 bg-green-600 text-white text-[10px] px-1.5 py-0.5 font-bold rounded-tr-md">
              Free Shipping
            </div>
          </div>
          <div class="p-2 flex flex-col flex-1">
            <h3 class="text-xs sm:text-sm text-gray-800 font-medium line-clamp-2 mb-1 leading-tight h-9">{{ item.name }}</h3>
            <div class="mt-auto">
              <div class="flex items-baseline space-x-1">
                <span class="text-red-600 font-black text-base sm:text-lg leading-none">${{ item.price }}</span>
                <span class="text-gray-400 text-[10px] line-through">${{ (item.price * 1.2).toFixed(2) }}</span>
              </div>
              <div class="flex items-center justify-between mt-1">
                 <div class="flex items-center">
                   <svg class="w-3 h-3 text-yellow-400 fill-current" viewBox="0 0 20 20"><path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z"/></svg>
                   <span class="text-[10px] text-gray-500 ml-0.5">4.8</span>
                 </div>
                 <span class="text-[10px] text-gray-500">{{ item.sold }} sold</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, watch, nextTick } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PRODUCT_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    const searchQuery = ref('')

    // Sync search query from store if present
    if (signatureStore.PRODUCT_LIST_has_searched) {
       // Maybe retrieve last query if we stored it, for now just ref
    }

    const items = computed(() => dataStore.products || [])

    const filteredItems = computed(() => {
      let result = [...items.value]
      
      // Search filter
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        result = result.filter(i => i.name.toLowerCase().includes(q))
      } else if (signatureStore.matched_item_id) {
        // If we navigated here with a matched item intention (though FSM sets matched_item_id on search action)
        // Just for visual consistency, show all or filtered
      }

      // Filter flag
      if (signatureStore.PRODUCT_LIST_filters_applied) {
         // Apply some dummy filter logic or randomized subset
         return result.filter((_, i) => i % 2 === 0 || i % 3 === 0)
      }
      
      return result
    })

    const getItemClass = (item) => {
      if (signatureStore.matched_item_id === item.id) return 'product-row-matched'
      if (signatureStore.PRODUCT_LIST_filters_applied) return 'product-row-filtered'
      return 'product-row-visible'
    }

    const handleBackCategory = async () => {
      signatureStore.currentPageId = 'CATEGORY_LIST'
      await router.push({ name: 'CATEGORY_LIST' })
    }

    const handleSearch = () => {
      // FSM: ACT_PRODUCT_LIST_SEARCH
      // In reality, we find a match and set it
      const match = items.value.find(i => i.name.toLowerCase().includes(searchQuery.value.toLowerCase()))
      if (match) {
        signatureStore.matched_item_id = match.id
      }
      signatureStore.PRODUCT_LIST_has_searched = true
    }

    const handleFilterFreeShipping = () => {
      signatureStore.PRODUCT_LIST_filters_applied = true
    }

    const handlePriceSlider = () => {
      signatureStore.PRODUCT_LIST_filters_applied = true
    }

    const handleSort = (type) => {
      signatureStore.PRODUCT_LIST_filters_applied = true
    }

    const handleOpenItem = async (item) => {
      // Determine which action to map to based on state
      // If searched -> ACT_PRODUCT_LIST_OPEN_MATCHED
      // If filtered -> ACT_PRODUCT_LIST_OPEN_FILTERED
      // Else -> ACT_PRODUCT_LIST_OPEN_ANY
      
      // Clear flags as per effects
      if (signatureStore.PRODUCT_LIST_has_searched) {
         signatureStore.PRODUCT_LIST_has_searched = null
      } else if (signatureStore.PRODUCT_LIST_filters_applied) {
         signatureStore.PRODUCT_LIST_filters_applied = null
      } else {
         signatureStore.PRODUCT_LIST_viewport_anchor_id = null
      }
      
      signatureStore.selected_item_id = item.id
      signatureStore.currentPageId = 'PRODUCT_DETAIL'
      await router.push({ name: 'PRODUCT_DETAIL' })
    }

    // Scroll handler
    watch(() => signatureStore.PRODUCT_LIST_viewport_anchor_id, async (newId) => {
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
      searchQuery,
      filteredItems,
      getItemClass,
      handleBackCategory,
      handleSearch,
      handleFilterFreeShipping,
      handlePriceSlider,
      handleSort,
      handleOpenItem
    }
  }
}
</script>