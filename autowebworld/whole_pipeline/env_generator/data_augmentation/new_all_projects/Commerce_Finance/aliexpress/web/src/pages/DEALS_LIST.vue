<template>
  <div class="min-h-screen bg-red-50 pb-20">
    <!-- Header -->
    <header class="sticky top-0 z-20 bg-red-600 text-white shadow-md px-4 py-3 flex items-center justify-between">
      <button 
        id="back-home" 
        class="p-2 hover:bg-red-700 rounded-full transition-colors"
        @click="handleBackHome"
      >
        <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
      </button>
      <div class="flex flex-col items-center">
        <h1 class="text-lg font-black tracking-wider uppercase">Flash Deals</h1>
        <span class="text-xs bg-white text-red-600 px-2 py-0.5 rounded-full font-bold">Ending in 03:21:45</span>
      </div>
      <div class="w-10"></div>
    </header>

    <!-- Filters -->
    <div class="bg-white px-4 py-4 sticky top-[60px] z-10 shadow-sm space-y-3">
      <div class="flex items-center justify-between">
        <!-- Free Shipping Checkbox -->
        <div 
          id="filter-free-shipping-checkbox" 
          class="flex items-center space-x-2 cursor-pointer"
          @click="handleFilterFreeShipping"
        >
          <div :class="['w-5 h-5 border-2 rounded flex items-center justify-center transition-colors', signatureStore.DEALS_LIST_filters_applied ? 'bg-red-600 border-red-600' : 'border-gray-300']">
             <svg v-if="signatureStore.DEALS_LIST_filters_applied" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
          </div>
          <span class="text-sm font-medium text-gray-700">Free Shipping Only</span>
        </div>

        <!-- Sort -->
        <div class="relative group">
          <button id="deals-sort-dropdown" class="text-sm font-bold text-red-600 flex items-center">
            Sort
            <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>
          <div class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl border border-gray-100 hidden group-hover:block z-50">
            <div 
              id="deals-sort-discount-desc" 
              class="px-4 py-3 text-sm text-gray-700 hover:bg-red-50 cursor-pointer border-b border-gray-50"
              @click="handleSort('biggest_discount')"
            >
              Biggest Discount
            </div>
            <div 
              id="deals-sort-price-low" 
              class="px-4 py-3 text-sm text-gray-700 hover:bg-red-50 cursor-pointer border-b border-gray-50"
              @click="handleSort('price_low_high')"
            >
              Price: Low to High
            </div>
            <div 
              id="deals-sort-price-high" 
              class="px-4 py-3 text-sm text-gray-700 hover:bg-red-50 cursor-pointer"
              @click="handleSort('price_high_low')"
            >
              Price: High to Low
            </div>
          </div>
        </div>
      </div>

      <!-- Price Slider -->
      <div class="pt-2 pb-1">
        <div class="flex justify-between text-xs text-gray-500 mb-1">
          <span>Price Range</span>
          <span class="font-bold text-gray-900">${{ priceRange }} - $1000+</span>
        </div>
        <input 
          id="deals-price-slider"
          type="range" 
          min="0" 
          max="1000" 
          step="10"
          v-model="priceRange"
          class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-red-600"
          @input="handlePriceSlider"
        />
      </div>
    </div>

    <!-- Deals Grid -->
    <div 
      id="deals-list-container" 
      class="container mx-auto px-2 py-4 grid grid-cols-2 gap-2"
    >
      <div 
        v-for="item in filteredItems" 
        :key="item.id"
        :class="[
          'bg-white rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-shadow cursor-pointer',
          signatureStore.DEALS_LIST_filters_applied ? 'deal-row-filtered' : 'deal-row-visible',
          `data-id-${item.id}`
        ]"
        @click="handleOpenItem(item)"
      >
        <div class="h-40 relative bg-gray-100">
          <img :src="item.image" class="w-full h-full object-cover" alt="Deal Item" />
          <div class="absolute bottom-0 left-0 right-0 bg-red-600/90 text-white text-xs font-bold px-2 py-1 text-center">
            Flash Deal
          </div>
          <div class="absolute top-2 right-2 bg-yellow-400 text-red-800 text-xs font-bold px-1.5 py-0.5 rounded">
            -{{ item.discount }}%
          </div>
        </div>
        <div class="p-3">
          <div class="mb-2">
            <span class="text-lg font-black text-red-600">${{ item.price }}</span>
            <span class="ml-2 text-xs text-gray-400 line-through">${{ item.originalPrice }}</span>
          </div>
          <div class="w-full bg-gray-200 rounded-full h-1.5 mb-1">
            <div class="bg-red-500 h-1.5 rounded-full" :style="`width: ${item.stock}%`"></div>
          </div>
          <div class="text-[10px] text-gray-500 flex justify-between">
            <span>Sold: {{ item.sold }}</span>
            <span class="text-red-500">🔥 Hot</span>
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
  name: 'DEALS_LIST',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    const priceRange = ref(0)
    
    // Use deals/products from data store
    const items = computed(() => dataStore.products || [])

    const filteredItems = computed(() => {
      let result = [...items.value]

      // Apply filters
      if (signatureStore.DEALS_LIST_filters_applied) {
        // Filter by Free Shipping
        result = result.filter(item => item.shipping === 'Free')
      }

      // Apply price range filter separately
      if (priceRange.value > 0) {
        result = result.filter(item => item.price > priceRange.value)
      }

      // Apply sorting
      if (signatureStore.DEALS_LIST_sort_type) {
        switch (signatureStore.DEALS_LIST_sort_type) {
          case 'biggest_discount':
            result.sort((a, b) => b.discount - a.discount)
            break
          case 'price_low_high':
            result.sort((a, b) => a.price - b.price)
            break
          case 'price_high_low':
            result.sort((a, b) => b.price - a.price)
            break
        }
      }

      return result
    })

    const handleBackHome = async () => {
      signatureStore.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    const handleFilterFreeShipping = () => {
      signatureStore.DEALS_LIST_filters_applied = true
    }

    const handlePriceSlider = () => {
      signatureStore.DEALS_LIST_filters_applied = true
    }

    const handleSort = (type) => {
      signatureStore.DEALS_LIST_sort_type = type
      signatureStore.DEALS_LIST_filters_applied = true
    }

    const handleOpenItem = async (item) => {
      if (signatureStore.DEALS_LIST_filters_applied) {
         signatureStore.DEALS_LIST_filters_applied = null
      } else {
         signatureStore.DEALS_LIST_viewport_anchor_id = null
      }
      signatureStore.selected_item_id = item.id
      signatureStore.currentPageId = 'PRODUCT_DETAIL'
      await router.push({ name: 'PRODUCT_DETAIL' })
    }

    // Scroll handler
    watch(() => signatureStore.DEALS_LIST_viewport_anchor_id, async (newId) => {
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
      priceRange,
      handleBackHome,
      handleFilterFreeShipping,
      handlePriceSlider,
      handleSort,
      handleOpenItem
    }
  }
}
</script>