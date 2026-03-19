<template>
  <div class="electronics-page min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-[#0071DC] text-white p-4 sticky top-0 z-30 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center gap-4">
        <!-- Breadcrumb / Back -->
        <div 
          id="breadcrumb-departments" 
          @click="handleBackToDepartments"
          class="cursor-pointer p-2 hover:bg-white/10 rounded-full transition-colors flex items-center gap-1"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
          <span class="text-sm font-medium">Departments</span>
        </div>
        
        <h1 class="text-lg font-bold flex-1">Electronics</h1>

        <!-- Search Bar -->
        <div class="relative w-full max-w-md hidden sm:block">
           <input 
             id="category-search-input"
             type="text" 
             v-model="searchQuery"
             @keydown.enter="handleSearch"
             placeholder="Search in Electronics" 
             class="w-full pl-4 pr-10 py-2 rounded-full text-gray-900 focus:outline-none focus:ring-2 focus:ring-[#FFC220]"
           />
           <div 
             @click="handleSearch"
             class="absolute right-1 top-1/2 -translate-y-1/2 p-1.5 bg-[#FFC220] rounded-full text-black cursor-pointer hover:bg-[#ffcf4d]"
           >
             <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
           </div>
        </div>
      </div>
      <!-- Mobile Search -->
      <div class="sm:hidden px-2 pb-2 mt-2">
           <input 
             type="text" 
             v-model="searchQuery"
             @keydown.enter="handleSearch"
             placeholder="Search in Electronics" 
             class="w-full pl-4 pr-10 py-2 rounded-full text-gray-900 focus:outline-none"
           />
      </div>
    </header>

    <div class="flex-1 max-w-7xl mx-auto w-full p-4 flex flex-col md:flex-row gap-6">
      
      <!-- Filters Sidebar -->
      <aside class="w-full md:w-64 flex-shrink-0 space-y-6">
        <div class="bg-white p-4 rounded-xl shadow-sm">
          <div class="flex items-center justify-between mb-4">
            <h2 class="font-bold text-lg">Filters</h2>
            <button class="text-sm text-blue-600 font-medium hover:underline">Clear all</button>
          </div>
          
          <!-- Brand Filter -->
          <div class="mb-6">
            <h3 class="font-semibold mb-2 text-sm text-gray-700">Brand</h3>
            <div class="space-y-2">
              <label class="flex items-center gap-2 cursor-pointer">
                <input 
                  id="filter-brand-checkbox"
                  type="checkbox" 
                  v-model="brandFilter"
                  @change="handleFilterBrand"
                  class="rounded text-blue-600 focus:ring-blue-500 w-4 h-4"
                />
                <span class="text-sm">Premium Brands</span>
              </label>
              <!-- Decorative checkboxes -->
              <label class="flex items-center gap-2 cursor-pointer text-gray-500">
                <input type="checkbox" class="rounded w-4 h-4" disabled />
                <span class="text-sm">Samsung</span>
              </label>
              <label class="flex items-center gap-2 cursor-pointer text-gray-500">
                <input type="checkbox" class="rounded w-4 h-4" disabled />
                <span class="text-sm">Apple</span>
              </label>
            </div>
          </div>

          <!-- Price Filter -->
          <div class="mb-6">
            <h3 class="font-semibold mb-2 text-sm text-gray-700">Price</h3>
            <div class="px-2">
              <input 
                id="price-slider"
                type="range" 
                v-model.number="priceFilter"
                :min="minPrice"
                :max="maxPrice"
                step="10"
                @input="handleFilterPrice"
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
              />
              <div class="flex justify-between text-xs text-gray-500 mt-2">
                <span>${{ minPrice }}</span>
                <span class="font-bold text-gray-900">Above ${{ priceFilter }}</span>
                <span>${{ maxPrice }}</span>
              </div>
            </div>
          </div>

        </div>
      </aside>

      <!-- Main Content -->
      <div class="flex-1">
        <!-- Sort & Results Header -->
        <div class="flex flex-col sm:flex-row justify-between items-center mb-4 gap-4">
          <div class="text-sm text-gray-600">{{ filteredProducts.length }} results</div>
          
          <div class="relative z-20">
             <button 
               id="sort-dropdown" 
               @click="showSort = !showSort"
               class="flex items-center gap-2 bg-white px-4 py-2 rounded-full shadow-sm text-sm font-medium hover:bg-gray-50 border border-gray-200"
             >
               Sort by: <span class="text-blue-600">{{ currentSortLabel }}</span>
               <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
             </button>
             
             <div v-if="showSort" class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl py-1 border border-gray-100">
               <div 
                 id="sort-option-bestseller" 
                 @click="handleSort('bestseller')"
                 class="px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer"
               >
                 Best Sellers
               </div>
               <div 
                 id="sort-option-price-low-high" 
                 @click="handleSort('price_low_high')"
                 class="px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer"
               >
                 Price: Low to High
               </div>
               <div 
                 id="sort-option-price-high-low" 
                 @click="handleSort('price_high_low')"
                 class="px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer"
               >
                 Price: High to Low
               </div>
               <div 
                 id="sort-option-top-rated" 
                 @click="handleSort('top_rated')"
                 class="px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer"
               >
                 Top Rated
               </div>
             </div>
          </div>
        </div>

        <!-- Product Grid -->
        <div 
          id="product-list"
          class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          <div 
            v-for="product in filteredProducts" 
            :key="product.id"
            :class="[
              'bg-white rounded-xl shadow-sm hover:shadow-lg transition-all p-4 flex flex-col cursor-pointer group relative border border-transparent hover:border-blue-500',
              getProductClass(product) // Assign class based on context (filtered, matched, visible)
            ]"
            :data-id="product.id"
            @click="handleProductClick(product)"
          >
            <!-- Image -->
            <div class="aspect-square mb-4 relative overflow-hidden rounded-lg bg-gray-100">
              <img :src="product.image" :alt="product.name" class="w-full h-full object-contain mix-blend-multiply group-hover:scale-105 transition-transform duration-300" />
              <!-- Favorite Button (Decorative) -->
              <div class="absolute top-2 right-2 p-1.5 bg-white rounded-full shadow-sm hover:bg-gray-50 text-gray-400 hover:text-red-500 transition-colors">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>
              </div>
            </div>
            
            <!-- Details -->
            <div class="flex-1 flex flex-col">
               <div class="flex items-center gap-1 mb-1">
                 <div class="flex text-yellow-400 text-xs">
                   <span v-for="i in 5" :key="i">★</span>
                 </div>
                 <span class="text-xs text-gray-500">({{ Math.floor(product.rating * 100) }})</span>
               </div>
               
               <h3 class="font-medium text-gray-900 mb-2 line-clamp-2 hover:text-[#0071DC] transition-colors">{{ product.name }}</h3>
               
               <div class="mt-auto">
                 <div class="text-2xl font-bold text-gray-900">${{ product.price.toFixed(2) }}</div>
                 <div class="text-xs text-gray-500 mt-1">Free shipping, arrives in 3+ days</div>
               </div>
            </div>
            
            <button class="mt-4 w-full py-2 rounded-full border border-gray-300 text-gray-700 font-medium hover:border-gray-800 transition-colors">
              Options
            </button>
          </div>
        </div>

        <!-- Empty State -->
        <div v-if="filteredProducts.length === 0" class="text-center py-20 bg-white rounded-xl shadow-sm mt-4">
           <div class="text-6xl mb-4">🔍</div>
           <h3 class="text-xl font-bold text-gray-900">No products found</h3>
           <p class="text-gray-500 mt-2">Try adjusting your filters or search query.</p>
        </div>

      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ELECTRONICS_CATEGORY',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    // State
    const searchQuery = ref('')
    const brandFilter = ref(false)
    const priceFilter = ref(0) // Start at min
    const currentSort = ref(null)
    const showSort = ref(false)

    // Derived Data
    const products = computed(() => dataStore.electronics)
    const minPrice = computed(() => Math.min(...products.value.map(p => p.price)) - 1 || 0)
    const maxPrice = computed(() => Math.max(...products.value.map(p => p.price)) + 1 || 1000)
    
    // Initialize price filter to min on mount
    onMounted(() => {
      priceFilter.value = minPrice.value
    })

    const filteredProducts = computed(() => {
      let res = [...products.value]

      // Search
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        res = res.filter(p => p.name.toLowerCase().includes(q) || p.brand.toLowerCase().includes(q))
      }

      // Brand Filter (Mock logic: Filter "Apple" or "Samsung" as 'Premium')
      if (brandFilter.value) {
        res = res.filter(p => ['Apple', 'Samsung', 'Sony'].includes(p.brand))
      }

      // Price Filter (Items ABOVE the slider value)
      // Note: Typically logic is "Below", but FSM says drag to RIGHT to filter? 
      // Actually FSM doesn't specify logic, just action. 
      // "Slider at 0 shows all" -> so condition is price > slider.value.
      // If slider is max, show none? Or slider is Min Price?
      // Let's implement: Show items with Price >= SliderValue.
      res = res.filter(p => p.price >= priceFilter.value)

      // Sort
      if (currentSort.value) {
        switch (currentSort.value) {
          case 'price_low_high':
            res.sort((a, b) => a.price - b.price)
            break
          case 'price_high_low':
            res.sort((a, b) => b.price - a.price)
            break
          case 'top_rated':
            res.sort((a, b) => b.rating - a.rating)
            break
          case 'bestseller':
             // Mock sort by ID
            res.sort((a, b) => a.id.localeCompare(b.id))
            break
        }
      }

      return res
    })

    const currentSortLabel = computed(() => {
      const map = {
        'bestseller': 'Best Sellers',
        'price_low_high': 'Price: Low to High',
        'price_high_low': 'Price: High to Low',
        'top_rated': 'Top Rated'
      }
      return map[currentSort.value] || 'Relevance'
    })

    // Determine CSS class for FSM selector targeting
    const getProductClass = (product) => {
      // Logic to distinguish 'filtered', 'matched', 'visible'
      // FSM has 3 actions opening products with different selectors:
      // 1. ACT_ELECTRONICS_OPEN_FILTERED_PRODUCT -> .product-card-filtered (Precond: filters_applied)
      // 2. ACT_ELECTRONICS_OPEN_MATCHED_PRODUCT -> .product-card-matched (Precond: matched_id, has_searched)
      // 3. ACT_ELECTRONICS_OPEN_ANY_PRODUCT -> .product-card-visible (Precond: viewport_anchor)
      
      // We apply ALL classes that apply to the current state.
      const classes = []
      
      // If filters are active (store state is set by actions)
      // Note: We check local state or store state? Actions update Store.
      // But for visual feedback, we rely on local computed. 
      // To strictly support FSM selectors, we should assume if we ARE filtered, add the class.
      const isFiltered = brandFilter.value || priceFilter.value > minPrice.value || currentSort.value
      if (isFiltered) classes.push('product-card-filtered')

      // If searched
      if (searchQuery.value) {
         // If this product matches the query, it's a "matched" product
         classes.push('product-card-matched')
      }

      // Default visible class
      classes.push('product-card-visible')

      return classes.join(' ')
    }

    // Handlers
    const handleSearch = () => {
      // FSM: ACT_ELECTRONICS_SEARCH_PRODUCTS
      // Logic: Update store + local state
      // Effect in FSM sets 'matched_product_id' to {ITEM_ANY} ?? No, 
      // FSM says parameter is item_id={ITEM_ANY} but user TYPES text. 
      // Wait, look at FSM ACT_ELECTRONICS_SEARCH_PRODUCTS:
      // parameters: item_id: {ITEM_ANY} ?? This seems like a placeholder error in FSM or intended for 'selecting' a result?
      // Ah, FSM effect: set matched_product_id = {item_id}.
      // AND set has_searched = true.
      // But typically search action takes a string. 
      // FSM gui_procedure: type_text -> text: "{search_query}".
      // So the parameter really driving this is the typed text.
      // But the Action Parameter definition has 'item_id'. 
      // The FSM assumes we might "select" a result from an autocomplete list?
      // OR it assumes the search action identifies a "target" item ID to become the "matched" one.
      // Let's assume the search just updates the query, and the "matched_product_id" is set to the FIRST result found.
      
      store.electronics_category_has_searched = true
      // We'll set matched_product_id to the first result's ID if exists
      if (filteredProducts.value.length > 0) {
        store.matched_product_id = filteredProducts.value[0].id
      }
    }

    const handleFilterBrand = () => {
      // FSM: ACT_ELECTRONICS_FILTER_BRANDS_CHECKBOX
      store.electronics_category_filters_applied = true
    }

    const handleFilterPrice = () => {
      // FSM: ACT_ELECTRONICS_FILTER_PRICE_SLIDER
      store.electronics_category_filters_applied = true
    }

    const handleSort = (sortType) => {
      // FSM: ACT_ELECTRONICS_FILTER_SORT
      currentSort.value = sortType
      showSort.value = false
      store.electronics_category_filters_applied = true
    }

    const handleProductClick = async (product) => {
      // Determine which action this maps to based on state.
      // FSM has 3 actions to go to product detail.
      // Ideally we trigger the one that matches current state conditions.
      // But practically, they all go to PRODUCT_DETAIL and set selected_product_id.
      
      store.selected_product_id = product.id
      
      // Clear flags as per effects
      if (store.electronics_category_filters_applied) {
        store.electronics_category_filters_applied = null // clear
      }
      if (store.electronics_category_has_searched) {
        store.electronics_category_has_searched = null // clear
      }
      if (store.electronics_category_viewport_anchor_id) {
         store.electronics_category_viewport_anchor_id = null // clear
      }
      
      store.currentPageId = 'PRODUCT_DETAIL'
      await router.push({ name: 'PRODUCT_DETAIL', params: { id: product.id } })
    }
    
    // For Scroll Action (ACT_ELECTRONICS_SCROLL_PRODUCT_INTO_VIEW)
    // We can simulate this by detecting scroll on #product-list, but typically not needed for basic nav.
    // However, clicking #product-list to 'drag' is the FSM op.
    // We can add a click handler on the container to set viewport_anchor if not clicking a product?
    // Or just assume product click covers it.
    // FSM: Action is 'click' #product-list then 'drag'. 
    // And effect sets viewport_anchor_id.
    // Then NEXT action is OPEN_ANY_PRODUCT.
    // We'll leave this implicit or bind a click to the container if needed, but product click logic above covers the transition.

    const handleBackToDepartments = async () => {
      store.currentPageId = 'DEPARTMENTS'
      await router.push({ name: 'DEPARTMENTS' })
    }

    return {
      searchQuery,
      brandFilter,
      priceFilter,
      minPrice,
      maxPrice,
      filteredProducts,
      showSort,
      currentSortLabel,
      handleSearch,
      handleFilterBrand,
      handleFilterPrice,
      handleSort,
      handleProductClick,
      handleBackToDepartments,
      getProductClass
    }
  }
}
</script>