<template>
  <div class="grocery-page min-h-screen bg-white flex flex-col">
    <!-- Grocery Header -->
    <header class="bg-[#2A8703] text-white p-4 sticky top-0 z-30 shadow-md">
      <div class="max-w-7xl mx-auto flex items-center gap-4">
        <div 
          id="grocery-breadcrumb-departments" 
          @click="handleBackToDepartments"
          class="cursor-pointer p-2 hover:bg-white/10 rounded-full transition-colors flex items-center gap-1"
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
          <span class="text-sm font-medium">Departments</span>
        </div>
        
        <h1 class="text-lg font-bold flex-1 flex items-center gap-2">
          <span class="text-2xl">🥦</span> Grocery
        </h1>

        <!-- Search Bar -->
        <div class="relative w-full max-w-md hidden sm:block">
           <input 
             id="grocery-category-search-input"
             type="text" 
             v-model="searchQuery"
             @keydown.enter="handleSearch"
             placeholder="Search groceries" 
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
    </header>

    <div class="flex-1 max-w-7xl mx-auto w-full p-4 flex flex-col md:flex-row gap-6">
      
      <!-- Filters Sidebar -->
      <aside class="w-full md:w-64 flex-shrink-0 space-y-6">
        <div class="bg-gray-50 p-4 rounded-xl border border-gray-100">
          <div class="flex items-center justify-between mb-4">
            <h2 class="font-bold text-lg text-gray-900">Filters</h2>
          </div>
          
          <!-- Organic Filter -->
          <div class="mb-6">
            <h3 class="font-semibold mb-2 text-sm text-gray-700">Preference</h3>
            <label class="flex items-center gap-2 cursor-pointer p-2 hover:bg-white rounded transition-colors">
              <input 
                id="filter-organic-checkbox"
                type="checkbox" 
                v-model="organicFilter"
                @change="handleFilterOrganic"
                class="rounded text-[#2A8703] focus:ring-[#2A8703] w-5 h-5"
              />
              <span class="text-sm font-medium">Organic Only</span>
            </label>
          </div>

          <!-- Price Filter -->
          <div class="mb-6">
            <h3 class="font-semibold mb-2 text-sm text-gray-700">Price</h3>
            <div class="px-2">
              <input 
                id="grocery-price-slider"
                type="range" 
                v-model.number="priceFilter"
                :min="minPrice"
                :max="maxPrice"
                step="1"
                @input="handleFilterPrice"
                class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#2A8703]"
              />
              <div class="flex justify-between text-xs text-gray-500 mt-2">
                <span>${{ minPrice }}</span>
                <span class="font-bold text-gray-900">Under ${{ priceFilter }}</span>
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
          <div class="text-sm text-gray-600">{{ filteredProducts.length }} items</div>
          
          <div class="relative z-20">
             <button 
               id="grocery-sort-dropdown" 
               @click="showSort = !showSort"
               class="flex items-center gap-2 bg-white px-4 py-2 rounded-full border border-gray-200 shadow-sm text-sm font-medium hover:bg-gray-50"
             >
               Sort: <span class="text-[#2A8703]">{{ currentSortLabel }}</span>
               <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" /></svg>
             </button>
             
             <div v-if="showSort" class="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-xl py-1 border border-gray-100">
               <div 
                 id="grocery-sort-option-popular" 
                 @click="handleSort('popular')"
                 class="px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer"
               >
                 Popular
               </div>
               <div 
                 id="grocery-sort-option-price-low-high" 
                 @click="handleSort('price_low_high')"
                 class="px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer"
               >
                 Price: Low to High
               </div>
               <div 
                 id="grocery-sort-option-price-high-low" 
                 @click="handleSort('price_high_low')"
                 class="px-4 py-2 text-sm hover:bg-gray-100 cursor-pointer"
               >
                 Price: High to Low
               </div>
             </div>
          </div>
        </div>

        <!-- Product Grid -->
        <div 
          id="grocery-product-list"
          class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4"
        >
          <div 
            v-for="product in filteredProducts" 
            :key="product.id"
            :class="[
              'bg-white rounded-xl shadow-sm hover:shadow-lg transition-all p-4 flex flex-col cursor-pointer group relative border border-transparent hover:border-[#2A8703]',
              getProductClass(product)
            ]"
            :data-id="product.id"
            @click="handleProductClick(product)"
          >
            <!-- Image -->
            <div class="aspect-square mb-3 relative overflow-hidden rounded-lg bg-gray-50">
              <img :src="product.image" :alt="product.name" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" />
              <!-- Organic Badge -->
              <div v-if="product.type === 'organic'" class="absolute top-2 left-2 bg-[#2A8703] text-white text-[10px] font-bold px-2 py-1 rounded-full uppercase tracking-wider">
                 Organic
              </div>
            </div>
            
            <!-- Details -->
            <div class="flex-1 flex flex-col">
               <div class="font-bold text-lg text-gray-900 mb-1">${{ product.price.toFixed(2) }} <span class="text-xs font-normal text-gray-500">/ {{ product.unit }}</span></div>
               <h3 class="font-medium text-gray-700 text-sm line-clamp-2 hover:text-[#2A8703] transition-colors mb-2">{{ product.name }}</h3>
               
               <button class="mt-auto w-full py-1.5 rounded-full bg-[#2A8703] text-white text-sm font-bold opacity-0 group-hover:opacity-100 transition-opacity transform translate-y-2 group-hover:translate-y-0">
                 Add
               </button>
            </div>
          </div>
        </div>

        <!-- Empty State -->
        <div v-if="filteredProducts.length === 0" class="text-center py-20 bg-white rounded-xl shadow-sm mt-4 border border-dashed border-gray-300">
           <div class="text-6xl mb-4">🥗</div>
           <h3 class="text-xl font-bold text-gray-900">No groceries found</h3>
           <p class="text-gray-500 mt-2">Try checking your spelling or adjusting filters.</p>
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
  name: 'GROCERY_CATEGORY',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const organicFilter = ref(false)
    const priceFilter = ref(100) // Start at max
    const currentSort = ref(null)
    const showSort = ref(false)

    const products = computed(() => dataStore.groceries)
    const minPrice = computed(() => 0)
    const maxPrice = computed(() => Math.max(...products.value.map(p => p.price)) + 1 || 50)

    onMounted(() => {
      priceFilter.value = maxPrice.value
    })

    const filteredProducts = computed(() => {
      let res = [...products.value]

      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase()
        res = res.filter(p => p.name.toLowerCase().includes(q))
      }

      if (organicFilter.value) {
        res = res.filter(p => p.type === 'organic')
      }

      // Logic: Under price (ACT says drag LEFT to filter, typically implies reducing max price)
      res = res.filter(p => p.price <= priceFilter.value)

      if (currentSort.value) {
        switch (currentSort.value) {
          case 'price_low_high':
            res.sort((a, b) => a.price - b.price)
            break
          case 'price_high_low':
            res.sort((a, b) => b.price - a.price)
            break
          case 'popular':
            res.sort((a, b) => a.id.localeCompare(b.id))
            break
        }
      }

      return res
    })

    const currentSortLabel = computed(() => {
      const map = {
        'popular': 'Popular',
        'price_low_high': 'Price: Low to High',
        'price_high_low': 'Price: High to Low'
      }
      return map[currentSort.value] || 'Featured'
    })

    const getProductClass = (product) => {
      const classes = ['product-card-visible']
      
      const isFiltered = organicFilter.value || priceFilter.value < maxPrice.value || currentSort.value
      if (isFiltered) classes.push('product-card-filtered')

      if (searchQuery.value && store.grocery_category_has_searched) {
        classes.push('product-card-matched')
      }

      return classes.join(' ')
    }

    const handleSearch = () => {
      // FSM: ACT_GROCERY_SEARCH_PRODUCTS
      store.grocery_category_has_searched = true
      if (filteredProducts.value.length > 0) {
        store.matched_product_id = filteredProducts.value[0].id
      }
    }

    const handleFilterOrganic = () => {
      // FSM: ACT_GROCERY_FILTER_TYPE_CHECKBOX
      store.grocery_category_filters_applied = true
    }

    const handleFilterPrice = () => {
      // FSM: ACT_GROCERY_FILTER_PRICE_SLIDER
      store.grocery_category_filters_applied = true
    }

    const handleSort = (val) => {
      // FSM: ACT_GROCERY_FILTER_SORT
      currentSort.value = val
      showSort.value = false
      store.grocery_category_filters_applied = true
    }

    const handleProductClick = async (product) => {
      store.selected_product_id = product.id
      
      // Clear flags
      store.grocery_category_filters_applied = null
      store.grocery_category_has_searched = null
      store.grocery_category_viewport_anchor_id = null

      store.currentPageId = 'GROCERY_PRODUCT_DETAIL'
      await router.push({ name: 'GROCERY_PRODUCT_DETAIL', params: { id: product.id } })
    }

    const handleBackToDepartments = async () => {
      // FSM: ACT_GROCERY_BACK_TO_DEPARTMENTS
      store.currentPageId = 'DEPARTMENTS'
      await router.push({ name: 'DEPARTMENTS' })
    }

    return {
      searchQuery, organicFilter, priceFilter, minPrice, maxPrice, filteredProducts,
      showSort, currentSortLabel,
      handleSearch, handleFilterOrganic, handleFilterPrice, handleSort,
      handleProductClick, handleBackToDepartments,
      getProductClass
    }
  }
}
</script>