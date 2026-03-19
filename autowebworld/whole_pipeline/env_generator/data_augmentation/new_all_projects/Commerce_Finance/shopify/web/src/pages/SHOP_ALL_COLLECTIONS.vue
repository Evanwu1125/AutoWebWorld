<template>
  <div class="min-h-screen bg-gray-50 text-gray-900 font-sans">
    
    <!-- Location Permission Modal -->
    <div v-if="showLocationPermission" class="fixed inset-0 z-[10000] flex items-center justify-center bg-black/50 backdrop-blur-sm">
      <div class="bg-white rounded-xl shadow-2xl p-8 max-w-md w-full mx-4 animate-scale-in">
        <div class="text-center mb-6">
           <div class="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-4">
               <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
           </div>
          <h2 class="text-2xl font-bold mb-2">Use Location?</h2>
          <p class="text-gray-600 text-sm leading-relaxed">
            We need your location to show products available in your area and calculate shipping estimates.
          </p>
        </div>
        <button 
          id="permission-location-allow" 
          @click="grantLocationPermission" 
          class="w-full bg-[#008060] hover:bg-[#004C3F] text-white font-semibold py-3 px-6 rounded-lg transition-colors duration-200 mb-3"
        >
          Allow Location Access
        </button>
      </div>
    </div>

    <!-- Header -->
    <header class="bg-white border-b border-gray-200 sticky top-0 z-30">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div id="logo-home" @click="goHome" class="text-xl font-bold text-[#008060] cursor-pointer tracking-tight flex items-center gap-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
            BACK TO HOME
        </div>
        <div class="flex items-center space-x-6">
             <!-- Search -->
             <div class="relative hidden md:block">
                 <input 
                    id="collection-search-input"
                    type="text" 
                    v-model="searchQuery" 
                    @keypress.enter="performSearch"
                    placeholder="Search products..." 
                    class="pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-[#008060] focus:border-transparent outline-none w-64 transition-all"
                 />
                 <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400 absolute left-3 top-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                  </svg>
             </div>
             
             <!-- Cart Icon -->
             <div id="cart-icon" @click="goToCart" class="relative cursor-pointer text-gray-600 hover:text-[#008060] transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 11V7a4 4 0 00-8 0v4M5 9h14l1 12H4L5 9z" />
                </svg>
                <span v-if="cartCount > 0" class="absolute -top-2 -right-2 bg-red-500 text-white text-xs font-bold w-5 h-5 rounded-full flex items-center justify-center shadow-sm">
                    {{ cartCount }}
                </span>
             </div>
        </div>
      </div>
    </header>
    
    <!-- Main Content -->
    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <h1 class="text-3xl font-extrabold text-gray-900 mb-8">Shop All Collections</h1>
      
      <div class="flex flex-col lg:flex-row gap-8">
        <!-- Sidebar Filters -->
        <aside class="w-full lg:w-64 flex-shrink-0 space-y-8">
            <!-- Sort -->
            <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 class="text-lg font-semibold mb-4">Sort By</h3>
                <div class="relative" id="sort-dropdown">
                    <!-- Custom Dropdown Trigger -->
                    <div
                        @click="toggleSortDropdown"
                        class="w-full border border-gray-300 rounded-lg shadow-sm py-2 px-3 bg-white cursor-pointer flex justify-between items-center hover:border-[#008060] transition-colors"
                    >
                        <span :class="selectedSort ? 'text-gray-900' : 'text-gray-500'">
                            {{ selectedSortLabel || 'Select option' }}
                        </span>
                        <svg class="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                        </svg>
                    </div>

                    <!-- Custom Dropdown Options -->
                    <div
                        v-if="sortDropdownOpen"
                        class="absolute z-10 w-full mt-2 bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden"
                    >
                        <div
                            id="sort-option-featured"
                            @click="selectSort('featured', 'Featured')"
                            :class="[
                                'px-3 py-2 cursor-pointer transition-colors',
                                selectedSort === 'featured'
                                    ? 'bg-[#008060] text-white'
                                    : 'hover:bg-gray-100 text-gray-700'
                            ]"
                        >
                            Featured
                        </div>
                        <div
                            id="sort-option-best-selling-desc"
                            @click="selectSort('best_selling', 'Best Selling')"
                            :class="[
                                'px-3 py-2 cursor-pointer transition-colors',
                                selectedSort === 'best_selling'
                                    ? 'bg-[#008060] text-white'
                                    : 'hover:bg-gray-100 text-gray-700'
                            ]"
                        >
                            Best Selling
                        </div>
                        <div
                            id="sort-option-price-asc"
                            @click="selectSort('price_asc', 'Price: Low to High')"
                            :class="[
                                'px-3 py-2 cursor-pointer transition-colors',
                                selectedSort === 'price_asc'
                                    ? 'bg-[#008060] text-white'
                                    : 'hover:bg-gray-100 text-gray-700'
                            ]"
                        >
                            Price: Low to High
                        </div>
                    </div>
                </div>
            </div>

            <!-- Price Filter -->
             <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                <h3 class="text-lg font-semibold mb-4">Price Range</h3>
                <div class="space-y-4">
                    <input 
                        id="filter-price-slider"
                        type="range" 
                        v-model.number="priceFilter" 
                        :min="0" 
                        :max="500" 
                        step="1"
                        @change="handleSliderChange"
                        class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#008060]"
                    />
                     <div class="flex justify-between text-sm text-gray-600 font-medium">
                        <span>$0</span>
                        <span>> ${{ priceFilter }}</span>
                        <span>$500+</span>
                    </div>
                </div>
            </div>

            <!-- Tags Filter -->
            <div class="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
                 <h3 class="text-lg font-semibold mb-4">Tags</h3>
                 <div class="space-y-3">
                    <label class="flex items-center space-x-3 cursor-pointer group">
                        <input 
                            id="filter-tag-sale-checkbox"
                            type="checkbox" 
                            v-model="saleFilter" 
                            @change="handleCheckboxChange"
                            class="form-checkbox h-5 w-5 text-[#008060] rounded focus:ring-[#008060] border-gray-300 transition duration-150 ease-in-out"
                        />
                        <span class="text-gray-700 group-hover:text-[#008060] transition-colors">On Sale</span>
                    </label>
                 </div>
            </div>
        </aside>

        <!-- Product Grid -->
        <div class="flex-1" id="product-grid">
            <div v-if="filteredProducts.length === 0" class="text-center py-20 bg-white rounded-xl shadow-sm border border-gray-100">
                <div class="text-6xl mb-4">🔍</div>
                <h3 class="text-xl font-medium text-gray-900">No products found</h3>
                <p class="text-gray-500 mt-2">Try adjusting your filters or search query.</p>
            </div>

            <div v-else class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                <div 
                    v-for="product in filteredProducts" 
                    :key="product.id" 
                    :class="[
                        'group bg-white rounded-xl shadow-sm hover:shadow-xl transition-all duration-300 overflow-hidden border border-gray-100 cursor-pointer',
                        `data-id-${product.id}`,
                        isFiltered ? 'product-card-filtered' : '',
                        isSearched ? 'product-card-matched' : '',
                        !isFiltered && !isSearched ? 'product-card-visible' : ''
                    ]"
                    @click="openProduct(product.id)"
                >
                    <div class="relative aspect-square overflow-hidden bg-gray-200">
                        <img :src="product.image" :alt="product.title" class="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700" />
                        <div v-if="product.compare_at_price" class="absolute top-3 left-3 bg-red-500 text-white text-xs font-bold px-2 py-1 rounded">SALE</div>
                    </div>
                    <div class="p-5">
                        <div class="text-xs text-gray-500 mb-1 uppercase tracking-wider font-semibold">{{ product.vendor }}</div>
                        <h3 class="font-bold text-gray-900 text-lg mb-2 group-hover:text-[#008060] transition-colors line-clamp-1">{{ product.title }}</h3>
                        <div class="flex items-baseline space-x-2">
                             <span class="text-lg font-bold text-gray-900">${{ product.price.toFixed(2) }}</span>
                             <span v-if="product.compare_at_price" class="text-sm text-gray-400 line-through">${{ product.compare_at_price.toFixed(2) }}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed, ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'SHOP_ALL_COLLECTIONS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const showLocationPermission = computed(() => signatureStore.location_permission_granted === null)
    const cartCount = computed(() => signatureStore.cart_items ? signatureStore.cart_items.reduce((sum, item) => sum + item.quantity, 0) : 0)

    const searchQuery = ref('')
    const priceFilter = ref(0)
    const saleFilter = ref(false)
    const selectedSort = ref('')
    const selectedSortLabel = ref('')
    const sortDropdownOpen = ref(false)

    // Derived state flags for FSM selectors
    const isFiltered = computed(() => signatureStore.collections_filters_applied === true)
    const isSearched = computed(() => signatureStore.collections_list_has_searched === true)

    // Load products
    onMounted(() => {
        // Reset transient state when entering
        signatureStore.resetTransientState()

        // Initialize filters from store if needed, or defaults
        priceFilter.value = 0
        saleFilter.value = false
        selectedSort.value = ''
        selectedSortLabel.value = ''
    })

    const filteredProducts = computed(() => {
        let items = [...dataStore.products]

        // Search
        if (searchQuery.value) {
            const q = searchQuery.value.toLowerCase()
            items = items.filter(p => p.title.toLowerCase().includes(q))
        }

        // Filters
        if (saleFilter.value) {
            items = items.filter(p => p.tags.includes('sale'))
        }
        
        if (priceFilter.value > 0) {
            items = items.filter(p => p.price > priceFilter.value)
        }

        // Sort
        if (selectedSort.value === 'price_asc') {
            items.sort((a, b) => a.price - b.price)
        } else if (selectedSort.value === 'best_selling') {
            // Sort by sales count (high to low)
            items.sort((a, b) => b.salesCount - a.salesCount)
        }
        // 'featured' is default/no sort

        return items
    })

    const grantLocationPermission = () => {
        signatureStore.location_permission_granted = true
    }

    const goHome = async () => {
        signatureStore.currentPageId = 'HOME'
        await router.push({ name: 'HOME' })
    }

    const goToCart = async () => {
        signatureStore.currentPageId = 'CART'
        await router.push({ name: 'CART' })
    }

    const performSearch = () => {
        if (searchQuery.value.trim() !== '') {
            signatureStore.collections_list_has_searched = true
            signatureStore.collections_matched_product_id = 'MATCHED_ANY' // FSM logic placeholder
        }
    }

    const handleCheckboxChange = () => {
        signatureStore.collections_filters_applied = true
    }

    const handleSliderChange = () => {
        signatureStore.collections_filters_applied = true
    }

    const toggleSortDropdown = () => {
        sortDropdownOpen.value = !sortDropdownOpen.value
    }

    const selectSort = (value, label) => {
        selectedSort.value = value
        selectedSortLabel.value = label
        sortDropdownOpen.value = false
        handleSort()
    }

    const handleSort = () => {
        signatureStore.collections_filters_applied = true
    }

    const openProduct = async (productId) => {
        if (isFiltered.value) {
            signatureStore.collections_selected_product_id = productId
            signatureStore.collections_filters_applied = null // Clear after use as per FSM
        } else if (isSearched.value) {
             signatureStore.collections_selected_product_id = productId
             signatureStore.collections_list_has_searched = null // Clear after use
        } else {
            // Scroll logic handled by FSM state check, here we just set selected
            signatureStore.collections_viewport_anchor_id = productId // Simulating "scrolled to"
            signatureStore.collections_selected_product_id = productId
            signatureStore.collections_viewport_anchor_id = null
        }
        
        signatureStore.currentPageId = 'PRODUCT_DETAIL'
        await router.push({ name: 'PRODUCT_DETAIL', params: { id: productId } })
    }

    return {
        showLocationPermission,
        grantLocationPermission,
        goHome,
        goToCart,
        searchQuery,
        performSearch,
        cartCount,
        priceFilter,
        saleFilter,
        selectedSort,
        selectedSortLabel,
        sortDropdownOpen,
        toggleSortDropdown,
        selectSort,
        handleCheckboxChange,
        handleSliderChange,
        handleSort,
        filteredProducts,
        openProduct,
        isFiltered,
        isSearched
    }
  }
}
</script>