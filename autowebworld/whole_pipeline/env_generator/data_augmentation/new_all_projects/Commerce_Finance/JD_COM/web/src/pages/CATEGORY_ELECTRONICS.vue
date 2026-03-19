<template>
  <div class="min-h-screen bg-[#F6F6F6] pb-12">
    <!-- Permission Modal -->
    <PermissionModal />

    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-4 flex items-center gap-4">
        <button id="back-home" @click="goHome" class="text-gray-500 hover:text-[#E1251B] flex items-center gap-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          Home
        </button>
        
        <!-- Search Bar -->
        <div class="flex-1 relative">
          <div class="flex items-center border-2 border-[#E1251B] rounded-full overflow-hidden max-w-xl mx-auto">
            <input 
              id="electronics-search" 
              type="text" 
              v-model="searchQuery" 
              @keydown.enter="performSearch"
              placeholder="Search electronics..." 
              class="w-full px-4 py-2 outline-none"
            />
            <button @click="performSearch" class="bg-[#E1251B] text-white px-6 py-2 font-bold">
              Search
            </button>
          </div>
        </div>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6">
      <!-- Filters -->
      <div class="bg-white rounded-lg shadow-sm p-4 mb-6">
        <div class="flex flex-wrap gap-6 items-center">
          <!-- Checkbox Filter -->
          <div class="flex items-center gap-2">
            <div 
              id="filter-brand-jd" 
              @click="toggleJDFilter"
              class="w-5 h-5 border-2 rounded cursor-pointer flex items-center justify-center transition-colors"
              :class="jdFilterActive ? 'bg-[#E1251B] border-[#E1251B]' : 'border-gray-300'"
            >
              <svg v-if="jdFilterActive" class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20"><path d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"></path></svg>
            </div>
            <label class="text-sm font-medium text-gray-700 cursor-pointer" @click="toggleJDFilter">JD Self-Operated</label>
          </div>

          <!-- Price Slider -->
          <div class="flex items-center gap-3 flex-1 max-w-xs">
            <span class="text-sm font-medium text-gray-700">Price:</span>
            <input 
              id="price-slider" 
              type="range" 
              min="0" 
              max="3000" 
              step="10"
              v-model.number="priceFilter"
              @change="applyFilters"
              class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#E1251B]"
            />
            <span class="text-sm text-gray-500 min-w-[4rem]">{{ priceFilter > 0 ? '> $' + priceFilter : 'All' }}</span>
          </div>

          <!-- Sort Dropdown -->
          <div class="relative" id="sort-dropdown">
            <button @click="toggleSort" class="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded hover:border-[#E1251B] hover:text-[#E1251B] transition-colors">
              {{ currentSortLabel }}
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div v-if="sortOpen" class="absolute top-full right-0 mt-1 w-40 bg-white shadow-lg rounded border border-gray-100 py-1 z-30">
              <div id="sort-option-sales" @click="sort('sales')" class="px-4 py-2 hover:bg-red-50 text-sm cursor-pointer">Sales Volume</div>
              <div id="sort-option-price-low" @click="sort('price_low_high')" class="px-4 py-2 hover:bg-red-50 text-sm cursor-pointer">Price: Low to High</div>
              <div id="sort-option-price-high" @click="sort('price_high_low')" class="px-4 py-2 hover:bg-red-50 text-sm cursor-pointer">Price: High to Low</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Product List -->
      <div id="electronics-list" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        <div 
          v-for="item in filteredItems" 
          :key="item.id"
          :class="[
            'bg-white rounded-xl shadow-sm hover:shadow-xl transition-shadow overflow-hidden group cursor-pointer',
            getItemClass(item),
            `data-id-${item.id}`
          ]"
          @click="openItem(item)"
        >
          <div class="relative aspect-square bg-gray-100 overflow-hidden">
            <img :src="item.image" :alt="item.name" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
            <div v-if="item.tags.includes('Flash Sale')" class="absolute top-2 left-2 bg-[#E1251B] text-white text-xs font-bold px-2 py-1 rounded">
              FLASH SALE
            </div>
          </div>
          <div class="p-4">
            <div class="flex items-baseline gap-1 mb-1">
              <span class="text-[#E1251B] text-lg font-bold">$</span>
              <span class="text-[#E1251B] text-2xl font-bold">{{ item.price }}</span>
            </div>
            <h3 class="font-medium text-gray-900 line-clamp-2 mb-2 h-12 group-hover:text-[#E1251B] transition-colors">
              <span v-if="item.tags.includes('Self-Operated')" class="bg-[#E1251B] text-white text-xs px-1 rounded mr-1 align-middle">JD</span>
              {{ item.name }}
            </h3>
            <div class="flex items-center text-xs text-gray-500 mb-2">
              <span>{{ item.rating }} ★</span>
              <span class="mx-1">|</span>
              <span>{{ item.sales }}+ sold</span>
            </div>
            <div class="text-xs text-gray-400">{{ item.brand }}</div>
          </div>
        </div>
      </div>
      
      <div v-if="filteredItems.length === 0" class="text-center py-20 text-gray-500">
        <div class="text-6xl mb-4">🔍</div>
        <p>No products found matching your criteria.</p>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';
import PermissionModal from '../components/PermissionModal.vue';

export default {
  name: 'CATEGORY_ELECTRONICS',
  components: {
    PermissionModal
  },
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const searchQuery = ref('');
    const jdFilterActive = ref(false);
    const priceFilter = ref(0);
    const sortOpen = ref(false);
    const currentSort = ref('');
    
    // State from signature
    const matchedId = computed(() => signatureStore.electronics_matched_item_id);
    const isFiltered = computed(() => signatureStore.electronics_list_filters_applied);
    const hasSearched = computed(() => signatureStore.electronics_list_has_searched);

    const items = computed(() => dataStore.electronics);

    const filteredItems = computed(() => {
      let result = [...items.value];

      // Apply Search
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase();
        result = result.filter(i => i.name.toLowerCase().includes(q) || i.brand.toLowerCase().includes(q));
      }

      // Apply Filters
      if (jdFilterActive.value) {
        result = result.filter(i => i.tags.includes('Self-Operated'));
      }
      if (priceFilter.value > 0) {
        result = result.filter(i => i.price > priceFilter.value);
      }

      // Apply Sort
      if (currentSort.value === 'sales') {
        result.sort((a, b) => b.sales - a.sales);
      } else if (currentSort.value === 'price_low_high') {
        result.sort((a, b) => a.price - b.price);
      } else if (currentSort.value === 'price_high_low') {
        result.sort((a, b) => b.price - a.price);
      }

      return result;
    });

    const currentSortLabel = computed(() => {
      if (currentSort.value === 'sales') return 'Sales Volume';
      if (currentSort.value === 'price_low_high') return 'Price: Low to High';
      if (currentSort.value === 'price_high_low') return 'Price: High to Low';
      return 'Sort By';
    });

    const getItemClass = (item) => {
      if (matchedId.value && item.id === matchedId.value) return 'row-matched';
      if (isFiltered.value) return 'row-filtered';
      return 'row-visible';
    };

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    const performSearch = () => {
      // FSM Effect: set electronics_list_has_searched = true, matched_item_id = first result
      if (filteredItems.value.length > 0) {
        signatureStore.electronics_matched_item_id = filteredItems.value[0].id;
      } else {
        signatureStore.electronics_matched_item_id = '';
      }
      signatureStore.electronics_list_has_searched = true;
    };

    const toggleJDFilter = () => {
      jdFilterActive.value = !jdFilterActive.value;
      signatureStore.electronics_list_filters_applied = true;
    };

    const applyFilters = () => {
      signatureStore.electronics_list_filters_applied = true;
    };

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value;
    };

    const sort = (type) => {
      currentSort.value = type;
      sortOpen.value = false;
      signatureStore.electronics_list_filters_applied = true;
    };

    const openItem = async (item) => {
      signatureStore.electronics_selected_item_id = item.id;
      signatureStore.electronics_list_filters_applied = null; // clear effect
      signatureStore.electronics_list_has_searched = null; // clear effect
      signatureStore.currentPageId = 'PRODUCT_DETAIL';
      await router.push({ name: 'PRODUCT_DETAIL', params: { id: item.id } });
    };

    return {
      searchQuery,
      jdFilterActive,
      priceFilter,
      sortOpen,
      currentSortLabel,
      filteredItems,
      getItemClass,
      goHome,
      performSearch,
      toggleJDFilter,
      applyFilters,
      toggleSort,
      sort,
      openItem
    };
  }
}
</script>