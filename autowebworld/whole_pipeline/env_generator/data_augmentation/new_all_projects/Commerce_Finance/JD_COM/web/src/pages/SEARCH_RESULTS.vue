<template>
  <div class="min-h-screen bg-[#F6F6F6]">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-4 flex items-center gap-4">
        <button id="back-home" @click="goHome" class="text-gray-500 hover:text-[#E1251B] flex items-center gap-1">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          Home
        </button>
        
        <!-- Search Bar -->
        <div class="flex-1 relative">
          <div class="flex items-center border-2 border-[#E1251B] bg-white rounded-full overflow-hidden max-w-2xl">
            <input 
              id="search-results-input" 
              type="text" 
              v-model="searchQuery" 
              @keydown.enter="performSearch"
              placeholder="Search again..." 
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
      <div class="bg-white rounded-lg shadow-sm p-4 mb-6 border border-gray-100">
        <div class="flex flex-wrap gap-6 items-center">
          <!-- Checkbox Filter -->
          <div class="flex items-center gap-2">
            <div 
              id="search-filter-self-operated" 
              @click="toggleSelfOperated"
              class="w-5 h-5 border-2 rounded cursor-pointer flex items-center justify-center transition-colors"
              :class="selfOperatedFilter ? 'bg-[#E1251B] border-[#E1251B]' : 'border-gray-300'"
            >
              <svg v-if="selfOperatedFilter" class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20"><path d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"></path></svg>
            </div>
            <label class="text-sm font-medium text-gray-700 cursor-pointer" @click="toggleSelfOperated">JD Self-Operated</label>
          </div>

          <!-- Sort Dropdown -->
          <div class="relative ml-auto" id="search-sort-dropdown">
            <button @click="toggleSort" class="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded hover:border-[#E1251B] hover:text-[#E1251B] transition-colors text-sm">
              {{ currentSortLabel }}
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div v-if="sortOpen" class="absolute top-full right-0 mt-1 w-48 bg-white shadow-lg rounded border border-gray-100 py-1 z-30">
              <div id="search-sort-option-comprehensive" @click="sort('comprehensive')" class="px-4 py-2 hover:bg-red-50 text-sm cursor-pointer">Comprehensive</div>
              <div id="search-sort-option-price-low" @click="sort('price_low_high')" class="px-4 py-2 hover:bg-red-50 text-sm cursor-pointer">Price: Low to High</div>
              <div id="search-sort-option-comment" @click="sort('comment')" class="px-4 py-2 hover:bg-red-50 text-sm cursor-pointer">Most Reviews</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Results List -->
      <div id="search-results-list" class="space-y-4">
        <div 
          v-for="item in filteredItems" 
          :key="item.id"
          :class="[
            'bg-white p-4 rounded-lg shadow-sm hover:shadow-md transition-shadow flex gap-4 cursor-pointer border border-transparent hover:border-[#E1251B]',
            getItemClass(item),
            `data-id-${item.id}`
          ]"
          @click="openItem(item)"
        >
          <div class="w-48 h-48 bg-gray-100 rounded overflow-hidden flex-shrink-0">
            <img :src="item.image" :alt="item.name" class="w-full h-full object-cover" />
          </div>
          <div class="flex-1 py-2">
            <div class="flex items-baseline gap-1 mb-2">
              <span class="text-[#E1251B] text-lg font-bold">$</span>
              <span class="text-[#E1251B] text-2xl font-bold">{{ item.price }}</span>
            </div>
            <h3 class="text-lg font-medium text-gray-900 mb-2 hover:text-[#E1251B] transition-colors">
              <span v-if="item.tags?.includes('Self-Operated')" class="bg-[#E1251B] text-white text-xs px-1 rounded mr-1 align-middle">JD</span>
              {{ item.name }}
            </h3>
            <div class="flex items-center gap-4 text-sm text-gray-500 mb-4">
              <span class="text-[#E1251B] font-bold">{{ item.sales }}+ Comments</span>
              <span>{{ item.rating }}% Positive</span>
            </div>
            <div class="flex items-center gap-2">
              <span class="text-xs border border-[#E1251B] text-[#E1251B] px-1 rounded">Free Shipping</span>
              <span v-if="item.tags?.includes('Flash Sale')" class="text-xs bg-[#E1251B] text-white px-1 rounded">Flash Sale</span>
            </div>
            <div class="mt-4 text-gray-400 text-sm flex items-center gap-1">
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"></path></svg>
              {{ item.brand }} Official Store
            </div>
          </div>
        </div>
      </div>

      <div v-if="filteredItems.length === 0" class="text-center py-20 text-gray-500">
        <div class="text-6xl mb-4">📦</div>
        <p class="text-xl">No results found.</p>
        <p class="text-sm mt-2">Try adjusting your filters or search query.</p>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'SEARCH_RESULTS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const searchQuery = ref('');
    const selfOperatedFilter = ref(false);
    const sortOpen = ref(false);
    const currentSort = ref('');
    
    const matchedId = computed(() => signatureStore.search_matched_item_id);
    const isFiltered = computed(() => signatureStore.search_list_filters_applied);
    const hasSearched = computed(() => signatureStore.search_list_has_searched);

    // Combine all products for search
    const allItems = computed(() => [...dataStore.electronics, ...dataStore.supermarket]);

    const filteredItems = computed(() => {
      let result = [...allItems.value];

      // Mock Search query logic if not already filtered by signature
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase();
        result = result.filter(i => i.name.toLowerCase().includes(q));
      }

      // Apply Filters
      if (selfOperatedFilter.value) {
        result = result.filter(i => i.tags?.includes('Self-Operated'));
      }

      // Apply Sort
      if (currentSort.value === 'price_low_high') {
        result.sort((a, b) => a.price - b.price);
      } else if (currentSort.value === 'comment') {
        result.sort((a, b) => b.sales - a.sales);
      } else {
        // Comprehensive default
      }

      return result;
    });

    const currentSortLabel = computed(() => {
      if (currentSort.value === 'price_low_high') return 'Price: Low to High';
      if (currentSort.value === 'comment') return 'Most Reviews';
      return 'Comprehensive';
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
      if (filteredItems.value.length > 0) {
        signatureStore.search_matched_item_id = filteredItems.value[0].id;
      }
      signatureStore.search_list_has_searched = true;
    };

    const toggleSelfOperated = () => {
      selfOperatedFilter.value = !selfOperatedFilter.value;
      signatureStore.search_list_filters_applied = true;
    };

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value;
    };

    const sort = (type) => {
      currentSort.value = type;
      sortOpen.value = false;
      signatureStore.search_list_filters_applied = true;
    };

    const openItem = async (item) => {
      signatureStore.search_selected_item_id = item.id;
      signatureStore.search_list_filters_applied = null;
      signatureStore.search_list_has_searched = null;
      signatureStore.currentPageId = 'PRODUCT_DETAIL';
      await router.push({ name: 'PRODUCT_DETAIL', params: { id: item.id } });
    };

    return {
      searchQuery,
      selfOperatedFilter,
      sortOpen,
      currentSortLabel,
      filteredItems,
      getItemClass,
      goHome,
      performSearch,
      toggleSelfOperated,
      toggleSort,
      sort,
      openItem
    };
  }
}
</script>