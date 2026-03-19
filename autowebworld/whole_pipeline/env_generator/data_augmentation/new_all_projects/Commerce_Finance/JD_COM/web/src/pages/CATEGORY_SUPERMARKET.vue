<template>
  <div class="min-h-screen bg-[#F6F6F6] pb-12">
    <!-- Header -->
    <header class="bg-[#009933] shadow-sm sticky top-0 z-20">
      <div class="container mx-auto px-4 py-4 flex items-center gap-4 text-white">
        <button id="back-home" @click="goHome" class="hover:text-green-100 flex items-center gap-1 font-medium">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          Back to Home
        </button>
        <h1 class="text-2xl font-bold tracking-tight">JD Supermarket</h1>
      </div>
    </header>

    <main class="container mx-auto px-4 py-6">
      <!-- Filters -->
      <div class="bg-white rounded-lg shadow-sm p-4 mb-6">
        <div class="flex flex-wrap gap-6 items-center">
          <!-- Checkbox Filter -->
          <div class="flex items-center gap-2">
            <div 
              id="filter-fresh-only" 
              @click="toggleFreshFilter"
              class="w-5 h-5 border-2 rounded cursor-pointer flex items-center justify-center transition-colors"
              :class="freshFilterActive ? 'bg-[#009933] border-[#009933]' : 'border-gray-300'"
            >
              <svg v-if="freshFilterActive" class="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20"><path d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"></path></svg>
            </div>
            <label class="text-sm font-medium text-gray-700 cursor-pointer" @click="toggleFreshFilter">Fresh Only</label>
          </div>

          <!-- Price Slider -->
          <div class="flex items-center gap-3 flex-1 max-w-xs">
            <span class="text-sm font-medium text-gray-700">Max Price:</span>
            <input 
              id="supermarket-price-slider" 
              type="range" 
              min="0" 
              max="50" 
              step="1"
              v-model.number="priceFilter"
              @change="applyFilters"
              class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#009933]"
            />
            <span class="text-sm text-gray-500 min-w-[4rem]">{{ priceFilter > 0 ? '> $' + priceFilter : 'All' }}</span>
          </div>

          <!-- Sort Dropdown -->
          <div class="relative" id="supermarket-sort-dropdown">
            <button @click="toggleSort" class="flex items-center gap-2 px-4 py-2 border border-gray-300 rounded hover:border-[#009933] hover:text-[#009933] transition-colors">
              {{ currentSortLabel }}
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
            </button>
            <div v-if="sortOpen" class="absolute top-full right-0 mt-1 w-40 bg-white shadow-lg rounded border border-gray-100 py-1 z-30">
              <div id="supermarket-sort-option-time" @click="sort('time')" class="px-4 py-2 hover:bg-green-50 text-sm cursor-pointer">Newest Arrival</div>
              <div id="supermarket-sort-option-price-low-inc" @click="sort('price_low_high')" class="px-4 py-2 hover:bg-green-50 text-sm cursor-pointer">Price: Low to High</div>
              <div id="supermarket-sort-option-rating-desc" @click="sort('rating')" class="px-4 py-2 hover:bg-green-50 text-sm cursor-pointer">Top Rated</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Product List -->
      <div id="supermarket-list" class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
        <div 
          v-for="item in filteredItems" 
          :key="item.id"
          :class="[
            'bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow overflow-hidden group cursor-pointer border border-transparent hover:border-[#009933]',
            getItemClass(item),
            `data-id-${item.id}`
          ]"
          @click="openItem(item)"
        >
          <div class="relative aspect-square bg-gray-100 overflow-hidden">
            <img :src="item.image" :alt="item.name" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" />
            <div v-if="item.fresh" class="absolute bottom-0 left-0 right-0 bg-green-500/80 text-white text-xs text-center py-1">
              FRESH GUARANTEE
            </div>
          </div>
          <div class="p-3">
            <div class="flex items-baseline gap-1 mb-1">
              <span class="text-[#E1251B] text-sm font-bold">$</span>
              <span class="text-[#E1251B] text-lg font-bold">{{ item.price }}</span>
            </div>
            <h3 class="font-medium text-gray-800 text-sm line-clamp-2 mb-2 h-10 group-hover:text-[#009933] transition-colors">
              {{ item.name }}
            </h3>
            <div class="flex items-center text-xs text-gray-500">
              <span class="bg-gray-100 px-1 rounded">{{ item.category }}</span>
              <span class="ml-auto">{{ item.sales }}+ sold</span>
            </div>
          </div>
        </div>
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
  name: 'CATEGORY_SUPERMARKET',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const freshFilterActive = ref(false);
    const priceFilter = ref(0);
    const sortOpen = ref(false);
    const currentSort = ref('');
    
    const isFiltered = computed(() => signatureStore.supermarket_list_filters_applied);
    const viewportAnchor = computed(() => signatureStore.supermarket_list_viewport_anchor_id);

    const items = computed(() => dataStore.supermarket);

    const filteredItems = computed(() => {
      let result = [...items.value];

      // Apply Filters
      if (freshFilterActive.value) {
        result = result.filter(i => i.fresh);
      }
      if (priceFilter.value > 0) {
        result = result.filter(i => i.price > priceFilter.value);
      }

      // Apply Sort
      if (currentSort.value === 'time') {
        // Mock sort by time (reverse id)
        result.reverse();
      } else if (currentSort.value === 'price_low_high') {
        result.sort((a, b) => a.price - b.price);
      } else if (currentSort.value === 'rating') {
        result.sort((a, b) => b.rating - a.rating);
      }

      return result;
    });

    const currentSortLabel = computed(() => {
      if (currentSort.value === 'time') return 'Newest';
      if (currentSort.value === 'price_low_high') return 'Price: Low to High';
      if (currentSort.value === 'rating') return 'Top Rated';
      return 'Sort By';
    });

    const getItemClass = (item) => {
      if (isFiltered.value) return 'row-filtered';
      return 'row-visible';
    };

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    const toggleFreshFilter = () => {
      freshFilterActive.value = !freshFilterActive.value;
      signatureStore.supermarket_list_filters_applied = true;
    };

    const applyFilters = () => {
      signatureStore.supermarket_list_filters_applied = true;
    };

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value;
    };

    const sort = (type) => {
      currentSort.value = type;
      sortOpen.value = false;
      signatureStore.supermarket_list_filters_applied = true;
    };

    const openItem = async (item) => {
      signatureStore.supermarket_selected_item_id = item.id;
      signatureStore.supermarket_list_filters_applied = null; // clear effect
      signatureStore.currentPageId = 'PRODUCT_DETAIL';
      await router.push({ name: 'PRODUCT_DETAIL', params: { id: item.id } });
    };

    return {
      freshFilterActive,
      priceFilter,
      sortOpen,
      currentSortLabel,
      filteredItems,
      getItemClass,
      goHome,
      toggleFreshFilter,
      applyFilters,
      toggleSort,
      sort,
      openItem
    };
  }
}
</script>