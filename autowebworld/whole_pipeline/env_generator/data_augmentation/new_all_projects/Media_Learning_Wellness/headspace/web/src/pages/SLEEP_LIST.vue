<template>
  <div class="min-h-screen bg-[#1A2338] text-white pb-20 font-sans">
    <!-- Header -->
    <header class="bg-[#1A2338] sticky top-0 z-20 shadow-md border-b border-gray-800">
      <div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <button id="sleep-back-home" @click="goHome" class="text-blue-300 hover:text-white font-medium flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clip-rule="evenodd" />
          </svg>
          Back Home
        </button>
        <h1 class="text-xl font-bold tracking-wide">Sleepcasts</h1>
        <div class="w-20"></div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 pt-8">
      <!-- Search & Sort -->
      <div class="flex flex-col md:flex-row gap-4 mb-8">
        <div class="relative flex-1">
          <input id="sleep-search-input" 
                 type="text" 
                 v-model="searchQuery"
                 @keydown.enter="handleSearch"
                 placeholder="Search sleep sounds..." 
                 class="w-full pl-10 pr-4 py-3 rounded-xl border-gray-700 bg-[#25304C] text-white placeholder-gray-400 focus:border-blue-400 focus:ring-blue-400 transition-all shadow-sm" />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 absolute left-3 top-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        <div class="relative">
          <button id="sleep-sort-dropdown" 
                  @click="toggleSortMenu"
                  class="bg-[#25304C] px-6 py-3 rounded-xl border border-gray-700 shadow-sm flex items-center gap-2 hover:border-blue-400 transition-colors w-full md:w-auto justify-between text-white">
            <span class="font-medium">Sort by: {{ currentSortLabel }}</span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>
          
          <div v-if="isSortMenuOpen" class="absolute right-0 mt-2 w-48 bg-[#25304C] rounded-xl shadow-xl border border-gray-700 z-30 py-2">
            <div id="sleep-sort-popular" @click="handleSort('popular')" class="px-4 py-2 hover:bg-[#324164] cursor-pointer">Popular</div>
            <div id="sleep-sort-new" @click="handleSort('new')" class="px-4 py-2 hover:bg-[#324164] cursor-pointer">Newest</div>
            <div id="sleep-sort-length-desc" @click="handleSort('length')" class="px-4 py-2 hover:bg-[#324164] cursor-pointer">Length</div>
          </div>
        </div>
      </div>

      <!-- Filters -->
      <div class="bg-[#25304C] p-6 rounded-2xl shadow-sm border border-gray-700 mb-8">
        <h3 class="font-bold text-white mb-4">Filters</h3>
        <div class="flex flex-col md:flex-row gap-8">
          <!-- Checkbox Filter -->
          <div class="flex-1">
            <label class="block text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wide">Duration</label>
            <div class="flex gap-4">
              <label class="flex items-center gap-2 cursor-pointer group">
                <input id="sleep-filter-long-checkbox" type="checkbox" 
                       v-model="filters.long"
                       class="w-5 h-5 rounded text-blue-500 bg-gray-700 focus:ring-blue-500 border-gray-600 transition-all" />
                <span class="group-hover:text-blue-300">Long Sessions (45m+)</span>
              </label>
            </div>
          </div>

          <!-- Slider Filter -->
          <div class="flex-1">
            <label class="block text-sm font-semibold text-gray-400 mb-3 uppercase tracking-wide">
              Max Intensity: {{ filters.intensity }}
            </label>
            <input id="sleep-intensity-slider" type="range" 
                   v-model.number="filters.intensity"
                   min="0" max="10" step="1"
                   class="w-full h-2 bg-gray-600 rounded-lg appearance-none cursor-pointer accent-blue-500" />
            <div class="flex justify-between text-xs text-gray-400 mt-2">
              <span>Low</span>
              <span>High</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Sleep List -->
      <div id="sleep-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div v-for="track in filteredTracks" :key="track.id"
             :class="getRowClass(track)"
             class="bg-[#25304C] rounded-2xl overflow-hidden shadow-lg hover:shadow-2xl hover:shadow-blue-900/20 transition-all duration-300 group cursor-pointer border border-gray-700"
             @click="openTrack(track)">
          
          <!-- Image -->
          <div class="h-48 overflow-hidden relative">
            <img :src="track.image" :alt="track.title" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500 opacity-80 group-hover:opacity-100" />
            <div class="absolute top-3 right-3 bg-black/60 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-bold text-white">
              {{ track.duration_min }} min
            </div>
          </div>

          <!-- Content -->
          <div class="p-5">
            <div class="flex items-center gap-2 mb-2">
              <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-blue-900/50 text-blue-300 border border-blue-800">
                {{ track.type }}
              </span>
              <span class="px-2 py-0.5 rounded text-[10px] font-medium bg-gray-700 text-gray-300">
                {{ track.published_date }}
              </span>
            </div>
            <h3 class="font-bold text-lg text-white mb-2 group-hover:text-blue-300 transition-colors">
              {{ track.title }}
            </h3>
            <p class="text-gray-400 text-sm line-clamp-2">{{ track.description }}</p>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredTracks.length === 0" class="text-center py-20">
        <div class="text-6xl mb-4 text-blue-200">🌙</div>
        <h3 class="text-xl font-bold text-white mb-2">No sleep tracks found</h3>
        <p class="text-gray-400">Try adjusting your filters.</p>
      </div>

    </main>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'SLEEP_LIST',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const searchQuery = ref('');
    const isSortMenuOpen = ref(false);
    const currentSort = ref(null);
    const filters = ref({
      long: false,
      intensity: 10 // Start max to show all
    });

    watch(filters, () => {
      signatureStore.sleep_filters_applied = true;
    }, { deep: true });

    watch(searchQuery, (newVal) => {
      if(newVal) signatureStore.sleep_has_searched = true;
    });

    const filteredTracks = computed(() => {
      let result = dataStore.sleep_tracks;

      if (searchQuery.value) {
        const query = searchQuery.value.toLowerCase();
        result = result.filter(t => t.title.toLowerCase().includes(query));
        if (result.length > 0) signatureStore.matched_sleep_id = result[0].id;
      }

      if (filters.value.long) {
        result = result.filter(t => t.duration_min >= 45);
      }

      // Filter by Intensity (Max filter)
      // Since slider starts at 10, initially shows everything <= 10
      // If user drags left, intensity filter reduces
      result = result.filter(t => t.intensity <= filters.value.intensity);

      if (currentSort.value === 'new') {
        result = [...result].sort((a, b) => new Date(b.published_date) - new Date(a.published_date));
      } else if (currentSort.value === 'length') {
        result = [...result].sort((a, b) => b.duration_min - a.duration_min);
      } else if (currentSort.value === 'popular') {
        result = [...result].sort((a, b) => a.id.localeCompare(b.id));
      }

      return result;
    });

    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default';
      if (currentSort.value === 'new') return 'Newest';
      if (currentSort.value === 'length') return 'Longest';
      return 'Popular';
    });

    const toggleSortMenu = () => isSortMenuOpen.value = !isSortMenuOpen.value;

    const handleSort = (type) => {
      currentSort.value = type;
      signatureStore.sleep_filters_applied = true;
      isSortMenuOpen.value = false;
    };

    const handleSearch = () => {
      signatureStore.sleep_has_searched = true;
      if (filteredTracks.value.length > 0) {
        signatureStore.matched_sleep_id = filteredTracks.value[0].id;
      }
    };

    const getRowClass = (track) => {
      const classes = [`data-id-${track.id}`];
      if (signatureStore.sleep_filters_applied) classes.push('sleep-row-filtered');
      if (signatureStore.sleep_has_searched && track.id === signatureStore.matched_sleep_id) classes.push('sleep-row-matched');
      if (!signatureStore.sleep_filters_applied && !signatureStore.sleep_has_searched) classes.push('sleep-row-visible');
      return classes.join(' ');
    };

    const openTrack = async (track) => {
      signatureStore.sleep_filters_applied = null;
      signatureStore.sleep_has_searched = null;
      signatureStore.sleep_viewport_anchor_id = null;
      signatureStore.selected_sleep_id = track.id;
      await router.push({ name: 'SLEEP_DETAIL', params: { id: track.id } });
    };

    const goHome = async () => {
      await router.push({ name: 'HOME' });
    };

    return {
      searchQuery,
      filters,
      isSortMenuOpen,
      currentSortLabel,
      filteredTracks,
      toggleSortMenu,
      handleSort,
      handleSearch,
      getRowClass,
      openTrack,
      goHome
    };
  }
}
</script>