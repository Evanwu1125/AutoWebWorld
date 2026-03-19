<template>
  <div class="min-h-screen bg-[#FFF4E6] text-gray-800 pb-20 font-sans">
    <!-- Header -->
    <header class="bg-white sticky top-0 z-20 shadow-sm border-b border-orange-100">
      <div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <button id="focus-back-home" @click="goHome" class="text-orange-500 hover:text-orange-600 font-medium flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clip-rule="evenodd" />
          </svg>
          Back Home
        </button>
        <h1 class="text-xl font-bold tracking-wide text-gray-800">Focus Music</h1>
        <div class="w-20"></div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 pt-8">
      <!-- Search & Sort -->
      <div class="flex flex-col md:flex-row gap-4 mb-8">
        <div class="relative flex-1">
          <input id="focus-search-input" 
                 type="text" 
                 v-model="searchQuery"
                 @keydown.enter="handleSearch"
                 placeholder="Search focus music..." 
                 class="w-full pl-10 pr-4 py-3 rounded-xl border-orange-200 bg-white text-gray-800 placeholder-gray-400 focus:border-orange-500 focus:ring-orange-500 transition-all shadow-sm" />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 absolute left-3 top-3.5 text-orange-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        <div class="relative">
          <button id="focus-sort-dropdown" 
                  @click="toggleSortMenu"
                  class="bg-white px-6 py-3 rounded-xl border border-orange-200 shadow-sm flex items-center gap-2 hover:border-orange-500 transition-colors w-full md:w-auto justify-between text-gray-700">
            <span class="font-medium">Sort by: {{ currentSortLabel }}</span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>
          
          <div v-if="isSortMenuOpen" class="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-orange-100 z-30 py-2">
            <div id="focus-sort-time-desc" @click="handleSort('date')" class="px-4 py-2 hover:bg-orange-50 cursor-pointer text-gray-700">Time</div>
            <div id="focus-sort-new" @click="handleSort('new')" class="px-4 py-2 hover:bg-orange-50 cursor-pointer text-gray-700">Newest</div>
            <div id="focus-sort-short" @click="handleSort('short')" class="px-4 py-2 hover:bg-orange-50 cursor-pointer text-gray-700">Shortest</div>
          </div>
        </div>
      </div>

      <!-- Filters -->
      <div class="bg-white p-6 rounded-2xl shadow-sm border border-orange-100 mb-8">
        <h3 class="font-bold text-gray-800 mb-4">Filters</h3>
        <div class="flex flex-col md:flex-row gap-8">
          <!-- Checkbox Filter -->
          <div class="flex-1">
            <label class="block text-sm font-semibold text-gray-500 mb-3 uppercase tracking-wide">Type</label>
            <div class="flex gap-4">
              <label class="flex items-center gap-2 cursor-pointer group">
                <input id="focus-filter-music-checkbox" type="checkbox" 
                       v-model="filters.music"
                       class="w-5 h-5 rounded text-orange-500 focus:ring-orange-500 border-gray-300 transition-all" />
                <span class="group-hover:text-orange-600">With Music</span>
              </label>
            </div>
          </div>

          <!-- Slider Filter -->
          <div class="flex-1">
            <label class="block text-sm font-semibold text-gray-500 mb-3 uppercase tracking-wide">
              Min Duration: {{ filters.length }} min
            </label>
            <input id="focus-length-slider" type="range" 
                   v-model.number="filters.length"
                   min="0" max="120" step="5"
                   class="w-full h-2 bg-orange-200 rounded-lg appearance-none cursor-pointer accent-orange-500" />
            <div class="flex justify-between text-xs text-gray-400 mt-2">
              <span>0m</span>
              <span>60m+</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Focus List -->
      <div id="focus-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div v-for="session in filteredSessions" :key="session.id"
             :class="getRowClass(session)"
             class="bg-white rounded-2xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-300 group cursor-pointer border border-orange-50"
             @click="openSession(session)">
          
          <!-- Image -->
          <div class="h-48 overflow-hidden relative">
            <img :src="session.image" :alt="session.title" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
            <div class="absolute top-3 right-3 bg-white/90 backdrop-blur-sm px-3 py-1 rounded-full text-xs font-bold text-gray-700">
              {{ session.duration_min }} min
            </div>
          </div>

          <!-- Content -->
          <div class="p-5">
            <div class="flex items-center gap-2 mb-2">
              <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-orange-100 text-orange-600">
                {{ session.music_type }}
              </span>
              <span v-if="session.has_music" class="text-xs">🎵</span>
              <span class="px-2 py-0.5 rounded text-[10px] font-medium bg-gray-100 text-gray-600">
                {{ session.published_date }}
              </span>
            </div>
            <h3 class="font-bold text-lg text-gray-900 mb-2 group-hover:text-orange-500 transition-colors">
              {{ session.title }}
            </h3>
            <p class="text-gray-500 text-sm line-clamp-2">{{ session.description }}</p>
          </div>
        </div>
      </div>

      <!-- Empty State -->
      <div v-if="filteredSessions.length === 0" class="text-center py-20">
        <div class="text-6xl mb-4 text-orange-200">🎧</div>
        <h3 class="text-xl font-bold text-gray-800 mb-2">No focus sessions found</h3>
        <p class="text-gray-500">Try adjusting your filters.</p>
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
  name: 'FOCUS_LIST',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const searchQuery = ref('');
    const isSortMenuOpen = ref(false);
    const currentSort = ref(null);
    const filters = ref({
      music: false,
      length: 0
    });

    watch(filters, () => {
      signatureStore.focus_filters_applied = true;
    }, { deep: true });

    watch(searchQuery, (newVal) => {
      if(newVal) signatureStore.focus_has_searched = true;
    });

    const filteredSessions = computed(() => {
      let result = dataStore.focus_sessions;

      if (searchQuery.value) {
        const query = searchQuery.value.toLowerCase();
        result = result.filter(s => s.title.toLowerCase().includes(query));
        if (result.length > 0) signatureStore.matched_focus_id = result[0].id;
      }

      if (filters.value.music) {
        result = result.filter(s => s.has_music);
      }

      if (filters.value.length > 0) {
        result = result.filter(s => s.duration_min >= filters.value.length);
      }

      if (currentSort.value === 'new') {
        result = [...result].sort((a, b) => new Date(b.published_date) - new Date(a.published_date));
      } else if (currentSort.value === 'short') {
        result = [...result].sort((a, b) => a.duration_min - b.duration_min);
      } else if (currentSort.value === 'popular') {
        result = [...result].sort((a, b) => a.id.localeCompare(b.id));
      } else if (currentSort.value === 'date') {
        result = [...result].sort((a, b) => b.duration_min - a.duration_min);
      }

      return result;
    });

    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default';
      if (currentSort.value === 'new') return 'Newest';
      if (currentSort.value === 'short') return 'Shortest';
      if (currentSort.value === 'date') return 'Time';
      return 'Popular';
    });

    const toggleSortMenu = () => isSortMenuOpen.value = !isSortMenuOpen.value;

    const handleSort = (type) => {
      currentSort.value = type;
      signatureStore.focus_filters_applied = true;
      isSortMenuOpen.value = false;
    };

    const handleSearch = () => {
      signatureStore.focus_has_searched = true;
      if (filteredSessions.value.length > 0) {
        signatureStore.matched_focus_id = filteredSessions.value[0].id;
      }
    };

    const getRowClass = (session) => {
      const classes = [`data-id-${session.id}`];
      if (signatureStore.focus_filters_applied) classes.push('focus-row-filtered');
      if (signatureStore.focus_has_searched && session.id === signatureStore.matched_focus_id) classes.push('focus-row-matched');
      if (!signatureStore.focus_filters_applied && !signatureStore.focus_has_searched) classes.push('focus-row-visible');
      return classes.join(' ');
    };

    const openSession = async (session) => {
      signatureStore.focus_filters_applied = null;
      signatureStore.focus_has_searched = null;
      signatureStore.focus_viewport_anchor_id = null;
      signatureStore.selected_focus_id = session.id;
      await router.push({ name: 'FOCUS_DETAIL', params: { id: session.id } });
    };

    const goHome = async () => {
      await router.push({ name: 'HOME' });
    };

    return {
      searchQuery,
      filters,
      isSortMenuOpen,
      currentSortLabel,
      filteredSessions,
      toggleSortMenu,
      handleSort,
      handleSearch,
      getRowClass,
      openSession,
      goHome
    };
  }
}
</script>