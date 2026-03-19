<template>
  <div class="min-h-screen bg-[#FDFBF7] pb-20">
    <!-- Permission Modal -->
    <PermissionModal 
      v-if="showPermissionModal" 
      @allow="handleAllowPermission"
      @deny="handleDenyPermission"
    />

    <!-- Header -->
    <header class="bg-white sticky top-0 z-20 shadow-sm">
      <div class="max-w-7xl mx-auto px-4 h-16 flex items-center justify-between">
        <button id="header-back-home" @click="goHome" class="text-gray-600 hover:text-orange-500 font-medium flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path fill-rule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clip-rule="evenodd" />
          </svg>
          Back Home
        </button>
        <h1 class="text-xl font-bold text-gray-800">Browse Library</h1>
        <div class="w-20"></div> <!-- Spacer for centering -->
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 pt-8">
      <!-- Search & Sort -->
      <div class="flex flex-col md:flex-row gap-4 mb-8">
        <div class="relative flex-1">
          <input id="browse-search-input" 
                 type="text" 
                 v-model="searchQuery"
                 @keydown.enter="handleSearch"
                 placeholder="Search meditations..." 
                 class="w-full pl-10 pr-4 py-3 rounded-xl border-gray-200 focus:border-orange-500 focus:ring-orange-500 transition-all shadow-sm" />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 absolute left-3 top-3.5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        <div class="relative">
          <button id="browse-sort-dropdown" 
                  @click="toggleSortMenu"
                  class="bg-white px-6 py-3 rounded-xl border border-gray-200 shadow-sm flex items-center gap-2 hover:border-orange-500 transition-colors w-full md:w-auto justify-between">
            <span class="text-gray-700 font-medium">Sort by: {{ currentSortLabel }}</span>
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" viewBox="0 0 20 20" fill="currentColor">
              <path fill-rule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clip-rule="evenodd" />
            </svg>
          </button>
          
          <div v-if="isSortMenuOpen" class="absolute right-0 mt-2 w-48 bg-white rounded-xl shadow-xl border border-gray-100 z-30 py-2">
            <div id="browse-sort-newest-desc" @click="handleSort('newest')" class="px-4 py-2 hover:bg-orange-50 cursor-pointer">Newest</div>
            <div id="browse-sort-shortest" @click="handleSort('shortest')" class="px-4 py-2 hover:bg-orange-50 cursor-pointer">Shortest</div>
            <div id="browse-sort-popular" @click="handleSort('popular')" class="px-4 py-2 hover:bg-orange-50 cursor-pointer">Popular</div>
          </div>
        </div>
      </div>

      <!-- Filters -->
      <div class="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 mb-8">
        <h3 class="font-bold text-gray-800 mb-4">Filters</h3>
        <div class="flex flex-col md:flex-row gap-8">
          <!-- Checkbox Filter -->
          <div class="flex-1">
            <label class="block text-sm font-semibold text-gray-500 mb-3 uppercase tracking-wide">Difficulty</label>
            <div class="flex gap-4">
              <label class="flex items-center gap-2 cursor-pointer group">
                <input id="filter-difficulty-beginner-checkbox" type="checkbox" 
                       v-model="filters.beginner"
                       class="w-5 h-5 rounded text-orange-500 focus:ring-orange-500 border-gray-300 transition-all" />
                <span class="group-hover:text-orange-600">Beginner</span>
              </label>
              <!-- Add other difficulty checkboxes if needed by FSM, but FSM only specifies beginner explicitly -->
            </div>
          </div>

          <!-- Slider Filter -->
          <div class="flex-1">
            <label class="block text-sm font-semibold text-gray-500 mb-3 uppercase tracking-wide">
              Min Duration: {{ filters.duration }} min
            </label>
            <input id="filter-duration-slider" type="range" 
                   v-model.number="filters.duration"
                   min="0" max="120" step="5"
                   class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-orange-500" />
            <div class="flex justify-between text-xs text-gray-400 mt-2">
              <span>0m</span>
              <span>60m</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Session List -->
      <div id="browse-session-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div v-for="session in filteredSessions" :key="session.id"
             :class="getRowClass(session)"
             class="bg-white rounded-2xl overflow-hidden shadow-sm hover:shadow-xl transition-all duration-300 group cursor-pointer border border-gray-50"
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
              <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider bg-blue-50 text-blue-600">
                {{ session.difficulty }}
              </span>
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
        <div class="text-6xl mb-4">🍃</div>
        <h3 class="text-xl font-bold text-gray-800 mb-2">No sessions found</h3>
        <p class="text-gray-500">Try adjusting your filters or search query.</p>
      </div>

    </main>
  </div>
</template>

<script>
import { ref, computed, watch, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';
import PermissionModal from '../components/PermissionModal.vue';

export default {
  name: 'BROWSE',
  components: { PermissionModal },
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    // Local State
    const searchQuery = ref('');
    const isSortMenuOpen = ref(false);
    const currentSort = ref(null);
    const filters = ref({
      beginner: false,
      duration: 0
    });

    // Permission Modal Logic
    const showPermissionModal = computed(() => signatureStore.location_permission_granted === null);

    const handleAllowPermission = () => {
      signatureStore.location_permission_granted = true;
      // Corresponds to ACT_BROWSE_LOCATION_PERMISSION_ALLOW effect
    };
    
    const handleDenyPermission = () => {
      // Not in FSM but handle gracefully
      signatureStore.location_permission_granted = false; 
    };

    // Filter Logic
    // Watch filters to update global state flag
    watch(filters, () => {
      signatureStore.browse_filters_applied = true;
    }, { deep: true });

    watch(searchQuery, (newVal) => {
      if(newVal) signatureStore.browse_has_searched = true;
    });

    const filteredSessions = computed(() => {
      let result = dataStore.browse_sessions;

      // Filter by Search
      if (searchQuery.value) {
        const query = searchQuery.value.toLowerCase();
        result = result.filter(s => s.title.toLowerCase().includes(query));
        // Update matched matched_session_id if we have results
        if (result.length > 0) {
          signatureStore.matched_session_id = result[0].id;
        }
      }

      // Filter by Difficulty (Beginner)
      if (filters.value.beginner) {
        result = result.filter(s => s.difficulty === 'Beginner');
      }

      // Filter by Duration
      if (filters.value.duration > 0) {
        result = result.filter(s => s.duration_min >= filters.value.duration);
      }

      // Sort
      if (currentSort.value === 'newest') {
        result = [...result].sort((a, b) => new Date(b.published_date) - new Date(a.published_date));
      } else if (currentSort.value === 'shortest') {
        result = [...result].sort((a, b) => a.duration_min - b.duration_min);
      } else if (currentSort.value === 'popular') {
        result = [...result].sort((a, b) => a.id.localeCompare(b.id));
      }

      return result;
    });

    // Helpers
    const currentSortLabel = computed(() => {
      if (!currentSort.value) return 'Default';
      return currentSort.value.charAt(0).toUpperCase() + currentSort.value.slice(1);
    });

    const toggleSortMenu = () => isSortMenuOpen.value = !isSortMenuOpen.value;

    const handleSort = (sortType) => {
      currentSort.value = sortType;
      signatureStore.browse_filters_applied = true;
      isSortMenuOpen.value = false;
    };

    const handleSearch = () => {
      // Triggered on Enter key
      signatureStore.browse_has_searched = true;
      // In FSM, effect sets matched_session_id based on search result
      if (filteredSessions.value.length > 0) {
        signatureStore.matched_session_id = filteredSessions.value[0].id;
      }
    };

    const getRowClass = (session) => {
      // Determine class based on state context
      const classes = [`data-id-${session.id}`];
      
      // session-row-filtered
      if (signatureStore.browse_filters_applied) {
        classes.push('session-row-filtered');
      }
      // session-row-matched
      if (signatureStore.browse_has_searched && session.id === signatureStore.matched_session_id) {
        classes.push('session-row-matched');
      }
      // session-row-visible (default fallback)
      if (!signatureStore.browse_filters_applied && !signatureStore.browse_has_searched) {
        classes.push('session-row-visible');
      }
      
      return classes.join(' ');
    };

    const openSession = async (session) => {
      // Clear flags based on FSM effects
      signatureStore.browse_filters_applied = null;
      signatureStore.browse_has_searched = null;
      signatureStore.browse_viewport_anchor_id = null;
      
      // Set selected ID
      signatureStore.selected_session_id = session.id;
      
      // Navigate
      await router.push({ name: 'SESSION_DETAIL', params: { id: session.id } });
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
      showPermissionModal,
      toggleSortMenu,
      handleSort,
      handleSearch,
      getRowClass,
      openSession,
      goHome,
      handleAllowPermission,
      handleDenyPermission
    };
  }
}
</script>