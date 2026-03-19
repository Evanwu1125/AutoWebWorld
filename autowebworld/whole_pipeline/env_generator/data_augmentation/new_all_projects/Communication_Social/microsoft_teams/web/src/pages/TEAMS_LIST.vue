<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <header class="bg-[#6264A7] text-white p-4 shadow-md flex justify-between items-center z-20">
      <div class="font-bold text-lg flex items-center">
        <button id="back-to-home" @click="goHome" class="mr-4 hover:bg-[#464775] p-1 rounded">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        Teams
      </div>
      <div class="flex items-center gap-4">
        <!-- Search ACT_TEAMS_SEARCH -->
        <div class="relative">
          <input 
            id="teams-search-input"
            type="text" 
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            placeholder="Search teams..."
            class="pl-10 pr-4 py-2 rounded bg-[#464775] text-white placeholder-gray-300 border-none focus:ring-2 focus:ring-white/50 w-64"
          />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-300 absolute left-3 top-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
        <!-- Create Team ACT_TEAMS_NEW_TEAM -->
        <button id="create-team-button" @click="createTeam" class="bg-white text-[#6264A7] px-4 py-2 rounded font-medium hover:bg-gray-100 transition-colors">
          + Join or create team
        </button>
      </div>
    </header>

    <div class="flex-1 flex overflow-hidden">
      <!-- Sidebar Filters -->
      <aside class="w-64 bg-white border-r border-gray-200 p-4 flex flex-col gap-6 overflow-y-auto">
        <div>
          <h3 class="font-semibold text-gray-700 mb-2">Filters</h3>
          <!-- Checkbox Filter ACT_TEAMS_FILTER_CHECKBOX -->
          <div class="flex items-center gap-2 mb-4">
            <input 
              id="filter-favorites-checkbox"
              type="checkbox" 
              v-model="showFavoritesOnly"
              class="w-4 h-4 text-[#6264A7] rounded focus:ring-[#6264A7]"
            />
            <label for="filter-favorites-checkbox" class="text-sm text-gray-600">Favorites only</label>
          </div>

          <!-- Slider Filter ACT_TEAMS_FILTER_SLIDER -->
          <div class="mb-4">
            <label class="text-sm text-gray-600 block mb-1">Activity Level: > {{ minActivity }}%</label>
            <input 
              id="filter-activity-slider"
              type="range" 
              min="0" 
              max="100" 
              v-model.number="minActivity"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#6264A7]"
            />
          </div>
        </div>

        <div>
          <h3 class="font-semibold text-gray-700 mb-2">Sort By</h3>
          <!-- Sort Dropdown ACT_TEAMS_FILTER_SORT -->
          <div id="teams-sort-dropdown" class="relative">
            <div 
              @click="toggleSort"
              class="w-full border rounded px-3 py-2 text-sm text-gray-700 bg-white cursor-pointer flex justify-between items-center"
            >
              {{ sortBy === 'recent' ? 'Most Recent' : (sortBy === 'name' ? 'Name (A-Z)' : 'Select...') }}
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            <div v-if="sortOpen" class="absolute top-full left-0 right-0 mt-1 bg-white border rounded shadow-lg z-10">
              <div id="teams-sort-recent" @click="setSort('recent')" class="px-3 py-2 text-sm hover:bg-gray-100 cursor-pointer">Most Recent</div>
              <div id="teams-sort-name-inc" @click="setSort('name')" class="px-3 py-2 text-sm hover:bg-gray-100 cursor-pointer">Name (A-Z)</div>
            </div>
          </div>
        </div>
      </aside>

      <!-- Main Content -->
      <main id="teams-list-container" class="flex-1 p-6 overflow-y-auto bg-gray-50">
        <h2 class="text-2xl font-bold text-gray-800 mb-6">Your Teams</h2>
        
        <div id="teams-list" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          <div 
            v-for="team in filteredTeams" 
            :key="team.id"
            :class="`data-id-${team.id} bg-white rounded-lg shadow-sm hover:shadow-md transition-shadow cursor-pointer overflow-hidden group border border-gray-100 flex flex-col ${getCardClass(team)}`"
            @click="openTeam(team)"
          >
            <div class="h-32 bg-gray-200 relative overflow-hidden">
               <img 
                :src="team.image" 
                class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500"
                alt="Team Cover"
                @error="$event.target.src = 'https://picsum.photos/400/200'"
              />
              <div class="absolute top-2 right-2" v-if="team.isFavorite">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-yellow-400 fill-current" viewBox="0 0 20 20">
                  <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                </svg>
              </div>
            </div>
            <div class="p-4 flex-1 flex flex-col">
              <h3 class="font-bold text-lg text-gray-800 mb-1 group-hover:text-[#6264A7] transition-colors">{{ team.name }}</h3>
              <p class="text-sm text-gray-500 line-clamp-2 mb-4 flex-1">{{ team.description }}</p>
              <div class="flex justify-between items-center text-xs text-gray-400 border-t pt-3">
                <span class="flex items-center gap-1">
                  <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Activity: {{ team.activity }}%
                </span>
                <span>Open &rarr;</span>
              </div>
            </div>
          </div>
          
          <!-- Empty State -->
          <div v-if="filteredTeams.length === 0" class="col-span-full flex flex-col items-center justify-center p-12 text-gray-500">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 mb-4 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <p class="text-lg">No teams found matching your filters.</p>
          </div>
        </div>
      </main>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'TEAMS_LIST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const searchQuery = ref('')
    const showFavoritesOnly = ref(false)
    const minActivity = ref(0)
    const sortBy = ref('')
    const sortOpen = ref(false)

    // Filter Logic
    const filteredTeams = computed(() => {
      let result = dataStore.teams;

      // Filter by Search (Action: ACT_TEAMS_SEARCH)
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase();
        result = result.filter(t => t.name.toLowerCase().includes(q) || t.description.toLowerCase().includes(q));
      }

      // Filter by Favorites (Action: ACT_TEAMS_FILTER_CHECKBOX)
      if (showFavoritesOnly.value) {
        result = result.filter(t => t.isFavorite);
      }

      // Filter by Slider (Action: ACT_TEAMS_FILTER_SLIDER)
      if (minActivity.value > 0) {
        result = result.filter(t => t.activity > minActivity.value);
      }

      // Sort (Action: ACT_TEAMS_FILTER_SORT)
      if (sortBy.value === 'name') {
        result = [...result].sort((a, b) => a.name.localeCompare(b.name));
      } else if (sortBy.value === 'recent') {
        // Mock "recent" by assuming ID order or activity reversed for now
        result = [...result].sort((a, b) => b.activity - a.activity);
      }

      return result;
    })

    const handleSearch = () => {
      store.teams_list_has_searched = true;
      store.matched_team_id = filteredTeams.value.length > 0 ? filteredTeams.value[0].id : null;
    }

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value
    }

    const setSort = (type) => {
      sortBy.value = type;
      sortOpen.value = false;
      store.teams_list_filters_applied = true;
    }

    // Determine specific classes for testing based on filtering state
    const getCardClass = (team) => {
      let classes = 'team-row-visible '; // ACT_TEAMS_OPEN_ANY_TEAM selector
      if (store.teams_list_filters_applied) classes += 'team-row-filtered '; // ACT_TEAMS_OPEN_FILTERED_TEAM selector
      if (store.teams_list_has_searched) classes += 'team-row-matched '; // ACT_TEAMS_OPEN_MATCHED_TEAM selector
      return classes;
    }

    const openTeam = async (team) => {
      store.selected_team_id = team.id;
      // Clear flags as per effects
      store.teams_list_filters_applied = null;
      store.teams_list_has_searched = null;
      store.teams_list_viewport_anchor_id = null;
      
      store.currentPageId = 'CHANNELS_LIST';
      await router.push({ name: 'CHANNELS_LIST', params: { teamId: team.id } });
    }

    const createTeam = async () => {
      store.currentPageId = 'CREATE_TEAM';
      await router.push({ name: 'CREATE_TEAM' });
    }

    const goHome = async () => {
      store.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    }

    return {
      searchQuery,
      showFavoritesOnly,
      minActivity,
      sortBy,
      sortOpen,
      filteredTeams,
      handleSearch,
      toggleSort,
      setSort,
      getCardClass,
      openTeam,
      createTeam,
      goHome,
      store
    }
  },
  watch: {
    showFavoritesOnly() {
      this.store.teams_list_filters_applied = true;
    },
    minActivity() {
      this.store.teams_list_filters_applied = true;
    }
  }
}
</script>