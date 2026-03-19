<template>
  <div class="min-h-screen bg-[#0d1117] text-white font-sans">
    <PermissionModal />
    
    <!-- Header -->
    <header class="bg-[#161b22] border-b border-gray-700 py-4 px-6 flex items-center justify-between">
      <div class="flex items-center gap-4">
        <div id="github-logo" class="cursor-pointer" @click="goHome">
          <svg height="32" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="32" data-view-component="true" class="octicon octicon-mark-github v-align-middle text-white fill-current">
            <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path>
          </svg>
        </div>
        <h1 class="text-xl font-semibold">Repositories</h1>
      </div>
      <button id="new-repo-button" @click="goToNewRepo" class="px-3 py-1 text-sm font-medium text-white bg-[#238636] rounded-md hover:bg-[#2ea44f]">
        New
      </button>
    </header>

    <main class="max-w-6xl mx-auto p-6">
      <!-- Search and Filter Bar -->
      <div class="flex flex-col md:flex-row gap-4 mb-6 items-start md:items-center justify-between">
        <div class="flex-grow relative max-w-lg">
          <input 
            id="repo-search-input"
            type="text" 
            v-model="searchQuery"
            @keyup.enter="handleSearch"
            placeholder="Find a repository..." 
            class="w-full px-3 py-1.5 bg-[#0d1117] border border-gray-600 rounded-md text-sm focus:border-blue-500 focus:ring-1 focus:ring-blue-500 outline-none"
          />
        </div>
        
        <div class="flex gap-3 items-center">
            <!-- Type Filter (Checkbox) -->
             <label class="flex items-center space-x-2 text-sm text-gray-300 cursor-pointer select-none">
              <input 
                id="filter-private-checkbox"
                type="checkbox" 
                class="form-checkbox h-4 w-4 text-blue-600 bg-gray-800 border-gray-600 rounded focus:ring-offset-gray-900"
                @change="handleFilterPrivate"
              />
              <span>Private</span>
            </label>

            <!-- Stars Filter (Slider) -->
            <div class="flex items-center space-x-2">
                <label class="text-sm text-gray-400">Stars > {{ starFilter }}</label>
                <input 
                    id="filter-stars-slider"
                    type="range" 
                    min="0" 
                    max="200000" 
                    step="1000"
                    v-model.number="starFilter"
                    @input="handleStarFilter"
                    class="w-32 h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                />
            </div>

            <!-- Sort Dropdown -->
            <div class="relative">
                <button 
                    id="repos-sort-dropdown"
                    @click="sortDropdownOpen = !sortDropdownOpen"
                    class="px-3 py-1.5 text-sm font-medium text-gray-300 bg-[#21262d] border border-gray-600 rounded-md hover:bg-gray-700"
                >
                    Sort: {{ currentSort || 'Default' }} ▾
                </button>
                <div v-if="sortDropdownOpen" class="absolute right-0 z-10 mt-2 w-40 bg-[#161b22] border border-gray-700 rounded-md shadow-lg">
                    <div id="repos-sort-updated-desc" @click="handleSort('updated')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Last updated</div>
                    <div id="repos-sort-stars" @click="handleSort('stars')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Stars</div>
                    <div id="repos-sort-name" @click="handleSort('name')" class="block px-4 py-2 text-sm text-gray-300 hover:bg-blue-600 hover:text-white cursor-pointer">Name</div>
                </div>
            </div>
        </div>
      </div>

      <!-- Repositories List -->
      <div id="repo-list" class="space-y-4">
        <div 
            v-for="repo in filteredRepos" 
            :key="repo.id" 
            :class="['p-4 border border-gray-700 rounded-md bg-[#161b22] hover:bg-[#1c2128] transition-colors']"
        >
          <div class="flex items-start justify-between">
            <div class="flex items-center gap-3">
                <div class="w-8 h-8 rounded-full overflow-hidden bg-gray-800 flex-shrink-0">
                    <img :src="repo.image" alt="repo" class="w-full h-full object-cover" />
                </div>
                <div>
                    <h3 class="text-xl font-semibold text-blue-400 hover:underline cursor-pointer">
                        <span 
                            :class="{
                                [`data-id-${repo.id}`]: true,
                                'repo-row-matched': isMatched(repo),
                                'repo-row-filtex xred': isFiltered(repo),
                                'repo-row-visible': !isMatched(repo) && !isFiltered(repo)
                            }"
                            @click="openRepo(repo)"
                        >
                            {{ repo.name }}
                        </span>
                        <span v-if="repo.private" class="ml-2 px-1.5 py-0.5 text-xs border border-gray-600 rounded-full text-gray-400">Private</span>
                        <span v-else class="ml-2 px-1.5 py-0.5 text-xs border border-gray-600 rounded-full text-gray-400">Public</span>
                    </h3>
                    <p class="text-gray-400 text-sm mt-1">{{ repo.description }}</p>
                    <div class="flex items-center gap-4 mt-3 text-xs text-gray-500">
                        <div class="flex items-center gap-1">
                            <span class="w-3 h-3 rounded-full bg-yellow-400"></span>
                            {{ repo.language }}
                        </div>
                        <div class="flex items-center gap-1">
                            <svg aria-hidden="true" height="16" viewBox="0 0 16 16" version="1.1" width="16" data-view-component="true" class="octicon octicon-star fill-gray-500"><path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.75.75 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25Z"></path></svg>
                            {{ repo.stars.toLocaleString() }}
                        </div>
                        <div>Updated on {{ repo.updated_at }}</div>
                    </div>
                </div>
            </div>
            <button class="px-3 py-1 text-xs font-medium text-gray-300 bg-[#21262d] border border-gray-600 rounded-md hover:bg-gray-700">
                Star
            </button>
          </div>
        </div>

        <!-- Empty State -->
        <div v-if="filteredRepos.length === 0" class="text-center py-12 border border-dashed border-gray-700 rounded-md">
            <h3 class="text-lg font-medium text-gray-300">No repositories found</h3>
            <p class="text-gray-500">Try adjusting your search or filters.</p>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';
import PermissionModal from '../components/PermissionModal.vue';
import { FSMRuntime } from '../fsm/FSMRuntime';
import fsmData from '../../fsm.json';

export default {
  name: 'REPOSITORIES_LIST',
  components: { PermissionModal },
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const dataStore = useDataStore();
    const fsmRuntime = new FSMRuntime(fsmData, { store, router });

    // State
    const searchQuery = ref('');
    const starFilter = ref(0);
    const showPrivate = ref(false);
    const sortDropdownOpen = ref(false);
    const currentSort = ref('');
    
    const matchedRepoId = ref(null); // For search results highlighting logic
    const filtersApplied = ref(false);

    // Computed
    const filteredRepos = computed(() => {
      let repos = [...dataStore.repositories];

      // Search
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase();
        repos = repos.filter(r => r.name.toLowerCase().includes(q));
      }

      // Private Filter
      if (showPrivate.value) {
         repos = repos.filter(r => r.private);
      }

      // Star Filter
      if (starFilter.value > 0) {
        repos = repos.filter(r => r.stars >= starFilter.value);
      }

      // Sort
      if (currentSort.value) {
        repos.sort((a, b) => {
          if (currentSort.value === 'stars') return b.stars - a.stars;
          if (currentSort.value === 'name') return a.name.localeCompare(b.name);
          if (currentSort.value === 'updated') return new Date(b.updated_at) - new Date(a.updated_at);
          return 0;
        });
      }

      return repos;
    });

    // FSM Helpers for selectors
    const isMatched = (repo) => store.signature.repos_matched_repo_id === repo.id;
    const isFiltered = (repo) => store.signature.repos_list_filters_applied === true;

    // Actions
    const goHome = async () => {
      store.setCurrentPageId('HOME');
      await router.push({ name: 'HOME' });
    };

    const goToNewRepo = async () => {
      store.setCurrentPageId('NEW_REPOSITORY');
      await router.push({ name: 'NEW_REPOSITORY' });
    };

    const handleSearch = () => {
      // Apply FSM effects for ACT_REPOS_SEARCH
      const action = fsmData.pages.find(p => p.id === 'REPOSITORIES_LIST').actions.find(a => a.id === 'ACT_REPOS_SEARCH');
      if (action) {
        // Logic: find matched ID if any
        const match = dataStore.repositories.find(r => r.name.toLowerCase().includes(searchQuery.value.toLowerCase()));
        const params = { item_id: match ? match.id : null };
        const nextSig = fsmRuntime.applyEffects(action, store.signature, params);
        store.signature.repos_matched_repo_id = nextSig.repos_matched_repo_id;
        store.signature.repos_has_searched = nextSig.repos_has_searched;
      }
    };

    const handleFilterPrivate = (e) => {
      showPrivate.value = e.target.checked;
      updateFilterState('ACT_REPOS_FILTER_PRIVATE_CHECKBOX');
    };

    const handleStarFilter = () => {
      updateFilterState('ACT_REPOS_FILTER_STAR_SLIDER', { widget: 'slider' });
    };

    const handleSort = (type) => {
      currentSort.value = type;
      sortDropdownOpen.value = false;
      updateFilterState('ACT_REPOS_SORT', { widget: 'dropdown' });
    };

    const updateFilterState = (actionId, params = {}) => {
       const action = fsmData.pages.find(p => p.id === 'REPOSITORIES_LIST').actions.find(a => a.id === actionId);
       if (action) {
           const nextSig = fsmRuntime.applyEffects(action, store.signature, params);
           store.signature.repos_list_filters_applied = nextSig.repos_list_filters_applied;
       }
    };

    const openRepo = async (repo) => {
      let actionId = 'ACT_REPOS_OPEN_ANY';
      if (isMatched(repo)) actionId = 'ACT_REPOS_OPEN_MATCHED';
      else if (isFiltered(repo)) actionId = 'ACT_REPOS_OPEN_FILTERED';

      const action = fsmData.pages.find(p => p.id === 'REPOSITORIES_LIST').actions.find(a => a.id === actionId);
      
      // Check preconditions
      if (action && fsmRuntime.checkPreconditions(action, store.signature)) {
         const params = { item_id: repo.id };
         const nextSig = fsmRuntime.applyEffects(action, store.signature, params);
         store.signature.repos_selected_repo_id = nextSig.repos_selected_repo_id;
         
         // Clear flags as per effects
         if (nextSig.repos_list_filters_applied === null) store.signature.repos_list_filters_applied = null;
         if (nextSig.repos_has_searched === null) store.signature.repos_has_searched = null;
         if (nextSig.repos_viewport_anchor_id === null) store.signature.repos_viewport_anchor_id = null;

         store.setCurrentPageId('REPOSITORY_DETAIL');
         await router.push({ name: 'REPOSITORY_DETAIL', params: { item_id: repo.id } });
      } else {
        // Fallback if preconditions fail (e.g., direct click without search)
        // ACT_REPOS_OPEN_ANY usually has weak preconditions or logic flow might allow direct
        // For robustness, if specific action fails, try generic open
        store.signature.repos_selected_repo_id = repo.id;
        store.setCurrentPageId('REPOSITORY_DETAIL');
        await router.push({ name: 'REPOSITORY_DETAIL', params: { item_id: repo.id } });
      }
    };

    return {
      filteredRepos,
      searchQuery,
      starFilter,
      sortDropdownOpen,
      currentSort,
      handleSearch,
      handleFilterPrivate,
      handleStarFilter,
      handleSort,
      openRepo,
      isMatched,
      isFiltered,
      goHome,
      goToNewRepo
    };
  }
}
</script>