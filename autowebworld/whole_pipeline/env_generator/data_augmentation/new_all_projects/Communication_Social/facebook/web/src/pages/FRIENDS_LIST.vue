<template>
  <div class="min-h-screen bg-gray-100 pb-10">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20 h-16 flex items-center px-4 justify-between">
      <div class="flex items-center gap-4">
        <button 
          id="back-home-from-friends"
          @click="goBack"
          class="p-2 hover:bg-gray-100 rounded-full transition-colors"
        >
          <svg class="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        <h1 class="text-xl font-bold text-gray-900">Friends</h1>
      </div>
      <button 
        id="friend-suggestions-link"
        @click="goToSuggestions"
        class="text-blue-600 font-semibold text-sm hover:bg-blue-50 px-3 py-2 rounded-lg transition-colors"
      >
        Suggestions
      </button>
    </header>

    <div class="max-w-4xl mx-auto px-4 py-6 flex flex-col md:flex-row gap-6">
      <!-- Sidebar Filters -->
      <div class="w-full md:w-64 space-y-6 flex-shrink-0">
        <h2 class="text-xl font-bold text-gray-900">Filters</h2>
        
        <!-- Sort Dropdown -->
        <div class="relative">
          <button 
            id="friends-sort-dropdown"
            @click="toggleSort"
            class="w-full flex items-center justify-between bg-white px-4 py-2 rounded-lg shadow-sm border border-gray-200 text-sm font-medium hover:bg-gray-50 transition-colors"
          >
            <span>Sort: {{ sortOption === 'name' ? 'Name' : 'Recently Added' }}</span>
            <svg class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          <div v-if="sortOpen" class="absolute left-0 mt-1 w-full bg-white rounded-lg shadow-lg py-1 z-10 ring-1 ring-black ring-opacity-5">
              <div
                id="friends-sort-option-name-inc"
                @click="selectSort('name')"
                class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
              >
                Name
              </div>
            <div 
              id="friends-sort-option-recently-added"
              @click="selectSort('recently_added')"
              class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
            >
              Recently Added
            </div>
          </div>
        </div>

        <!-- Mutual Friends Filter -->
        <label class="flex items-center gap-3 cursor-pointer bg-white p-3 rounded-lg shadow-sm border border-gray-200 hover:bg-gray-50 transition-colors">
          <div 
            id="filter-mutual-friends-checkbox"
            class="w-5 h-5 border-2 border-gray-300 rounded flex items-center justify-center transition-colors"
            :class="{ 'bg-blue-600 border-blue-600': filters.mutualOnly }"
            @click.prevent="toggleMutualOnly"
          >
            <svg v-if="filters.mutualOnly" class="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <span class="text-sm font-medium text-gray-700">Mutual Friends Only</span>
        </label>

        <!-- Activity Slider -->
        <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
          <label class="block text-sm font-medium text-gray-700 mb-2">Activity Level</label>
          <input 
            id="friends-activity-slider"
            type="range" 
            min="0" 
            max="100" 
            step="25"
            v-model="filters.activity"
            @input="applyFilters"
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
          <div class="flex justify-between text-xs text-gray-500 mt-1">
            <span>Low</span>
            <span>High</span>
          </div>
        </div>
      </div>

      <!-- Friends Grid -->
      <div class="flex-1">
        <div id="friends-list" class="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div 
            v-for="friend in filteredFriends" 
            :key="friend.id" 
            class="bg-white rounded-lg shadow-sm border border-gray-200 p-4 flex items-center gap-4 hover:shadow-md transition-shadow cursor-pointer"
            :class="{ 'friend-visible': true, 'friend-filtered': isFiltered }"
            :data-id-value="friend.id"
            @click="openProfile(friend)"
          >
            <img :src="friend.avatar" class="h-20 w-20 rounded-full object-cover border border-gray-100" :alt="friend.name" />
            <div class="flex-1 min-w-0">
              <h3 
                class="text-base font-semibold text-gray-900 truncate"
                :class="`data-id-${friend.id}`"
              >
                {{ friend.name }}
              </h3>
              <p class="text-sm text-gray-500">{{ friend.mutual }} mutual friends</p>
            </div>
            <button class="text-gray-400 hover:bg-gray-100 p-2 rounded-full">
              <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h.01M12 12h.01M19 12h.01M6 12a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0z" />
              </svg>
            </button>
          </div>
          
          <div v-if="filteredFriends.length === 0" class="col-span-full text-center py-10 bg-white rounded-lg border border-gray-200 border-dashed">
            <p class="text-gray-500">No friends match your filters.</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';
import { orderBy } from 'lodash-es';

export default {
  name: 'FRIENDS_LIST',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const sortOpen = ref(false);
    const sortOption = ref('name');
    const filters = ref({
      mutualOnly: false,
      activity: 0
    });

    const isFiltered = computed(() => {
      return filters.value.mutualOnly || filters.value.activity > 0;
    });

    const filteredFriends = computed(() => {
      let result = [...dataStore.friends];

      if (filters.value.mutualOnly) {
        result = result.filter(f => f.mutual > 10); // Mock logic for mutual > 10
      }
      
      // Mock activity filter
      if (filters.value.activity > 50) {
        // Assume friends with odd IDs are "highly active"
        // This is just deterministic mock logic
        result = result.filter(f => f.id.length % 2 !== 0);
      }

      if (sortOption.value === 'name') {
        result = orderBy(result, ['name'], ['asc']);
      } else {
        // Recently Added -> Sort by ID descending (mock)
        result = orderBy(result, ['id'], ['desc']);
      }

      return result;
    });

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value;
    };

    const selectSort = (option) => {
      sortOption.value = option;
      sortOpen.value = false;
      signatureStore.friend_requests_list_filters_applied = true; // FSM Effect
    };

    const toggleMutualOnly = () => {
      filters.value.mutualOnly = !filters.value.mutualOnly;
      signatureStore.friend_requests_list_filters_applied = true; // FSM Effect
    };

    const applyFilters = () => {
      signatureStore.friend_requests_list_filters_applied = true; // FSM Effect
    };

    const openProfile = async (friend) => {
      signatureStore.selected_user_id = friend.id;
      // Clear viewport anchor (FSM Effect)
      signatureStore.friend_requests_list_viewport_anchor_id = null;
      // Clear filters (FSM Effect for filtered click)
      if (isFiltered.value) {
        signatureStore.friend_requests_list_filters_applied = null;
      }
      
      await router.push({ name: 'PROFILE_TIMELINE', params: { id: friend.id } });
    };

    const goToSuggestions = async () => {
      signatureStore.currentPageId = 'FRIEND_SUGGESTIONS';
      await router.push({ name: 'FRIEND_SUGGESTIONS' });
    };

    const goBack = async () => {
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    return {
      sortOpen,
      sortOption,
      filters,
      isFiltered,
      filteredFriends,
      toggleSort,
      selectSort,
      toggleMutualOnly,
      applyFilters,
      openProfile,
      goToSuggestions,
      goBack
    };
  }
}
</script>