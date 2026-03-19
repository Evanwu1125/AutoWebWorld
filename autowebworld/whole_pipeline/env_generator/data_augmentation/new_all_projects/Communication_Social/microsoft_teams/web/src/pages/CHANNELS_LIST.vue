<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <header class="bg-[#6264A7] text-white p-4 shadow-md flex justify-between items-center z-20">
      <div class="font-bold text-lg flex items-center">
        <button id="back-to-teams" @click="goBack" class="mr-4 hover:bg-[#464775] p-1 rounded">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        {{ currentTeam?.name || 'Team Channels' }}
      </div>
      <div class="flex items-center gap-4">
        <!-- Search ACT_CHANNELS_SEARCH -->
        <div class="relative">
          <input 
            id="channels-search-input"
            type="text" 
            v-model="searchQuery"
            @keypress.enter="handleSearch"
            placeholder="Search channels..."
            class="pl-10 pr-4 py-2 rounded bg-[#464775] text-white placeholder-gray-300 border-none focus:ring-2 focus:ring-white/50 w-64"
          />
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-300 absolute left-3 top-2.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>
      </div>
    </header>

    <div class="flex-1 flex overflow-hidden">
      <!-- Sidebar Filters -->
      <aside class="w-64 bg-white border-r border-gray-200 p-4 flex flex-col gap-6 overflow-y-auto">
        <div>
          <h3 class="font-semibold text-gray-700 mb-2">Filters</h3>
          <!-- Checkbox Filter ACT_CHANNELS_FILTER_CHECKBOX -->
          <div class="flex items-center gap-2 mb-4">
            <input 
              id="filter-show-private-checkbox"
              type="checkbox" 
              v-model="showPrivateOnly"
              class="w-4 h-4 text-[#6264A7] rounded focus:ring-[#6264A7]"
            />
            <label for="filter-show-private-checkbox" class="text-sm text-gray-600">Show private only</label>
          </div>

          <!-- Slider Filter ACT_CHANNELS_FILTER_SLIDER -->
          <div class="mb-4">
            <label class="text-sm text-gray-600 block mb-1">Activity Level: >= {{ minActivity }}%</label>
            <input 
              id="channel-activity-slider"
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
          <!-- Sort Dropdown ACT_CHANNELS_FILTER_SORT -->
          <div id="channels-sort-dropdown" class="relative">
            <div 
              @click="toggleSort"
              class="w-full border rounded px-3 py-2 text-sm text-gray-700 bg-white cursor-pointer flex justify-between items-center"
            >
              {{ sortBy === 'unread' ? 'Unread Messages' : (sortBy === 'alphabetical' ? 'Alphabetical' : 'Select...') }}
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            <div v-if="sortOpen" class="absolute top-full left-0 right-0 mt-1 bg-white border rounded shadow-lg z-10">
              <div id="channels-sort-unread-desc" @click="setSort('unread')" class="px-3 py-2 text-sm hover:bg-gray-100 cursor-pointer">Unread Messages</div>
              <div id="channels-sort-alpha" @click="setSort('alphabetical')" class="px-3 py-2 text-sm hover:bg-gray-100 cursor-pointer">Alphabetical</div>
            </div>
          </div>
        </div>
      </aside>

      <!-- Main Content -->
      <main id="channels-list-container" class="flex-1 p-6 overflow-y-auto bg-gray-50">
        <h2 class="text-2xl font-bold text-gray-800 mb-6">Channels</h2>
        
        <div id="channels-list" class="flex flex-col gap-3">
          <div 
            v-for="channel in filteredChannels" 
            :key="channel.id"
            :class="`data-id-${channel.id} bg-white p-4 rounded-lg shadow-sm hover:shadow-md transition-all cursor-pointer border-l-4 border-transparent hover:border-[#6264A7] flex justify-between items-center group ${getChannelClass(channel)}`"
            @click="openChannel(channel)"
          >
            <div class="flex items-center gap-4">
              <div class="bg-gray-100 p-2 rounded text-gray-500 group-hover:bg-purple-50 group-hover:text-[#6264A7] transition-colors">
                <span v-if="channel.type === 'private'">🔒</span>
                <span v-else>#</span>
              </div>
              <div>
                <h3 class="font-bold text-gray-800 group-hover:text-[#6264A7] transition-colors">{{ channel.name }}</h3>
                <p class="text-xs text-gray-500 mt-1">Activity: {{ channel.activity }}%</p>
              </div>
            </div>

            <div v-if="channel.unread > 0" class="bg-red-500 text-white text-xs font-bold px-2 py-1 rounded-full">
              {{ channel.unread }}
            </div>
          </div>
          
          <!-- Empty State -->
          <div v-if="filteredChannels.length === 0" class="flex flex-col items-center justify-center p-12 text-gray-500">
            <p>No channels found.</p>
          </div>
        </div>
      </main>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'CHANNELS_LIST',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const teamId = route.params.teamId
    const currentTeam = computed(() => dataStore.teams.find(t => t.id === teamId))

    const searchQuery = ref('')
    const showPrivateOnly = ref(false)
    const minActivity = ref(0)
    const sortBy = ref('')
    const sortOpen = ref(false)

    // Filter Logic
    const filteredChannels = computed(() => {
      let result = dataStore.channels.filter(c => c.teamId === teamId);

      // Filter by Search (Action: ACT_CHANNELS_SEARCH)
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase();
        result = result.filter(c => c.name.toLowerCase().includes(q));
      }

      // Filter by Private (Action: ACT_CHANNELS_FILTER_CHECKBOX)
      if (showPrivateOnly.value) {
        result = result.filter(c => c.type === 'private');
      }

      // Filter by Slider (Action: ACT_CHANNELS_FILTER_SLIDER)
      if (minActivity.value > 0) {
        result = result.filter(c => c.activity >= minActivity.value);
      }

      // Sort (Action: ACT_CHANNELS_FILTER_SORT)
      if (sortBy.value === 'alphabetical') {
        result = [...result].sort((a, b) => a.name.localeCompare(b.name));
      } else if (sortBy.value === 'unread') {
        result = [...result].sort((a, b) => b.unread - a.unread);
      }

      return result;
    })

    const handleSearch = () => {
      store.channels_list_has_searched = true;
      store.matched_channel_id = filteredChannels.value.length > 0 ? filteredChannels.value[0].id : null;
    }

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value
    }

    const setSort = (type) => {
      sortBy.value = type;
      sortOpen.value = false;
      store.channels_list_filters_applied = true;
    }

    const getChannelClass = (channel) => {
      let classes = 'channel-row-visible ';
      if (store.channels_list_filters_applied) classes += 'channel-row-filtered ';
      if (store.channels_list_has_searched) classes += 'channel-row-matched ';
      return classes;
    }

    const openChannel = async (channel) => {
      store.selected_channel_id = channel.id;
      // Clear flags
      store.channels_list_filters_applied = null;
      store.channels_list_has_searched = null;
      store.channels_list_viewport_anchor_id = null;
      
      store.currentPageId = 'CHANNEL_POST_COMPOSE';
      await router.push({ name: 'CHANNEL_POST_COMPOSE', params: { teamId, channelId: channel.id } });
    }

    const goBack = async () => {
      store.currentPageId = 'TEAMS_LIST';
      await router.push({ name: 'TEAMS_LIST' });
    }

    return {
      currentTeam,
      searchQuery,
      showPrivateOnly,
      minActivity,
      sortBy,
      sortOpen,
      filteredChannels,
      handleSearch,
      toggleSort,
      setSort,
      getChannelClass,
      openChannel,
      goBack,
      store
    }
  },
  watch: {
    showPrivateOnly() {
      this.store.channels_list_filters_applied = true;
    },
    minActivity() {
      this.store.channels_list_filters_applied = true;
    }
  }
}
</script>