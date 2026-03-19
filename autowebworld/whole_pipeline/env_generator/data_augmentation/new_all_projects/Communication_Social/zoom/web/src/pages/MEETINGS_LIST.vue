<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm border-b border-gray-200 sticky top-0 z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <h1 class="text-xl font-bold text-gray-900">Upcoming Meetings</h1>
        <button 
          id="meetings-back-dashboard" 
          @click="goDashboard"
          class="text-blue-600 hover:text-blue-700 font-medium"
        >
          Dashboard
        </button>
      </div>
    </header>

    <main class="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
      <!-- Controls -->
      <div class="bg-white rounded-lg shadow-sm p-6 mb-6 space-y-6 md:space-y-0 md:flex md:items-end md:gap-6">
        <!-- Search -->
        <div class="flex-1">
          <label class="block text-sm font-medium text-gray-700 mb-1">Search</label>
          <div class="relative">
            <input 
              id="meetings-search-input"
              v-model="searchQuery"
              @input="handleSearch"
              @keyup.enter="handleSearchEnter"
              type="text" 
              class="w-full border border-gray-300 rounded-md pl-10 pr-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              placeholder="Search by topic or host..."
            />
            <svg class="w-5 h-5 text-gray-400 absolute left-3 top-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
          </div>
        </div>

        <!-- Filter Checkbox -->
        <div class="pb-3">
           <div class="flex items-center cursor-pointer" @click="toggleMyMeetings">
            <div 
              id="meetings-filter-my-meetings-checkbox" 
              class="w-5 h-5 rounded border border-gray-300 flex items-center justify-center mr-2 transition-colors"
              :class="{'bg-blue-600 border-blue-600': filterMyMeetings}"
            >
               <svg v-if="filterMyMeetings" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <span class="text-sm font-medium text-gray-700">My Meetings Only</span>
          </div>
        </div>

        <!-- Duration Slider -->
        <div class="w-full md:w-64">
          <div class="flex justify-between text-sm mb-1">
            <label class="font-medium text-gray-700">Min Duration</label>
            <span class="text-blue-600 font-medium">{{ minDuration }} min</span>
          </div>
          <input 
            id="meetings-duration-slider"
            type="range" 
            v-model="minDuration"
            @input="handleSlider"
            min="0" 
            max="180" 
            step="15"
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
          <div class="flex justify-between text-xs text-gray-400 mt-1">
            <span>0m</span>
            <span>180m</span>
          </div>
        </div>

        <!-- Sort -->
        <div class="relative w-full md:w-48">
          <label class="block text-sm font-medium text-gray-700 mb-1">Sort By</label>
          <button 
            id="meetings-sort-dropdown"
            @click="toggleSort"
            class="w-full bg-white border border-gray-300 rounded-md px-4 py-2 text-left flex justify-between items-center hover:bg-gray-50"
          >
            <span>{{ currentSortLabel }}</span>
            <svg class="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>
          
          <div v-if="sortOpen" class="absolute z-10 w-full bg-white border border-gray-300 rounded-md shadow-lg mt-1 right-0">
            <div 
              id="meetings-sort-upcoming" 
              @click="selectSort('upcoming')" 
              class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
            >Upcoming</div>
            <div 
              id="meetings-sort-date" 
              @click="selectSort('date')" 
              class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
            >Date</div>
            <div 
              id="meetings-sort-host" 
              @click="selectSort('host')" 
              class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
            >Host</div>
          </div>
        </div>
      </div>

      <!-- List -->
      <div 
        id="meetings-list"
        class="bg-white rounded-lg shadow-sm overflow-hidden min-h-[400px]"
        @scroll="handleScroll"
      >
        <div v-if="filteredMeetings.length === 0" class="p-10 text-center text-gray-500">
          No meetings found matching your criteria.
        </div>
        
        <div 
          v-for="meeting in filteredMeetings" 
          :key="meeting.id"
          :class="getRowClass(meeting)"
          @click="selectMeeting(meeting)"
          class="group border-b border-gray-100 last:border-0 hover:bg-blue-50 cursor-pointer transition-colors p-4 sm:p-6 flex flex-col sm:flex-row gap-4 items-start sm:items-center"
        >
          <!-- Thumbnail -->
          <div class="w-full sm:w-48 h-32 sm:h-28 bg-gray-200 rounded-lg overflow-hidden flex-shrink-0 relative">
            <img :src="meeting.image" :alt="meeting.topic" class="w-full h-full object-cover transition-transform group-hover:scale-105" />
            <div class="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-0.5 rounded">
              {{ meeting.duration }}m
            </div>
          </div>

          <!-- Info -->
          <div class="flex-grow">
            <div class="text-sm text-blue-600 font-semibold mb-1">{{ formatDate(meeting.date, meeting.time) }}</div>
            <h3 class="text-lg font-bold text-gray-900 mb-2 group-hover:text-blue-700">{{ meeting.topic }}</h3>
            <div class="flex items-center text-gray-500 text-sm">
              <div class="flex items-center mr-6">
                <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"></path></svg>
                {{ meeting.host }}
              </div>
              <div class="flex items-center">
                <svg class="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
                Zoom Meeting
              </div>
            </div>
          </div>

          <!-- Action -->
          <div class="self-end sm:self-center">
            <button class="px-4 py-2 bg-white border border-blue-600 text-blue-600 rounded-full font-medium text-sm hover:bg-blue-50 group-hover:bg-blue-600 group-hover:text-white transition-colors">
              View Details
            </button>
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
  name: 'MEETINGS_LIST',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const dataStore = useDataStore();

    const searchQuery = ref('');
    const filterMyMeetings = ref(false);
    const minDuration = ref(0);
    const sortOpen = ref(false);
    const sortBy = ref('upcoming');

    const toggleSort = () => sortOpen.value = !sortOpen.value;
    const currentSortLabel = computed(() => {
      const map = { 'upcoming': 'Upcoming', 'date': 'Date', 'host': 'Host' };
      return map[sortBy.value];
    });

    const filteredMeetings = computed(() => {
      let results = [...dataStore.meetings];

      // Search
      if (searchQuery.value) {
        const q = searchQuery.value.toLowerCase();
        results = results.filter(m => 
          m.topic.toLowerCase().includes(q) || 
          m.host.toLowerCase().includes(q)
        );
      }

      // Filter Checkbox (Mock logic: filter by host name 'Me' or similar, or just random subset for demo)
      if (filterMyMeetings.value) {
        // Since mock data hosts are diverse, let's assume 'Sarah Johnson' (first item host) is 'Me' for demo
        // Or just filter odd IDs to show change
        // results = results.filter((m, i) => i % 2 === 0); 
        // Better: filter by logged in user if we had one. 
        // Let's just simulate it by filtering hosts starting with 'S' or 'M'.
        // Actually, let's filter by duration > 60 just to show effect if checkbox logic is weak.
        // No, checkbox says "My Meetings". Let's assume my name is "John Doe".
        // Mock data doesn't have John Doe.
        // I'll just return first 5 items.
        results = results.slice(0, 5);
      }

      // Filter Slider (Duration > min)
      if (minDuration.value > 0) {
        results = results.filter(m => m.duration >= minDuration.value);
      }

      // Sort
      if (sortBy.value === 'date' || sortBy.value === 'upcoming') {
        results.sort((a, b) => new Date(a.date + 'T' + a.time) - new Date(b.date + 'T' + b.time));
      } else if (sortBy.value === 'host') {
        results.sort((a, b) => a.host.localeCompare(b.host));
      }

      return results;
    });

    const getRowClass = (meeting) => {
      const classes = [`data-id-${meeting.id}`];
      
      // Determine if matched by search
      if (store.meetings_list_has_searched && store.meetings_matched_id === meeting.id) {
        classes.push('row-matched');
      }
      // Determine if filtered
      else if (store.meetings_list_filters_applied) {
        classes.push('row-filtered');
      }
      // Default visible
      else {
        classes.push('row-visible');
      }
      
      return classes.join(' ');
    };

    const formatDate = (date, time) => {
      const d = new Date(date + 'T' + time);
      return d.toLocaleString('en-US', { weekday: 'short', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
    };

    const handleSearch = (e) => {
      // Typing action not explicitly defined separately from search button in FSM 
      // but ACT_MEETINGS_SEARCH is type+enter.
      // We'll update local ref. 
    };

    const handleSearchEnter = () => {
      // Set first match ID if any
      const firstMatch = filteredMeetings.value[0]?.id;
      if (firstMatch) {
        store.handleAction('ACT_MEETINGS_SEARCH', { item_id: firstMatch });
      }
    };

    const toggleMyMeetings = () => {
      filterMyMeetings.value = !filterMyMeetings.value;
      store.handleAction('ACT_MEETINGS_FILTER_CHECKBOX');
    };

    const handleSlider = () => {
      store.handleAction('ACT_MEETINGS_FILTER_SLIDER', { widget: 'slider' });
    };

    const selectSort = (val) => {
      sortBy.value = val;
      store.handleAction('ACT_MEETINGS_SORT', { widget: 'dropdown' });
      sortOpen.value = false;
    };

    const handleScroll = () => {
      // Debounce drag action if needed, or just trigger once.
      // FSM uses drag on list.
      // We can just simulate it via method if needed.
      // For user interaction, scrolling is natural.
      // store.handleAction('ACT_MEETINGS_SCROLL_INTO_VIEW', { item_id: ... });
    };

    const selectMeeting = async (meeting) => {
      // Determine which action to trigger based on context
      let action = 'ACT_MEETINGS_OPEN_ANY';
      
      if (store.meetings_list_has_searched && store.meetings_matched_id === meeting.id) {
        action = 'ACT_MEETINGS_OPEN_MATCHED';
      } else if (store.meetings_list_filters_applied) {
        action = 'ACT_MEETINGS_OPEN_FILTERED';
      }

      if (store.handleAction(action, { item_id: meeting.id })) {
        await router.push({ name: 'MEETING_DETAIL' });
      }
    };

    const goDashboard = async () => {
      if (store.handleAction('ACT_MEETINGS_BACK_DASHBOARD')) {
        await router.push({ name: 'DASHBOARD' });
      }
    };

    return {
      store,
      dataStore,
      searchQuery,
      filterMyMeetings,
      minDuration,
      sortOpen,
      sortBy,
      currentSortLabel,
      filteredMeetings,
      getRowClass,
      formatDate,
      handleSearch,
      handleSearchEnter,
      toggleMyMeetings,
      handleSlider,
      toggleSort,
      selectSort,
      handleScroll,
      selectMeeting,
      goDashboard
    };
  }
}
</script>