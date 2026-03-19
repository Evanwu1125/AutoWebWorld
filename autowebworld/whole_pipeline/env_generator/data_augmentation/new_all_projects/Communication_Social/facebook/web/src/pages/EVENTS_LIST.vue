<template>
  <div class="min-h-screen bg-gray-100 pb-10">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20 h-16 flex items-center px-4 justify-between">
      <div class="flex items-center gap-4">
        <button 
          id="events-back-home"
          @click="goBack"
          class="p-2 hover:bg-gray-100 rounded-full transition-colors"
        >
          <svg class="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        <h1 class="text-xl font-bold text-gray-900">Events</h1>
      </div>
      <button 
        id="create-event-button"
        @click="createEvent"
        class="bg-blue-600 text-white px-4 py-2 rounded-lg font-semibold text-sm hover:bg-blue-700 transition-colors flex items-center gap-2"
      >
        <span class="text-lg">+</span> Create Event
      </button>
    </header>

    <div class="max-w-6xl mx-auto px-4 py-6 flex flex-col md:flex-row gap-6">
      <!-- Sidebar Filters -->
      <div class="w-full md:w-64 space-y-6 flex-shrink-0">
        <h2 class="text-xl font-bold text-gray-900">Find Events</h2>
        
        <!-- Sort Dropdown -->
        <div class="relative">
           <button 
             id="events-sort-dropdown"
             @click="toggleSort"
             class="w-full flex items-center justify-between bg-white px-4 py-2 rounded-lg shadow-sm border border-gray-200 text-sm font-medium hover:bg-gray-50 transition-colors"
           >
             <span>Sort: {{ sortOption === 'upcoming' ? 'Upcoming' : 'Past' }}</span>
             <svg class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
             </svg>
           </button>
           
           <div v-if="sortOpen" class="absolute left-0 mt-1 w-full bg-white rounded-lg shadow-lg py-1 z-10 ring-1 ring-black ring-opacity-5">
             <div 
               id="events-sort-option-upcoming"
               @click="selectSort('upcoming')"
               class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
             >
               Upcoming
             </div>
             <div 
               id="events-sort-option-past"
               @click="selectSort('past')"
               class="px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
             >
               Past
             </div>
           </div>
        </div>

        <!-- Checkbox Filter -->
        <label class="flex items-center gap-3 cursor-pointer bg-white p-3 rounded-lg shadow-sm border border-gray-200 hover:bg-gray-50 transition-colors">
          <div 
            id="events-filter-going-checkbox"
            class="w-5 h-5 border-2 border-gray-300 rounded flex items-center justify-center transition-colors"
            :class="{ 'bg-blue-600 border-blue-600': filters.goingOnly }"
            @click.prevent="toggleGoingOnly"
          >
            <svg v-if="filters.goingOnly" class="w-3 h-3 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
            </svg>
          </div>
          <span class="text-sm font-medium text-gray-700">Going</span>
        </label>

        <!-- Date Slider -->
        <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
          <label class="block text-sm font-medium text-gray-700 mb-2">Date Range</label>
          <input 
            id="events-date-range-slider"
            type="range" 
            min="0" 
            max="100" 
            step="25"
            v-model="filters.dateRange"
            @input="applyFilters"
            class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
          <div class="flex justify-between text-xs text-gray-500 mt-1">
            <span>Any time</span>
            <span>This Week</span>
          </div>
        </div>
      </div>

      <!-- Events Grid -->
      <div class="flex-1">
        <div id="events-list" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <div 
            v-for="event in filteredEvents" 
            :key="event.id" 
            class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden flex flex-col hover:shadow-md transition-shadow cursor-pointer h-full"
            :class="{ 'event-visible': true, 'event-filtered': isFiltered }"
            :data-id-value="event.id"
            @click="openEvent(event)"
          >
            <!-- Event Image -->
            <div class="h-40 w-full relative">
              <img :src="event.image" class="w-full h-full object-cover" :alt="event.name" />
              <div class="absolute top-2 left-2 bg-white rounded-md px-2 py-1 shadow-sm text-center min-w-[50px]">
                <span class="block text-xs font-bold text-red-500 uppercase">{{ new Date(event.date).toLocaleString('default', { month: 'short' }) }}</span>
                <span class="block text-lg font-bold text-gray-900 leading-none">{{ new Date(event.date).getDate() }}</span>
              </div>
            </div>

            <!-- Event Content -->
            <div class="p-4 flex flex-col flex-1">
              <div class="mb-1 text-xs font-semibold text-red-500 uppercase">{{ new Date(event.date).toDateString() }}</div>
              <h3 
                class="text-lg font-bold text-gray-900 mb-1 leading-tight line-clamp-2"
                :class="`data-id-${event.id}`"
              >
                {{ event.name }}
              </h3>
              <p class="text-sm text-gray-500 mb-4">{{ event.location }}</p>
              
              <div class="mt-auto flex items-center justify-between text-xs text-gray-500">
                 <span>{{ event.attending }} people going</span>
                 <div class="flex -space-x-1">
                    <div class="h-6 w-6 rounded-full bg-gray-300 border-2 border-white"></div>
                    <div class="h-6 w-6 rounded-full bg-gray-400 border-2 border-white"></div>
                    <div class="h-6 w-6 rounded-full bg-gray-500 border-2 border-white flex items-center justify-center text-[8px] text-white font-bold">+99</div>
                 </div>
              </div>

              <div class="mt-4 flex gap-2">
                 <button class="flex-1 border border-gray-300 rounded-md py-1.5 text-sm font-semibold text-gray-700 hover:bg-gray-50 transition-colors">Interested</button>
                 <button class="flex-1 bg-blue-100 rounded-md py-1.5 text-sm font-semibold text-blue-700 hover:bg-blue-200 transition-colors">Going</button>
              </div>
            </div>
          </div>
          
          <div v-if="filteredEvents.length === 0" class="col-span-full text-center py-20 bg-white rounded-lg border border-gray-200 border-dashed">
            <p class="text-gray-500 text-lg">No events match your filters.</p>
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
  name: 'EVENTS_LIST',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const sortOpen = ref(false);
    const sortOption = ref('upcoming');
    const filters = ref({
      goingOnly: false,
      dateRange: 0
    });

    const isFiltered = computed(() => {
      return filters.value.goingOnly || filters.value.dateRange > 0;
    });

    const filteredEvents = computed(() => {
      let result = [...dataStore.events];

      if (filters.value.goingOnly) {
         // Mock: odd IDs are "going"
         result = result.filter(e => e.id.length % 2 !== 0);
      }
      
      if (filters.value.dateRange > 50) {
         // Mock: only recent/this week
         // Just filter a subset for demo
         result = result.slice(0, 5);
      }

      if (sortOption.value === 'upcoming') {
         // Sort by date asc
         result = orderBy(result, ['date'], ['asc']);
      } else {
         // Past -> date desc
         result = orderBy(result, ['date'], ['desc']);
      }

      return result;
    });

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value;
    };

    const selectSort = (option) => {
      sortOption.value = option;
      sortOpen.value = false;
      signatureStore.events_list_filters_applied = true; // FSM Effect
    };

    const toggleGoingOnly = () => {
      filters.value.goingOnly = !filters.value.goingOnly;
      signatureStore.events_list_filters_applied = true; // FSM Effect
    };

    const applyFilters = () => {
      signatureStore.events_list_filters_applied = true; // FSM Effect
    };

    const openEvent = async (event) => {
      signatureStore.selected_event_id = event.id;
      // Clear anchor
      signatureStore.events_list_viewport_anchor_id = null;
      // Clear filters
      if (isFiltered.value) {
        signatureStore.events_list_filters_applied = null;
      }
      
      await router.push({ name: 'EVENT_DETAIL', params: { id: event.id } });
    };

    const createEvent = async () => {
      signatureStore.currentPageId = 'CREATE_EVENT_DETAILS';
      await router.push({ name: 'CREATE_EVENT_DETAILS' });
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
      filteredEvents,
      toggleSort,
      selectSort,
      toggleGoingOnly,
      applyFilters,
      openEvent,
      createEvent,
      goBack
    };
  }
}
</script>