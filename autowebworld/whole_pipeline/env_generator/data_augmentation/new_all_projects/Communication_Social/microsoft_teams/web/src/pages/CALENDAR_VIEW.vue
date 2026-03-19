<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <header class="bg-[#6264A7] text-white p-4 shadow-md flex justify-between items-center z-20">
      <div class="font-bold text-lg flex items-center">
        <button id="calendar-back-home" @click="goHome" class="mr-4 hover:bg-[#464775] p-1 rounded">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        Calendar
      </div>
      <div class="flex items-center gap-4">
        <!-- Meet Now ACT_CALENDAR_MEET_NOW -->
        <button id="meet-now-button" @click="meetNow" class="bg-transparent border border-white text-white px-4 py-2 rounded font-medium hover:bg-white/10 transition-colors flex items-center gap-2">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
          Meet Now
        </button>
        <!-- New Meeting ACT_CALENDAR_NEW_MEETING -->
        <button id="new-meeting-button" @click="newMeeting" class="bg-white text-[#6264A7] px-4 py-2 rounded font-medium hover:bg-gray-100 transition-colors">
          + New meeting
        </button>
      </div>
    </header>

    <div class="flex-1 flex overflow-hidden">
      <!-- Sidebar Filters -->
      <aside class="w-64 bg-white border-r border-gray-200 p-4 flex flex-col gap-6 overflow-y-auto">
        <div>
          <h3 class="font-semibold text-gray-700 mb-2">View Settings</h3>
          <!-- Checkbox Filter ACT_CALENDAR_FILTER_CHECKBOX -->
          <div class="flex items-center gap-2 mb-4">
            <input 
              id="filter-my-meetings-checkbox"
              type="checkbox" 
              v-model="myMeetingsOnly"
              class="w-4 h-4 text-[#6264A7] rounded focus:ring-[#6264A7]"
            />
            <label for="filter-my-meetings-checkbox" class="text-sm text-gray-600">My meetings only</label>
          </div>

          <!-- Slider Filter ACT_CALENDAR_FILTER_SLIDER -->
          <div class="mb-4">
            <label class="text-sm text-gray-600 block mb-1">Duration: > {{ minDuration }} mins</label>
            <input 
              id="calendar-range-slider"
              type="range" 
              min="0" 
              max="120"
              step="15" 
              v-model.number="minDuration"
              class="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-[#6264A7]"
            />
          </div>
        </div>

        <div>
          <h3 class="font-semibold text-gray-700 mb-2">Sort By</h3>
          <!-- Sort Dropdown ACT_CALENDAR_FILTER_SORT -->
          <div id="calendar-sort-dropdown" class="relative">
            <div 
              @click="toggleSort"
              class="w-full border rounded px-3 py-2 text-sm text-gray-700 bg-white cursor-pointer flex justify-between items-center"
            >
              {{ sortBy === 'start_time' ? 'Start Time' : (sortBy === 'title' ? 'Title' : 'Select...') }}
              <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
              </svg>
            </div>
            <div v-if="sortOpen" class="absolute top-full left-0 right-0 mt-1 bg-white border rounded shadow-lg z-10">
              <div id="calendar-sort-start-time-inc" @click="setSort('start_time')" class="px-3 py-2 text-sm hover:bg-gray-100 cursor-pointer">Start Time</div>
              <div id="calendar-sort-title" @click="setSort('title')" class="px-3 py-2 text-sm hover:bg-gray-100 cursor-pointer">Title</div>
            </div>
          </div>
        </div>
      </aside>

      <!-- Main Content -->
      <main class="flex-1 p-6 overflow-y-auto bg-gray-50">
        <h2 class="text-2xl font-bold text-gray-800 mb-6">Upcoming Meetings</h2>
        
        <div class="space-y-4">
          <div 
            v-for="event in filteredEvents" 
            :key="event.id"
            class="bg-white p-4 rounded-lg shadow-sm border-l-4 border-[#6264A7] flex justify-between items-center hover:shadow-md transition-shadow"
          >
            <div>
              <h3 class="font-bold text-lg text-gray-800">{{ event.title }}</h3>
              <p class="text-sm text-gray-500 mt-1 flex items-center gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                {{ event.start_time }} - {{ event.end_time }}
              </p>
              <span class="inline-block bg-purple-100 text-[#6264A7] text-xs px-2 py-1 rounded mt-2 capitalize">{{ event.type }}</span>
            </div>
            <button class="bg-[#6264A7] text-white px-4 py-2 rounded text-sm font-medium hover:bg-[#464775]">Join</button>
          </div>
          
          <div v-if="filteredEvents.length === 0" class="text-center py-12 text-gray-500">
            No meetings found.
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
  name: 'CALENDAR_VIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const myMeetingsOnly = ref(false)
    const minDuration = ref(0)
    const sortBy = ref('')
    const sortOpen = ref(false)

    const filteredEvents = computed(() => {
      let result = dataStore.calendarEvents;

      if (myMeetingsOnly.value) {
        result = result.filter(e => e.organizer === 'me' || e.organizer === dataStore.currentUser.id);
      }

      // Duration logic (mock calculation)
      if (minDuration.value > 0) {
        // Assume mock dates parseable
        result = result.filter(e => {
            // Simplified logic: just allow all for mock or calculate
            // In real app, calculate diff between start_time and end_time
            return true; 
        });
      }

      if (sortBy.value === 'title') {
        result = [...result].sort((a, b) => a.title.localeCompare(b.title));
      } else if (sortBy.value === 'start_time') {
        result = [...result].sort((a, b) => a.start_time.localeCompare(b.start_time));
      }

      return result;
    })

    const toggleSort = () => {
      sortOpen.value = !sortOpen.value
    }

    const setSort = (type) => {
      sortBy.value = type;
      sortOpen.value = false;
      store.calendar_filters_applied = true;
    }

    const meetNow = async () => {
      store.currentPageId = 'MEET_NOW_SETUP';
      await router.push({ name: 'MEET_NOW_SETUP' });
    }

    const newMeeting = async () => {
      store.currentPageId = 'MEETING_DETAILS';
      await router.push({ name: 'MEETING_DETAILS' });
    }

    const goHome = async () => {
      store.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    }

    return {
      myMeetingsOnly,
      minDuration,
      sortBy,
      sortOpen,
      filteredEvents,
      toggleSort,
      setSort,
      meetNow,
      newMeeting,
      goHome,
      store
    }
  },
  watch: {
    myMeetingsOnly() {
      this.store.calendar_filters_applied = true;
    },
    minDuration() {
      this.store.calendar_filters_applied = true;
    }
  }
}
</script>