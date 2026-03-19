<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-lg overflow-hidden flex flex-col h-[500px]">
      <!-- Header -->
      <div class="relative px-4 py-3 border-b border-gray-100 flex items-center justify-center bg-white z-10">
        <div 
          id="event-back-date"
          @click="goBackDate"
          class="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          <svg class="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
        </div>
        <h2 class="text-lg font-bold text-gray-900">Review Event</h2>
        <div 
          id="event-cancel-from-review" 
          @click="cancelReview"
          class="absolute right-4 top-1/2 transform -translate-y-1/2 p-1 text-gray-500 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          Cancel
        </div>
      </div>

      <!-- Content -->
      <div class="flex-1 p-6 space-y-6">
        <div class="h-32 bg-gray-200 rounded-lg overflow-hidden relative">
           <img src="/images/Event.jpg" class="w-full h-full object-cover" alt="Event Cover" />
           <div class="absolute bottom-2 left-2 bg-white px-2 py-1 rounded text-xs font-bold text-gray-700">Preview</div>
        </div>
        
        <div>
           <h1 class="text-2xl font-bold text-gray-900">{{ eventName }}</h1>
           <div class="mt-4 space-y-3">
              <div class="flex items-center gap-3">
                 <div class="bg-red-50 p-2 rounded-full text-red-500"><svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg></div>
                 <div>
                    <div class="font-medium text-gray-900">{{ eventDate ? new Date(eventDate).toLocaleString() : '' }}</div>
                 </div>
              </div>
              <div class="flex items-center gap-3">
                 <div class="bg-blue-50 p-2 rounded-full text-blue-500"><svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg></div>
                 <div>
                    <div class="font-medium text-gray-900">{{ eventLocation }}</div>
                 </div>
              </div>
           </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="p-4 border-t border-gray-100 bg-gray-50">
        <button 
          id="event-create-button"
          @click="createEvent"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-sm hover:bg-blue-700 transition-colors"
        >
          Create Event
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'CREATE_EVENT_REVIEW',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();
    
    const eventName = computed(() => signatureStore.event_name);
    const eventLocation = computed(() => signatureStore.event_location);
    const eventDate = computed(() => signatureStore.event_date);

    const createEvent = async () => {
      // Mock create logic
      const newEvent = {
        id: `event_new_${Date.now()}`,
        name: eventName.value,
        location: eventLocation.value,
        date: eventDate.value,
        image: '/images/Event.jpg',
        attending: 1,
        time: '10:00 AM' // default
      };
      
      dataStore.events.unshift(newEvent);
      
      signatureStore.currentPageId = 'EVENT_CREATE_SUCCESS';
      await router.push({ name: 'EVENT_CREATE_SUCCESS' });
    };

    const goBackDate = async () => {
      signatureStore.currentPageId = 'CREATE_EVENT_DATE';
      await router.push({ name: 'CREATE_EVENT_DATE' });
    };

    const cancelReview = async () => {
      // Clear
      signatureStore.event_name = null;
      signatureStore.event_location = null;
      signatureStore.event_date = null;
      signatureStore.currentPageId = 'EVENTS_LIST';
      await router.push({ name: 'EVENTS_LIST' });
    };

    return {
      eventName,
      eventLocation,
      eventDate,
      createEvent,
      goBackDate,
      cancelReview
    };
  }
}
</script>