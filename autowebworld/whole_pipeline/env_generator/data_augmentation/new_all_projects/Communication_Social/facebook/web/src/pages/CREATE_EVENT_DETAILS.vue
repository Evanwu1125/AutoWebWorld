<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-lg overflow-hidden flex flex-col h-[500px]">
      <!-- Header -->
      <div class="relative px-4 py-3 border-b border-gray-100 flex items-center justify-center bg-white z-10">
        <h2 class="text-lg font-bold text-gray-900">Create Event</h2>
        <div 
          id="event-cancel" 
          @click="goBack"
          class="absolute right-4 top-1/2 transform -translate-y-1/2 p-1 text-gray-500 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          Cancel
        </div>
      </div>

      <!-- Form -->
      <div class="flex-1 p-6 space-y-6">
        <div>
           <label class="block text-sm font-medium text-gray-700 mb-1">Event Name</label>
           <input 
             id="event-name-input"
             type="text" 
             v-model="eventName"
             @input="handleNameInput"
             placeholder="Event Name"
             class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all"
           />
        </div>
        
        <div>
           <label class="block text-sm font-medium text-gray-700 mb-1">Location</label>
           <div class="relative">
              <input 
                id="event-location-input"
                type="text" 
                v-model="eventLocation"
                @input="handleLocationInput"
                placeholder="Where is it?"
                class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none transition-all pl-10"
              />
              <div class="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400">
                 <svg class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                 </svg>
              </div>
           </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="p-4 border-t border-gray-100 bg-gray-50">
        <button 
          id="event-next-date"
          @click="goToDate"
          :disabled="!canProceed"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-lg shadow-sm hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          Next
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'CREATE_EVENT_DETAILS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    const eventName = ref(signatureStore.event_name || '');
    const eventLocation = ref(signatureStore.event_location || '');
    
    const canProceed = computed(() => {
      return eventName.value.length > 0 && eventLocation.value.length > 0;
    });

    const handleNameInput = () => {
      signatureStore.event_name = eventName.value;
    };

    const handleLocationInput = () => {
      signatureStore.event_location = eventLocation.value;
    };

    const goToDate = async () => {
      if (canProceed.value) {
        signatureStore.currentPageId = 'CREATE_EVENT_DATE';
        await router.push({ name: 'CREATE_EVENT_DATE' });
      }
    };

    const goBack = async () => {
      // Clear
      signatureStore.event_name = null;
      signatureStore.event_location = null;
      signatureStore.currentPageId = 'EVENTS_LIST';
      await router.push({ name: 'EVENTS_LIST' });
    };

    return {
      eventName,
      eventLocation,
      canProceed,
      handleNameInput,
      handleLocationInput,
      goToDate,
      goBack
    };
  }
}
</script>