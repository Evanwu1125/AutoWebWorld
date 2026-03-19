<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-lg overflow-hidden flex flex-col h-auto min-h-[500px]">
      <!-- Header -->
      <div class="relative px-4 py-3 border-b border-gray-100 flex items-center justify-center bg-white z-10">
        <div 
          id="event-back-details"
          @click="goBack"
          class="absolute left-4 top-1/2 transform -translate-y-1/2 p-2 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          <svg class="h-6 w-6 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
        </div>
        <h2 class="text-lg font-bold text-gray-900">Date and Time</h2>
      </div>

      <!-- Date Picker -->
      <div class="flex-1 p-6 flex flex-col items-center justify-center">
        <DateTimePicker 
          id="date-picker"
          @change="handleDateChange"
        />
        
        <div v-if="selectedDate" class="mt-4 text-center">
            <span class="text-gray-500 text-sm">Selected:</span>
            <div class="text-lg font-bold text-gray-900">{{ new Date(selectedDate).toLocaleString() }}</div>
        </div>
      </div>

      <!-- Footer -->
      <div class="p-4 border-t border-gray-100 bg-gray-50">
        <button 
          id="event-next-review"
          @click="goToReview"
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
import DateTimePicker from '../components/widgets/DateTimePicker.vue';

export default {
  name: 'CREATE_EVENT_DATE',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    const selectedDate = ref(signatureStore.event_date || null);
    
    const canProceed = computed(() => {
      return selectedDate.value !== null;
    });

    const handleDateChange = (date) => {
      selectedDate.value = date;
      signatureStore.event_date = date; // FSM Effect
    };

    const goToReview = async () => {
      if (canProceed.value) {
        signatureStore.currentPageId = 'CREATE_EVENT_REVIEW';
        await router.push({ name: 'CREATE_EVENT_REVIEW' });
      }
    };

    const goBack = async () => {
      signatureStore.currentPageId = 'CREATE_EVENT_DETAILS';
      await router.push({ name: 'CREATE_EVENT_DETAILS' });
    };

    return {
      selectedDate,
      canProceed,
      handleDateChange,
      goToReview,
      goBack
    };
  }
}
</script>