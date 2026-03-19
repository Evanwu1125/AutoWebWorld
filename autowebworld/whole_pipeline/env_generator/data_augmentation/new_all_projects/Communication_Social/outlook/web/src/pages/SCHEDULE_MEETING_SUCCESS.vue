<template>
  <div class="h-screen flex flex-col items-center justify-center bg-[#F0F2F5] p-4 text-center">
      <div class="w-24 h-24 bg-teal-100 rounded-full flex items-center justify-center mb-6">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
      </div>
      <h1 class="text-3xl font-bold text-gray-900 mb-2">Meeting Scheduled</h1>
      <p class="text-gray-600 mb-8 max-w-md">Your event has been added to the calendar.</p>
      
      <div class="flex gap-4">
          <button id="schedule-success-home" class="px-6 py-2 border border-gray-300 bg-white rounded-md hover:bg-gray-50 font-medium text-gray-700 transition-colors" @click="goHome">
              Back to Home
          </button>
          <button id="schedule-success-calendar" class="px-6 py-2 bg-[#0078D4] text-white rounded-md hover:bg-[#005A9E] font-medium transition-colors shadow-sm" @click="goCalendar">
              Go to Calendar
          </button>
      </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'SCHEDULE_MEETING_SUCCESS',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const goHome = async () => {
        await signatureStore.handleAction('ACT_SCHEDULE_SUCCESS_GO_HOME');
        router.push({ name: 'HOME' });
    };

    const goCalendar = async () => {
        await signatureStore.handleAction('ACT_SCHEDULE_SUCCESS_GO_CALENDAR');
        router.push({ name: 'CALENDAR_MONTH' });
    };

    return { goHome, goCalendar };
  }
}
</script>