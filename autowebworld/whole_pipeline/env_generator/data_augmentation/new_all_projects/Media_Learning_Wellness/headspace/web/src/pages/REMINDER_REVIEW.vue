<template>
  <div class="min-h-screen bg-[#FDFBF7] flex flex-col items-center py-10 px-4">
    <div class="bg-white rounded-3xl shadow-xl w-full max-w-lg border border-gray-100 p-8 text-center">
      
      <h1 class="text-2xl font-bold text-gray-800 mb-8">Confirm Reminder</h1>

      <div class="space-y-6 mb-10 text-left bg-gray-50 p-6 rounded-2xl">
        <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Label</label>
          <p class="text-lg font-medium text-gray-800">{{ label || 'Not set' }}</p>
        </div>
        <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Time</label>
          <p class="text-lg font-medium text-gray-800">{{ time || 'Not set' }}</p>
        </div>
      </div>

      <div class="flex flex-col gap-4">
        <button id="reminder-confirm-button" 
                @click="confirmReminder"
                class="w-full bg-orange-500 hover:bg-orange-600 text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all transform hover:scale-[1.02]">
          Set Reminder
        </button>
        
        <button id="reminder-review-back-form" 
                @click="goBack"
                class="w-full text-gray-500 hover:text-gray-700 font-medium py-2">
          Make Changes
        </button>
      </div>

    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'REMINDER_REVIEW',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const label = computed(() => signatureStore.reminder_label_text);
    const time = computed(() => signatureStore.reminder_time_choice);

    const confirmReminder = async () => {
      signatureStore.reminder_set = true;
      await router.push({ name: 'REMINDER_SET_SUCCESS' });
    };

    const goBack = async () => {
      await router.push({ name: 'REMINDER_FORM' });
    };

    return {
      label,
      time,
      confirmReminder,
      goBack
    };
  }
}
</script>