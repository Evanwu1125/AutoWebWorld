<template>
  <div class="min-h-screen bg-[#FDFBF7] flex flex-col items-center py-10 px-4">
    <div class="bg-white rounded-3xl shadow-xl w-full max-w-lg border border-gray-100 p-8 text-center">
      
      <h1 class="text-2xl font-bold text-gray-800 mb-8">Confirm Enrollment</h1>

      <div class="space-y-6 mb-10 text-left bg-gray-50 p-6 rounded-2xl">
        <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Motivation</label>
          <p class="text-lg font-medium text-gray-800">{{ reason || 'Not set' }}</p>
        </div>
        <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Frequency</label>
          <p class="text-lg font-medium text-gray-800">{{ frequency || 'Not set' }}</p>
        </div>
      </div>

      <div class="flex flex-col gap-4">
        <button id="enroll-confirm-button" 
                @click="confirmEnroll"
                class="w-full bg-orange-500 hover:bg-orange-600 text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all transform hover:scale-[1.02]">
          Confirm & Enroll
        </button>
        
        <button id="enroll-review-back-form" 
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
  name: 'COURSE_ENROLL_REVIEW',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const reason = computed(() => signatureStore.enroll_reason_text);
    const frequency = computed(() => signatureStore.enroll_frequency_choice);

    const confirmEnroll = async () => {
      signatureStore.course_enrolled = true;
      await router.push({ name: 'COURSE_ENROLLED_SUCCESS' });
    };

    const goBack = async () => {
      await router.push({ name: 'COURSE_ENROLL_FORM' });
    };

    return {
      reason,
      frequency,
      confirmEnroll,
      goBack
    };
  }
}
</script>