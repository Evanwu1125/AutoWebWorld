<template>
  <div class="min-h-screen bg-[#FDFBF7] flex flex-col items-center py-10 px-4">
    <div class="bg-white rounded-3xl shadow-xl w-full max-w-lg border border-gray-100 p-8 text-center">
      
      <h1 class="text-2xl font-bold text-gray-800 mb-8">Ready to begin?</h1>

      <div class="space-y-6 mb-10 text-left bg-gray-50 p-6 rounded-2xl">
        <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Intention</label>
          <p class="text-lg font-medium text-gray-800">{{ intention || 'Not set' }}</p>
        </div>
        <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Environment</label>
          <p class="text-lg font-medium text-gray-800">{{ environment || 'Not set' }}</p>
        </div>
         <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Duration</label>
          <p class="text-lg font-medium text-gray-800">{{ duration }} minutes</p>
        </div>
      </div>

      <div class="flex flex-col gap-4">
        <button id="start-session-confirm" 
                @click="confirmStart"
                class="w-full bg-orange-500 hover:bg-orange-600 text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all transform hover:scale-[1.02]">
          Start Now
        </button>
        
        <button id="review-back-to-form" 
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
  name: 'SESSION_REVIEW',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const intention = computed(() => signatureStore.session_intention_text);
    const environment = computed(() => signatureStore.session_environment_choice);
    const duration = computed(() => signatureStore.chosen_duration_minutes || 10);

    const confirmStart = async () => {
      signatureStore.session_started = true;
      await router.push({ name: 'SESSION_COMPLETED_SUCCESS' });
    };

    const goBack = async () => {
      await router.push({ name: 'SESSION_START_FORM' });
    };

    return {
      intention,
      environment,
      duration,
      confirmStart,
      goBack
    };
  }
}
</script>