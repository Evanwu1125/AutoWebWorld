<template>
  <div class="min-h-screen bg-[#1A2338] text-white flex flex-col items-center py-10 px-4">
    <div class="bg-[#25304C] rounded-3xl shadow-xl w-full max-w-lg border border-gray-700 p-8 text-center">
      
      <h1 class="text-2xl font-bold text-white mb-8">Ready to sleep?</h1>

      <div class="space-y-6 mb-10 text-left bg-[#1A2338] p-6 rounded-2xl border border-gray-800">
        <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Bedtime</label>
          <p class="text-lg font-medium text-white">{{ bedtime || 'Not set' }}</p>
        </div>
        <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Environment</label>
          <p class="text-lg font-medium text-white">{{ environment || 'Not set' }}</p>
        </div>
      </div>

      <div class="flex flex-col gap-4">
        <button id="sleep-start-confirm" 
                @click="confirmStart"
                class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all transform hover:scale-[1.02]">
          Start Sleepcast
        </button>
        
        <button id="sleep-review-back-form" 
                @click="goBack"
                class="w-full text-gray-400 hover:text-white font-medium py-2 transition-colors">
          Adjust Settings
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
  name: 'SLEEP_REVIEW',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const bedtime = computed(() => signatureStore.sleep_bedtime_text);
    const environment = computed(() => signatureStore.sleep_environment_choice);

    const confirmStart = async () => {
      signatureStore.sleep_session_started = true;
      await router.push({ name: 'SLEEP_SESSION_COMPLETED_SUCCESS' });
    };

    const goBack = async () => {
      await router.push({ name: 'SLEEP_START_FORM' });
    };

    return {
      bedtime,
      environment,
      confirmStart,
      goBack
    };
  }
}
</script>