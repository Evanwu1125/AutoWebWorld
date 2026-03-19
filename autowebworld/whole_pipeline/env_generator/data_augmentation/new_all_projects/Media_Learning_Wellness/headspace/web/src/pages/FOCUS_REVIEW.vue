<template>
  <div class="min-h-screen bg-[#FFF4E6] flex flex-col items-center py-10 px-4">
    <div class="bg-white rounded-3xl shadow-xl w-full max-w-lg border border-orange-100 p-8 text-center">
      
      <h1 class="text-2xl font-bold text-gray-800 mb-8">Ready to focus?</h1>

      <div class="space-y-6 mb-10 text-left bg-orange-50 p-6 rounded-2xl">
        <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Task</label>
          <p class="text-lg font-medium text-gray-800">{{ task || 'Not set' }}</p>
        </div>
        <div>
          <label class="text-xs font-bold text-gray-400 uppercase tracking-wider">Duration</label>
          <p class="text-lg font-medium text-gray-800">{{ duration || 'Not set' }}</p>
        </div>
      </div>

      <div class="flex flex-col gap-4">
        <button id="focus-start-confirm" 
                @click="confirmStart"
                class="w-full bg-orange-500 hover:bg-orange-600 text-white font-bold py-4 px-6 rounded-xl shadow-lg transition-all transform hover:scale-[1.02]">
          Start Focusing
        </button>
        
        <button id="focus-review-back-form" 
                @click="goBack"
                class="w-full text-gray-500 hover:text-gray-700 font-medium py-2">
          Change Settings
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
  name: 'FOCUS_REVIEW',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const task = computed(() => signatureStore.focus_task_text);
    const duration = computed(() => signatureStore.focus_duration_choice);

    const confirmStart = async () => {
      signatureStore.focus_session_started = true;
      await router.push({ name: 'FOCUS_SESSION_COMPLETED_SUCCESS' });
    };

    const goBack = async () => {
      await router.push({ name: 'FOCUS_START_FORM' });
    };

    return {
      task,
      duration,
      confirmStart,
      goBack
    };
  }
}
</script>