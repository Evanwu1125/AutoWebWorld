<template>
  <div class="min-h-screen bg-[#0F1523] text-white flex flex-col items-center justify-center p-4">
    <div class="text-center max-w-md animate-fade-in-up">
      <div class="w-24 h-24 bg-blue-900/50 rounded-full flex items-center justify-center mx-auto mb-6 text-5xl shadow-lg shadow-blue-900/20">
        😴
      </div>
      
      <h1 class="text-3xl font-bold text-white mb-4">Sweet Dreams</h1>
      <p class="text-blue-200 mb-8 text-lg">
        The sleepcast has started. Relax and drift off.
      </p>

      <button id="sleep-complete-go-home" 
              @click="goHome"
              class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-4 px-10 rounded-full shadow-lg hover:shadow-blue-500/30 transition-all transform hover:-translate-y-1">
        Return Home
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';

export default {
  name: 'SLEEP_SESSION_COMPLETED_SUCCESS',
  setup() {
    const router = useRouter();

    const goHome = async () => {
      await router.push({ name: 'HOME' });
    };

    return {
      goHome
    };
  }
}
</script>

<style scoped>
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to { opacity: 1; transform: translateY(0); }
}
.animate-fade-in-up {
  animation: fadeInUp 0.6s ease-out forwards;
}
</style>