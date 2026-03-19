<template>
  <div class="min-h-screen bg-orange-50 flex flex-col items-center justify-center p-4">
    <div class="text-center max-w-md animate-fade-in-up">
      <div class="w-24 h-24 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-6 text-5xl shadow-sm">
        🔥
      </div>
      
      <h1 class="text-3xl font-bold text-gray-900 mb-4">Focus Mode On</h1>
      <p class="text-gray-600 mb-8 text-lg">
        Good luck with your task. You got this!
      </p>

      <button id="focus-complete-go-home" 
              @click="goHome"
              class="bg-orange-500 hover:bg-orange-600 text-white font-bold py-4 px-10 rounded-full shadow-lg hover:shadow-orange-500/30 transition-all transform hover:-translate-y-1">
        Return Home
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';

export default {
  name: 'FOCUS_SESSION_COMPLETED_SUCCESS',
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