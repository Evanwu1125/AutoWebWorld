<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-6">
    <div class="bg-white rounded-xl shadow-xl p-8 text-center max-w-md w-full animate-fade-in-up">
      <div class="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </div>
      
      <h1 class="text-2xl font-bold text-gray-900 mb-2">Base Created!</h1>
      <p class="text-gray-600 mb-8">
        Your new base is ready. You can start adding records, configuring views, and collaborating with your team.
      </p>
      
      <div class="space-y-3">
        <button 
          id="view-bases-button"
          @click="goToBases"
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg shadow-md transition-colors flex items-center justify-center gap-2"
        >
          View All Bases
          <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 8l4 4m0 0l-4 4m4-4H3" />
          </svg>
        </button>
        
        <button 
          id="go-home"
          @click="goHome"
          class="w-full bg-white hover:bg-gray-50 text-gray-600 font-medium py-3 px-4 rounded-lg border border-gray-200 transition-colors"
        >
          Back to Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'BASE_CREATED_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goToBases = async () => {
      store.setCurrentPageId('BASES_DASHBOARD')
      await router.push({ name: 'BASES_DASHBOARD' })
    }

    const goHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      goToBases,
      goHome
    }
  }
}
</script>

<style scoped>
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translate3d(0, 20px, 0);
  }
  to {
    opacity: 1;
    transform: translate3d(0, 0, 0);
  }
}

.animate-fade-in-up {
  animation: fadeInUp 0.5s ease-out;
}
</style>