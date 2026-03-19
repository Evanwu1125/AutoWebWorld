<template>
  <div class="h-screen flex flex-col bg-gray-50 items-center justify-center">
    <div class="bg-white rounded-xl shadow-2xl p-8 max-w-md w-full text-center border-t-4 border-purple-500">
      <div class="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-6 animate-pulse">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
        </svg>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-800 mb-2">Team Created!</h2>
      <p class="text-gray-600 mb-8">{{ store.success_message || 'Your new team is ready.' }}</p>
      
      <button 
        id="team-created-go-home"
        @click="goHome"
        class="w-full bg-[#6264A7] hover:bg-[#464775] text-white font-semibold py-3 px-6 rounded-lg shadow transition-colors"
      >
        Return to Home
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'TEAM_CREATED_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHome = async () => {
      store.success_message = "Team created successfully";
      store.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    }

    return {
      store,
      goHome
    }
  }
}
</script>