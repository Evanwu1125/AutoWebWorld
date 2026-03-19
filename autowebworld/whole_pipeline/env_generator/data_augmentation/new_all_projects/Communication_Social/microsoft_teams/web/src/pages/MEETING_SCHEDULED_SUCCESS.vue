<template>
  <div class="h-screen flex flex-col bg-gray-50 items-center justify-center">
    <div class="bg-white rounded-xl shadow-2xl p-8 max-w-md w-full text-center border-t-4 border-blue-500">
      <div class="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6 animate-bounce">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
        </svg>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-800 mb-2">Meeting Scheduled!</h2>
      <p class="text-gray-600 mb-8">{{ store.success_message || 'Invitations have been sent.' }}</p>
      
      <button 
        id="meeting-scheduled-go-home"
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
import { onMounted } from 'vue'

export default {
  name: 'MEETING_SCHEDULED_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const goHome = async () => {
      store.success_message = "Meeting scheduled";
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