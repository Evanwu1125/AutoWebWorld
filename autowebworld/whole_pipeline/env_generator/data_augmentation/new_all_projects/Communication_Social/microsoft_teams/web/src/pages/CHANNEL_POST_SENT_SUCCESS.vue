<template>
  <div class="h-screen flex flex-col bg-gray-50 items-center justify-center">
    <div class="bg-white rounded-xl shadow-2xl p-8 max-w-md w-full text-center border-t-4 border-green-500">
      <div class="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6 animate-bounce">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
        </svg>
      </div>
      
      <h2 class="text-2xl font-bold text-gray-800 mb-2">Success!</h2>
      <p class="text-gray-600 mb-8">{{ store.success_message || 'Channel post sent successfully.' }}</p>
      
      <button 
        id="success-go-home"
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
  name: 'CHANNEL_POST_SENT_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    onMounted(() => {
      // Set success message as per FSM effect of ACT_CHANNEL_POST_SUCCESS_GO_HOME (simulated here for display)
      // Actually FSM says effect happens ON click. 
      // But we want to show something.
    })

    const goHome = async () => {
      store.success_message = "Channel post sent";
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