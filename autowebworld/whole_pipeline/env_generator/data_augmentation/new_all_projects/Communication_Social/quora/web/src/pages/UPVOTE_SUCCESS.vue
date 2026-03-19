<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl p-10 max-w-md w-full text-center">
      <div class="w-20 h-20 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14 10h4.764a2 2 0 011.789 2.894l-3.5 7A2 2 0 0115.263 21h-4.017c-.163 0-.326-.02-.485-.06L7 20m7-10V5a2 2 0 00-2-2h-.095c-.5 0-.905.405-.905.905 0 .714-.211 1.412-.608 2.006L7 11v9m7-10h-2M7 20H5a2 2 0 01-2-2v-6a2 2 0 012-2h2.5"></path></svg>
      </div>
      
      <h2 class="text-3xl font-bold text-gray-900 mb-2">Upvoted!</h2>
      <p class="text-gray-600 mb-8 text-lg">Thanks for supporting quality content.</p>
      
      <div class="space-y-4">
        <button 
          id="upvote-success-home" 
          @click="goHome" 
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-xl transition-colors shadow-lg"
        >
          Return to Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'UPVOTE_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    onMounted(() => {
      store.success_message = "Your upvote has been recorded."
    })

    async function goHome() {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      goHome
    }
  }
}
</script>