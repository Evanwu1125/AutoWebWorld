<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl p-10 max-w-md w-full text-center">
      <div class="w-20 h-20 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-6">
        <svg class="w-10 h-10 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z"></path></svg>
      </div>
      
      <h2 class="text-3xl font-bold text-gray-900 mb-2">Topic Followed</h2>
      <p class="text-gray-600 mb-8 text-lg">You will now see updates from this topic in your feed.</p>
      
      <div class="space-y-4">
        <button 
          id="follow-topic-success-home" 
          @click="goHome" 
          class="w-full bg-[#B92B27] hover:bg-[#a02521] text-white font-bold py-3 px-6 rounded-xl transition-colors shadow-lg"
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
  name: 'FOLLOW_TOPIC_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    onMounted(() => {
      store.success_message = "You are now following this topic."
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