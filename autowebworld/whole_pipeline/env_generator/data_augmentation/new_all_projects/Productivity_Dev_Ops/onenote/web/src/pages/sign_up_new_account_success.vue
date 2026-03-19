<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-2xl shadow-xl max-w-md w-full p-8 text-center animate-fade-in">
      
      <!-- Celebration Icon -->
      <div class="mb-6 flex justify-center">
        <div class="w-24 h-24 bg-gradient-to-tr from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white shadow-lg">
          <svg class="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.828 14.828a4 4 0 01-5.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
        </div>
      </div>
      
      <h1 class="text-3xl font-bold text-gray-900 mb-2">Welcome Aboard!</h1>
      <p class="text-gray-600 mb-8 leading-relaxed">
        Your new account has been created successfully. You're all set to start organizing your life.
      </p>

      <div class="bg-purple-50 rounded-lg p-4 mb-8 text-sm text-purple-800">
        <p class="font-bold">What's Next?</p>
        <p>Try creating your first notebook or explore the quick notes feature.</p>
      </div>

      <button 
        id="sign-up-success-go-home"
        @click="goHome"
        class="w-full bg-purple-600 hover:bg-purple-700 text-white font-bold py-3.5 rounded-xl shadow-lg hover:shadow-xl transition-all transform hover:-translate-y-0.5"
      >
        Get Started
      </button>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { onMounted } from 'vue'

export default {
  name: 'sign_up_new_account_success',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    onMounted(() => {
      // Set the success message in store as per FSM effect
      store.sign_up_message = "Your new account has been created successfully."
    })

    const goHome = async () => {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      goHome
    }
  }
}
</script>

<style scoped>
.animate-fade-in {
  animation: fadeIn 0.6s ease-out;
}
@keyframes fadeIn {
  from { opacity: 0; transform: scale(0.95); }
  to { opacity: 1; transform: scale(1); }
}
</style>