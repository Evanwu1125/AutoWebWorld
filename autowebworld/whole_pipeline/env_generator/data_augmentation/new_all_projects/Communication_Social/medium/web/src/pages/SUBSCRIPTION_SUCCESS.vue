<template>
  <div class="min-h-screen bg-white flex flex-col items-center justify-center p-4">
     <div class="text-center max-w-md animate-fade-in-up">
        <div class="w-24 h-24 bg-gradient-to-r from-green-400 to-blue-500 text-white rounded-full flex items-center justify-center mx-auto mb-8 shadow-lg">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
           </svg>
        </div>
        <h2 class="text-4xl font-bold font-serif mb-4">Welcome to Medium!</h2>
        <p class="text-gray-500 font-sans text-xl mb-12">You are now a member. Enjoy unlimited access to everything.</p>
        
        <button class="bg-black text-white px-8 py-3 rounded-full text-lg font-sans font-medium hover:bg-gray-800 transition-colors">Start Reading</button>
     </div>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SUBSCRIPTION_SUCCESS',
  setup() {
     const signatureStore = useSignatureStore()
     
     onMounted(() => {
        signatureStore.success_message = "Subscription successful"
     })
  }
}
</script>

<style scoped>
.animate-fade-in-up {
  animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(40px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>