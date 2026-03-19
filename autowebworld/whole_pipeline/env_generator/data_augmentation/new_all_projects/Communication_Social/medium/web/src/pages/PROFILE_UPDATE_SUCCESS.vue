<template>
  <div class="min-h-screen bg-white flex flex-col items-center justify-center p-4">
     <div class="text-center max-w-md animate-fade-in-up">
        <div class="w-16 h-16 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-6">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
           </svg>
        </div>
        <h2 class="text-2xl font-bold font-serif mb-2">Profile Updated</h2>
        <p class="text-gray-500 font-sans">Your profile changes have been saved.</p>
     </div>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PROFILE_UPDATE_SUCCESS',
  setup() {
     const signatureStore = useSignatureStore()
     
     onMounted(() => {
        signatureStore.success_message = "Profile updated successfully"
     })
  }
}
</script>

<style scoped>
.animate-fade-in-up {
  animation: fadeInUp 0.6s ease-out;
}

@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}
</style>