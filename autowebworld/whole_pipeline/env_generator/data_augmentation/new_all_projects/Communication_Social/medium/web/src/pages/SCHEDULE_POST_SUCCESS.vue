<template>
  <div class="min-h-screen bg-white flex flex-col items-center justify-center p-4">
     <div class="text-center max-w-md animate-fade-in-up">
        <div class="w-20 h-20 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-8">
           <svg xmlns="http://www.w3.org/2000/svg" class="h-10 w-10" fill="none" viewBox="0 0 24 24" stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
           </svg>
        </div>
        <h2 class="text-3xl font-bold font-serif mb-4">Story Scheduled</h2>
        <p class="text-gray-500 font-sans text-lg mb-8">Your story has been scheduled and will be published automatically.</p>
     </div>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SCHEDULE_POST_SUCCESS',
  setup() {
     const signatureStore = useSignatureStore()
     
     onMounted(() => {
        signatureStore.success_message = "Story scheduled successfully"
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