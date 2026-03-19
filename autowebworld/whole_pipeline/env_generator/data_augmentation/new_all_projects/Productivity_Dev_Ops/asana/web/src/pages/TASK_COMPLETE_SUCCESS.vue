<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="max-w-md w-full bg-white rounded-xl shadow-lg p-8 text-center animate-bounce-in">
       <!-- Celebration Animation Placeholder -->
       <div class="w-24 h-24 bg-gradient-to-br from-green-400 to-emerald-600 rounded-full flex items-center justify-center mx-auto mb-6 shadow-lg shadow-green-200">
         <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7" />
         </svg>
       </div>
       
       <h2 class="text-3xl font-extrabold text-gray-900 mb-2">Nice Work!</h2>
       <p class="text-gray-600 mb-8 text-lg">
         Task marked as complete. Keep up the momentum!
       </p>
       
       <div class="space-y-4">
         <button 
           id="task-complete-success-back-board"
           @click="goBoard"
           class="w-full bg-indigo-600 text-white px-6 py-3 rounded-xl font-bold hover:bg-indigo-700 transition-all shadow-md hover:shadow-lg transform hover:-translate-y-0.5"
         >
           Back to Board
         </button>
         
         <button 
           id="task-complete-success-go-home"
           @click="goHome"
           class="w-full bg-white text-gray-600 border-2 border-gray-100 px-6 py-3 rounded-xl font-bold hover:bg-gray-50 hover:border-gray-200 transition-all"
         >
           Go Home
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
  name: 'TASK_COMPLETE_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    onMounted(() => {
        store.success_message = "Task completed successfully"
    })

    const goBoard = async () => {
      await router.push({ name: 'PROJECT_BOARD' })
    }

    const goHome = async () => {
      await router.push({ name: 'HOME' })
    }

    return {
      goBoard,
      goHome
    }
  }
}
</script>

<style scoped>
@keyframes bounceIn {
  0% { opacity: 0; transform: scale(0.3); }
  50% { opacity: 1; transform: scale(1.05); }
  70% { transform: scale(0.9); }
  100% { transform: scale(1); }
}
.animate-bounce-in {
  animation: bounceIn 0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}
</style>