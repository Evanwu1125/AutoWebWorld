<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="max-w-sm w-full bg-white rounded-xl shadow-lg p-8 text-center">
       <div class="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
         <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
           <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
         </svg>
       </div>
       
       <h2 class="text-xl font-bold text-gray-900 mb-2">Comment Posted</h2>
       <p class="text-gray-500 mb-6">
         Your feedback has been added to the task.
       </p>
       
       <div class="space-y-3">
         <button 
           id="comment-success-back-task"
           @click="goTask"
           class="w-full bg-indigo-600 text-white px-4 py-2 rounded-lg font-medium hover:bg-indigo-700 transition-colors shadow-sm"
         >
           Back to Task
         </button>
         
         <button 
           id="comment-success-go-home"
           @click="goHome"
           class="w-full bg-white text-gray-700 border border-gray-300 px-4 py-2 rounded-lg font-medium hover:bg-gray-50 transition-colors"
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
  name: 'COMMENT_ADD_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    onMounted(() => {
        store.success_message = "Comment added successfully"
    })

    const goTask = async () => {
        // FSM has specific Back to Task action
        const taskId = store.selected_task_id
        if (taskId) {
            await router.push({ name: 'TASK_DETAIL', params: { id: taskId } })
        } else {
             // Fallback
            await router.push({ name: 'PROJECT_BOARD' })
        }
    }

    const goHome = async () => {
      await router.push({ name: 'HOME' })
    }

    return {
      goTask,
      goHome
    }
  }
}
</script>