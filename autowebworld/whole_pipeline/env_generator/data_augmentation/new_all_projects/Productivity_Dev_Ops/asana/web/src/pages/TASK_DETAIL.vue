<template>
  <div class="min-h-screen bg-gray-50 flex flex-col items-center p-4 md:p-8">
    <div class="bg-white rounded-xl shadow-lg border border-gray-100 max-w-4xl w-full overflow-hidden flex flex-col md:flex-row min-h-[600px]">
       
       <!-- Main Content -->
       <div class="flex-grow p-8 flex flex-col border-r border-gray-100">
          <!-- Header -->
          <div class="flex items-start justify-between mb-6">
             <div class="flex items-center gap-3">
                <button 
                  id="task-complete-checkbox"
                  @click="markComplete"
                  class="w-8 h-8 rounded-full border-2 border-gray-300 hover:border-green-500 hover:bg-green-50 transition-colors flex items-center justify-center group"
                >
                   <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-transparent group-hover:text-green-500 transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
                   </svg>
                </button>
                <div class="flex flex-col">
                    <h1 class="text-2xl font-bold text-gray-900 leading-tight">{{ task?.name }}</h1>
                    <div class="flex items-center gap-2 mt-1 text-sm text-gray-500">
                        <span>in <span class="font-medium text-gray-700">{{ projectName }}</span></span>
                        <span>•</span>
                        <span class="bg-gray-100 px-2 py-0.5 rounded text-xs">{{ sectionName }}</span>
                    </div>
                </div>
             </div>
             
             <button 
               id="task-detail-back-board"
               @click="goBack"
               class="text-gray-400 hover:text-gray-600 p-2 hover:bg-gray-100 rounded-full transition-colors"
             >
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
                </svg>
             </button>
          </div>

          <!-- Description -->
          <div class="mb-8">
             <h3 class="text-sm font-semibold text-gray-900 mb-2 uppercase tracking-wide">Description</h3>
             <p class="text-gray-600 leading-relaxed whitespace-pre-line">{{ task?.description || 'No description provided.' }}</p>
          </div>

          <!-- Hero Image if exists -->
           <div v-if="task?.image" class="mb-8 rounded-lg overflow-hidden h-64 w-full bg-gray-100">
               <img :src="task.image" class="w-full h-full object-cover" />
           </div>

          <!-- Comments Section -->
          <div class="mt-auto">
             <h3 class="text-sm font-semibold text-gray-900 mb-4 uppercase tracking-wide">Comments</h3>
             
             <!-- Comment List -->
             <div class="space-y-4 mb-6 max-h-64 overflow-y-auto pr-2 custom-scrollbar">
                <div v-for="comment in comments" :key="comment.id" class="flex gap-3">
                   <div class="flex-shrink-0">
                      <img :src="getUserAvatar(comment.user_id)" class="h-8 w-8 rounded-full" />
                   </div>
                   <div class="flex-grow">
                      <div class="flex items-baseline gap-2">
                         <span class="font-semibold text-gray-900 text-sm">{{ getUserName(comment.user_id) }}</span>
                         <span class="text-xs text-gray-400">{{ formatDate(comment.created_at) }}</span>
                      </div>
                      <p class="text-gray-600 text-sm mt-0.5">{{ comment.text }}</p>
                   </div>
                </div>
             </div>

             <!-- Add Comment -->
             <div class="flex gap-3 items-start bg-gray-50 p-4 rounded-lg border border-gray-100">
                <div class="flex-shrink-0">
                   <img src="/images/photo1765161152.jpg" class="h-8 w-8 rounded-full" />
                </div>
                <div class="flex-grow">
                   <textarea 
                     id="task-comment-input"
                     v-model="newComment"
                     rows="2"
                     class="w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-sm p-2 resize-none bg-white"
                     placeholder="Ask a question or post an update..."
                   ></textarea>
                   <div class="flex justify-end mt-2">
                      <button 
                        id="task-comment-submit"
                        @click="submitComment"
                        :disabled="!newComment"
                        class="bg-indigo-600 text-white px-3 py-1.5 rounded text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                      >
                         Comment
                      </button>
                   </div>
                </div>
             </div>
          </div>
       </div>

       <!-- Sidebar Meta -->
       <div class="w-full md:w-72 bg-gray-50 p-8 flex flex-col gap-6 border-t md:border-t-0 md:border-l border-gray-200">
          <div>
             <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Assignee</label>
             <div class="flex items-center gap-3 bg-white p-2 rounded-lg border border-gray-200 shadow-sm">
                <img :src="getUserAvatar(task?.assignee_id)" class="h-8 w-8 rounded-full border border-gray-100" />
                <span class="text-sm font-medium text-gray-700">{{ getUserName(task?.assignee_id) || 'Unassigned' }}</span>
             </div>
          </div>

          <div>
             <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Due Date</label>
             <div class="flex items-center gap-2 text-gray-700 bg-white p-2 rounded-lg border border-gray-200 shadow-sm">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                   <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span class="text-sm font-medium">{{ formatDate(task?.due_date) || 'No date' }}</span>
             </div>
          </div>

          <div>
             <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Priority</label>
             <div class="flex items-center gap-2">
                <div class="h-2 flex-grow bg-gray-200 rounded-full overflow-hidden">
                   <div 
                     class="h-full rounded-full transition-all duration-500"
                     :class="getPriorityColorClass(task?.priority)"
                     :style="{ width: `${task?.priority || 0}%` }"
                   ></div>
                </div>
                <span class="text-xs font-medium text-gray-600">{{ task?.priority }}</span>
             </div>
          </div>
          
          <div>
             <label class="block text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2">Status</label>
             <span 
                class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium"
                :class="task?.completed ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'"
             >
                {{ task?.completed ? 'Completed' : 'In Progress' }}
             </span>
          </div>
       </div>

    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'TASK_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const taskId = computed(() => route.params.id || signatureStore.selected_task_id || 't1')
    const task = computed(() => dataStore.tasks.find(t => t.id === taskId.value))
    
    // Meta Data
    const project = computed(() => dataStore.projects.find(p => p.id === task.value?.project_id))
    const projectName = computed(() => project.value?.name || 'Unknown Project')
    
    const section = computed(() => dataStore.sections.find(s => s.id === task.value?.section_id))
    const sectionName = computed(() => section.value?.name || 'General')

    const comments = computed(() => dataStore.comments.filter(c => c.task_id === taskId.value))

    // Form
    const newComment = ref('')

    // Methods
    const markComplete = async () => {
        // Effect: set task_completed = true
        signatureStore.task_completed = true
        
        // Data update
        if (task.value) task.value.completed = true

        await router.push({ name: 'TASK_COMPLETE_SUCCESS' })
    }

    const submitComment = async () => {
        if (!newComment.value) return

        // Effect: set new_comment_text
        signatureStore.new_comment_text = newComment.value

        // Data update
        dataStore.addComment({
            id: `c${Date.now()}`,
            task_id: taskId.value,
            user_id: 'u1',
            text: newComment.value,
            created_at: new Date().toISOString()
        })

        await router.push({ name: 'COMMENT_ADD_SUCCESS' })
    }

    const goBack = async () => {
        // Determine where to go back to (Board or My Tasks)
        // FSM says TASK_DETAIL_BACK_TO_BOARD -> PROJECT_BOARD
        // Real app might check history, but we follow FSM strictly for this action ID
        await router.push({ name: 'PROJECT_BOARD', params: { id: task.value?.project_id } })
    }

    // Helpers
    const getUserName = (uid) => {
        const u = dataStore.users.find(u => u.id === uid)
        return u ? u.name : 'Unknown'
    }
    const getUserAvatar = (uid) => {
        const u = dataStore.users.find(u => u.id === uid)
        return u ? u.avatar : '/images/photo1765161152.jpg'
    }
    const formatDate = (iso) => {
        if (!iso) return ''
        return new Date(iso).toLocaleDateString()
    }
    const getPriorityColorClass = (p) => {
        if (p >= 80) return 'bg-red-500'
        if (p >= 50) return 'bg-yellow-500'
        return 'bg-green-500'
    }

    return {
       task,
       projectName,
       sectionName,
       comments,
       newComment,
       markComplete,
       submitComment,
       goBack,
       getUserName,
       getUserAvatar,
       formatDate,
       getPriorityColorClass
    }
  }
}
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 4px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: transparent; 
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: #cbd5e1; 
  border-radius: 2px;
}
</style>