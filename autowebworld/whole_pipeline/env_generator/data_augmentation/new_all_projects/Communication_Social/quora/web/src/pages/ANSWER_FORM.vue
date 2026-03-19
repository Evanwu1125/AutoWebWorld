<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center py-10 px-4">
    <div class="bg-white w-full max-w-3xl rounded-xl shadow-xl overflow-hidden flex flex-col h-[80vh]">
      <!-- Header -->
      <div class="px-6 py-4 border-b border-gray-100 bg-white flex justify-between items-center flex-shrink-0">
        <h2 class="text-lg font-bold text-gray-900">Write Answer</h2>
        <button id="answer-cancel" @click="cancel" class="text-gray-400 hover:text-gray-600">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
      </div>

      <!-- Question Context (Read Only) -->
      <div class="bg-gray-50 px-6 py-4 border-b border-gray-100 flex-shrink-0">
        <h3 class="font-bold text-gray-800 mb-1 line-clamp-1">{{ questionTitle }}</h3>
        <p class="text-sm text-gray-500 line-clamp-2">{{ questionDetails }}</p>
      </div>

      <!-- Editor Area -->
      <div class="flex-1 p-6 overflow-y-auto">
        <textarea 
          id="answer-body-input"
          v-model="body"
          @input="updateBody"
          class="w-full h-full resize-none focus:outline-none text-lg leading-relaxed text-gray-800 placeholder-gray-300"
          placeholder="Write your answer here..."
        ></textarea>
      </div>

      <!-- Footer -->
      <div class="px-6 py-4 border-t border-gray-100 bg-white flex justify-end gap-3 flex-shrink-0">
        <button @click="cancel" class="px-6 py-2 text-gray-500 font-medium hover:bg-gray-100 rounded-full transition-colors">
          Cancel
        </button>
        <button 
          id="answer-submit" 
          @click="submitAnswer" 
          :disabled="!isValid"
          :class="[
            'px-8 py-2 text-white font-bold rounded-full transition-all shadow-sm',
            isValid ? 'bg-blue-600 hover:bg-blue-700' : 'bg-blue-300 cursor-not-allowed'
          ]"
        >
          Submit Answer
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ANSWER_FORM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const body = ref('')
    
    // Get context
    const question = computed(() => dataStore.questions.find(q => q.id === store.selected_question_id))
    const questionTitle = computed(() => question.value?.title || 'Unknown Question')
    const questionDetails = computed(() => question.value?.details || '')

    const isValid = computed(() => body.value.trim().length > 0)

    function updateBody() {
      store.answer_body_draft = body.value
    }

    function cancel() {
      store.setCurrentPageId('QUESTION_DETAIL')
      router.push({ name: 'QUESTION_DETAIL', params: { id: store.selected_question_id } })
    }

    async function submitAnswer() {
      if (!isValid.value) return
      
      // FSM Logic: Append to answers
      const newAnswer = {
        id: `a_${Date.now()}`,
        question_id: store.selected_question_id,
        author: store.profile_name || 'Anonymous', // Use current profile name
        body: body.value,
        image: '/images/UserAvatar.jpg', // Use current user avatar
        upvotes: 0
      }
      
      dataStore.answers.push(newAnswer)
      
      // Also mark question as answered
      if (question.value) question.value.answered = true;

      store.setCurrentPageId('ANSWER_QUESTION_SUCCESS')
      await router.push({ name: 'ANSWER_QUESTION_SUCCESS' })
    }

    return {
      body,
      questionTitle,
      questionDetails,
      isValid,
      updateBody,
      cancel,
      submitAnswer
    }
  }
}
</script>