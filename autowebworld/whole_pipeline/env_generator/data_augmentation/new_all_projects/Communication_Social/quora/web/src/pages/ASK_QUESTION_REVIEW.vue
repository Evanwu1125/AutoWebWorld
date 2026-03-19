<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center py-10 px-4">
    <div class="bg-white w-full max-w-2xl rounded-xl shadow-xl overflow-hidden">
      <!-- Header -->
      <div class="px-8 py-6 border-b border-gray-100 bg-white">
        <h2 class="text-2xl font-bold text-gray-900">Review Your Question</h2>
        <p class="text-gray-500 text-sm mt-1">Please verify your question details before posting.</p>
      </div>

      <!-- Review Content -->
      <div class="p-8 space-y-8">
        <div class="space-y-2">
          <h4 class="text-xs font-bold text-gray-400 uppercase tracking-wider">Title</h4>
          <p class="text-xl font-bold text-gray-900 font-serif">{{ title }}</p>
        </div>
        
        <div class="space-y-2">
          <h4 class="text-xs font-bold text-gray-400 uppercase tracking-wider">Topic</h4>
           <span class="inline-block bg-blue-50 text-blue-700 px-3 py-1 rounded-full text-sm font-medium border border-blue-100">
             {{ topicLabel }}
           </span>
        </div>

        <div class="space-y-2">
          <h4 class="text-xs font-bold text-gray-400 uppercase tracking-wider">Context / Details</h4>
          <div class="bg-gray-50 p-4 rounded-lg border border-gray-100 text-gray-700 leading-relaxed whitespace-pre-wrap">{{ details }}</div>
        </div>
      </div>

      <!-- Footer Actions -->
      <div class="px-8 py-6 bg-gray-50 border-t border-gray-200 flex justify-between items-center">
        <button 
          id="ask-back-edit" 
          @click="goBack" 
          class="text-gray-600 font-medium hover:text-gray-900 transition-colors flex items-center gap-2"
        >
          <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          Edit Draft
        </button>
        
        <button 
          id="ask-submit" 
          @click="submitQuestion" 
          class="bg-[#B92B27] hover:bg-[#a02521] text-white font-bold py-2.5 px-8 rounded-full shadow-md transform transition-all hover:-translate-y-0.5"
        >
          Post Question
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'ASK_QUESTION_REVIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const title = computed(() => store.draft_question_title)
    const details = computed(() => store.draft_question_details)
    const topicId = computed(() => store.draft_question_topic_id)
    
    const topicLabel = computed(() => {
      if (topicId.value === 'topic_any') return 'General / Any'
      if (topicId.value === 'topic_other') return 'Other / Specific'
      return topicId.value
    })

    function goBack() {
      store.setCurrentPageId('ASK_QUESTION_FORM')
      router.push({ name: 'ASK_QUESTION_FORM' })
    }

    async function submitQuestion() {
      // FSM Logic: Append to questions list
      // id generation is mocked
      const newQuestion = {
        id: `q_${Date.now()}`,
        title: title.value,
        details: details.value,
        topic_id: topicId.value,
        upvotes: 0,
        time: 0, // Just now
        image: '/images/photo1765097739.jpg', // Placeholder
        answered: false
      }
      
      dataStore.questions.unshift(newQuestion)
      
      store.setCurrentPageId('ASK_QUESTION_SUCCESS')
      await router.push({ name: 'ASK_QUESTION_SUCCESS' })
    }

    return {
      title,
      details,
      topicLabel,
      goBack,
      submitQuestion
    }
  }
}
</script>