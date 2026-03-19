<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center py-10 px-4">
    <div class="bg-white w-full max-w-2xl rounded-xl shadow-xl overflow-hidden">
      <!-- Header -->
      <div class="px-8 py-6 border-b border-gray-100 bg-white flex justify-between items-center">
        <h2 class="text-2xl font-bold text-gray-900">Add Question</h2>
        <button id="ask-cancel" @click="cancel" class="text-gray-400 hover:text-gray-600 transition-colors">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
      </div>

      <!-- Form Content -->
      <div class="p-8 space-y-6">
        
        <!-- Tips Box -->
        <div class="bg-blue-50 text-blue-800 p-4 rounded-lg text-sm border border-blue-100 mb-6">
          <h4 class="font-bold mb-1">Tips on getting good answers quickly</h4>
          <ul class="list-disc list-inside space-y-1 opacity-90">
            <li>Make sure your question has not been asked already</li>
            <li>Keep your question short and to the point</li>
            <li>Double-check grammar and spelling</li>
          </ul>
        </div>

        <!-- Title Input -->
        <div class="space-y-2">
          <label class="block text-sm font-bold text-gray-700">Question Title</label>
          <input 
            id="ask-title-input"
            v-model="title"
            @input="updateTitle"
            type="text" 
            placeholder='Start your question with "What", "How", "Why", etc.' 
            class="w-full text-lg border-b-2 border-gray-200 focus:border-blue-600 focus:outline-none py-2 transition-colors placeholder-gray-300"
          />
        </div>

        <!-- Topic Selection (Dropdown) -->
        <div class="space-y-2 relative">
           <label class="block text-sm font-bold text-gray-700">Topic</label>
           <div 
             id="ask-topic-dropdown" 
             @click="toggleTopicDropdown"
             class="w-full border border-gray-300 rounded-lg px-4 py-2.5 bg-white cursor-pointer hover:border-blue-400 transition-all flex justify-between items-center text-gray-700"
           >
             <span>{{ selectedTopicName || 'Select a topic...' }}</span>
             <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
           </div>
           
           <div v-if="isTopicDropdownOpen" class="absolute top-full left-0 w-full bg-white border border-gray-200 rounded-lg shadow-xl mt-1 z-20 max-h-60 overflow-y-auto">
             <!-- Static options from FSM first, then maybe dynamic ones if needed, but FSM defines specific selectors -->
             <!-- FSM: #ask-topic-option1 (value: topic_any), #ask-topic-option2 (value: topic_other) -->
             <div id="ask-topic-option1" @click="selectTopic('topic_any')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700 border-b border-gray-50">General / Any</div>
             <div id="ask-topic-option2" @click="selectTopic('topic_other')" class="px-4 py-3 hover:bg-blue-50 cursor-pointer text-gray-700">Other / Specific</div>
           </div>
        </div>

        <!-- Details Input -->
        <div class="space-y-2">
          <label class="block text-sm font-bold text-gray-700">Context / Details (Optional)</label>
          <textarea 
            id="ask-details-input"
            v-model="details"
            @input="updateDetails"
            rows="4"
            placeholder="Include any details that would help answer your question..."
            class="w-full border border-gray-300 rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-100 focus:border-blue-500 focus:outline-none transition-all resize-none"
          ></textarea>
        </div>

      </div>

      <!-- Footer Actions -->
      <div class="px-8 py-5 bg-gray-50 border-t border-gray-200 flex justify-end gap-3">
        <button 
          @click="cancel" 
          class="px-6 py-2.5 text-gray-600 font-medium hover:bg-gray-200 rounded-full transition-colors"
        >
          Cancel
        </button>
        <button 
          id="ask-next-review" 
          @click="submit" 
          :disabled="!isValid"
          :class="[
            'px-8 py-2.5 text-white font-bold rounded-full transition-all shadow-sm',
            isValid ? 'bg-blue-600 hover:bg-blue-700 transform hover:-translate-y-0.5' : 'bg-blue-300 cursor-not-allowed'
          ]"
        >
          Next Step
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'ASK_QUESTION_FORM',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const title = ref('')
    const details = ref('')
    const topicId = ref('')
    const isTopicDropdownOpen = ref(false)

    // Validation
    const isValid = computed(() => {
      return title.value.trim().length > 0 && 
             details.value.trim().length > 0 && 
             topicId.value.length > 0
    })
    
    const selectedTopicName = computed(() => {
      if (topicId.value === 'topic_any') return 'General / Any'
      if (topicId.value === 'topic_other') return 'Other / Specific'
      return ''
    })

    // Actions
    function updateTitle() {
      store.draft_question_title = title.value
    }

    function updateDetails() {
      store.draft_question_details = details.value
    }

    function toggleTopicDropdown() {
      isTopicDropdownOpen.value = !isTopicDropdownOpen.value
    }

    function selectTopic(id) {
      topicId.value = id
      store.draft_question_topic_id = id
      isTopicDropdownOpen.value = false
    }

    function cancel() {
      store.setCurrentPageId('FEED')
      router.push({ name: 'FEED' })
    }

    async function submit() {
      if (!isValid.value) return
      store.setCurrentPageId('ASK_QUESTION_REVIEW')
      await router.push({ name: 'ASK_QUESTION_REVIEW' })
    }

    return {
      title,
      details,
      isTopicDropdownOpen,
      selectedTopicName,
      isValid,
      updateTitle,
      updateDetails,
      toggleTopicDropdown,
      selectTopic,
      cancel,
      submit
    }
  }
}
</script>