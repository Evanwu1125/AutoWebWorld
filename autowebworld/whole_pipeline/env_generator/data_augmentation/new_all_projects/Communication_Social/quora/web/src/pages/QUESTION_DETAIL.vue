<template>
  <div class="min-h-screen bg-[#F1F2F2]">
    <nav class="bg-white shadow-sm sticky top-0 z-50">
      <div class="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <button id="question-back-feed" @click="goBack" class="text-gray-500 hover:text-gray-700 p-2 rounded-full hover:bg-gray-100 transition-colors">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          </button>
          <h1 class="text-[#B92B27] text-xl font-bold font-serif cursor-pointer" @click="goBack">Quora</h1>
        </div>
      </div>
    </nav>

    <main class="max-w-4xl mx-auto px-4 py-8">
      <div class="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
        <!-- Question Header -->
        <div class="p-6 md:p-8">
          <div v-if="question">
            <!-- Topic Tag -->
            <div class="flex items-center gap-2 mb-4">
              <span class="bg-gray-100 text-gray-600 px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wide">
                {{ getTopicName(question.topic_id) }}
              </span>
            </div>

            <h1 class="text-3xl font-bold text-gray-900 mb-4 font-serif leading-tight">{{ question.title }}</h1>
            
            <div class="flex items-start gap-4 mb-6">
               <img v-if="question.image" :src="question.image" class="w-full md:w-1/3 rounded-lg object-cover h-48 shadow-sm" alt="Question Context" />
               <p class="text-gray-700 text-lg leading-relaxed flex-1">{{ question.details }}</p>
            </div>

            <!-- Action Bar -->
            <div class="flex items-center gap-4 border-t border-b border-gray-100 py-3 mt-6">
              <button 
                id="write-answer-button" 
                @click="writeAnswer" 
                class="flex items-center gap-2 text-blue-600 font-medium px-4 py-2 rounded-full hover:bg-blue-50 transition-colors border border-blue-600"
              >
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z"></path></svg>
                Answer
              </button>
              
              <button 
                id="question-upvote-button" 
                @click="upvote" 
                class="flex items-center gap-2 text-gray-500 font-medium px-4 py-2 rounded-full hover:bg-gray-100 transition-colors"
              >
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path></svg>
                Upvote <span class="text-sm">({{ question.upvotes }})</span>
              </button>
              
              <button class="flex items-center gap-2 text-gray-500 font-medium px-4 py-2 rounded-full hover:bg-gray-100 transition-colors ml-auto">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.684 13.342C8.886 12.938 9 12.482 9 12c0-.482-.114-.938-.316-1.342m0 2.684a3 3 0 110-2.684m0 2.684l6.632 3.316m-6.632-6l6.632-3.316m0 0a3 3 0 105.367-2.684 3 3 0 00-5.367 2.684zm0 9.316a3 3 0 105.368 2.684 3 3 0 00-5.368-2.684z"></path></svg>
                Share
              </button>
            </div>
          </div>
          <div v-else class="text-center py-10">
            <p class="text-gray-500">Question not found.</p>
          </div>
        </div>
        
        <!-- Answers Section -->
        <div class="bg-gray-50 p-6 border-t border-gray-200">
          <h3 class="font-bold text-gray-800 mb-4">{{ answers.length }} Answers</h3>
          
          <div v-for="answer in answers" :key="answer.id" class="bg-white p-6 rounded-lg shadow-sm border border-gray-200 mb-4">
            <div class="flex items-center gap-3 mb-3">
              <img :src="answer.image" class="w-10 h-10 rounded-full object-cover border border-gray-200" />
              <div>
                <div class="font-bold text-gray-900 text-sm">{{ answer.author }}</div>
                <div class="text-xs text-gray-500">Answered just now</div>
              </div>
            </div>
            <p class="text-gray-800 leading-relaxed mb-4">{{ answer.body }}</p>
            <div class="flex items-center gap-4 text-sm text-gray-500">
               <span class="flex items-center gap-1 hover:text-blue-600 cursor-pointer">
                 <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18"></path></svg>
                 {{ answer.upvotes }}
               </span>
               <span class="cursor-pointer hover:underline">Reply</span>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'QUESTION_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const questionId = computed(() => signatureStore.selected_question_id || route.params.id)
    
    const question = computed(() => {
      return dataStore.questions.find(q => q.id === questionId.value)
    })

    const answers = computed(() => {
      return dataStore.answers.filter(a => a.question_id === questionId.value)
    })

    function getTopicName(topicId) {
      const topic = dataStore.topics.find(t => t.id === topicId)
      return topic ? topic.name : 'General'
    }

    function goBack() {
      signatureStore.setCurrentPageId('FEED')
      router.push({ name: 'FEED' })
    }

    function writeAnswer() {
      signatureStore.setCurrentPageId('ANSWER_FORM')
      router.push({ name: 'ANSWER_FORM' })
    }

    function upvote() {
      signatureStore.setCurrentPageId('UPVOTE_SUCCESS')
      router.push({ name: 'UPVOTE_SUCCESS' })
    }

    return {
      question,
      answers,
      getTopicName,
      goBack,
      writeAnswer,
      upvote
    }
  }
}
</script>