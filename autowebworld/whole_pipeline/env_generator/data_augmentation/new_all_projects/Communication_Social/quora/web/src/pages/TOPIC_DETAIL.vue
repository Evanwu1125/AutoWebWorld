<template>
  <div class="min-h-screen bg-[#F1F2F2]">
    <nav class="bg-white shadow-sm sticky top-0 z-50">
      <div class="max-w-5xl mx-auto px-4 h-14 flex items-center justify-between">
        <div class="flex items-center gap-4">
          <button id="topic-back" @click="goBack" class="text-gray-500 hover:text-gray-700 p-2 rounded-full hover:bg-gray-100 transition-colors">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
          </button>
          <h1 class="text-[#B92B27] text-xl font-bold font-serif">Topic</h1>
        </div>
      </div>
    </nav>

    <main class="max-w-5xl mx-auto px-4 py-8" v-if="topic">
      <!-- Topic Header Card -->
      <div class="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden mb-6">
        <div class="relative h-48 md:h-64">
           <img :src="topic.image" class="w-full h-full object-cover" :alt="topic.name" />
           <div class="absolute inset-0 bg-gradient-to-t from-black/80 via-black/30 to-transparent"></div>
           <div class="absolute bottom-6 left-6 md:left-10 text-white">
             <h1 class="text-4xl md:text-5xl font-bold mb-2 shadow-sm">{{ topic.name }}</h1>
             <p class="text-white/90 text-sm font-medium">{{ (topic.followers / 1000).toFixed(1) }}k Followers · High Activity</p>
           </div>
        </div>
        
        <div class="p-4 md:p-6 flex items-center gap-4 bg-white">
          <button 
            id="topic-follow-button" 
            @click="followTopic" 
            class="bg-blue-600 text-white px-8 py-2.5 rounded-full font-bold hover:bg-blue-700 transition-colors shadow-md flex items-center gap-2"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path></svg>
            Follow Topic
          </button>
          
          <button 
            id="topic-ask-question" 
            @click="askQuestion" 
            class="bg-white text-gray-700 border border-gray-300 px-6 py-2.5 rounded-full font-bold hover:bg-gray-50 transition-colors flex items-center gap-2"
          >
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            Ask Question
          </button>
        </div>
      </div>

      <!-- Topic Content (Questions) -->
      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="md:col-span-2 space-y-4">
           <h3 class="font-bold text-gray-700 text-sm uppercase tracking-wide mb-2">Popular in {{ topic.name }}</h3>
           
           <div v-for="question in topicQuestions" :key="question.id" class="bg-white p-5 rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow cursor-pointer">
             <h3 class="text-xl font-bold text-gray-900 mb-2 font-serif">{{ question.title }}</h3>
             <p class="text-gray-600 line-clamp-2 text-sm mb-3">{{ question.details }}</p>
             <div class="flex items-center gap-4 text-xs text-gray-500">
                <span>{{ question.upvotes }} Upvotes</span>
                <span>{{ question.time }}h ago</span>
             </div>
           </div>
           
           <div v-if="topicQuestions.length === 0" class="bg-white p-8 rounded-lg text-center border border-gray-200">
             <p class="text-gray-500">No questions in this topic yet.</p>
           </div>
        </div>

        <div class="md:col-span-1 space-y-6">
          <div class="bg-white p-4 rounded-lg shadow-sm border border-gray-200">
            <h4 class="font-bold text-gray-800 mb-3 border-b pb-2">Related Topics</h4>
            <div class="flex flex-wrap gap-2">
              <span class="bg-gray-100 text-gray-600 px-3 py-1 rounded-full text-xs hover:bg-gray-200 cursor-pointer">Science</span>
              <span class="bg-gray-100 text-gray-600 px-3 py-1 rounded-full text-xs hover:bg-gray-200 cursor-pointer">Technology</span>
              <span class="bg-gray-100 text-gray-600 px-3 py-1 rounded-full text-xs hover:bg-gray-200 cursor-pointer">News</span>
            </div>
          </div>
        </div>
      </div>

    </main>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'TOPIC_DETAIL',
  setup() {
    const router = useRouter()
    const route = useRoute()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()

    const topicId = computed(() => signatureStore.selected_topic_id || route.params.id)
    
    const topic = computed(() => {
      return dataStore.topics.find(t => t.id === topicId.value)
    })

    const topicQuestions = computed(() => {
      return dataStore.questions.filter(q => q.topic_id === topicId.value)
    })

    function goBack() {
      signatureStore.setCurrentPageId('TOPIC_LIST')
      router.push({ name: 'TOPIC_LIST' })
    }

    function followTopic() {
      signatureStore.setCurrentPageId('FOLLOW_TOPIC_SUCCESS')
      router.push({ name: 'FOLLOW_TOPIC_SUCCESS' })
    }

    function askQuestion() {
      signatureStore.setCurrentPageId('ASK_QUESTION_FORM')
      router.push({ name: 'ASK_QUESTION_FORM' })
    }

    return {
      topic,
      topicQuestions,
      goBack,
      followTopic,
      askQuestion
    }
  }
}
</script>