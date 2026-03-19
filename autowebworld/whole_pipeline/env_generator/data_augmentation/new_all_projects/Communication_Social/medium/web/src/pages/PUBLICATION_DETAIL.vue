<template>
  <div class="min-h-screen bg-white">
    <!-- Pub Header -->
    <header v-if="publication" class="bg-gray-50 border-b border-gray-200">
       <div class="max-w-5xl mx-auto px-4 py-12 flex items-center gap-8">
          <img :src="publication.icon" class="w-24 h-24 rounded-lg shadow-sm" />
          <div class="flex-1">
             <h1 class="text-4xl font-bold font-serif mb-2">{{ publication.name }}</h1>
             <p class="text-gray-600 font-serif text-lg mb-6">{{ publication.description }}</p>
             <div class="flex items-center gap-4">
                <button id="publication-follow-button" class="bg-black text-white px-6 py-2 rounded-full text-sm font-sans font-medium hover:bg-gray-800">Follow</button>
                <span class="text-sm text-gray-500 font-sans">{{ (publication.member_count / 1000).toFixed(1) }}k Followers</span>
             </div>
          </div>
       </div>
    </header>
    
    <!-- Nav Strip -->
    <div class="border-b border-gray-200 sticky top-0 bg-white z-20">
       <div class="max-w-5xl mx-auto px-4 h-12 flex items-center gap-8 text-sm font-sans text-gray-500">
          <button id="publication-stories-link" @click="handleOpenStories" class="hover:text-black h-full border-b-2 border-transparent hover:border-gray-300">Stories</button>
          <button class="hover:text-black h-full border-b-2 border-transparent hover:border-gray-300">About</button>
          <button id="publication-submit-story" @click="handleSubmitStory" class="hover:text-black text-green-600 font-medium ml-auto">Submit a story</button>
       </div>
    </div>
    
    <div class="max-w-5xl mx-auto px-4 py-12">
       <button id="publication-back-list" @click="handleBackList" class="flex items-center gap-2 text-gray-400 hover:text-gray-600 mb-8 text-sm font-sans">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Back to all publications
       </button>

       <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
          <!-- Fake stories for the publication -->
          <div v-for="i in 6" :key="i" class="flex flex-col gap-4">
             <div class="aspect-video bg-gray-100 rounded mb-2"></div>
             <h3 class="font-bold font-serif text-xl">Sample Publication Story {{ i }}</h3>
             <p class="text-gray-500 font-serif text-sm line-clamp-3">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.</p>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'PUBLICATION_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    const pubId = route.params.id || signatureStore.publication_selected_id
    const publication = computed(() => dataStore.getPublicationById(pubId))

    const handleOpenStories = async () => {
       // FSM says to POST_LIST. Wait?
       // Yes, ACT_PUBLICATION_DETAIL_OPEN_STORIES -> POST_LIST
       signatureStore.setCurrentPageId('POST_LIST')
       await router.push({ name: 'POST_LIST' })
    }

    const handleSubmitStory = async () => {
       signatureStore.setCurrentPageId('NEW_STORY_EDITOR')
       await router.push({ name: 'NEW_STORY_EDITOR' })
    }

    const handleBackList = async () => {
       signatureStore.setCurrentPageId('PUBLICATION_LIST')
       await router.push({ name: 'PUBLICATION_LIST' })
    }

    return {
       publication,
       handleOpenStories,
       handleSubmitStory,
       handleBackList
    }
  }
}
</script>