<template>
  <div class="min-h-screen bg-white pb-32">
    <nav class="border-b border-gray-200 bg-white sticky top-0 z-20">
      <div class="max-w-3xl mx-auto px-4 h-16 flex items-center justify-between">
        <div class="flex items-center gap-4">
           <button id="post-detail-back-home" class="text-gray-500 hover:text-gray-900 font-serif font-bold text-xl" @click="handleBackHome">Medium</button>
        </div>
        <div class="flex items-center gap-4">
           <button id="post-detail-back" class="text-sm text-gray-500 hover:text-gray-900 font-sans" @click="handleBackList">Back to list</button>
        </div>
      </div>
    </nav>

    <article v-if="post" class="max-w-2xl mx-auto px-4 py-12">
      <h1 class="text-4xl md:text-5xl font-bold font-serif text-gray-900 mb-4 leading-tight">{{ post.title }}</h1>
      <h2 class="text-2xl font-serif text-gray-500 mb-8 font-normal leading-snug">{{ post.subtitle }}</h2>
      
      <div class="flex items-center justify-between mb-10">
         <div class="flex items-center gap-3 cursor-pointer" id="post-detail-author-link" @click="handleOpenProfile">
            <img :src="author.avatar" class="w-12 h-12 rounded-full border border-gray-100" />
            <div>
               <div class="font-sans font-medium text-gray-900">{{ author.name }}</div>
               <div class="font-sans text-sm text-gray-500">{{ formatDate(post.published_date) }} · {{ post.length_minutes }} min read</div>
            </div>
         </div>
         
         <div class="flex items-center gap-4 border-t border-b border-gray-100 py-3 px-4 rounded-full">
             <button id="post-detail-clap-button" @click="handleClap" class="flex items-center gap-2 text-gray-500 hover:text-black transition-colors group">
               <span class="text-2xl group-hover:scale-110 transition-transform">👏</span>
               <span class="font-sans text-sm">{{ post.claps + (clapped ? 1 : 0) }}</span>
             </button>
             
             <button id="post-detail-add-response" @click="handleOpenComments" class="flex items-center gap-2 text-gray-500 hover:text-black transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <span class="font-sans text-sm">{{ post.responses }}</span>
             </button>
         </div>
      </div>
      
      <img :src="post.image" class="w-full h-auto rounded-sm mb-12" :alt="post.title" />
      
      <div class="prose prose-lg prose-serif max-w-none mb-20 text-gray-800">
         <p>{{ post.content }}</p>
         <p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.</p>
         <p>Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.</p>
      </div>
      
      <!-- Bottom Actions -->
      <div class="flex items-center justify-between border-t border-gray-200 pt-8">
         <div class="flex gap-2">
            <span class="bg-gray-100 px-3 py-1 rounded-full text-sm text-gray-700 font-sans">{{ post.tag }}</span>
         </div>
         <button id="post-detail-bookmark-button" @click="handleBookmark" class="text-gray-400 hover:text-black transition-colors p-2">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" :class="{ 'fill-current text-black': bookmarked }">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
            </svg>
         </button>
      </div>
    </article>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'POST_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    const postId = route.params.id
    const post = computed(() => dataStore.getPostById(postId))
    const author = computed(() => post.value ? dataStore.getUserById(post.value.author_id) : null)
    
    const clapped = ref(false)
    const bookmarked = ref(false)

    const formatDate = (dateStr) => {
      return new Date(dateStr).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
    }

    const handleClap = () => {
       clapped.value = true
       signatureStore.post_clapped = true
    }

    const handleBookmark = () => {
       bookmarked.value = !bookmarked.value
       signatureStore.post_is_bookmarked = true
    }

    const handleOpenComments = async () => {
       signatureStore.setCurrentPageId('COMMENT_FORM')
       await router.push({ name: 'COMMENT_FORM', params: { id: postId } })
    }

    const handleOpenProfile = async () => {
       signatureStore.setCurrentPageId('PROFILE_OVERVIEW')
       await router.push({ name: 'PROFILE_OVERVIEW' })
    }

    const handleBackList = async () => {
       signatureStore.setCurrentPageId('POST_LIST')
       await router.push({ name: 'POST_LIST' })
    }

    const handleBackHome = async () => {
       signatureStore.setCurrentPageId('HOME')
       await router.push({ name: 'HOME' })
    }

    return {
       post,
       author,
       clapped,
       bookmarked,
       formatDate,
       handleClap,
       handleBookmark,
       handleOpenComments,
       handleOpenProfile,
       handleBackList,
       handleBackHome
    }
  }
}
</script>