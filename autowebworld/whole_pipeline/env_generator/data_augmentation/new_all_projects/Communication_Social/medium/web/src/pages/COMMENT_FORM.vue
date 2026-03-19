<template>
  <div class="fixed inset-0 bg-white z-50 flex flex-col">
    <!-- Header -->
    <div class="flex items-center justify-between px-6 py-4 border-b border-gray-100">
       <h2 class="text-lg font-bold font-sans">Responses ({{ post.responses }})</h2>
       <button id="comment-back" @click="handleBack" class="text-gray-500 hover:text-gray-900">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
          </svg>
       </button>
    </div>

    <!-- Content -->
    <div class="flex-1 overflow-y-auto p-6 max-w-2xl mx-auto w-full">
       <div class="bg-white rounded shadow-sm p-4 border border-gray-200 mb-8">
          <div class="flex items-center gap-3 mb-4">
             <img :src="currentUser.avatar" class="w-8 h-8 rounded-full" />
             <span class="text-sm font-medium font-sans">{{ currentUser.name }}</span>
          </div>
          <textarea 
            id="comment-editor" 
            v-model="commentText" 
            placeholder="What are your thoughts?" 
            class="w-full min-h-[120px] p-0 border-none resize-none focus:ring-0 text-lg font-serif placeholder-gray-400"
          ></textarea>
          
          <div class="flex items-center justify-between mt-4 pt-4 border-t border-gray-50">
             <div class="flex gap-2">
                <button class="p-2 text-gray-400 hover:text-gray-600 rounded-full hover:bg-gray-100">
                   <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                     <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                   </svg>
                </button>
             </div>
             <div class="flex gap-3">
                <button 
                   v-if="commentText.length > 0"
                   id="comment-preview-button" 
                   @click="handlePreview" 
                   class="px-4 py-2 text-sm font-medium text-gray-600 hover:text-gray-900 font-sans transition-colors"
                >
                   Preview
                </button>
                <button 
                   id="comment-submit-button" 
                   @click="handleSubmit" 
                   :disabled="commentText.length === 0"
                   :class="{
                      'px-4 py-2 rounded-full text-sm font-medium font-sans transition-colors': true,
                      'bg-green-600 text-white hover:bg-green-700': commentText.length > 0,
                      'bg-green-200 text-white cursor-not-allowed': commentText.length === 0
                   }"
                >
                   Respond
                </button>
             </div>
          </div>
       </div>

       <!-- Preview Area -->
       <div v-if="previewShown" class="bg-gray-50 p-6 rounded-lg mb-8 border border-gray-200">
          <h4 class="text-xs font-bold uppercase tracking-wide text-gray-500 mb-4 font-sans">Preview</h4>
          <div class="prose prose-serif">
             <p>{{ commentText }}</p>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'COMMENT_FORM',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    const postId = route.params.id || signatureStore.post_detail_post_id || 'post_1' // Fallback for dev
    const post = computed(() => dataStore.getPostById(postId))
    const currentUser = computed(() => dataStore.getUserById(signatureStore.current_user_id))
    
    const commentText = ref('')
    const previewShown = ref(false)

    const handlePreview = () => {
       if (commentText.value.length > 0) {
          previewShown.value = true
          signatureStore.comment_text = commentText.value
          signatureStore.comment_preview_shown = true
       }
    }

    const handleSubmit = async () => {
       if (commentText.value.length > 0) {
          signatureStore.comment_text = commentText.value
          signatureStore.setCurrentPageId('COMMENT_SUBMIT_SUCCESS')
          await router.push({ name: 'COMMENT_SUBMIT_SUCCESS' })
       }
    }

    const handleBack = async () => {
       signatureStore.setCurrentPageId('POST_DETAIL')
       await router.push({ name: 'POST_DETAIL', params: { id: postId } })
    }

    return {
       post,
       currentUser,
       commentText,
       previewShown,
       handlePreview,
       handleSubmit,
       handleBack
    }
  }
}
</script>