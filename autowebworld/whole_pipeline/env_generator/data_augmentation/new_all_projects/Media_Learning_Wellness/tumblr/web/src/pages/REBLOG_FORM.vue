<template>
  <div v-if="post" class="min-h-screen bg-slate-900/95 flex items-center justify-center p-4">
    <div class="w-full max-w-2xl bg-white rounded-lg shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
      <!-- Header -->
      <div class="p-4 border-b border-gray-200 flex justify-between items-center bg-gray-50">
        <span class="font-bold text-gray-700">Reblogging {{ getBlog(post.blog_id)?.name }}</span>
        <button 
          id="reblog-back-post" 
          @click="goBack"
          class="text-gray-400 hover:text-gray-600"
        >
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
        </button>
      </div>

      <div class="overflow-y-auto flex-1 p-6">
        <!-- Original Post Preview -->
        <div class="mb-6 border-l-4 border-gray-300 pl-4 opacity-70 pointer-events-none select-none scale-95 origin-left">
           <div v-if="post.type === 'photo'" class="h-32 w-full overflow-hidden rounded-lg mb-2">
             <img :src="post.content" class="w-full h-full object-cover" />
           </div>
           <p v-else class="text-sm font-serif line-clamp-3">{{ post.content }}</p>
           <div class="text-xs text-gray-500 mt-2">Source: {{ getBlog(post.blog_id)?.handle }}</div>
        </div>

        <!-- Inputs -->
        <div class="space-y-4">
           <div 
             id="reblog-textarea"
             @click="focusText"
             class="min-h-[150px] p-4 text-lg outline-none cursor-text text-gray-800"
           >
             <textarea 
               ref="textArea"
               placeholder="Add a caption..." 
               class="w-full h-full resize-none outline-none bg-transparent"
               :value="store.reblog_text"
               @input="handleTextInput"
             ></textarea>
           </div>
           
           <div 
             id="reblog-tags-input" 
             @click="focusTags"
             class="border-t border-gray-100 pt-4 flex items-center gap-2 text-gray-500"
           >
             <span>#</span>
             <input 
               ref="tagsInput"
               type="text" 
               placeholder="Add tags" 
               class="flex-1 outline-none bg-transparent"
               :value="store.reblog_tags"
               @input="handleTagsInput"
             />
           </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="p-4 bg-gray-50 border-t border-gray-200 flex justify-between items-center">
        <button @click="goBack" class="px-6 py-2 rounded-full font-bold text-gray-500 hover:bg-gray-200 transition-colors">Close</button>
        <button 
          id="reblog-submit-button" 
          @click="submitReblog"
          class="px-8 py-2 rounded-full font-bold text-white bg-blue-500 hover:bg-blue-600 transition-all shadow-md transform hover:scale-105"
        >
          Reblog
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'REBLOG_FORM',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()
    
    const textArea = ref(null)
    const tagsInput = ref(null)

    const postId = computed(() => route.params.id || store.selected_post_id)
    const post = computed(() => dataStore.posts.find(p => p.id === postId.value))

    const getBlog = (id) => dataStore.blogs.find(b => b.id === id)

    const focusText = () => textArea.value?.focus()
    const focusTags = () => tagsInput.value?.focus()

    const handleTextInput = (e) => store.reblog_text = e.target.value
    const handleTagsInput = (e) => store.reblog_tags = e.target.value

    const goBack = async () => {
      store.currentPageId = 'POST_DETAIL'
      await router.push({ name: 'POST_DETAIL', params: { id: postId.value } })
    }

    const submitReblog = async () => {
      store.success_message = "Reblogged successfully!"
      store.currentPageId = 'POST_PUBLISH_SUCCESS'
      await router.push({ name: 'POST_PUBLISH_SUCCESS' })
    }

    onMounted(() => {
      if (!postId.value) router.push({ name: 'DASHBOARD_FEED' })
    })

    return {
      store,
      post,
      getBlog,
      textArea,
      tagsInput,
      focusText,
      focusTags,
      handleTextInput,
      handleTagsInput,
      goBack,
      submitReblog
    }
  }
}
</script>