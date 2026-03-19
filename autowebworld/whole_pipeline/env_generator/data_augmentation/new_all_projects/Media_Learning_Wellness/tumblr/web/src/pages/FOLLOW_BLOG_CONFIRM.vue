<template>
  <div v-if="blog" class="min-h-screen bg-slate-900 flex flex-col items-center justify-center p-4">
    <div class="bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl max-w-md w-full overflow-hidden">
      <!-- Header -->
      <div class="p-6 text-center border-b border-slate-700">
        <h2 class="text-xl font-bold text-white">Follow {{ blog.name }}?</h2>
        <p class="text-slate-400 text-sm mt-1">You'll see their posts on your dashboard.</p>
      </div>

      <!-- Body -->
      <div class="p-8 flex flex-col items-center gap-6">
        <img :src="blog.avatar" class="w-24 h-24 rounded-full border-4 border-slate-700 shadow-lg" />
        
        <!-- Optional Note -->
        <div class="w-full">
          <label class="block text-xs font-bold text-slate-500 uppercase mb-2">Add a note (optional)</label>
          <div 
            id="follow-notes-input" 
            @click="focusInput"
            class="bg-slate-900 border border-slate-700 rounded-lg p-3 cursor-text"
          >
            <input 
              ref="noteInput"
              type="text"
              placeholder="Say hello..."
              class="w-full bg-transparent outline-none text-white placeholder-slate-500"
              :value="store.confirm_follow_notes"
              @input="handleInput"
            />
          </div>
        </div>
      </div>

      <!-- Footer Actions -->
      <div class="p-6 bg-slate-900/50 flex gap-4">
        <button 
          id="follow-confirm-back-overview" 
          @click="goBack"
          class="flex-1 py-3 px-4 rounded-lg font-bold text-slate-400 hover:bg-slate-800 transition-colors"
        >
          Cancel
        </button>
        <button 
          id="follow-confirm-submit" 
          @click="submitFollow"
          class="flex-1 py-3 px-4 rounded-lg font-bold text-white bg-blue-500 hover:bg-blue-600 shadow-lg shadow-blue-500/20 transition-all transform hover:scale-105"
        >
          Confirm Follow
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
  name: 'FOLLOW_BLOG_CONFIRM',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()
    const noteInput = ref(null)

    const blogId = computed(() => route.params.id || store.selected_blog_id)
    const blog = computed(() => dataStore.blogs.find(b => b.id === blogId.value))

    const focusInput = () => {
      noteInput.value?.focus()
    }

    const handleInput = (e) => {
      store.confirm_follow_notes = e.target.value
    }

    const goBack = async () => {
      store.currentPageId = 'BLOG_OVERVIEW'
      await router.push({ name: 'BLOG_OVERVIEW', params: { id: blogId.value } })
    }

    const submitFollow = async () => {
      // Logic: Update mock data to reflect following state (optional but good for realism)
      const b = dataStore.blogs.find(b => b.id === blogId.value)
      if (b) b.following = true

      store.success_message = `You are now following ${b.name}!`
      store.currentPageId = 'FOLLOW_BLOG_SUCCESS'
      await router.push({ name: 'FOLLOW_BLOG_SUCCESS' })
    }

    onMounted(() => {
      if (!blogId.value) router.push({ name: 'EXPLORE' })
    })

    return {
      store,
      blog,
      noteInput,
      focusInput,
      handleInput,
      goBack,
      submitFollow
    }
  }
}
</script>