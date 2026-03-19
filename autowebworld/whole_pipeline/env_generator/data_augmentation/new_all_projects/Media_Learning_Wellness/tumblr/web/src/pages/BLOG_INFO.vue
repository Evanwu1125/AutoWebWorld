<template>
  <div v-if="blog" class="min-h-screen bg-slate-900 text-white p-6 flex flex-col items-center justify-center relative">
    <!-- Back Button -->
    <button 
      id="blog-info-back-overview" 
      @click="goBackOverview"
      class="absolute top-6 left-6 p-2 hover:bg-slate-800 rounded-full transition-colors text-slate-400 hover:text-white"
    >
      <svg xmlns="http://www.w3.org/2000/svg" class="h-8 w-8" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
    </button>

    <div class="max-w-md w-full text-center space-y-8">
      <img :src="blog.avatar" class="w-32 h-32 rounded-full border-4 border-slate-700 mx-auto shadow-2xl" />
      
      <div>
        <h1 class="text-4xl font-bold mb-2">{{ blog.name }}</h1>
        <p class="text-xl text-slate-400 mb-6">{{ blog.handle }}</p>
        
        <div class="bg-slate-800/50 p-8 rounded-2xl border border-slate-700/50 backdrop-blur-sm">
          <p class="text-lg leading-relaxed text-slate-200">
            {{ blog.description }}
          </p>
          <div class="mt-6 pt-6 border-t border-slate-700/50 flex justify-between text-slate-400 font-mono text-sm">
             <span>Registered since 2023</span>
             <span>{{ (blog.followers / 1000).toFixed(1) }}k Followers</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'BLOG_INFO',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const blogId = computed(() => route.params.id || store.selected_blog_id)
    const blog = computed(() => dataStore.blogs.find(b => b.id === blogId.value))

    const goBackOverview = async () => {
      store.currentPageId = 'BLOG_OVERVIEW'
      await router.push({ name: 'BLOG_OVERVIEW', params: { id: blogId.value } })
    }

    onMounted(() => {
      if (!blogId.value) router.push({ name: 'EXPLORE' })
    })

    return {
      blog,
      goBackOverview
    }
  }
}
</script>