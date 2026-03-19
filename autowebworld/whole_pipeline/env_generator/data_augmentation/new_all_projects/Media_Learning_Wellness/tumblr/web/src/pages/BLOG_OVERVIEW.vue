<template>
  <div v-if="blog" class="min-h-screen bg-slate-900 text-white pb-20">
    <!-- Hero Header -->
    <div class="h-[40vh] md:h-[50vh] relative">
      <img :src="blog.cover" class="w-full h-full object-cover" alt="Cover" />
      <div class="absolute inset-0 bg-gradient-to-t from-slate-900 to-transparent"></div>
      
      <!-- Navigation Overlay -->
      <button 
        id="blog-overview-back-explore" 
        @click="goBackExplore"
        class="absolute top-6 left-6 bg-black/50 hover:bg-black/70 p-2 rounded-full text-white backdrop-blur-sm transition-all"
      >
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
      </button>
    </div>

    <!-- Blog Info Container -->
    <div class="max-w-4xl mx-auto px-6 -mt-32 relative z-10">
      <div class="flex flex-col md:flex-row items-end md:items-center gap-6 mb-8">
        <img :src="blog.avatar" class="w-32 h-32 md:w-40 md:h-40 rounded-2xl border-4 border-slate-900 shadow-2xl bg-slate-800 object-cover" alt="Avatar" />
        <div class="flex-1 mb-2">
          <h1 class="text-4xl md:text-5xl font-bold tracking-tight mb-2">{{ blog.name }}</h1>
          <p class="text-slate-400 font-mono text-lg">{{ blog.handle }}</p>
        </div>
        
        <!-- Follow Button -->
        <button 
          id="blog-follow-button" 
          @click="goFollowConfirm"
          class="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-8 rounded-full shadow-lg shadow-blue-500/30 transform hover:scale-105 transition-all mb-4 md:mb-0"
        >
          Follow
        </button>
      </div>

      <!-- Navigation Links -->
      <div class="grid grid-cols-2 gap-4 max-w-md mb-12">
        <button 
          id="blog-overview-posts-link" 
          @click="goPosts"
          class="bg-slate-800 hover:bg-slate-700 border border-slate-700 p-6 rounded-xl text-left group transition-all"
        >
          <div class="text-2xl mb-2 group-hover:scale-110 origin-left transition-transform">📝</div>
          <div class="font-bold text-lg">Posts</div>
          <div class="text-slate-400 text-sm">View all entries</div>
        </button>

        <button 
          id="blog-overview-info-link" 
          @click="goInfo"
          class="bg-slate-800 hover:bg-slate-700 border border-slate-700 p-6 rounded-xl text-left group transition-all"
        >
          <div class="text-2xl mb-2 group-hover:scale-110 origin-left transition-transform">ℹ️</div>
          <div class="font-bold text-lg">About</div>
          <div class="text-slate-400 text-sm">Bio & Info</div>
        </button>
      </div>

      <!-- Latest 3 Posts Preview (Decorative) -->
      <h3 class="text-xl font-bold mb-6 border-b border-slate-800 pb-2">Latest Updates</h3>
      <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div v-for="post in latestPosts" :key="post.id" class="aspect-square bg-slate-800 rounded-lg overflow-hidden relative group cursor-not-allowed opacity-70">
           <img v-if="post.type === 'photo'" :src="post.content" class="w-full h-full object-cover" />
           <div v-else class="p-4 text-sm text-slate-300">{{ post.title || 'Text Post' }}</div>
        </div>
      </div>
    </div>
  </div>
  
  <div v-else class="min-h-screen flex items-center justify-center bg-slate-900 text-white">
    <p>Loading blog...</p>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'BLOG_OVERVIEW',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const blogId = computed(() => route.params.id || store.selected_blog_id)
    const blog = computed(() => dataStore.blogs.find(b => b.id === blogId.value))

    // Precondition Check
    onMounted(() => {
      if (!blogId.value) {
        // Fallback or error handling
        router.push({ name: 'EXPLORE' })
      }
      // Ensure store is synced if came via URL
      if (store.selected_blog_id !== blogId.value) {
        store.selected_blog_id = blogId.value
      }
    })

    const latestPosts = computed(() => {
       return dataStore.posts.filter(p => p.blog_id === blogId.value).slice(0, 3)
    })

    const goBackExplore = async () => {
      store.currentPageId = 'EXPLORE'
      await router.push({ name: 'EXPLORE' })
    }

    const goPosts = async () => {
      store.currentPageId = 'BLOG_POSTS_LIST'
      await router.push({ name: 'BLOG_POSTS_LIST', params: { id: blogId.value } })
    }

    const goInfo = async () => {
      store.currentPageId = 'BLOG_INFO'
      await router.push({ name: 'BLOG_INFO', params: { id: blogId.value } })
    }

    const goFollowConfirm = async () => {
      store.currentPageId = 'FOLLOW_BLOG_CONFIRM'
      await router.push({ name: 'FOLLOW_BLOG_CONFIRM', params: { id: blogId.value } })
    }

    return {
      blog,
      latestPosts,
      goBackExplore,
      goPosts,
      goInfo,
      goFollowConfirm
    }
  }
}
</script>