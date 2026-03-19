<template>
  <div v-if="post" class="min-h-screen bg-slate-900 text-white flex flex-col items-center py-10 px-4 relative">
    <!-- Navigation Overlay -->
    <div class="fixed top-0 left-0 w-full p-6 flex justify-between z-50 pointer-events-none">
       <button 
         id="post-detail-back-dashboard" 
         @click="goDashboard"
         class="pointer-events-auto bg-slate-800/80 p-3 rounded-full hover:bg-slate-700 backdrop-blur-sm transition-colors"
       >
         <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
       </button>
       
       <!-- Optional link back to blog posts specifically -->
       <button 
         id="post-detail-back-blog" 
         @click="goBlogPosts"
         class="pointer-events-auto bg-slate-800/80 p-3 rounded-full hover:bg-slate-700 backdrop-blur-sm transition-colors"
       >
         <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" /></svg>
       </button>
    </div>

    <div class="w-full max-w-[600px] bg-slate-800 rounded-lg overflow-hidden shadow-2xl border border-slate-700 mt-8">
      <!-- Header -->
      <div class="p-4 flex items-center gap-3 border-b border-slate-700/50 bg-slate-800">
        <img :src="getBlog(post.blog_id)?.avatar" class="w-10 h-10 rounded-lg object-cover" />
        <div class="flex-1">
          <div class="font-bold text-white">{{ getBlog(post.blog_id)?.name }}</div>
        </div>
        <button class="text-blue-400 font-bold text-sm hover:underline">Follow</button>
      </div>

      <!-- Content -->
      <div class="relative">
         <img v-if="post.type === 'photo'" :src="post.content" class="w-full h-auto" />
         
         <div v-else-if="post.type === 'text'" class="p-8">
            <h1 v-if="post.title" class="text-3xl font-bold mb-4 font-serif">{{ post.title }}</h1>
            <p class="text-lg leading-relaxed text-slate-200 whitespace-pre-line">{{ post.content }}</p>
         </div>

         <div v-else-if="post.type === 'quote'" class="p-10 bg-serif font-serif">
            <blockquote class="text-3xl italic text-white leading-tight mb-6">
              {{ post.content }}
            </blockquote>
            <cite class="block text-right text-slate-400 not-italic">— {{ post.source }}</cite>
         </div>
         
         <div v-if="post.caption" class="p-6 text-slate-300 border-t border-slate-700/30" v-html="post.caption"></div>
      </div>

      <!-- Actions Bar -->
      <div class="p-4 bg-slate-800 border-t border-slate-700 flex justify-between items-center">
         <div class="text-slate-400 font-bold text-sm">{{ post.notes }} notes</div>
         
         <div class="flex gap-4">
           <!-- Reblog -->
           <button 
             id="post-reblog-button" 
             @click="goReblog"
             class="p-2 hover:bg-slate-700 rounded-full transition-colors text-slate-400 hover:text-green-500"
           >
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" /></svg>
           </button>
           
           <!-- Reply (Compose Reply) -->
           <button 
             id="post-reply-button" 
             @click="goReply"
             class="p-2 hover:bg-slate-700 rounded-full transition-colors text-slate-400 hover:text-blue-500"
           >
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6" /></svg>
           </button>

           <!-- Like (Decorative) -->
           <button class="p-2 hover:bg-slate-700 rounded-full transition-colors text-slate-400 hover:text-pink-500">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" /></svg>
           </button>
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
  name: 'POST_DETAIL',
  setup() {
    const route = useRoute()
    const router = useRouter()
    const store = useSignatureStore()
    const dataStore = useDataStore()

    const postId = computed(() => route.params.id || store.selected_post_id)
    const post = computed(() => dataStore.posts.find(p => p.id === postId.value))

    const getBlog = (id) => dataStore.blogs.find(b => b.id === id)

    const goDashboard = async () => {
      store.currentPageId = 'DASHBOARD_FEED'
      await router.push({ name: 'DASHBOARD_FEED' })
    }

    const goBlogPosts = async () => {
      if (post.value) {
         store.selected_blog_id = post.value.blog_id
         store.currentPageId = 'BLOG_POSTS_LIST'
         await router.push({ name: 'BLOG_POSTS_LIST', params: { id: post.value.blog_id } })
      }
    }

    const goReblog = async () => {
      store.currentPageId = 'REBLOG_FORM'
      await router.push({ name: 'REBLOG_FORM', params: { id: postId.value } })
    }

    const goReply = async () => {
      // Mapping to COMPOSE_TEXT_POST as per FSM for reply action, 
      // though typically reply is inline or modal. FSM says "to: COMPOSE_TEXT_POST".
      store.currentPageId = 'COMPOSE_TEXT_POST'
      await router.push({ name: 'COMPOSE_TEXT_POST' })
    }

    onMounted(() => {
      if (!postId.value) router.push({ name: 'DASHBOARD_FEED' })
    })

    return {
      post,
      getBlog,
      goDashboard,
      goBlogPosts,
      goReblog,
      goReply
    }
  }
}
</script>