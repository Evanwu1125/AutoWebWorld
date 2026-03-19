<template>
  <div class="min-h-screen bg-white">
    <!-- Editor Nav -->
    <nav class="max-w-4xl mx-auto px-4 h-16 flex items-center justify-between">
       <div class="flex items-center gap-4">
          <button id="editor-back-home" @click="handleBackHome" class="p-2 hover:bg-gray-100 rounded-full transition-colors">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
             </svg>
          </button>
          <span class="text-sm text-gray-500 font-sans">Draft in {{ currentUser.name }}</span>
       </div>
       <div class="flex items-center gap-4">
          <!-- Check Ready Button -->
          <button 
             v-if="!canPublish"
             id="editor-check-ready" 
             @click="handleCheckReady"
             class="px-3 py-1.5 text-sm text-green-600 hover:text-green-700 font-sans font-medium"
          >
             Check if ready
          </button>
          
          <!-- Publish Button (Opens Modal) -->
          <button 
             v-if="canPublish"
             id="editor-open-publish"
             @click="handleOpenPublish"
             class="bg-green-600 hover:bg-green-700 text-white px-4 py-1.5 rounded-full text-sm font-sans font-medium transition-colors"
          >
             Publish
          </button>
          
          <button class="text-gray-400 hover:text-gray-900">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
             </svg>
          </button>
          <img :src="currentUser.avatar" class="w-8 h-8 rounded-full" />
       </div>
    </nav>

    <!-- Editor Area -->
    <main class="max-w-3xl mx-auto px-4 py-12">
       <!-- Title -->
       <input 
          id="editor-title"
          v-model="title"
          type="text" 
          placeholder="Title" 
          class="w-full text-5xl font-serif font-bold border-none focus:ring-0 placeholder-gray-300 p-0 mb-4"
       />
       
       <!-- Subtitle -->
       <input 
          id="editor-subtitle"
          v-model="subtitle"
          type="text" 
          placeholder="Tell your story..." 
          class="w-full text-2xl font-serif text-gray-500 border-none focus:ring-0 placeholder-gray-300 p-0 mb-8"
       />

       <!-- Body -->
       <textarea 
          id="editor-body"
          v-model="body"
          placeholder="Start writing..." 
          class="w-full min-h-[400px] text-xl font-serif leading-relaxed border-none focus:ring-0 resize-none placeholder-gray-300 p-0 mb-12"
       ></textarea>
       
       <!-- Tag Selector -->
       <div class="relative inline-block w-full max-w-xs">
          <label class="block text-sm font-medium text-gray-500 mb-2 font-sans uppercase tracking-wide">Add a topic</label>
          <div 
             id="editor-tag-dropdown" 
             @click="toggleTagMenu"
             class="w-full p-3 bg-gray-50 rounded border border-gray-200 cursor-pointer flex justify-between items-center font-sans text-sm"
          >
             <span :class="selectedTag ? 'text-black' : 'text-gray-400'">{{ selectedTag ? selectedTag : 'Select a topic...' }}</span>
             <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
             </svg>
          </div>
          
          <div v-if="tagMenuOpen" class="absolute top-full left-0 w-full mt-1 bg-white border border-gray-100 shadow-lg rounded-md z-10">
             <div id="editor-tag-option-tech" @click="handleSelectTag('technology')" class="px-4 py-2 hover:bg-gray-50 cursor-pointer font-sans text-sm">Technology</div>
             <div id="editor-tag-option-culture" @click="handleSelectTag('culture')" class="px-4 py-2 hover:bg-gray-50 cursor-pointer font-sans text-sm">Culture</div>
             <div id="editor-tag-option-productivity" @click="handleSelectTag('productivity')" class="px-4 py-2 hover:bg-gray-50 cursor-pointer font-sans text-sm">Productivity</div>
          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import { useDataStore } from '../stores/data'

export default {
  name: 'NEW_STORY_EDITOR',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    const dataStore = useDataStore()
    
    const currentUser = computed(() => dataStore.getUserById(signatureStore.current_user_id))
    
    const title = ref('')
    const subtitle = ref('')
    const body = ref('')
    const selectedTag = ref(null)
    const tagMenuOpen = ref(false)
    
    const canPublish = computed(() => signatureStore.draft_can_publish === true)

    // Watch inputs to update store
    watch(title, (val) => { if(val) signatureStore.draft_title = 'typed' })
    watch(subtitle, (val) => { if(val) signatureStore.draft_subtitle = 'typed' })
    watch(body, (val) => { if(val) signatureStore.draft_body = 'typed' })

    const toggleTagMenu = () => {
       tagMenuOpen.value = !tagMenuOpen.value
    }

    const handleSelectTag = (tag) => {
       selectedTag.value = tag
       signatureStore.draft_tag = 'selected'
       tagMenuOpen.value = false
    }

    const handleCheckReady = () => {
       if (title.value.length > 0 && body.value.length > 0) {
          signatureStore.draft_can_publish = true
       }
    }

    const handleOpenPublish = async () => {
       if (canPublish.value) {
          signatureStore.setCurrentPageId('PUBLISH_OPTIONS')
          await router.push({ name: 'PUBLISH_OPTIONS' })
       }
    }

    const handleBackHome = async () => {
       signatureStore.setCurrentPageId('HOME')
       await router.push({ name: 'HOME' })
    }

    return {
       currentUser,
       title,
       subtitle,
       body,
       selectedTag,
       tagMenuOpen,
       canPublish,
       toggleTagMenu,
       handleSelectTag,
       handleCheckReady,
       handleOpenPublish,
       handleBackHome
    }
  }
}
</script>