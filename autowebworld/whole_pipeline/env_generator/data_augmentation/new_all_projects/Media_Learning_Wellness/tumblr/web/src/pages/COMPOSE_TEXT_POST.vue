<template>
  <div class="min-h-screen bg-white flex flex-col items-center pt-10 pb-20 relative">
    <!-- Top Bar -->
    <div class="w-full max-w-2xl px-6 flex justify-between items-center mb-8">
      <div class="font-bold text-gray-400 text-sm uppercase tracking-wide">
        Text Post
      </div>
      <div 
        id="compose-visibility-dropdown"
        class="relative"
      >
        <button 
           @click="visibilityOpen = !visibilityOpen"
           class="flex items-center gap-1 text-gray-500 font-bold hover:text-gray-800"
        >
          {{ store.compose_visibility === 'private' ? 'Private' : 'Public' }} 
          <span class="text-xs">▼</span>
        </button>
        
        <div v-if="visibilityOpen" class="absolute right-0 mt-2 w-32 bg-white rounded-lg shadow-xl border border-gray-100 overflow-hidden z-20">
          <div id="visibility-option-public" @click="setVisibility('public')" class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm font-medium">Public</div>
          <div id="visibility-option-private" @click="setVisibility('private')" class="px-4 py-2 hover:bg-gray-50 cursor-pointer text-sm font-medium">Private</div>
        </div>
      </div>
    </div>

    <!-- Editor -->
    <div class="w-full max-w-2xl bg-white">
      <div 
        id="compose-title-input" 
        @click="focusTitle"
        class="px-6 py-2"
      >
        <input 
          ref="titleInput"
          type="text" 
          placeholder="Title" 
          class="w-full text-4xl font-bold placeholder-gray-300 outline-none font-serif"
          :value="store.compose_title"
          @input="handleTitleInput"
        />
      </div>

      <div 
        id="compose-body-textarea" 
        @click="focusBody"
        class="px-6 py-4 min-h-[300px] cursor-text"
      >
        <textarea 
          ref="bodyInput"
          placeholder="Your text here..." 
          class="w-full h-full min-h-[300px] resize-none outline-none text-lg leading-relaxed placeholder-gray-300 font-serif"
          :value="store.compose_body"
          @input="handleBodyInput"
        ></textarea>
      </div>

      <div 
        id="compose-tags-input" 
        @click="focusTags"
        class="px-6 py-4 border-t border-gray-100 flex items-center gap-2 text-gray-400"
      >
        <span>#</span>
        <input 
          ref="tagsInput"
          type="text" 
          placeholder="tags" 
          class="flex-1 outline-none text-gray-600 italic"
          :value="store.compose_tags"
          @input="handleTagsInput"
        />
      </div>
    </div>

    <!-- Footer Actions -->
    <div class="fixed bottom-0 left-0 w-full p-4 bg-white/90 backdrop-blur-md border-t border-gray-100 flex justify-between items-center max-w-2xl mx-auto right-0">
      <button 
        id="compose-back-dashboard" 
        @click="goDashboard"
        class="px-6 py-2 rounded-full font-bold text-gray-400 hover:bg-gray-100 hover:text-gray-600 transition-colors"
      >
        Close
      </button>

      <div class="flex gap-3">
         <button 
           id="compose-schedule-button" 
           @click="goSchedule"
           :disabled="!isValid"
           :class="[
             'w-10 h-10 rounded-full flex items-center justify-center transition-all',
             isValid ? 'bg-gray-100 text-blue-500 hover:bg-blue-100' : 'bg-gray-50 text-gray-300 cursor-not-allowed'
           ]"
         >
           <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
         </button>
         
         <button 
           id="compose-publish-button" 
           @click="publish"
           :disabled="!isValid"
           :class="[
             'px-8 py-2 rounded-full font-bold text-white transition-all transform shadow-md',
             isValid ? 'bg-blue-500 hover:bg-blue-600 hover:scale-105 shadow-blue-500/30' : 'bg-gray-200 cursor-not-allowed'
           ]"
         >
           Post now
         </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'COMPOSE_TEXT_POST',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const titleInput = ref(null)
    const bodyInput = ref(null)
    const tagsInput = ref(null)
    const visibilityOpen = ref(false)

    const focusTitle = () => titleInput.value?.focus()
    const focusBody = () => bodyInput.value?.focus()
    const focusTags = () => tagsInput.value?.focus()

    const handleTitleInput = (e) => store.compose_title = e.target.value
    const handleBodyInput = (e) => store.compose_body = e.target.value
    const handleTagsInput = (e) => store.compose_tags = e.target.value

    const setVisibility = (val) => {
      store.compose_visibility = val
      visibilityOpen.value = false
    }

    const isValid = computed(() => {
      return (store.compose_body?.length > 0) && (store.compose_visibility?.length > 0)
    })

    const goDashboard = async () => {
      store.currentPageId = 'DASHBOARD_FEED'
      await router.push({ name: 'DASHBOARD_FEED' })
    }

    const goSchedule = async () => {
      if (!isValid.value) return
      store.currentPageId = 'SCHEDULE_POST'
      await router.push({ name: 'SCHEDULE_POST' })
    }

    const publish = async () => {
      if (!isValid.value) return
      store.success_message = "Posted successfully!"
      store.currentPageId = 'POST_PUBLISH_SUCCESS'
      await router.push({ name: 'POST_PUBLISH_SUCCESS' })
    }

    return {
      store,
      titleInput,
      bodyInput,
      tagsInput,
      visibilityOpen,
      focusTitle,
      focusBody,
      focusTags,
      handleTitleInput,
      handleBodyInput,
      handleTagsInput,
      setVisibility,
      isValid,
      goDashboard,
      goSchedule,
      publish
    }
  }
}
</script>