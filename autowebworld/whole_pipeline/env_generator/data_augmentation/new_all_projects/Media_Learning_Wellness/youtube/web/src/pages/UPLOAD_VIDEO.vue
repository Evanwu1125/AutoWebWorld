<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col items-center justify-center p-6 relative">
    <!-- Background Overlay -->
    <div class="absolute inset-0 bg-[#0F0F0F] z-0"></div>
    
    <div class="bg-[#282828] w-full max-w-4xl h-[80vh] rounded-xl shadow-2xl flex flex-col relative z-10 overflow-hidden border border-gray-700">
      <!-- Header -->
      <div class="flex items-center justify-between px-6 py-4 border-b border-gray-700">
        <h2 class="text-xl font-bold">Upload videos</h2>
        <div class="flex gap-4">
          <button id="upload-feedback" class="text-gray-400 hover:text-white"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"></path></svg></button>
          <button id="upload-cancel" @click="goHome" class="text-gray-400 hover:text-white"><svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg></button>
        </div>
      </div>

      <!-- Content -->
      <div class="flex-1 overflow-y-auto p-8 flex flex-col items-center">
        <!-- Step Indicator -->
        <div class="flex items-center gap-4 mb-8 text-sm font-medium text-gray-400">
           <span class="text-blue-500">1. Details</span>
           <span class="w-8 h-[1px] bg-gray-600"></span>
           <span>2. Video Elements</span>
           <span class="w-8 h-[1px] bg-gray-600"></span>
           <span>3. Visibility</span>
        </div>

        <!-- File Select Area -->
        <div 
          id="upload-file-input"
          @click="selectFile"
          class="w-full max-w-2xl border-2 border-dashed border-gray-600 rounded-xl h-48 flex flex-col items-center justify-center cursor-pointer hover:bg-[#333] hover:border-gray-500 transition-all mb-8 group"
          :class="{'border-blue-500 bg-blue-900/10': store.file_selected}"
        >
          <div class="w-16 h-16 bg-[#1F1F1F] rounded-full flex items-center justify-center mb-4 group-hover:-translate-y-2 transition-transform shadow-lg">
             <svg class="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
          </div>
          <p class="text-gray-300 font-medium mb-1" v-if="!store.file_selected">Drag and drop video files to upload</p>
          <p class="text-gray-300 font-medium mb-1" v-else>File Selected: my_awesome_video.mp4</p>
          <p class="text-gray-500 text-xs">Your videos will be private until you publish them.</p>
          <button class="mt-4 bg-[#3EA6FF] text-black font-bold px-4 py-2 rounded-sm hover:bg-blue-400 transition-colors uppercase text-sm">Select Files</button>
        </div>

        <!-- Meta Inputs (Shown mostly after file select, but visible for FSM flow) -->
        <div class="w-full max-w-2xl space-y-6 animate-fade-in-up">
           <div class="group">
              <div class="flex justify-between mb-1">
                 <label class="text-sm font-medium text-gray-300">Title (required)</label>
                 <span class="text-xs text-gray-500">0/100</span>
              </div>
              <input 
                id="upload-title-input"
                v-model="title"
                @input="handleTitleInput"
                type="text"
                placeholder="Add a title that describes your video"
                class="w-full bg-[#1F1F1F] border border-gray-600 rounded px-3 py-2 focus:border-[#3EA6FF] focus:outline-none transition-colors"
              >
           </div>

           <div class="group">
              <div class="flex justify-between mb-1">
                 <label class="text-sm font-medium text-gray-300">Description</label>
                 <span class="text-xs text-gray-500">0/5000</span>
              </div>
              <textarea 
                id="upload-description-input"
                v-model="description"
                @input="handleDescriptionInput"
                placeholder="Tell viewers about your video"
                rows="4"
                class="w-full bg-[#1F1F1F] border border-gray-600 rounded px-3 py-2 focus:border-[#3EA6FF] focus:outline-none transition-colors resize-none"
              ></textarea>
           </div>

           <div class="group relative">
              <label class="block text-sm font-medium text-gray-300 mb-1">Audience</label>
              <div 
                 id="audience-dropdown"
                 @click="isAudienceOpen = !isAudienceOpen"
                 class="w-full bg-[#1F1F1F] border border-gray-600 rounded px-3 py-2 cursor-pointer flex justify-between items-center"
              >
                 <span class="capitalize">{{ audience || 'Select Audience' }}</span>
                 <svg class="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
              </div>

              <div v-if="isAudienceOpen" class="absolute w-full bg-[#282828] border border-gray-600 rounded mt-1 z-20 shadow-xl">
                 <div id="audience-option-public" @click="selectAudience('public')" class="px-3 py-2 hover:bg-gray-700 cursor-pointer">Yes, it's made for kids</div>
                 <div id="audience-option-unlisted" @click="selectAudience('unlisted')" class="px-3 py-2 hover:bg-gray-700 cursor-pointer">No, it's not made for kids</div>
                 <div id="audience-option-private" @click="selectAudience('private')" class="px-3 py-2 hover:bg-gray-700 cursor-pointer">Age restriction (advanced)</div>
              </div>
           </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="border-t border-gray-700 p-4 flex justify-between items-center bg-[#1F1F1F]">
         <div class="flex items-center gap-2">
            <div class="w-6 h-6 rounded-full border-2 border-gray-500 border-t-blue-500 animate-spin" v-if="uploading"></div>
            <span class="text-xs text-gray-400" v-if="uploading">Uploading... 45%</span>
            <span class="text-xs text-gray-400" v-else>Upload complete</span>
         </div>
         <button 
           id="upload-next-details" 
           @click="goNextDetails"
           class="bg-[#3EA6FF] text-black font-bold px-6 py-2 rounded-sm hover:bg-blue-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
           :disabled="!isValid"
         >
           NEXT
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
  name: 'UPLOAD_VIDEO',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const title = ref('')
    const description = ref('')
    const audience = ref(null)
    const isAudienceOpen = ref(false)
    const uploading = ref(false)

    // FSM Logic Requirements
    const isValid = computed(() => {
      return store.file_selected && store.title_entered
    })

    const selectFile = () => {
      uploading.value = true
      setTimeout(() => uploading.value = false, 1500)
      store.file_selected = true
    }

    const handleTitleInput = () => {
      if (title.value.length > 0) store.title_entered = 'typed'
      else store.title_entered = null
    }

    const handleDescriptionInput = () => {
      if (description.value.length > 0) store.description_entered = 'typed'
    }

    const selectAudience = (val) => {
      audience.value = val
      store.audience_selected = 'public' // FSM sets simplified value
      isAudienceOpen.value = false
    }

    const goNextDetails = () => {
      if (isValid.value) {
        store.currentPageId = 'UPLOAD_DETAILS'
        router.push({ name: 'UPLOAD_DETAILS' })
      }
    }

    const goHome = () => {
      store.currentPageId = 'HOME'
      router.push({ name: 'HOME' })
    }

    return {
      store,
      title,
      description,
      audience,
      isAudienceOpen,
      uploading,
      isValid,
      selectFile,
      handleTitleInput,
      handleDescriptionInput,
      selectAudience,
      goNextDetails,
      goHome
    }
  }
}
</script>