<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col items-center justify-center p-6 relative">
    <div class="absolute inset-0 bg-[#0F0F0F] z-0"></div>
    
    <div class="bg-[#282828] w-full max-w-4xl h-[80vh] rounded-xl shadow-2xl flex flex-col relative z-10 overflow-hidden border border-gray-700">
      <!-- Header -->
      <div class="flex items-center justify-between px-6 py-4 border-b border-gray-700">
        <h2 class="text-xl font-bold">Video elements</h2>
      </div>

      <!-- Content -->
      <div class="flex-1 overflow-y-auto p-8">
        <!-- Step Indicator -->
        <div class="flex items-center gap-4 mb-8 text-sm font-medium text-gray-400 justify-center">
           <span class="text-green-500 flex items-center gap-1"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg> Details</span>
           <span class="w-8 h-[1px] bg-blue-500"></span>
           <span class="text-blue-500">2. Video Elements</span>
           <span class="w-8 h-[1px] bg-gray-600"></span>
           <span>3. Visibility</span>
        </div>

        <div class="max-w-2xl mx-auto space-y-8">
           <!-- Tags -->
           <div>
              <label class="block text-lg font-medium mb-2">Tags</label>
              <p class="text-gray-400 text-sm mb-3">Tags can be useful if content in your video is commonly misspelled. Otherwise, tags play a minimal role in helping viewers find your video.</p>
              <input 
                id="tags-input"
                v-model="tags"
                @input="handleTagsInput"
                type="text"
                placeholder="Add tags (comma separated)"
                class="w-full bg-[#1F1F1F] border border-gray-600 rounded px-3 py-2 focus:border-[#3EA6FF] focus:outline-none"
              >
           </div>

           <!-- Thumbnails -->
           <div>
              <label class="block text-lg font-medium mb-2">Thumbnail</label>
              <p class="text-gray-400 text-sm mb-4">Select or upload a picture that shows what's in your video. A good thumbnail stands out and draws viewers' attention.</p>
              
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                 <!-- Upload Custom -->
                 <div class="aspect-video border border-dashed border-gray-600 rounded flex flex-col items-center justify-center cursor-pointer hover:bg-[#333]">
                    <svg class="w-6 h-6 text-gray-400 mb-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                    <span class="text-xs text-gray-400">Upload thumbnail</span>
                 </div>
                 
                 <!-- Option 1 -->
                 <div 
                   id="thumbnail-option-1"
                   @click="selectThumbnail(1)"
                   class="aspect-video bg-gray-800 rounded overflow-hidden cursor-pointer relative group border-2 transition-all"
                   :class="selectedThumbnail === 1 ? 'border-white' : 'border-transparent'"
                 >
                    <div class="absolute inset-0 bg-black/50 group-hover:bg-transparent transition-colors"></div>
                    <div class="absolute inset-0 flex items-center justify-center text-white text-xs font-bold">Auto-generated 1</div>
                 </div>

                 <!-- Option 2 -->
                 <div class="aspect-video bg-gray-800 rounded overflow-hidden relative opacity-50 cursor-not-allowed">
                    <div class="absolute inset-0 flex items-center justify-center text-white text-xs font-bold">Auto-generated 2</div>
                 </div>
                 
                 <!-- Option 3 -->
                 <div class="aspect-video bg-gray-800 rounded overflow-hidden relative opacity-50 cursor-not-allowed">
                    <div class="absolute inset-0 flex items-center justify-center text-white text-xs font-bold">Auto-generated 3</div>
                 </div>
              </div>
           </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="border-t border-gray-700 p-4 flex justify-between items-center bg-[#1F1F1F]">
         <button 
           id="details-back-upload" 
           @click="goBackUpload"
           class="text-gray-300 font-bold px-6 py-2 hover:text-white uppercase text-sm"
         >
           Back
         </button>
         <button 
           id="details-next-visibility" 
           @click="goNextVisibility"
           class="bg-[#3EA6FF] text-black font-bold px-6 py-2 rounded-sm hover:bg-blue-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
           :disabled="!store.thumbnail_selected"
         >
           NEXT
         </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'UPLOAD_DETAILS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const tags = ref('')
    const selectedThumbnail = ref(null)

    const handleTagsInput = () => {
      if (tags.value.length > 0) store.tags_entered = 'typed'
    }

    const selectThumbnail = (id) => {
      selectedThumbnail.value = id
      store.thumbnail_selected = true
    }

    const goBackUpload = () => {
      store.currentPageId = 'UPLOAD_VIDEO'
      router.push({ name: 'UPLOAD_VIDEO' })
    }

    const goNextVisibility = () => {
      if (store.thumbnail_selected) {
        store.currentPageId = 'UPLOAD_VISIBILITY'
        router.push({ name: 'UPLOAD_VISIBILITY' })
      }
    }

    return {
      store,
      tags,
      selectedThumbnail,
      handleTagsInput,
      selectThumbnail,
      goBackUpload,
      goNextVisibility
    }
  }
}
</script>