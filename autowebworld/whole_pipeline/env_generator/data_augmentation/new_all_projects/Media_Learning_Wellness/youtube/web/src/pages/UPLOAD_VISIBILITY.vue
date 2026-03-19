<template>
  <div class="min-h-screen bg-[#0F0F0F] text-white flex flex-col items-center justify-center p-6 relative">
    <div class="absolute inset-0 bg-[#0F0F0F] z-0"></div>
    
    <div class="bg-[#282828] w-full max-w-4xl h-[80vh] rounded-xl shadow-2xl flex flex-col relative z-10 overflow-hidden border border-gray-700">
      <!-- Header -->
      <div class="flex items-center justify-between px-6 py-4 border-b border-gray-700">
        <h2 class="text-xl font-bold">Visibility</h2>
      </div>

      <!-- Content -->
      <div class="flex-1 overflow-y-auto p-8">
        <!-- Step Indicator -->
        <div class="flex items-center gap-4 mb-8 text-sm font-medium text-gray-400 justify-center">
           <span class="text-green-500 flex items-center gap-1"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg> Details</span>
           <span class="w-8 h-[1px] bg-green-500"></span>
           <span class="text-green-500 flex items-center gap-1"><svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg> Video Elements</span>
           <span class="w-8 h-[1px] bg-blue-500"></span>
           <span class="text-blue-500">3. Visibility</span>
        </div>

        <div class="max-w-2xl mx-auto">
           <h3 class="text-lg font-bold mb-2">Save or publish</h3>
           <p class="text-gray-400 text-sm mb-6">Make your video public, unlisted, or private</p>

           <div class="border border-gray-700 rounded-lg p-4 bg-[#1F1F1F]">
              <div 
                 id="visibility-dropdown"
                 @click="isVisibilityOpen = !isVisibilityOpen"
                 class="w-full bg-[#121212] border border-gray-600 rounded px-4 py-3 cursor-pointer flex justify-between items-center"
              >
                 <span class="capitalize font-medium">{{ visibility || 'Select Visibility Mode' }}</span>
                 <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
              </div>

              <div v-if="isVisibilityOpen" class="mt-2 space-y-2 animate-fade-in-up">
                 <div 
                   id="visibility-option-private" 
                   @click="selectVisibility('private')"
                   class="flex items-start gap-3 p-3 hover:bg-gray-700 rounded-lg cursor-pointer"
                 >
                    <div class="mt-1 w-4 h-4 rounded-full border border-gray-400 flex items-center justify-center">
                       <div v-if="visibility === 'private'" class="w-2 h-2 rounded-full bg-white"></div>
                    </div>
                    <div>
                       <div class="font-bold text-sm">Private</div>
                       <div class="text-xs text-gray-400">Only you and people you choose can watch your video</div>
                    </div>
                 </div>

                 <div 
                   id="visibility-option-unlisted" 
                   @click="selectVisibility('unlisted')"
                   class="flex items-start gap-3 p-3 hover:bg-gray-700 rounded-lg cursor-pointer"
                 >
                    <div class="mt-1 w-4 h-4 rounded-full border border-gray-400 flex items-center justify-center">
                       <div v-if="visibility === 'unlisted'" class="w-2 h-2 rounded-full bg-white"></div>
                    </div>
                    <div>
                       <div class="font-bold text-sm">Unlisted</div>
                       <div class="text-xs text-gray-400">Anyone with the video link can watch your video</div>
                    </div>
                 </div>

                 <div 
                   id="visibility-option-public" 
                   @click="selectVisibility('public')"
                   class="flex items-start gap-3 p-3 hover:bg-gray-700 rounded-lg cursor-pointer"
                 >
                    <div class="mt-1 w-4 h-4 rounded-full border border-gray-400 flex items-center justify-center">
                       <div v-if="visibility === 'public'" class="w-2 h-2 rounded-full bg-white"></div>
                    </div>
                    <div>
                       <div class="font-bold text-sm">Public</div>
                       <div class="text-xs text-gray-400">Everyone can watch your video</div>
                    </div>
                 </div>
              </div>
           </div>

           <div class="mt-6 bg-[#1F1F1F] p-4 rounded-lg border border-gray-700">
              <h4 class="font-bold mb-2">Before you publish, check the following:</h4>
              <ul class="text-sm text-gray-400 list-disc list-inside space-y-1">
                 <li>Do children appear in this video?</li>
                 <li>Looking for overall content guidance?</li>
              </ul>
           </div>
        </div>
      </div>

      <!-- Footer -->
      <div class="border-t border-gray-700 p-4 flex justify-between items-center bg-[#1F1F1F]">
         <button 
           id="visibility-back-details" 
           @click="goBackDetails"
           class="text-gray-300 font-bold px-6 py-2 hover:text-white uppercase text-sm"
         >
           Back
         </button>
         <button 
           id="publish-button" 
           @click="publishVideo"
           class="bg-[#3EA6FF] text-black font-bold px-6 py-2 rounded-sm hover:bg-blue-400 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
           :disabled="!store.visibility_selected"
         >
           PUBLISH
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
  name: 'UPLOAD_VISIBILITY',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const visibility = ref(null)
    const isVisibilityOpen = ref(false) // Open by default for better UX in demo

    const selectVisibility = (val) => {
      visibility.value = val
      store.visibility_selected = 'public' // FSM simplifies to length check
    }

    const goBackDetails = () => {
      store.currentPageId = 'UPLOAD_DETAILS'
      router.push({ name: 'UPLOAD_DETAILS' })
    }

    const publishVideo = () => {
      if (store.visibility_selected) {
        store.currentPageId = 'UPLOAD_PUBLISH_SUCCESS'
        router.push({ name: 'UPLOAD_PUBLISH_SUCCESS' })
      }
    }

    return {
      store,
      visibility,
      isVisibilityOpen,
      selectVisibility,
      goBackDetails,
      publishVideo
    }
  }
}
</script>