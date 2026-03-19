<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden items-center justify-center">
    <div class="w-full max-w-lg bg-[#282828] rounded-xl shadow-2xl p-8 border border-[#3E3E3E] relative">
       <!-- Close/Back -->
       <div id="back-playlist" @click="handleBackPlaylist" class="absolute top-4 right-4 cursor-pointer text-[#B3B3B3] hover:text-white">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
       </div>

       <h1 class="text-2xl font-bold mb-6 text-center">Share Playlist</h1>

       <div class="space-y-6">
          <!-- Target Dropdown -->
          <div id="share-target-dropdown" class="relative group">
             <label class="block text-xs font-bold uppercase text-[#B3B3B3] mb-2">Share To</label>
             <div class="bg-[#181818] border border-transparent rounded p-3 text-white cursor-pointer flex justify-between items-center hover:bg-[#3E3E3E] transition-colors">
                <span>{{ selectedTargetLabel || 'Select Platform' }}</span>
                <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
             </div>
             <div class="hidden group-hover:block absolute w-full left-0 top-full mt-1 bg-[#282828] border border-[#3E3E3E] rounded shadow-xl z-50">
                <div id="share-target-copy-link" class="px-4 py-3 hover:bg-[#3E3E3E] cursor-pointer" @click="handleSelectTarget('copy_link')">Copy Link</div>
                <div id="share-target-facebook" class="px-4 py-3 hover:bg-[#3E3E3E] cursor-pointer" @click="handleSelectTarget('facebook')">Facebook</div>
                <div id="share-target-twitter" class="px-4 py-3 hover:bg-[#3E3E3E] cursor-pointer" @click="handleSelectTarget('twitter')">Twitter / X</div>
             </div>
          </div>

          <!-- Message -->
          <div>
             <label class="block text-xs font-bold uppercase text-[#B3B3B3] mb-2">Add a Message</label>
             <textarea 
               id="share-message-input"
               v-model="message"
               @input="handleInputMessage"
               rows="3"
               placeholder="Check out this playlist..."
               class="w-full bg-[#181818] border border-transparent focus:border-white rounded p-3 text-white placeholder-[#535353] outline-none transition-colors resize-none"
             ></textarea>
          </div>

          <button 
             id="share-submit-button"
             @click="handleSubmit"
             class="w-full bg-[#1DB954] text-black font-bold py-3 rounded-full uppercase tracking-widest hover:scale-105 transition-transform"
          >
             Share
          </button>
       </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'

export default {
  name: 'PLAYLIST_SHARE',
  setup() {
    const store = useSignatureStore()
    const router = useRouter()

    const message = ref('')
    
    const selectedTargetLabel = computed(() => {
       const map = {
          'copy_link': 'Copy Link',
          'facebook': 'Facebook',
          'twitter': 'Twitter / X'
       }
       return map[store.share_target] || ''
    })

    const handleBackPlaylist = async () => {
       store.setCurrentPageId('PLAYLIST_DETAIL')
       await router.push({ name: 'PLAYLIST_DETAIL' })
    }

    const handleSelectTarget = (val) => {
       store.share_target = val
    }

    const handleInputMessage = () => {
       store.share_message = message.value
    }

    const handleSubmit = async () => {
       if (store.share_target) {
          store.setCurrentPageId('PLAYLIST_SHARED_SUCCESS')
          await router.push({ name: 'PLAYLIST_SHARED_SUCCESS' })
       }
    }

    return {
       message,
       selectedTargetLabel,
       handleBackPlaylist,
       handleSelectTarget,
       handleInputMessage,
       handleSubmit
    }
  }
}
</script>