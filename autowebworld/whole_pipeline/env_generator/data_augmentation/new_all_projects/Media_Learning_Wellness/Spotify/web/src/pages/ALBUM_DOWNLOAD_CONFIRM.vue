<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden items-center justify-center">
    <div class="w-full max-w-md bg-[#242424] rounded-xl shadow-2xl p-8 border border-[#3E3E3E] text-center">
       <h1 class="text-2xl font-bold mb-4">Download Album?</h1>
       <p class="text-[#B3B3B3] mb-8">This will use approximately 150MB of data or storage.</p>

       <div class="flex flex-col space-y-4">
          <button 
             id="confirm-download-button"
             @click="handleConfirm"
             class="w-full bg-[#1DB954] text-black font-bold py-3 rounded-full uppercase tracking-widest hover:scale-105 transition-transform"
          >
             Download Now
          </button>
          
          <button 
             id="back-album"
             @click="handleBackAlbum"
             class="w-full bg-transparent border border-[#727272] hover:border-white text-white font-bold py-3 rounded-full transition-colors"
          >
             Cancel
          </button>
       </div>
    </div>
  </div>
</template>

<script>
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'

export default {
  name: 'ALBUM_DOWNLOAD_CONFIRM',
  setup() {
    const store = useSignatureStore()
    const router = useRouter()

    const handleConfirm = async () => {
       store.album_download_confirmed = true
       store.setCurrentPageId('ALBUM_DOWNLOAD_SUCCESS')
       await router.push({ name: 'ALBUM_DOWNLOAD_SUCCESS' })
    }

    const handleBackAlbum = async () => {
       store.setCurrentPageId('ALBUM_DETAIL')
       await router.push({ name: 'ALBUM_DETAIL' })
    }

    return {
       handleConfirm,
       handleBackAlbum
    }
  }
}
</script>