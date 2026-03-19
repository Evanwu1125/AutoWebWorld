<template>
  <div class="flex h-screen bg-black text-white font-sans overflow-hidden">
    <aside class="w-64 bg-black flex-shrink-0 p-6 border-r border-[#282828] hidden md:block">
      <div id="back-library" @click="handleBackLibrary" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold mb-8">
         <svg class="w-6 h-6" fill="currentColor" viewBox="0 0 24 24"><path d="M12.5 3.247a1 1 0 0 0-1 0L4 8.75v9a1 1 0 0 0 1 1h5v-5h4v5h5a1 1 0 0 0 1-1v-9l-7.5-5.503z"/></svg>
         <span>Your Library</span>
      </div>
    </aside>

    <main class="flex-1 overflow-y-auto p-8 md:p-12 max-w-3xl mx-auto w-full flex flex-col justify-center">
       <h1 class="text-4xl font-bold mb-8">Create Playlist</h1>

       <div class="bg-[#181818] p-8 rounded-xl border border-[#282828]">
          <div class="flex flex-col md:flex-row gap-8">
             <!-- Cover Placeholder -->
             <div class="w-48 h-48 bg-[#282828] shadow-lg flex items-center justify-center group cursor-pointer relative">
                <svg class="w-16 h-16 text-[#535353] group-hover:text-white transition-colors" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4"/></svg>
                <div class="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-opacity">
                   <span class="text-xs font-bold uppercase tracking-widest">Choose Photo</span>
                </div>
             </div>

             <!-- Form -->
             <div class="flex-1 space-y-6">
                <div>
                   <label class="block text-xs font-bold uppercase text-[#B3B3B3] mb-2">Name</label>
                   <input 
                     id="playlist-name-input"
                     v-model="form.name"
                     @input="handleInputName"
                     type="text" 
                     placeholder="My Awesome Playlist"
                     class="w-full bg-[#3E3E3E] border border-transparent focus:border-white rounded p-3 text-white placeholder-[#B3B3B3] outline-none transition-colors font-bold text-lg"
                   />
                </div>

                <div>
                   <label class="block text-xs font-bold uppercase text-[#B3B3B3] mb-2">Description</label>
                   <textarea 
                     id="playlist-description-input"
                     v-model="form.description"
                     @input="handleInputDescription"
                     rows="3"
                     placeholder="Add an optional description"
                     class="w-full bg-[#3E3E3E] border border-transparent focus:border-white rounded p-3 text-white placeholder-[#B3B3B3] outline-none transition-colors resize-none"
                   ></textarea>
                </div>

                <!-- Visibility -->
                <div id="playlist-visibility-dropdown" class="relative group w-48">
                   <label class="block text-xs font-bold uppercase text-[#B3B3B3] mb-2">Visibility</label>
                   <div class="bg-[#3E3E3E] border border-transparent rounded p-3 text-white cursor-pointer flex justify-between items-center">
                      <span>{{ form.visibility === 'public' ? 'Public' : (form.visibility === 'private' ? 'Private' : 'Select') }}</span>
                      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
                   </div>
                   <div class="hidden group-hover:block absolute w-full left-0 top-full mt-1 bg-[#282828] border border-[#3E3E3E] rounded shadow-xl z-50">
                      <div id="playlist-visibility-public" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer" @click="handleSelectVisibility('public')">Public</div>
                      <div id="playlist-visibility-private" class="px-4 py-2 hover:bg-[#3E3E3E] cursor-pointer" @click="handleSelectVisibility('private')">Private</div>
                   </div>
                </div>
             </div>
          </div>

          <div class="mt-8 pt-6 border-t border-[#282828] flex justify-end">
             <button 
                id="playlist-create-submit"
                @click="handleSubmit"
                class="bg-white text-black font-bold py-3 px-8 rounded-full hover:scale-105 transition-transform uppercase tracking-widest text-sm"
             >
                Create
             </button>
          </div>
       </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'

export default {
  name: 'PLAYLIST_CREATE',
  setup() {
    const store = useSignatureStore()
    const router = useRouter()

    const form = ref({
       name: '',
       description: '',
       visibility: ''
    })

    const handleBackLibrary = async () => {
       store.setCurrentPageId('YOUR_LIBRARY')
       await router.push({ name: 'YOUR_LIBRARY' })
    }

    const handleInputName = () => store.playlist_name = form.value.name
    const handleInputDescription = () => store.playlist_description = form.value.description
    
    const handleSelectVisibility = (val) => {
       form.value.visibility = val
       store.playlist_visibility = val
    }

    const handleSubmit = async () => {
       if (store.playlist_name && store.playlist_visibility) {
          store.setCurrentPageId('PLAYLIST_CREATED_SUCCESS')
          await router.push({ name: 'PLAYLIST_CREATED_SUCCESS' })
       }
    }

    return {
       form,
       handleBackLibrary,
       handleInputName,
       handleInputDescription,
       handleSelectVisibility,
       handleSubmit
    }
  }
}
</script>