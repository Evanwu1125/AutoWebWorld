<template>
  <div class="min-h-screen bg-white flex items-center justify-center p-4">
    <div class="w-full max-w-2xl grid grid-cols-1 md:grid-cols-2 gap-12">
       <!-- Left: Preview -->
       <div class="space-y-4">
          <div class="font-bold font-sans text-sm uppercase text-gray-500">Story Preview</div>
          <div class="bg-gray-50 aspect-[4/3] flex items-center justify-center text-gray-300 rounded">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
               <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
             </svg>
          </div>
          <h3 class="font-bold font-serif text-xl leading-tight">Story Title Preview</h3>
          <p class="text-gray-500 font-serif text-sm">Story subtitle preview...</p>
          <div class="text-xs text-gray-400 font-sans">Note: Changes here will affect how your story appears in public places like Medium’s homepage.</div>
       </div>

       <!-- Right: Options -->
       <div class="flex flex-col justify-between">
          <div class="space-y-8">
             <div>
                <div class="font-sans text-sm mb-2">Publishing to: <strong>Self</strong></div>
                <p class="text-xs text-gray-500 mb-4 font-sans">Add or change tags (up to 5) so readers know what your story is about.</p>
             </div>

             <div class="space-y-4">
                <label class="flex items-start gap-3 cursor-pointer group">
                   <input 
                      type="checkbox" 
                      id="publish-option-publication-checkbox" 
                      v-model="toPublication"
                      @change="handleTogglePublication"
                      class="mt-1 rounded text-green-600 focus:ring-green-500 border-gray-300" 
                   />
                   <div>
                      <span class="block font-medium font-sans text-sm text-gray-900 group-hover:text-black">Submit to publication</span>
                      <span class="block text-xs text-gray-500">Submit this story to a publication you are a writer for.</span>
                   </div>
                </label>

                <label class="flex items-start gap-3 cursor-pointer group">
                   <input 
                      type="checkbox" 
                      id="publish-option-responses-checkbox" 
                      v-model="allowResponses"
                      @change="handleToggleResponses"
                      class="mt-1 rounded text-green-600 focus:ring-green-500 border-gray-300" 
                   />
                   <div>
                      <span class="block font-medium font-sans text-sm text-gray-900 group-hover:text-black">Allow responses</span>
                      <span class="block text-xs text-gray-500">Let readers leave responses on your story.</span>
                   </div>
                </label>
             </div>
          </div>

          <div class="mt-12 flex items-center justify-end gap-4">
             <button id="publish-options-back" @click="handleBack" class="px-4 py-2 text-sm font-sans text-gray-500 hover:text-gray-900">Back</button>
             <button id="publish-options-continue" @click="handleContinue" class="bg-green-600 hover:bg-green-700 text-white px-6 py-2 rounded-full text-sm font-sans font-medium">Publish now</button>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PUBLISH_OPTIONS',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    
    const toPublication = ref(false)
    const allowResponses = ref(false)

    const handleTogglePublication = () => {
       signatureStore.publish_to_publication = true
    }

    const handleToggleResponses = () => {
       signatureStore.allow_responses = true
    }

    const handleContinue = async () => {
       signatureStore.selected_publish_option = 'now'
       signatureStore.setCurrentPageId('PUBLISH_CONFIRM')
       await router.push({ name: 'PUBLISH_CONFIRM' })
    }

    const handleBack = async () => {
       signatureStore.setCurrentPageId('NEW_STORY_EDITOR')
       await router.push({ name: 'NEW_STORY_EDITOR' })
    }

    return {
       toPublication,
       allowResponses,
       handleTogglePublication,
       handleToggleResponses,
       handleContinue,
       handleBack
    }
  }
}
</script>