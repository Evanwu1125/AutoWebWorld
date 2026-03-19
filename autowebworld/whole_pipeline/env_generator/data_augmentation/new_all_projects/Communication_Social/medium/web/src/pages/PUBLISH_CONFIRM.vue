<template>
  <div class="min-h-screen bg-white flex flex-col items-center justify-center p-4">
    <div class="w-full max-w-md text-center space-y-8">
       <h2 class="text-3xl font-bold font-serif">Ready to publish?</h2>
       <p class="text-gray-500 font-sans">Double check your story settings before sending it out to the world.</p>
       
       <div class="bg-gray-50 p-6 rounded-lg border border-gray-100 text-left space-y-4">
          <div class="flex justify-between text-sm font-sans">
             <span class="text-gray-500">Visibility</span>
             <span class="font-medium">Public</span>
          </div>
          <div class="flex justify-between text-sm font-sans">
             <span class="text-gray-500">License</span>
             <span class="font-medium">All rights reserved</span>
          </div>
          
          <button 
             id="publish-confirm-review" 
             @click="handleReview" 
             class="w-full text-center text-green-600 hover:text-green-700 text-sm font-medium font-sans mt-4"
          >
             Review settings (Click to confirm ready)
          </button>
       </div>

       <div class="flex flex-col gap-3">
          <button 
             v-if="readyToPublish"
             id="publish-confirm-publish-now" 
             @click="handlePublishNow" 
             class="w-full bg-green-600 hover:bg-green-700 text-white py-3 rounded-full font-medium font-sans transition-colors shadow-sm"
          >
             Publish Now
          </button>
          
          <button 
             id="publish-confirm-schedule" 
             @click="handleSchedule" 
             class="w-full bg-white border border-gray-300 hover:border-gray-400 text-gray-700 py-3 rounded-full font-medium font-sans transition-colors"
          >
             Schedule for later
          </button>
       </div>
       
       <button id="publish-confirm-back" @click="handleBack" class="text-gray-400 hover:text-gray-600 text-sm font-sans mt-4">
          Back to options
       </button>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'PUBLISH_CONFIRM',
  setup() {
    const router = useRouter()
    const signatureStore = useSignatureStore()
    
    const readyToPublish = computed(() => signatureStore.ready_to_publish === true)

    const handleReview = () => {
       signatureStore.ready_to_publish = true
    }

    const handlePublishNow = async () => {
       if (readyToPublish.value) {
          signatureStore.setCurrentPageId('PUBLISH_POST_SUCCESS')
          await router.push({ name: 'PUBLISH_POST_SUCCESS' })
       }
    }

    const handleSchedule = async () => {
       signatureStore.setCurrentPageId('SCHEDULE_PICKER')
       await router.push({ name: 'SCHEDULE_PICKER' })
    }

    const handleBack = async () => {
       signatureStore.setCurrentPageId('PUBLISH_OPTIONS')
       await router.push({ name: 'PUBLISH_OPTIONS' })
    }

    return {
       readyToPublish,
       handleReview,
       handlePublishNow,
       handleSchedule,
       handleBack
    }
  }
}
</script>