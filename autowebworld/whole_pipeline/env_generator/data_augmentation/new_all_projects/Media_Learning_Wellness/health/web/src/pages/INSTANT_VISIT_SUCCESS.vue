<template>
  <div class="min-h-screen bg-gray-900 flex flex-col items-center justify-center text-white px-4">
    <!-- Simulated Video Call UI -->
    <div class="relative w-full max-w-4xl bg-black rounded-2xl overflow-hidden shadow-2xl aspect-video border-4 border-gray-800">
       <img src="/images/VideoCallDoctor.jpg" alt="Doctor on video call" class="w-full h-full object-cover opacity-80" />
       
       <div class="absolute inset-0 flex items-center justify-center bg-black/40">
          <div class="text-center">
             <div class="mx-auto h-16 w-16 rounded-full bg-green-500 flex items-center justify-center mb-4">
               <svg class="h-8 w-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
             </div>
             <h2 class="text-3xl font-bold mb-2">Visit Complete</h2>
             <p class="text-gray-300">Thank you for using Teladoc Health.</p>
          </div>
       </div>

       <!-- Controls Overlay -->
       <div class="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/80 to-transparent">
          <div class="flex justify-center space-x-6">
             <button
               id="instant-success-go-dashboard"
               @click="handleGoDashboard"
               class="bg-gray-700 hover:bg-gray-600 text-white px-6 py-3 rounded-full font-medium transition-colors"
             >
               Go to Dashboard
             </button>
             <button
               id="instant-success-go-home"
               @click="handleGoHome"
               class="bg-[#009CDE] hover:bg-[#007bb0] text-white px-6 py-3 rounded-full font-bold transition-colors shadow-lg"
             >
               Return Home
             </button>
          </div>
       </div>
    </div>
  </div>
</template>

<script>
import { onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'INSTANT_VISIT_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleGoHome = async () => {
      // ACT_INSTANT_SUCCESS_GO_HOME
      // Effect: visit_started = true
      store.visit_started = true
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    const handleGoDashboard = async () => {
      // ACT_INSTANT_SUCCESS_GO_DASH
      store.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    return {
      handleGoHome,
      handleGoDashboard
    }
  }
}
</script>