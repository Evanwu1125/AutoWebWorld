<template>
  <div class="h-screen flex flex-col bg-gray-900 text-white">
    <!-- Header -->
    <header class="p-4 flex justify-between items-center z-20">
      <button id="meet-now-back-calendar" @click="goBack" class="hover:bg-gray-800 p-2 rounded-full">
        <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
        </svg>
      </button>
      <div class="font-bold">Meet Now</div>
      <div class="w-10"></div>
    </header>

    <main class="flex-1 flex flex-col items-center justify-center p-8">
      <div class="max-w-2xl w-full flex flex-col items-center gap-8">
        
        <!-- Video Preview -->
        <div class="relative bg-black rounded-lg aspect-video w-full flex items-center justify-center overflow-hidden border border-gray-700 shadow-2xl">
          <div v-if="cameraOn" class="w-full h-full bg-gray-800 flex items-center justify-center">
             <!-- Placeholder for camera feed -->
             <img src="https://picsum.photos/800/450" class="w-full h-full object-cover opacity-80" alt="Camera Preview" />
             <div class="absolute bottom-4 left-4 bg-black/50 px-2 py-1 rounded text-xs">You</div>
          </div>
          <div v-else class="flex flex-col items-center">
             <div class="w-24 h-24 bg-purple-600 rounded-full flex items-center justify-center text-3xl font-bold mb-4">AM</div>
             <p class="text-gray-400">Camera is off</p>
          </div>

          <!-- Controls Overlay -->
          <div class="absolute bottom-4 left-0 right-0 flex justify-center gap-4">
             <button 
                id="toggle-mic" 
                @click="toggleMic"
                :class="`p-3 rounded-full transition-colors ${micOn ? 'bg-gray-700 hover:bg-gray-600' : 'bg-red-600 hover:bg-red-700'}`"
             >
                <svg v-if="micOn" xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
                <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" clip-rule="evenodd" />
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2" />
                </svg>
             </button>
             <button 
                id="toggle-camera" 
                @click="toggleCamera"
                :class="`p-3 rounded-full transition-colors ${cameraOn ? 'bg-gray-700 hover:bg-gray-600' : 'bg-red-600 hover:bg-red-700'}`"
             >
                <svg v-if="cameraOn" xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <svg v-else xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M18.364 18.364A9 9 0 005.636 5.636m12.728 12.728A9 9 0 015.636 5.636m12.728 12.728L5.636 5.636" />
                </svg>
             </button>
          </div>
        </div>

        <!-- Meeting Settings -->
        <div class="w-full flex flex-col md:flex-row gap-4 items-center justify-between">
           <div class="flex-1 w-full">
              <label class="block text-sm text-gray-400 mb-1">Meeting Name</label>
              <input 
                id="meet-now-subject-input"
                type="text" 
                v-model="subject"
                placeholder="Name your meeting"
                class="w-full bg-gray-800 border border-gray-700 rounded-md px-4 py-2 text-white focus:ring-[#6264A7] focus:border-[#6264A7]"
              />
           </div>
           <div class="flex-none">
              <button 
                id="start-meet-now-button"
                @click="startMeeting"
                :disabled="!isValid"
                class="bg-[#6264A7] hover:bg-[#52548e] text-white font-bold py-3 px-8 rounded-lg shadow-lg disabled:opacity-50 disabled:cursor-not-allowed transition-all transform hover:scale-105"
              >
                Join now
              </button>
           </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed, watch } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'MEET_NOW_SETUP',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const subject = ref('')
    const cameraOn = ref(false)
    const micOn = ref(false)

    const isValid = computed(() => {
      return subject.value.trim().length > 0
    })

    // Watch for changes and sync to store
    watch(subject, (val) => {
      store.meet_now_subject = val
    })

    const toggleMic = () => {
      micOn.value = !micOn.value
      store.microphone_on = micOn.value
    }

    const toggleCamera = () => {
      cameraOn.value = !cameraOn.value
      store.camera_on = cameraOn.value
    }

    const startMeeting = async () => {
      if (!isValid.value) return;
      store.meet_now_subject = subject.value;
      store.currentPageId = 'MEET_NOW_STARTED_SUCCESS';
      await router.push({ name: 'MEET_NOW_STARTED_SUCCESS' });
    }

    const goBack = async () => {
      store.currentPageId = 'CALENDAR_VIEW';
      await router.push({ name: 'CALENDAR_VIEW' });
    }

    return {
      subject,
      cameraOn,
      micOn,
      isValid,
      toggleMic,
      toggleCamera,
      startMeeting,
      goBack
    }
  }
}
</script>