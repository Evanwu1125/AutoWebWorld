<template>
  <div class="min-h-screen bg-black flex flex-col items-center justify-center p-4 relative">
    <!-- Video Preview Area -->
    <div class="w-full max-w-4xl aspect-video bg-gray-900 rounded-2xl overflow-hidden relative shadow-2xl mb-8 border border-gray-800">
      <div v-if="store.video_on" class="w-full h-full">
         <!-- Placeholder for actual camera feed using ImageGetter -->
         <img src="/images/CameraFeed.jpg" alt="Camera Feed" class="w-full h-full object-cover" />
      </div>
      <div v-else class="w-full h-full flex items-center justify-center flex-col text-gray-500">
         <div class="w-24 h-24 bg-gray-700 rounded-full flex items-center justify-center mb-4">
           <span class="text-3xl text-gray-400 font-bold">{{ getInitials }}</span>
         </div>
         <p>Video is off</p>
      </div>

      <!-- Overlay Controls -->
      <div class="absolute bottom-6 left-1/2 transform -translate-x-1/2 flex gap-6">
        <button 
          id="join-preview-audio-toggle" 
          @click="toggleAudio"
          class="flex flex-col items-center gap-2 group"
        >
          <div class="w-12 h-12 rounded-full flex items-center justify-center transition-all" 
               :class="store.audio_join_with_computer ? 'bg-gray-700 text-white' : 'bg-red-600 text-white'">
             <svg v-if="store.audio_join_with_computer" class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"></path></svg>
             <svg v-else class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><line x1="1" y1="1" x2="23" y2="23"></line><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 9v3a3 3 0 0 0 5.12 2.12M15 9.34V4a3 3 0 0 0-5.94-.6"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16.95A7 7 0 0 1 5 12v-2m14 0v2a7 7 0 0 1-.11 1.23"></path><line x1="12" y1="19" x2="12" y2="23"></line><line x1="8" y1="23" x2="16" y2="23"></line></svg>
          </div>
          <span class="text-xs text-white font-medium">{{ store.audio_join_with_computer ? 'Mute' : 'Unmute' }}</span>
        </button>

        <button 
          id="join-preview-video-toggle" 
          @click="toggleVideo"
          class="flex flex-col items-center gap-2 group"
        >
          <div class="w-12 h-12 rounded-full flex items-center justify-center transition-all" 
               :class="store.video_on ? 'bg-gray-700 text-white' : 'bg-red-600 text-white'">
            <svg v-if="store.video_on" class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
            <svg v-else class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path><line x1="1" y1="1" x2="23" y2="23"></line></svg>
          </div>
          <span class="text-xs text-white font-medium">{{ store.video_on ? 'Stop Video' : 'Start Video' }}</span>
        </button>
      </div>
    </div>

    <div class="w-full max-w-md space-y-4">
      <button 
        id="join-preview-join-button" 
        @click="joinMeeting"
        class="w-full py-4 bg-blue-600 hover:bg-blue-700 text-white font-bold text-lg rounded-lg transition-colors shadow-lg"
      >
        Join Meeting
      </button>
      
      <button 
        id="join-preview-back-button" 
        @click="goBack"
        class="w-full py-3 text-gray-400 hover:text-white font-medium transition-colors"
      >
        Cancel
      </button>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'JOIN_MEETING_PREVIEW',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const getInitials = computed(() => {
      return store.meeting_name_input?.substring(0, 2).toUpperCase() || 'ME';
    });

    const toggleAudio = () => {
      store.audio_join_with_computer = !store.audio_join_with_computer;
      store.handleAction('ACT_JOIN_PREVIEW_TOGGLE_AUDIO');
    };

    const toggleVideo = () => {
      store.video_on = !store.video_on;
      store.handleAction('ACT_JOIN_PREVIEW_TOGGLE_VIDEO');
    };

    const joinMeeting = async () => {
      if (store.handleAction('ACT_JOIN_PREVIEW_JOIN')) {
        await router.push({ name: 'JOIN_MEETING_SUCCESS' });
      }
    };

    const goBack = async () => {
      if (store.handleAction('ACT_JOIN_PREVIEW_BACK_TO_FORM')) {
        await router.push({ name: 'JOIN_MEETING_FORM' });
      }
    };

    return {
      store,
      getInitials,
      toggleAudio,
      toggleVideo,
      joinMeeting,
      goBack
    };
  }
}
</script>