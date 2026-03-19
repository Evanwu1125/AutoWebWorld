<template>
  <div class="min-h-screen bg-gray-900 flex flex-col items-center justify-center p-4 text-white">
    <h1 class="text-3xl font-bold mb-2">Meeting Lobby</h1>
    <p class="text-gray-400 mb-10">Choose your audio and video settings before starting.</p>

    <div class="w-full max-w-md bg-gray-800 rounded-xl p-8 shadow-2xl border border-gray-700">
      <!-- Audio Selection -->
      <div class="mb-6">
        <label class="block text-sm font-medium text-gray-300 mb-2">Audio Connection</label>
        <div class="relative">
          <button 
            id="instant-audio-dropdown"
            @click="toggleAudioDropdown"
            class="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-left flex justify-between items-center hover:bg-gray-600 transition-colors"
          >
            <span>{{ audioLabel }}</span>
            <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
          </button>

          <div v-if="audioDropdownOpen" class="absolute z-10 w-full bg-gray-700 border border-gray-600 rounded-lg shadow-xl mt-2 overflow-hidden">
            <div 
              id="instant-audio-computer"
              @click="selectAudio('computer_audio', 'Computer Audio')"
              class="px-4 py-3 hover:bg-gray-600 cursor-pointer border-b border-gray-600 last:border-0"
            >
              Computer Audio
            </div>
            <div 
              id="instant-audio-phone"
              @click="selectAudio('phone_call', 'Phone Call')"
              class="px-4 py-3 hover:bg-gray-600 cursor-pointer border-b border-gray-600 last:border-0"
            >
              Phone Call
            </div>
            <div 
              id="instant-audio-none"
              @click="selectAudio('no_audio', 'No Audio')"
              class="px-4 py-3 hover:bg-gray-600 cursor-pointer"
            >
              No Audio
            </div>
          </div>
        </div>
      </div>

      <!-- Video Toggle -->
      <div class="mb-8 flex items-center justify-between bg-gray-700/50 p-4 rounded-lg">
        <span class="text-gray-300">Video</span>
        <button 
          id="instant-video-toggle"
          @click="toggleVideo"
          class="relative inline-flex items-center h-6 rounded-full w-11 transition-colors focus:outline-none"
          :class="store.video_on ? 'bg-green-500' : 'bg-gray-600'"
        >
          <span 
            class="inline-block w-4 h-4 transform bg-white rounded-full transition-transform"
            :class="store.video_on ? 'translate-x-6' : 'translate-x-1'"
          />
        </button>
      </div>

      <!-- Actions -->
      <button 
        id="instant-start-button"
        @click="startMeeting"
        class="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-6 rounded-lg mb-4 transition-colors shadow-lg disabled:opacity-50 disabled:cursor-not-allowed"
        :disabled="!store.audio_option"
      >
        Start Meeting
      </button>

      <button 
        id="instant-back-dashboard"
        @click="goBack"
        class="w-full bg-transparent hover:bg-gray-700 text-gray-300 font-medium py-3 px-6 rounded-lg transition-colors border border-gray-600"
      >
        Cancel
      </button>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'INSTANT_MEETING_LOBBY',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    
    const audioDropdownOpen = ref(false);
    const audioLabel = computed(() => {
      const map = {
        'computer_audio': 'Computer Audio',
        'phone_call': 'Phone Call',
        'no_audio': 'No Audio'
      };
      return map[store.audio_option] || 'Select Audio...';
    });

    const toggleAudioDropdown = () => {
      audioDropdownOpen.value = !audioDropdownOpen.value;
    };

    const selectAudio = (option, label) => {
      // FSM: ACT_INSTANT_SELECT_AUDIO
      // The FSM effect sets audio_option to 'computer_audio' hardcoded in one effect example, 
      // but for generic handling we should ideally pass params or let logic handle it.
      // The provided FSM example sets it to computer_audio.
      // In a real app, we'd set what was clicked.
      // I'll set store directly to simulate selection, then call action to confirm.
      store.audio_option = option;
      // The FSM signature says widget: "dropdown".
      store.handleAction('ACT_INSTANT_SELECT_AUDIO'); 
      audioDropdownOpen.value = false;
    };

    const toggleVideo = () => {
      store.video_on = !store.video_on;
      store.handleAction('ACT_INSTANT_TOGGLE_VIDEO');
    };

    const startMeeting = async () => {
      if (store.handleAction('ACT_INSTANT_MEETING_START')) {
        await router.push({ name: 'START_INSTANT_MEETING_SUCCESS' });
      }
    };

    const goBack = async () => {
      if (store.handleAction('ACT_INSTANT_LOBBY_BACK_DASHBOARD')) {
        await router.push({ name: 'DASHBOARD' });
      }
    };

    return {
      store,
      audioDropdownOpen,
      audioLabel,
      toggleAudioDropdown,
      selectAudio,
      toggleVideo,
      startMeeting,
      goBack
    };
  }
}
</script>