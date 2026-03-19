<template>
  <div class="min-h-screen bg-gray-50 p-8">
    <div class="max-w-4xl mx-auto bg-white rounded-xl shadow-lg overflow-hidden min-h-[600px] flex">
      <!-- Sidebar -->
      <div class="w-64 bg-gray-100 p-4 border-r border-gray-200">
        <h2 class="text-lg font-bold text-gray-700 mb-6 px-2">Settings</h2>
        <div class="space-y-1">
          <button 
            id="settings-video-back-general"
            @click="goToGeneral"
            class="w-full text-left px-4 py-2 text-gray-600 hover:bg-gray-200 rounded-md font-medium"
          >General</button>
          <button class="w-full text-left px-4 py-2 bg-blue-100 text-blue-700 rounded-md font-medium">Video</button>
        </div>
        
        <div class="mt-10 pt-10 border-t border-gray-200">
           <button 
            id="settings-video-back-profile" 
            @click="goBack"
            class="w-full text-left px-4 py-2 text-gray-500 hover:text-gray-800 flex items-center"
          >
            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
            Back to Profile
          </button>
        </div>
      </div>

      <!-- Content -->
      <div class="flex-1 p-8">
        <h1 class="text-2xl font-bold text-gray-900 mb-8">Video Settings</h1>
        
        <div class="mb-8">
          <div class="aspect-video bg-gray-900 rounded-lg mb-4 overflow-hidden relative">
            <!-- Placeholder Camera Preview -->
             <img src="/images/CameraPreview.jpg" alt="Camera Preview" class="w-full h-full object-cover" :class="{'transform scale-x-[-1]': store.mirror_my_video}" />
             <div class="absolute bottom-2 left-2 text-xs text-white bg-black/50 px-2 py-1 rounded">Preview</div>
          </div>
          <div class="text-sm text-gray-500">Camera: Integrated Webcam</div>
        </div>

        <div class="space-y-4">
          <div class="flex items-center">
            <div 
              id="settings-video-mirror-checkbox" 
              @click="toggleMirror"
              class="w-5 h-5 rounded border border-gray-300 flex items-center justify-center mr-3 cursor-pointer transition-colors"
              :class="{'bg-blue-600 border-blue-600': store.mirror_my_video}"
            >
              <svg v-if="store.mirror_my_video" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <span class="text-gray-700 cursor-pointer" @click="toggleMirror">Mirror my video</span>
          </div>

          <div class="flex items-center">
            <div 
              id="settings-video-touchup-checkbox" 
              @click="toggleTouchUp"
              class="w-5 h-5 rounded border border-gray-300 flex items-center justify-center mr-3 cursor-pointer transition-colors"
              :class="{'bg-blue-600 border-blue-600': store.touch_up_appearance}"
            >
               <svg v-if="store.touch_up_appearance" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <span class="text-gray-700 cursor-pointer" @click="toggleTouchUp">Touch up my appearance</span>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'SETTINGS_VIDEO',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const toggleMirror = () => {
      store.mirror_my_video = !store.mirror_my_video;
      store.handleAction('ACT_SETTINGS_VIDEO_TOGGLE_MIRROR');
    };

    const toggleTouchUp = () => {
      store.touch_up_appearance = !store.touch_up_appearance;
      store.handleAction('ACT_SETTINGS_VIDEO_TOGGLE_TOUCH_UP');
    };

    const goToGeneral = async () => {
      if (store.handleAction('ACT_SETTINGS_VIDEO_BACK_GENERAL')) {
        await router.push({ name: 'SETTINGS_GENERAL' });
      }
    };

    const goBack = async () => {
      if (store.handleAction('ACT_SETTINGS_VIDEO_BACK_PROFILE')) {
        await router.push({ name: 'PROFILE_OVERVIEW' });
      }
    };

    return {
      store,
      toggleMirror,
      toggleTouchUp,
      goToGeneral,
      goBack
    };
  }
}
</script>