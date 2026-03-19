<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg max-w-md w-full p-8">
      <div class="flex justify-between items-center mb-8">
        <h1 class="text-2xl font-bold text-gray-900">Join Meeting</h1>
        <button 
          id="join-meeting-back-dashboard" 
          @click="goDashboard"
          class="text-gray-500 hover:text-gray-700"
        >
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>
        </button>
      </div>

      <div class="space-y-6">
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Meeting ID</label>
          <input 
            id="join-meeting-id-input"
            v-model="meetingId"
            @input="updateMeetingId"
            type="text" 
            class="w-full border border-gray-300 rounded-md px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-lg"
            placeholder="Enter Meeting ID"
          />
        </div>

        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Your Name</label>
          <input 
            id="join-meeting-name-input"
            v-model="meetingName"
            @input="updateMeetingName"
            type="text" 
            class="w-full border border-gray-300 rounded-md px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none text-lg"
            placeholder="Enter your name"
          />
        </div>

        <div class="flex items-center" @click="toggleRememberName">
          <div id="join-remember-name-checkbox" class="w-5 h-5 rounded border border-gray-300 flex items-center justify-center mr-3 cursor-pointer" :class="{'bg-blue-600 border-blue-600': store.remember_name}">
            <svg v-if="store.remember_name" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
          </div>
          <span class="text-sm text-gray-600 cursor-pointer">Remember my name for future meetings</span>
        </div>

        <button 
          id="join-meeting-continue-button"
          @click="handleContinue"
          class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg transition-colors shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="!isValid"
        >
          Join
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'JOIN_MEETING_FORM',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const meetingId = computed({
      get: () => store.meeting_id_input,
      set: (val) => store.meeting_id_input = val // Direct sync for v-model if action is just typing
    });

    const meetingName = computed({
      get: () => store.meeting_name_input,
      set: (val) => store.meeting_name_input = val
    });

    const isValid = computed(() => {
      return store.meeting_id_input?.length > 0 && store.meeting_name_input?.length > 0;
    });

    const updateMeetingId = (e) => {
      store.handleAction('ACT_JOIN_MEETING_TYPE_MEETING_ID', { input_text: e.target.value });
    };

    const updateMeetingName = (e) => {
      store.handleAction('ACT_JOIN_MEETING_TYPE_NAME', { input_text: e.target.value });
    };

    const toggleRememberName = () => {
      store.remember_name = !store.remember_name;
      store.handleAction('ACT_JOIN_MEETING_TOGGLE_REMEMBER_NAME');
    };

    const handleContinue = async () => {
      if (store.handleAction('ACT_JOIN_MEETING_CONTINUE')) {
        await router.push({ name: 'JOIN_MEETING_PREVIEW' });
      }
    };

    const goDashboard = async () => {
      if (store.handleAction('ACT_JOIN_MEETING_BACK_TO_DASHBOARD')) {
        await router.push({ name: 'DASHBOARD' });
      }
    };

    return {
      store,
      meetingId,
      meetingName,
      isValid,
      updateMeetingId,
      updateMeetingName,
      toggleRememberName,
      handleContinue,
      goDashboard
    };
  }
}
</script>