<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg max-w-2xl w-full p-8">
      <h1 class="text-2xl font-bold text-gray-900 mb-6">Review Meeting Details</h1>
      
      <div class="bg-blue-50 rounded-lg p-6 mb-8 space-y-4">
        <div class="flex justify-between">
          <span class="text-gray-600">Topic</span>
          <span class="font-semibold text-gray-900">{{ store.meeting_topic }}</span>
        </div>
        <div class="flex justify-between">
          <span class="text-gray-600">Time</span>
          <span class="font-semibold text-gray-900">{{ formattedDate }}</span>
        </div>
        <div class="flex justify-between">
          <span class="text-gray-600">Duration</span>
          <span class="font-semibold text-gray-900">{{ store.meeting_duration_minutes }} min</span>
        </div>
      </div>

      <div class="space-y-4 mb-8">
        <div class="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 cursor-pointer" @click="toggleWaitingRoom">
          <div class="flex items-center">
            <div id="review-waiting-room-checkbox" class="w-5 h-5 rounded border border-gray-300 flex items-center justify-center mr-3" :class="{'bg-blue-600 border-blue-600': store.waiting_room_enabled}">
              <svg v-if="store.waiting_room_enabled" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <span class="text-gray-700">Enable Waiting Room</span>
          </div>
        </div>

        <div class="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 cursor-pointer" @click="toggleHostVideo">
          <div class="flex items-center">
             <div id="review-host-video-checkbox" class="w-5 h-5 rounded border border-gray-300 flex items-center justify-center mr-3" :class="{'bg-blue-600 border-blue-600': store.host_video_on}">
              <svg v-if="store.host_video_on" class="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="3" d="M5 13l4 4L19 7"></path></svg>
            </div>
            <span class="text-gray-700">Turn Host Video On</span>
          </div>
        </div>
      </div>

      <div class="flex justify-between gap-4">
        <button 
          id="review-back-button" 
          @click="goBack"
          class="px-6 py-3 border border-gray-300 rounded-lg text-gray-700 font-medium hover:bg-gray-50 transition-colors"
        >
          Back
        </button>
        <button 
          id="review-schedule-button" 
          @click="confirmSchedule"
          class="flex-1 px-6 py-3 bg-blue-600 text-white rounded-lg font-bold hover:bg-blue-700 transition-colors shadow-md"
        >
          Schedule
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
  name: 'SCHEDULE_MEETING_REVIEW',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const formattedDate = computed(() => {
      if (!store.meeting_date_time) return 'Not set';
      return new Date(store.meeting_date_time).toLocaleString();
    });

    const toggleWaitingRoom = () => {
      store.waiting_room_enabled = !store.waiting_room_enabled;
      store.handleAction('ACT_SCHEDULE_REVIEW_ENABLE_WAITING_ROOM');
    };

    const toggleHostVideo = () => {
      store.host_video_on = !store.host_video_on;
      store.handleAction('ACT_SCHEDULE_REVIEW_TURN_HOST_VIDEO_ON');
    };

    const confirmSchedule = async () => {
      if (store.handleAction('ACT_SCHEDULE_REVIEW_CONFIRM')) {
        await router.push({ name: 'SCHEDULE_MEETING_SUCCESS' });
      }
    };

    const goBack = async () => {
      if (store.handleAction('ACT_SCHEDULE_REVIEW_BACK_TO_FORM')) {
        await router.push({ name: 'SCHEDULE_MEETING_FORM' });
      }
    };

    return {
      store,
      formattedDate,
      toggleWaitingRoom,
      toggleHostVideo,
      confirmSchedule,
      goBack
    };
  }
}
</script>