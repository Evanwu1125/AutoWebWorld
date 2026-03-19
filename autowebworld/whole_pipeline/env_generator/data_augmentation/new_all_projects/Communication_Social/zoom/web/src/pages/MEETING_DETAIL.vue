<template>
  <div class="min-h-screen bg-gray-50 p-4 flex items-center justify-center">
    <div class="bg-white rounded-xl shadow-xl max-w-3xl w-full overflow-hidden">
      <!-- Header Image -->
      <div class="h-48 w-full bg-gray-200 relative">
        <img :src="meeting?.image || '/images/Meeting.jpg'" class="w-full h-full object-cover" alt="Meeting Header" />
        <button 
          id="meeting-detail-back-list" 
          @click="goBack"
          class="absolute top-4 left-4 bg-black/50 hover:bg-black/70 text-white p-2 rounded-full transition-colors"
        >
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
        </button>
      </div>

      <div class="p-8">
        <div class="flex justify-between items-start mb-6">
          <div>
            <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ meeting?.topic || 'Meeting Details' }}</h1>
            <p class="text-gray-500">Hosted by {{ meeting?.host }}</p>
          </div>
          <div class="bg-blue-50 text-blue-700 px-4 py-2 rounded-lg font-bold text-lg">
            {{ meeting?.duration }} min
          </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
          <div class="flex items-start space-x-3">
            <div class="bg-gray-100 p-2 rounded-lg text-gray-600">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
            </div>
            <div>
              <div class="font-semibold text-gray-900">Date</div>
              <div class="text-gray-600">{{ formattedDate }}</div>
            </div>
          </div>

          <div class="flex items-start space-x-3">
            <div class="bg-gray-100 p-2 rounded-lg text-gray-600">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            </div>
            <div>
              <div class="font-semibold text-gray-900">Time</div>
              <div class="text-gray-600">{{ formattedTime }}</div>
            </div>
          </div>
          
           <div class="flex items-start space-x-3">
            <div class="bg-gray-100 p-2 rounded-lg text-gray-600">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
            </div>
            <div>
              <div class="font-semibold text-gray-900">Meeting ID</div>
              <div class="text-gray-600">832 948 2910</div>
            </div>
          </div>
          
          <div class="flex items-start space-x-3">
            <div class="bg-gray-100 p-2 rounded-lg text-gray-600">
              <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"></path></svg>
            </div>
            <div>
              <div class="font-semibold text-gray-900">Password</div>
              <div class="text-gray-600">******</div>
            </div>
          </div>
        </div>

        <div class="border-t border-gray-200 pt-8 flex gap-4">
          <button 
            id="meeting-detail-edit-button"
            @click="editMeeting"
            class="flex-1 py-3 border border-blue-600 text-blue-600 rounded-lg font-bold hover:bg-blue-50 transition-colors"
          >
            Edit
          </button>
          <button 
            id="meeting-detail-start-button"
            @click="startMeeting"
            class="flex-1 py-3 bg-blue-600 text-white rounded-lg font-bold hover:bg-blue-700 transition-colors shadow-lg"
          >
            Start Meeting
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'MEETING_DETAIL',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    const dataStore = useDataStore();

    const meeting = computed(() => {
      return dataStore.meetings.find(m => m.id === store.meetings_selected_id);
    });

    const formattedDate = computed(() => {
      if (!meeting.value) return '';
      const d = new Date(meeting.value.date + 'T' + meeting.value.time);
      return d.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
    });

    const formattedTime = computed(() => {
      if (!meeting.value) return '';
      return meeting.value.time;
    });

    const editMeeting = async () => {
      if (store.handleAction('ACT_MEETING_DETAIL_EDIT')) {
        // Pre-fill edit form in store if needed
        if (meeting.value) {
          store.meeting_topic = meeting.value.topic;
          store.meeting_duration_minutes = meeting.value.duration;
          store.meeting_date_time = meeting.value.date + 'T' + meeting.value.time;
        }
        await router.push({ name: 'SCHEDULE_MEETING_FORM' });
      }
    };

    const startMeeting = async () => {
      if (store.handleAction('ACT_MEETING_DETAIL_START')) {
        await router.push({ name: 'START_INSTANT_MEETING_SUCCESS' });
      }
    };

    const goBack = async () => {
      if (store.handleAction('ACT_MEETING_DETAIL_BACK_LIST')) {
        await router.push({ name: 'MEETINGS_LIST' });
      }
    };

    return {
      store,
      meeting,
      formattedDate,
      formattedTime,
      editMeeting,
      startMeeting,
      goBack
    };
  }
}
</script>