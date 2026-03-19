<template>
  <div class="h-screen flex flex-col bg-gray-50">
    <!-- Header -->
    <header class="bg-[#6264A7] text-white p-4 shadow-md flex justify-between items-center z-20">
      <div class="font-bold text-lg flex items-center">
        <button id="meeting-review-back" @click="goBack" class="mr-4 hover:bg-[#464775] p-1 rounded">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
        </button>
        Review Meeting
      </div>
    </header>

    <main class="flex-1 flex flex-col p-8 items-center justify-center">
      <div class="bg-white rounded-lg shadow-lg p-8 w-full max-w-3xl border border-gray-200">
        <h2 class="text-2xl font-bold text-gray-800 mb-6 border-b pb-4">Review & Send</h2>
        
        <div class="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
            <div>
                <h3 class="font-semibold text-gray-500 text-sm uppercase mb-1">Title</h3>
                <p class="text-lg font-bold text-gray-900">{{ store.meeting_title }}</p>
                
                <h3 class="font-semibold text-gray-500 text-sm uppercase mt-4 mb-1">Description</h3>
                <p class="text-gray-700">{{ store.meeting_description || 'No description' }}</p>
            </div>
            <div>
                <h3 class="font-semibold text-gray-500 text-sm uppercase mb-1">When</h3>
                <div class="flex items-center gap-2 text-[#6264A7] font-medium">
                    <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    {{ store.meeting_date }} at {{ store.meeting_time }}
                </div>
            </div>
        </div>

        <div class="bg-gray-50 p-6 rounded-lg mb-8">
            <label class="block text-sm font-semibold text-gray-700 mb-2">Add Invitees</label>
            <div class="flex gap-2">
                <input 
                  id="meeting-invitees-input"
                  type="text" 
                  v-model="inviteeInput"
                  @keypress.enter="addInvitee"
                  placeholder="Type name and press Enter"
                  class="flex-1 rounded-md border-gray-300 shadow-sm focus:border-[#6264A7] focus:ring-[#6264A7] px-4 py-2 border"
                />
            </div>
            
            <div class="mt-4 flex flex-wrap gap-2">
                <span 
                    v-for="(invitee, idx) in invitees" 
                    :key="idx" 
                    class="bg-white border border-gray-200 text-gray-700 px-3 py-1 rounded-full text-sm flex items-center shadow-sm"
                >
                    {{ invitee }}
                </span>
            </div>
        </div>

        <div class="flex justify-end">
          <button 
            id="schedule-meeting-button"
            @click="schedule"
            class="bg-[#6264A7] hover:bg-[#464775] text-white font-semibold py-2 px-8 rounded shadow-sm transition-colors"
          >
            Send Invitation
          </button>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'MEETING_REVIEW',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const inviteeInput = ref('')
    // We keep local list for display, also sync with store if needed
    // The FSM action ACT_MEETING_REVIEW_ADD_INVITEE appends to store.invitees
    // But since `type` action replaces value, the "append" logic is in the effect.
    // Here we just simulate adding to a list.
    const invitees = ref([])

    const addInvitee = () => {
        if (inviteeInput.value) {
            invitees.value.push(inviteeInput.value);
            // Simulate FSM effect: append to invitees
            if (!store.invitees) store.invitees = [];
            store.invitees.push({ id: Date.now().toString(), name: inviteeInput.value });
            inviteeInput.value = '';
        }
    }

    const schedule = async () => {
      store.currentPageId = 'MEETING_SCHEDULED_SUCCESS';
      await router.push({ name: 'MEETING_SCHEDULED_SUCCESS' });
    }

    const goBack = async () => {
      store.currentPageId = 'MEETING_DETAILS';
      await router.push({ name: 'MEETING_DETAILS' });
    }

    return {
      store,
      inviteeInput,
      invitees,
      addInvitee,
      schedule,
      goBack
    }
  }
}
</script>