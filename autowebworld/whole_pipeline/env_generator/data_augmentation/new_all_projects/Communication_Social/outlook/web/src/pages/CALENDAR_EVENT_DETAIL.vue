<template>
  <div class="h-screen flex flex-col bg-white">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md shrink-0">
        <button id="event-detail-back-day" class="mr-4 hover:bg-[#005A9E] p-1 rounded" @click="goBackDay">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
        </button>
        <div class="font-semibold">Event Details</div>
    </header>

    <div class="flex-1 p-8 max-w-3xl mx-auto w-full">
        <div class="bg-blue-50 border-l-8 border-[#0078D4] rounded-r-lg shadow-sm p-8 mb-6">
            <h1 class="text-3xl font-light text-[#0078D4] mb-4">Team Sync</h1>
            
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="flex items-center gap-3">
                    <span class="text-xl text-gray-500">🕒</span>
                    <div>
                        <div class="font-semibold">Wednesday, Oct 22</div>
                        <div class="text-gray-600">10:00 AM - 12:00 PM</div>
                    </div>
                </div>
                
                <div class="flex items-center gap-3">
                    <span class="text-xl text-gray-500">📍</span>
                    <div>
                        <div class="font-semibold">Conference Room A</div>
                        <div class="text-gray-600">Building 4</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="prose max-w-none text-gray-700">
            <h3 class="text-gray-900 font-medium mb-2">Description</h3>
            <p>Weekly team sync to discuss project progress, blockers, and upcoming milestones. Please update your status in the shared doc before the meeting.</p>
        </div>
        
        <div class="mt-8 border-t border-gray-100 pt-6">
            <h3 class="text-gray-900 font-medium mb-4">Attendees (8)</h3>
            <div class="flex flex-wrap gap-2">
                <div class="w-10 h-10 rounded-full bg-gray-200 border-2 border-white shadow-sm flex items-center justify-center text-xs font-bold text-gray-600">JS</div>
                <div class="w-10 h-10 rounded-full bg-blue-200 border-2 border-white shadow-sm flex items-center justify-center text-xs font-bold text-blue-600">AB</div>
                <div class="w-10 h-10 rounded-full bg-green-200 border-2 border-white shadow-sm flex items-center justify-center text-xs font-bold text-green-600">CD</div>
                <div class="w-10 h-10 rounded-full bg-yellow-200 border-2 border-white shadow-sm flex items-center justify-center text-xs font-bold text-yellow-600">EF</div>
                <div class="w-10 h-10 rounded-full bg-gray-100 border-2 border-white shadow-sm flex items-center justify-center text-xs text-gray-500">+4</div>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'CALENDAR_EVENT_DETAIL',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const goBackDay = async () => {
        await signatureStore.handleAction('ACT_EVENT_DETAIL_BACK_DAY');
        router.push({ name: 'CALENDAR_DAY' });
    };

    return {
        goBackDay
    };
  }
}
</script>