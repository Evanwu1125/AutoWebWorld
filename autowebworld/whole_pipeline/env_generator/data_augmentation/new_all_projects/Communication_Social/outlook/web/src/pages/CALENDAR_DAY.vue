<template>
  <div class="h-screen flex flex-col bg-white overflow-hidden">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md z-20 shrink-0">
        <button id="day-back-month" class="mr-4 hover:bg-[#005A9E] p-1 rounded" @click="goBackMonth">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" /></svg>
        </button>
        <div class="font-semibold">Wednesday, October 22, 2025</div>
    </header>

    <!-- Day View -->
    <div id="day-view" class="flex-1 overflow-y-auto relative bg-white">
        <!-- Time Grid -->
        <div class="flex">
            <div class="w-16 border-r border-gray-200 bg-gray-50 text-xs text-gray-500 text-right pr-2 py-4 space-y-12">
                <div v-for="hour in 24" :key="hour" class="h-12 relative top-[-6px]">
                    {{ hour - 1 }}:00
                </div>
            </div>
            <div class="flex-1 relative">
                 <div v-for="hour in 24" :key="hour" class="h-12 border-b border-gray-100"></div>
                 
                 <!-- Event Slot -->
                 <div class="absolute top-40 left-2 right-2 bg-blue-100 border-l-4 border-[#0078D4] p-2 rounded cursor-pointer hover:bg-blue-200 event-slot h-24 shadow-sm"
                      @click="openEvent">
                     <div class="font-bold text-[#0078D4]">Team Sync</div>
                     <div class="text-xs text-blue-800">10:00 AM - 12:00 PM</div>
                     <div class="text-xs text-gray-600 mt-1">Conference Room A</div>
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
  name: 'CALENDAR_DAY',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const openEvent = async () => {
        await signatureStore.handleAction('ACT_DAY_OPEN_EVENT', { item_id: 'event-1' });
        router.push({ name: 'CALENDAR_EVENT_DETAIL', params: { id: 'event-1' } });
    };

    const goBackMonth = async () => {
        await signatureStore.handleAction('ACT_DAY_BACK_MONTH');
        router.push({ name: 'CALENDAR_MONTH' });
    };

    return {
        openEvent,
        goBackMonth
    };
  }
}
</script>