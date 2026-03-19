<template>
  <div class="h-screen flex flex-col bg-white overflow-hidden">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md z-20 shrink-0 justify-between">
        <div class="flex items-center gap-4">
             <button id="calendar-back-home" class="hover:bg-[#005A9E] p-1 rounded" @click="goHome">
                 <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" /></svg>
             </button>
             <span class="font-semibold">Calendar - October 2025</span>
        </div>
        <button id="button-new-event" class="bg-white text-[#0078D4] px-4 py-1 rounded font-semibold hover:bg-gray-100 flex items-center gap-2" @click="newEvent">
            <span>+ New Event</span>
        </button>
    </header>

    <!-- Calendar Grid -->
    <div id="calendar-grid" class="flex-1 p-4 overflow-y-auto" @scroll="handleScroll">
        <div class="grid grid-cols-7 gap-px bg-gray-200 border border-gray-200 rounded-lg overflow-hidden shadow-sm">
            <!-- Header -->
            <div class="bg-gray-50 p-2 text-center text-sm font-semibold text-gray-500">Sun</div>
            <div class="bg-gray-50 p-2 text-center text-sm font-semibold text-gray-500">Mon</div>
            <div class="bg-gray-50 p-2 text-center text-sm font-semibold text-gray-500">Tue</div>
            <div class="bg-gray-50 p-2 text-center text-sm font-semibold text-gray-500">Wed</div>
            <div class="bg-gray-50 p-2 text-center text-sm font-semibold text-gray-500">Thu</div>
            <div class="bg-gray-50 p-2 text-center text-sm font-semibold text-gray-500">Fri</div>
            <div class="bg-gray-50 p-2 text-center text-sm font-semibold text-gray-500">Sat</div>

            <!-- Days (Mock for Oct 2025) -->
            <!-- Previous month filler -->
            <div class="bg-white p-2 h-32 text-gray-300">28</div>
            <div class="bg-white p-2 h-32 text-gray-300">29</div>
            <div class="bg-white p-2 h-32 text-gray-300">30</div>
            
            <!-- Active Month Days -->
            <div v-for="day in 31" :key="day" 
                 :id="`calendar-day-${day}`"
                 class="bg-white p-2 h-32 hover:bg-blue-50 cursor-pointer border-t border-gray-100 relative group transition-colors"
                 @click="openDay(day)">
                 <span class="font-semibold text-gray-700" :class="{'text-blue-600': day === 22}">{{ day }}</span>
                 
                 <!-- Mock Events -->
                 <div v-if="day === 22" class="mt-2 text-xs bg-blue-100 text-blue-700 p-1 rounded truncate">
                     Team Sync
                 </div>
                 <div v-if="day === 15" class="mt-2 text-xs bg-purple-100 text-purple-700 p-1 rounded truncate">
                     Project Review
                 </div>
            </div>
            
            <!-- Next month filler -->
            <div class="bg-white p-2 h-32 text-gray-300">1</div>
        </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'CALENDAR_MONTH',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const openDay = async (day) => {
        // FSM specific requirement: Only day 1 has explicit selector in example but generalized usually.
        // FSM has "id": "ACT_CALENDAR_OPEN_DAY", selector: "#calendar-day-1".
        // This implies for FSM traversal we need to click day 1. 
        // For real usage, any day should work. We map all to this action for now, or just day 1 for strict FSM.
        // Since it is navigation, we allow all.
        await signatureStore.handleAction('ACT_CALENDAR_OPEN_DAY');
        router.push({ name: 'CALENDAR_DAY', query: { day } });
    };

    const newEvent = async () => {
        await signatureStore.handleAction('ACT_CALENDAR_NEW_EVENT');
        router.push({ name: 'CALENDAR_NEW_EVENT' });
    };

    const goHome = async () => {
        await signatureStore.handleAction('ACT_CALENDAR_BACK_HOME');
        router.push({ name: 'HOME' });
    };

    const handleScroll = () => {
        signatureStore.handleAction('ACT_CALENDAR_SCROLL', { item_id: 'day-1' });
    };

    return {
        openDay,
        newEvent,
        goHome,
        handleScroll
    };
  }
}
</script>