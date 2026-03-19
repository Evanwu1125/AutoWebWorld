<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div class="flex items-center cursor-pointer" id="dashboard-home-logo" @click="goHome">
          <img src="/images/Zoom.jpg" alt="Zoom" class="h-8 w-auto" />
        </div>
        
        <div class="flex items-center space-x-4">
           <div id="dashboard-profile-link" @click="goToProfile" class="flex items-center space-x-2 cursor-pointer hover:bg-gray-100 px-3 py-2 rounded-lg transition-colors">
             <div class="h-8 w-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 font-bold">JD</div>
             <span class="text-gray-700 font-medium hidden sm:block">John Doe</span>
           </div>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 w-full">
      <!-- Quick Actions Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        <!-- Schedule Meeting Card -->
        <div 
          id="dashboard-schedule-meeting" 
          @click="goToSchedule"
          class="bg-blue-600 rounded-2xl p-6 text-white cursor-pointer transform transition-all hover:scale-105 hover:shadow-xl flex flex-col items-center justify-center aspect-square"
        >
          <div class="bg-blue-500 p-4 rounded-2xl mb-4">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="18" x="3" y="4" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/></svg>
          </div>
          <h3 class="text-xl font-bold">Schedule</h3>
          <p class="text-blue-100 text-sm mt-2">Plan a meeting</p>
        </div>

        <!-- Join Meeting Card -->
        <div 
          id="dashboard-join-meeting" 
          @click="goToJoin"
          class="bg-white rounded-2xl p-6 text-gray-800 cursor-pointer transform transition-all hover:scale-105 hover:shadow-xl flex flex-col items-center justify-center aspect-square border border-gray-100"
        >
          <div class="bg-blue-50 p-4 rounded-2xl mb-4 text-blue-600">
            <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><line x1="19" y1="8" x2="19" y2="14"/><line x1="22" y1="11" x2="16" y2="11"/></svg>
          </div>
          <h3 class="text-xl font-bold">Join</h3>
          <p class="text-gray-500 text-sm mt-2">Join via ID</p>
        </div>
        
        <!-- Decorative / Placeholder Cards for Layout -->
        <div class="bg-white rounded-2xl p-6 text-gray-800 cursor-pointer hover:shadow-lg flex flex-col items-center justify-center aspect-square border border-gray-100 opacity-60">
           <div class="bg-orange-50 p-4 rounded-2xl mb-4 text-orange-500">
             <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="20" height="16" x="2" y="4" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg>
           </div>
           <h3 class="text-xl font-bold">Mail</h3>
        </div>

        <div class="bg-white rounded-2xl p-6 text-gray-800 cursor-pointer hover:shadow-lg flex flex-col items-center justify-center aspect-square border border-gray-100 opacity-60">
           <div class="bg-green-50 p-4 rounded-2xl mb-4 text-green-500">
             <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
           </div>
           <h3 class="text-xl font-bold">Team Chat</h3>
        </div>
      </div>

      <!-- Dashboard Content -->
      <div class="bg-white rounded-2xl shadow-sm border border-gray-100 p-8 text-center">
        <h2 class="text-2xl font-bold text-gray-800 mb-4">Good Morning, John!</h2>
        <p class="text-gray-500 mb-8">You have no upcoming meetings today.</p>
        <img src="/images/photo1764907552.jpg" alt="No meetings" class="mx-auto w-64 h-auto opacity-80" />
      </div>
    </main>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'DASHBOARD',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const goToSchedule = async () => {
      if (store.handleAction('ACT_DASHBOARD_GO_TO_SCHEDULE_MEETING')) {
        await router.push({ name: 'SCHEDULE_MEETING_FORM' });
      }
    };

    const goToJoin = async () => {
      if (store.handleAction('ACT_DASHBOARD_GO_TO_JOIN_MEETING')) {
        await router.push({ name: 'JOIN_MEETING_FORM' });
      }
    };

    const goToProfile = async () => {
      if (store.handleAction('ACT_DASHBOARD_GO_TO_PROFILE')) {
        await router.push({ name: 'PROFILE_OVERVIEW' });
      }
    };

    const goHome = async () => {
      if (store.handleAction('ACT_DASHBOARD_GO_HOME')) {
        await router.push({ name: 'HOME' });
      }
    };

    return {
      goToSchedule,
      goToJoin,
      goToProfile,
      goHome
    };
  }
}
</script>