<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <!-- Header/Navigation Placeholder -->
    <header class="bg-[#6264A7] text-white p-4 shadow-md flex justify-between items-center z-20 relative">
      <div class="text-xl font-bold flex items-center gap-2">
        <span>Microsoft Teams</span>
      </div>
    </header>

    <main class="flex-1 flex flex-col items-center justify-center p-6 relative">
      <!-- Hero Section with Background Image -->
      <div class="absolute inset-0 z-0 overflow-hidden">
        <img 
            :src="'/images/Collaboration.jpg'" 
            class="w-full h-full object-cover opacity-10" 
            alt="Collaboration Background"
            @error="$event.target.src = 'https://picsum.photos/1920/1080?blur=2'"
        />
      </div>

      <div class="z-10 bg-white/90 backdrop-blur-sm p-8 rounded-xl shadow-2xl max-w-4xl w-full text-center border border-gray-100">
        <h1 class="text-4xl font-bold text-gray-800 mb-6">Welcome to Teams</h1>
        <p class="text-xl text-gray-600 mb-10 max-w-2xl mx-auto">Collaborate, meet, and chat from anywhere. Your workspace for real-time teamwork.</p>
        
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          
          <!-- ACT_HOME_GO_TEAMS_DIRECT -->
          <div 
            id="nav-teams-direct"
            @click="goTeams"
            class="group cursor-pointer bg-white p-6 rounded-lg shadow-md hover:shadow-xl hover:-translate-y-1 transition-all duration-300 border border-transparent hover:border-purple-200"
          >
            <div class="w-14 h-14 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-purple-600 transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7 text-purple-600 group-hover:text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
              </svg>
            </div>
            <h3 class="font-semibold text-lg text-gray-800">Teams</h3>
            <p class="text-sm text-gray-500 mt-2">Join or create teams</p>
          </div>

          <!-- ACT_HOME_GO_CHAT_MENU -->
          <div class="relative group" id="left-rail-menu" @click.stop="toggleChatMenu">
             <div 
              class="cursor-pointer bg-white p-6 rounded-lg shadow-md hover:shadow-xl hover:-translate-y-1 transition-all duration-300 border border-transparent hover:border-blue-200"
            >
              <div class="w-14 h-14 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-blue-600 transition-colors">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7 text-blue-600 group-hover:text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <h3 class="font-semibold text-lg text-gray-800">Chat</h3>
              <p class="text-sm text-gray-500 mt-2">Private conversations</p>
            </div>

            <!-- Dropdown Menu -->
            <div v-if="chatMenuOpen" class="absolute top-full left-0 right-0 mt-2 bg-white rounded-lg shadow-xl py-2 z-50 border border-gray-200 animate-in fade-in slide-in-from-top-2">
                <div id="left-rail-activity" class="px-4 py-2 hover:bg-gray-100 cursor-pointer text-left">Activity</div>
                <div id="left-rail-chat" @click="goChat" class="px-4 py-2 hover:bg-gray-100 cursor-pointer text-left font-medium text-blue-600 bg-blue-50">Open Chat</div>
                <div id="left-rail-teams" class="px-4 py-2 hover:bg-gray-100 cursor-pointer text-left">Teams</div>
                <div id="left-rail-calendar" class="px-4 py-2 hover:bg-gray-100 cursor-pointer text-left">Calendar</div>
            </div>
          </div>

          <!-- ACT_HOME_GO_CALENDAR_HOVER -->
          <div 
            id="app-bar-calendar"
            class="relative group cursor-pointer bg-white p-6 rounded-lg shadow-md hover:shadow-xl hover:-translate-y-1 transition-all duration-300 border border-transparent hover:border-green-200"
            @mouseenter="calendarHover = true"
            @mouseleave="calendarHover = false"
          >
            <div class="w-14 h-14 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-green-600 transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7 text-green-600 group-hover:text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <h3 class="font-semibold text-lg text-gray-800">Calendar</h3>
            <p class="text-sm text-gray-500 mt-2">Meetings & Schedule</p>

            <!-- Hover Menu -->
            <div v-if="calendarHover" class="absolute top-full left-0 right-0 mt-2 bg-white rounded-lg shadow-xl py-2 z-50 border border-gray-200">
                <div class="option-open-calendar px-4 py-2 hover:bg-gray-100 cursor-pointer text-left text-green-700 font-medium" @click="goCalendar">Open Calendar</div>
                <div class="option-open-meet-now px-4 py-2 hover:bg-gray-100 cursor-pointer text-left">Meet Now</div>
            </div>
          </div>

          <!-- ACT_HOME_GO_CALLS_DIRECT -->
          <div 
            id="nav-calls-direct"
            @click="goCalls"
            class="group cursor-pointer bg-white p-6 rounded-lg shadow-md hover:shadow-xl hover:-translate-y-1 transition-all duration-300 border border-transparent hover:border-pink-200"
          >
            <div class="w-14 h-14 bg-pink-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-pink-600 transition-colors">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-7 w-7 text-pink-600 group-hover:text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
              </svg>
            </div>
            <h3 class="font-semibold text-lg text-gray-800">Calls</h3>
            <p class="text-sm text-gray-500 mt-2">History & Contacts</p>
          </div>

        </div>
      </div>
    </main>

    <footer class="p-4 text-center text-gray-500 text-sm">
      &copy; 2025 Microsoft Corporation. All rights reserved.
    </footer>
  </div>
</template>

<script>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'HOME',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    const calendarHover = ref(false)
    const chatMenuOpen = ref(false)

    const toggleChatMenu = () => {
      chatMenuOpen.value = !chatMenuOpen.value
    }

    const goTeams = async () => {
      // Check precondition: cookie_consent_given == true
      if (store.cookie_consent_given !== true) return;
      store.currentPageId = 'TEAMS_LIST'
      await router.push({ name: 'TEAMS_LIST' })
    }

    const goCalendar = async () => {
      if (store.cookie_consent_given !== true) return;
      store.currentPageId = 'CALENDAR_VIEW'
      await router.push({ name: 'CALENDAR_VIEW' })
    }

    const goChat = async () => {
      if (store.cookie_consent_given !== true) return;
      store.currentPageId = 'CHAT_LIST'
      await router.push({ name: 'CHAT_LIST' })
    }

    const goCalls = async () => {
      if (store.cookie_consent_given !== true) return;
      store.currentPageId = 'CALLS_HUB'
      await router.push({ name: 'CALLS_HUB' })
    }

    return {
      calendarHover,
      chatMenuOpen,
      toggleChatMenu,
      goTeams,
      goCalendar,
      goChat,
      goCalls
    }
  }
}
</script>