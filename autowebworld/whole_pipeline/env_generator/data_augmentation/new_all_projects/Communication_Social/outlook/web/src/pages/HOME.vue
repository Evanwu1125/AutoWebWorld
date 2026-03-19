<template>
  <div class="h-screen flex flex-col bg-gray-100 overflow-hidden">
    <!-- Top Bar -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md z-20 justify-between">
      <div class="flex items-center gap-4">
        <div class="grid grid-cols-3 gap-0.5 p-2 hover:bg-[#005A9E] cursor-pointer rounded-sm" id="app-launcher">
             <div class="w-1 h-1 bg-white rounded-full"></div>
             <div class="w-1 h-1 bg-white rounded-full"></div>
             <div class="w-1 h-1 bg-white rounded-full"></div>
             <div class="w-1 h-1 bg-white rounded-full"></div>
             <div class="w-1 h-1 bg-white rounded-full"></div>
             <div class="w-1 h-1 bg-white rounded-full"></div>
             <div class="w-1 h-1 bg-white rounded-full"></div>
             <div class="w-1 h-1 bg-white rounded-full"></div>
             <div class="w-1 h-1 bg-white rounded-full"></div>
             
             <!-- Hover Menu Options (Hidden by default, shown on hover/click) -->
             <div class="absolute top-12 left-0 bg-white text-black shadow-lg rounded-md p-2 w-64 hidden group-hover:block border border-gray-200 z-50"
                  style="display: none;"> <!-- Controlled by logic, but for simplicity we rely on click for navigation in FSM -->
             </div>
        </div>
        <span class="font-semibold text-lg tracking-wide">Outlook</span>
      </div>
      
      <!-- Search Bar -->
      <div class="flex-1 max-w-2xl mx-4">
        <div class="relative">
          <input type="text" placeholder="Search" class="w-full bg-[#C3E0FA] text-black placeholder-gray-600 rounded-md py-1.5 px-10 border-none focus:bg-white focus:outline-none focus:ring-2 focus:ring-[#0078D4] transition-colors" />
          <div class="absolute left-3 top-2 text-[#005A9E]">
             <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" /></svg>
          </div>
        </div>
      </div>
      
      <!-- User Menu -->
      <div class="relative">
        <div id="user-menu-toggle" class="flex items-center gap-2 cursor-pointer hover:bg-[#005A9E] p-1 rounded-full" @click="toggleUserMenu">
           <div class="w-8 h-8 rounded-full bg-[#C3E0FA] text-[#0078D4] flex items-center justify-center font-bold border border-white">JS</div>
        </div>
        
        <!-- Dropdown -->
        <div v-if="showUserMenu" id="user-menu" class="absolute right-0 top-12 bg-white text-black shadow-xl rounded-md w-72 border border-gray-200 z-50 overflow-hidden">
            <div class="p-4 border-b border-gray-100 flex items-center gap-3">
               <div class="w-12 h-12 rounded-full bg-[#0078D4] text-white flex items-center justify-center font-bold text-xl">JS</div>
               <div>
                  <div class="font-semibold">Jane Smith</div>
                  <div class="text-sm text-gray-500">jane.smith@outlook.com</div>
               </div>
            </div>
            <div class="py-2">
                <div id="user-menu-account" class="px-4 py-2 hover:bg-gray-100 cursor-pointer flex items-center gap-3">
                   <span class="text-gray-500">👤</span> View account
                </div>
                <div id="user-menu-profile" class="px-4 py-2 hover:bg-gray-100 cursor-pointer flex items-center gap-3">
                   <span class="text-gray-500">📝</span> My profile
                </div>
                <div id="user-menu-settings" class="px-4 py-2 hover:bg-gray-100 cursor-pointer flex items-center gap-3" @click="goToSettings">
                   <span class="text-gray-500">⚙️</span> Settings
                </div>
            </div>
        </div>
      </div>
    </header>

    <div class="flex flex-1 overflow-hidden relative">
      <!-- Left Rail Navigation -->
      <nav class="w-16 bg-[#F0F2F5] flex flex-col items-center py-4 gap-4 border-r border-gray-200 z-10">
         <div id="nav-mail" class="w-10 h-10 flex items-center justify-center rounded-md cursor-pointer hover:bg-white hover:shadow-sm text-[#0078D4]" @click="goToInbox">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
         </div>
         <div class="w-10 h-10 flex items-center justify-center rounded-md cursor-pointer hover:bg-white hover:shadow-sm text-gray-500 relative group" @click="handleCalendarHover">
            <!-- This is the trigger for the hover menu in FSM -->
            <div class="tile-calendar w-full h-full flex items-center justify-center">
              <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
            </div>
            
            <!-- Hover Menu for App Launcher Simulation (reused here for FSM flow visualization) -->
            <div class="hidden group-hover:block absolute left-12 top-0 bg-white shadow-md p-2 rounded w-40 z-50">
               <div class="tile-outlook p-2 hover:bg-gray-100 cursor-pointer">Outlook</div>
               <div class="tile-calendar p-2 hover:bg-gray-100 cursor-pointer text-[#0078D4] font-semibold">Calendar</div>
               <div class="tile-people p-2 hover:bg-gray-100 cursor-pointer">People</div>
            </div>
         </div>
         <div id="nav-people" class="w-10 h-10 flex items-center justify-center rounded-md cursor-pointer hover:bg-white hover:shadow-sm text-gray-500" @click="goToPeople">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4.354a4 4 0 110 5.292M15 21H3v-1a6 6 0 0112 0v1zm0 0h6v-1a6 6 0 00-9-5.197M13 7a4 4 0 11-8 0 4 4 0 018 0z" /></svg>
         </div>
      </nav>

      <!-- Main Content Area (Hero / Dashboard) -->
      <main class="flex-1 bg-[#FAF9F8] p-8 overflow-y-auto relative">
         <div class="max-w-4xl mx-auto">
             <div class="bg-white rounded-lg shadow-sm p-8 mb-8 flex items-center gap-8">
                 <div class="flex-1">
                     <h1 class="text-3xl font-light mb-4 text-[#323130]">Good morning, Jane</h1>
                     <p class="text-gray-600 mb-6">You're all caught up! No new emails in your Focused inbox.</p>
                     <button class="bg-[#0078D4] text-white px-6 py-2 rounded-sm hover:bg-[#005A9E] transition-colors shadow-sm" @click="goToInbox">
                         Go to Inbox
                     </button>
                 </div>
                 <div class="w-64 h-48 bg-gray-100 rounded-md overflow-hidden relative">
                     <!-- Using ImageGetter via tool call for dynamic image -->
                     <img src="https://images.unsplash.com/photo-1486312338219-ce68d2c6f44d?ixlib=rb-4.0.3&auto=format&fit=crop&w=1172&q=80" alt="Productivity" class="w-full h-full object-cover" />
                 </div>
             </div>

             <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                 <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
                     <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
                        <span class="text-[#0078D4]">📅</span> Today's Agenda
                     </h2>
                     <div class="space-y-3">
                         <div class="p-3 bg-blue-50 border-l-4 border-blue-500 rounded-r-md">
                             <div class="font-medium text-blue-900">Project Sync</div>
                             <div class="text-sm text-blue-700">10:00 AM - 11:00 AM</div>
                         </div>
                         <div class="p-3 bg-gray-50 border-l-4 border-gray-300 rounded-r-md">
                             <div class="font-medium text-gray-900">Lunch Break</div>
                             <div class="text-sm text-gray-500">12:30 PM - 1:30 PM</div>
                         </div>
                     </div>
                 </div>
                 
                 <div class="bg-white p-6 rounded-lg shadow-sm border border-gray-100">
                     <h2 class="text-lg font-semibold mb-4 flex items-center gap-2">
                        <span class="text-[#0078D4]">👥</span> Recent Contacts
                     </h2>
                     <div class="flex gap-2">
                         <div class="w-10 h-10 rounded-full bg-yellow-200 flex items-center justify-center text-yellow-700 font-bold text-sm cursor-pointer hover:ring-2 hover:ring-offset-1 hover:ring-yellow-400">AB</div>
                         <div class="w-10 h-10 rounded-full bg-green-200 flex items-center justify-center text-green-700 font-bold text-sm cursor-pointer hover:ring-2 hover:ring-offset-1 hover:ring-green-400">CD</div>
                         <div class="w-10 h-10 rounded-full bg-purple-200 flex items-center justify-center text-purple-700 font-bold text-sm cursor-pointer hover:ring-2 hover:ring-offset-1 hover:ring-purple-400">EF</div>
                         <div class="w-10 h-10 rounded-full border border-dashed border-gray-300 flex items-center justify-center text-gray-400 cursor-pointer hover:border-gray-500 hover:text-gray-600">
                             <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor"><path fill-rule="evenodd" d="M10 5a1 1 0 011 1v3h3a1 1 0 110 2h-3v3a1 1 0 11-2 0v-3H6a1 1 0 110-2h3V6a1 1 0 011-1z" clip-rule="evenodd" /></svg>
                         </div>
                     </div>
                 </div>
             </div>
         </div>
      </main>
    </div>

    <!-- Cookie Consent Modal -->
    <div v-if="showCookieModal" class="fixed inset-0 bg-black/50 backdrop-blur-sm z-[10000] flex items-end md:items-center justify-center p-4">
        <div class="bg-white rounded-xl shadow-2xl p-6 max-w-md w-full animate-fade-in-up">
            <div class="flex items-start gap-4 mb-4">
                <div class="text-4xl">🍪</div>
                <div>
                    <h3 class="text-xl font-bold text-gray-900 mb-2">We Value Your Privacy</h3>
                    <p class="text-gray-600 text-sm leading-relaxed">
                        We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
                    </p>
                </div>
            </div>
            <div class="flex justify-end gap-3">
                <button class="px-4 py-2 text-gray-600 hover:bg-gray-100 rounded-md text-sm font-medium transition-colors">Preferences</button>
                <button id="cookie-accept" class="px-6 py-2 bg-[#0078D4] text-white rounded-md text-sm font-medium hover:bg-[#005A9E] shadow-sm transition-colors" @click="acceptCookies">Accept All</button>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { ref, onMounted } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'HOME',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    const showCookieModal = ref(false);
    const showUserMenu = ref(false);

    onMounted(() => {
        // Check cookie consent
        if (!signatureStore.cookie_consent_given) {
            showCookieModal.value = true;
        }
    });

    const acceptCookies = () => {
        signatureStore.handleAction('ACT_HOME_ACCEPT_COOKIES');
        showCookieModal.value = false;
    };

    const goToInbox = async () => {
        await signatureStore.handleAction('ACT_HOME_GO_INBOX_DIRECT');
        router.push({ name: 'MAIL_INBOX' });
    };

    const handleCalendarHover = async () => {
        await signatureStore.handleAction('ACT_HOME_GO_CALENDAR_HOVER', { widget: 'hover_menu' });
        router.push({ name: 'CALENDAR_MONTH' });
    };

    const toggleUserMenu = () => {
        showUserMenu.value = !showUserMenu.value;
    };

    const goToSettings = async () => {
        await signatureStore.handleAction('ACT_HOME_GO_SETTINGS_MENU', { widget: 'dropdown' });
        router.push({ name: 'MAIL_SETTINGS_GENERAL' });
    };

    const goToPeople = async () => {
        await signatureStore.handleAction('ACT_HOME_GO_PEOPLE_DIRECT');
        router.push({ name: 'PEOPLE_LIST' });
    };

    return {
        showCookieModal,
        acceptCookies,
        goToInbox,
        handleCalendarHover,
        showUserMenu,
        toggleUserMenu,
        goToSettings,
        goToPeople
    };
  }
}
</script>