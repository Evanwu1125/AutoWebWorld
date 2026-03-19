<template>
  <div class="min-h-screen bg-gray-50">
    <!-- Navbar -->
    <nav class="bg-[#005DAA] shadow-lg">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-between h-16">
          <div class="flex items-center">
            <div id="logo-home" @click="handleBackHome" class="text-white font-bold text-xl cursor-pointer">
              Teladoc Health
            </div>
            <div class="hidden md:block">
              <div class="ml-10 flex items-baseline space-x-4">
                <!-- Hover Menu: Appointments -->
                <div class="relative group">
                  <button id="nav-appointments" class="text-white hover:bg-[#004a87] px-3 py-2 rounded-md text-sm font-medium">
                    Appointments
                  </button>
                  <div class="absolute left-0 mt-2 w-48 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 hidden group-hover:block z-50">
                     <a id="nav-appointments-upcoming" @click="handleToAppts" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">
                       Upcoming & Past
                     </a>
                  </div>
                </div>

                <!-- Hover Menu: Billing -->
                <div class="relative group">
                  <button id="nav-billing" class="text-white hover:bg-[#004a87] px-3 py-2 rounded-md text-sm font-medium">
                    Billing
                  </button>
                  <div class="absolute left-0 mt-2 w-48 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 hidden group-hover:block z-50">
                    <a id="nav-billing-overview" @click="handleToBilling" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">
                      Overview
                    </a>
                  </div>
                </div>
              </div>
            </div>
          </div>
          
          <!-- Profile Menu -->
          <div class="relative">
            <button id="dashboard-profile-menu" @click="toggleProfileMenu" class="max-w-xs bg-[#005DAA] rounded-full flex items-center text-sm focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#005DAA] focus:ring-white">
               <img class="h-8 w-8 rounded-full border-2 border-white" src="/images/UserAvatar.jpg" alt="User Avatar" />
            </button>
             <div v-if="profileMenuOpen" class="origin-top-right absolute right-0 mt-2 w-48 rounded-md shadow-lg py-1 bg-white ring-1 ring-black ring-opacity-5 focus:outline-none z-50">
                <a id="dashboard-profile-settings" @click="handleToSettings" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">
                  Settings
                </a>
             </div>
          </div>
        </div>
      </div>
    </nav>

    <!-- Main Content -->
    <main class="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
      <div class="px-4 py-6 sm:px-0">
        <h1 class="text-3xl font-bold text-gray-900 mb-6">Welcome Back, John</h1>
        
        <div class="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-3">
          <!-- Schedule Visit Card -->
          <div class="bg-white overflow-hidden shadow-xl rounded-lg hover:shadow-2xl transition-shadow duration-300">
            <div class="p-6">
              <div class="flex items-center">
                <div class="flex-shrink-0 bg-blue-100 rounded-md p-3">
                  <svg class="h-6 w-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"></path></svg>
                </div>
                <div class="ml-5 w-0 flex-1">
                  <dt class="text-sm font-medium text-gray-500 truncate">
                    Need a doctor?
                  </dt>
                  <dd class="flex items-baseline">
                    <div class="text-2xl font-semibold text-gray-900">
                      Schedule a Visit
                    </div>
                  </dd>
                </div>
              </div>
            </div>
            <div class="bg-gray-50 px-5 py-3">
              <div class="text-sm">
                <a id="dashboard-schedule-visit" @click="handleToSchedule" class="font-medium text-[#009CDE] hover:text-[#007bb0] cursor-pointer">
                  Book appointment <span aria-hidden="true">&rarr;</span>
                </a>
              </div>
            </div>
          </div>

          <!-- Instant Visit Card (Menu Trigger) -->
           <div class="bg-white overflow-hidden shadow-xl rounded-lg hover:shadow-2xl transition-shadow duration-300 relative">
            <div class="p-6">
              <div class="flex items-center">
                <div class="flex-shrink-0 bg-green-100 rounded-md p-3">
                  <svg class="h-6 w-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
                </div>
                <div class="ml-5 w-0 flex-1">
                  <dt class="text-sm font-medium text-gray-500 truncate">
                    Urgent Care
                  </dt>
                  <dd class="flex items-baseline">
                    <div class="text-2xl font-semibold text-gray-900">
                      Request Instant Visit
                    </div>
                  </dd>
                </div>
              </div>
            </div>
            <div class="bg-gray-50 px-5 py-3 flex justify-between items-center">
               <button id="dashboard-menu" @click="toggleInstantMenu" class="font-medium text-[#009CDE] hover:text-[#007bb0] focus:outline-none">
                  Get started <span aria-hidden="true">&rarr;</span>
               </button>
               <div v-if="instantMenuOpen" class="absolute bottom-12 right-4 w-48 bg-white shadow-lg rounded-md border border-gray-100 z-10">
                  <a id="dashboard-menu-instant" @click="handleToInstant" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer">
                    Start Triage
                  </a>
               </div>
            </div>
          </div>

          <!-- Appointments Card -->
          <div class="bg-white overflow-hidden shadow-xl rounded-lg hover:shadow-2xl transition-shadow duration-300">
            <img src="/images/Appointments.jpg" alt="Calendar" class="w-full h-32 object-cover" />
            <div class="p-4">
              <h3 class="text-lg font-medium text-gray-900">Your Appointments</h3>
              <p class="mt-1 text-sm text-gray-500">View upcoming and past visits.</p>
              <div class="mt-4">
                 <button @click="handleToAppts" class="w-full bg-[#005DAA] text-white py-2 px-4 rounded-md hover:bg-[#004a87] transition-colors">
                   View Calendar
                 </button>
              </div>
            </div>
          </div>
        </div>

        <div class="mt-8 bg-white shadow rounded-lg p-6">
           <h3 class="text-lg leading-6 font-medium text-gray-900">Health Conditions</h3>
           <div class="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
              <div class="relative rounded-lg border border-gray-300 bg-white px-6 py-5 shadow-sm flex items-center space-x-3 hover:border-gray-400 focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                <div class="flex-shrink-0">
                  <img class="h-10 w-10 rounded-full" src="/images/ColdFlu.jpg" alt="Cold & Flu">
                </div>
                <div class="flex-1 min-w-0">
                  <span class="absolute inset-0" aria-hidden="true"></span>
                  <p class="text-sm font-medium text-gray-900">Cold & Flu</p>
                  <p class="text-sm text-gray-500 truncate">Common symptoms</p>
                </div>
              </div>
               <div class="relative rounded-lg border border-gray-300 bg-white px-6 py-5 shadow-sm flex items-center space-x-3 hover:border-gray-400 focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-indigo-500">
                <div class="flex-shrink-0">
                  <img class="h-10 w-10 rounded-full" src="/images/Allergies.jpg" alt="Allergies">
                </div>
                <div class="flex-1 min-w-0">
                  <span class="absolute inset-0" aria-hidden="true"></span>
                  <p class="text-sm font-medium text-gray-900">Allergies</p>
                  <p class="text-sm text-gray-500 truncate">Seasonal relief</p>
                </div>
              </div>
           </div>
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
  name: 'DASHBOARD',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const profileMenuOpen = ref(false)
    const instantMenuOpen = ref(false)

    const toggleProfileMenu = () => profileMenuOpen.value = !profileMenuOpen.value
    const toggleInstantMenu = () => instantMenuOpen.value = !instantMenuOpen.value

    const handleToSettings = async () => {
      // ACT_DASH_TO_SETTINGS
      store.setCurrentPageId('SETTINGS_ACCOUNT')
      await router.push({ name: 'SETTINGS_ACCOUNT' })
    }

    const handleBackHome = async () => {
      // ACT_DASH_BACK_HOME
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    const handleToAppts = async () => {
      // ACT_DASH_TO_APPTS
      store.setCurrentPageId('APPOINTMENTS_LIST')
      await router.push({ name: 'APPOINTMENTS_LIST' })
    }

    const handleToBilling = async () => {
      // ACT_DASH_TO_BILLING
      store.setCurrentPageId('BILLING_OVERVIEW')
      await router.push({ name: 'BILLING_OVERVIEW' })
    }

    const handleToSchedule = async () => {
      // ACT_DASH_TO_SCHEDULE_DIRECT
      store.setCurrentPageId('VISIT_TYPE_SELECTION')
      await router.push({ name: 'VISIT_TYPE_SELECTION' })
    }

    const handleToInstant = async () => {
      // ACT_DASH_TO_INSTANT_VISIT_MENU
      store.setCurrentPageId('INSTANT_VISIT_TRIAGE')
      await router.push({ name: 'INSTANT_VISIT_TRIAGE' })
    }

    return {
      profileMenuOpen,
      instantMenuOpen,
      toggleProfileMenu,
      toggleInstantMenu,
      handleToSettings,
      handleBackHome,
      handleToAppts,
      handleToBilling,
      handleToSchedule,
      handleToInstant
    }
  }
}
</script>