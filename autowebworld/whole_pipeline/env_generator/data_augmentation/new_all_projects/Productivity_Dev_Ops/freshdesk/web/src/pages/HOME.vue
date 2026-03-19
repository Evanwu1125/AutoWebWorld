<template>
  <div class="h-screen bg-slate-50 flex flex-col font-inter text-slate-900">
    <!-- Header/Navigation -->
    <header class="bg-white border-b border-slate-200 z-20 sticky top-0">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex">
            <div class="flex-shrink-0 flex items-center">
              <span class="text-2xl font-bold text-blue-600">HelpDesk</span>
            </div>
            <div class="hidden sm:ml-6 sm:flex sm:space-x-8">
              <!-- Direct Nav -->
              <a id="nav-tickets-direct" 
                 href="#" 
                 @click.prevent="handleGoToTicketsDirect"
                 class="border-transparent text-slate-500 hover:border-blue-500 hover:text-slate-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200">
                Tickets (Direct)
              </a>
              <a id="nav-dashboard" 
                 href="#" 
                 @click.prevent="handleGoToDashboard"
                 class="border-transparent text-slate-500 hover:border-blue-500 hover:text-slate-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium transition-colors duration-200">
                Dashboard
              </a>
              
              <!-- Hover Menu (Simulated with Click/Hover logic) -->
              <div id="menu-support" class="relative group inline-flex items-center h-full">
                <button class="border-transparent text-slate-500 hover:text-slate-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
                  Support (Hover)
                </button>
                <div class="absolute left-0 top-16 w-48 bg-white rounded-md shadow-lg py-1 border border-slate-100 hidden group-hover:block transition-all duration-200">
                  <a href="#" class="tickets-link block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50" @click.prevent="handleGoToTicketsHover('tickets')">Tickets</a>
                  <a href="#" class="contacts-link block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50" @click.prevent="handleGoToTicketsHover('contacts')">Contacts</a>
                  <a href="#" class="reports-link block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50" @click.prevent="handleGoToTicketsHover('reports')">Reports</a>
                </div>
              </div>

              <!-- Dropdown Menu -->
              <div class="relative inline-flex items-center h-full ml-4">
                 <button id="nav-dropdown" 
                         @click="toggleNavDropdown"
                         class="text-slate-500 hover:text-slate-700 inline-flex items-center text-sm font-medium">
                   More <span class="ml-1">▼</span>
                 </button>
                 <div v-if="navDropdownOpen" id="nav-dropdown-menu" class="absolute left-0 top-16 w-48 bg-white rounded-md shadow-lg py-1 border border-slate-100 z-50">
                    <div id="nav-tickets-item" class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer" @click="handleGoToContactsMenu('tickets')">Tickets</div>
                    <div id="nav-contacts-item" class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer" @click="handleGoToContactsMenu('contacts')">Contacts</div>
                    <div id="nav-reports-item" class="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-50 cursor-pointer" @click="handleGoToContactsMenu('reports')">Reports</div>
                 </div>
              </div>

            </div>
          </div>
          <div class="flex items-center">
            <div class="flex-shrink-0">
               <img class="h-8 w-8 rounded-full" src="/images/photo1765352430.jpg" alt="Agent Profile">
            </div>
          </div>
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <main class="flex-1 relative z-0 overflow-y-auto">
      <!-- Hero Section with Background -->
      <div class="relative bg-blue-800 pb-32 overflow-hidden">
         <!-- Use ImageGetter implicitly by providing a descriptive image path that doesn't exist yet -->
        <div class="absolute inset-0">
          <img class="w-full h-full object-cover opacity-30" src="/images/SupportTeam.jpg" alt="Support Team Office">
          <div class="absolute inset-0 bg-blue-900 mix-blend-multiply" aria-hidden="true"></div>
        </div>
        <div class="relative max-w-7xl mx-auto py-24 px-4 sm:py-32 sm:px-6 lg:px-8 flex flex-col items-center text-center">
          <h1 class="text-4xl font-extrabold tracking-tight text-white sm:text-5xl lg:text-6xl">
            Deliver exceptional support
          </h1>
          <p class="mt-6 text-xl text-blue-100 max-w-3xl">
            Manage tickets, support customers, and analyze performance all in one place. Fast, intuitive, and reliable.
          </p>
          <div class="mt-10 max-w-sm mx-auto sm:max-w-none sm:flex sm:justify-center">
             <div class="space-y-4 sm:space-y-0 sm:mx-auto sm:inline-grid sm:grid-cols-2 sm:gap-5">
                <a href="#" @click.prevent="handleGoToTicketsDirect" class="flex items-center justify-center px-4 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-blue-700 bg-white hover:bg-blue-50 sm:px-8">
                  View Tickets
                </a>
                <a href="#" @click.prevent="handleGoToDashboard" class="flex items-center justify-center px-4 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-blue-600 hover:bg-blue-700 sm:px-8">
                  Dashboard
                </a>
             </div>
          </div>
        </div>
      </div>

      <!-- Features Section -->
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 -mt-24 relative z-10">
         <div class="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <div class="bg-white overflow-hidden shadow rounded-lg hover:shadow-lg transition-shadow duration-300">
               <div class="p-6">
                  <div class="flex items-center">
                     <div class="flex-shrink-0 bg-blue-500 rounded-md p-3">
                        <!-- Icon placeholder -->
                        <span class="text-white text-xl">🎫</span>
                     </div>
                     <div class="ml-4">
                        <h3 class="text-lg font-medium text-slate-900">Ticket Management</h3>
                        <p class="mt-1 text-sm text-slate-500">Organize and prioritize support requests.</p>
                     </div>
                  </div>
               </div>
            </div>
            <div class="bg-white overflow-hidden shadow rounded-lg hover:shadow-lg transition-shadow duration-300">
               <div class="p-6">
                  <div class="flex items-center">
                     <div class="flex-shrink-0 bg-green-500 rounded-md p-3">
                        <span class="text-white text-xl">👥</span>
                     </div>
                     <div class="ml-4">
                        <h3 class="text-lg font-medium text-slate-900">Contact Database</h3>
                        <p class="mt-1 text-sm text-slate-500">Manage customer information effectively.</p>
                     </div>
                  </div>
               </div>
            </div>
            <div class="bg-white overflow-hidden shadow rounded-lg hover:shadow-lg transition-shadow duration-300">
               <div class="p-6">
                  <div class="flex items-center">
                     <div class="flex-shrink-0 bg-purple-500 rounded-md p-3">
                        <span class="text-white text-xl">📊</span>
                     </div>
                     <div class="ml-4">
                        <h3 class="text-lg font-medium text-slate-900">Analytics</h3>
                        <p class="mt-1 text-sm text-slate-500">Track performance and customer satisfaction.</p>
                     </div>
                  </div>
               </div>
            </div>
         </div>
      </div>
    </main>

    <!-- Cookie Consent Modal -->
    <div v-if="!signatureStore.cookie_consent_given" class="fixed inset-0 bg-slate-900 bg-opacity-50 z-[10000] flex items-end justify-center sm:items-center">
      <div class="bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6 animate-fade-in-up">
        <div class="sm:flex sm:items-start">
          <div class="mx-auto flex-shrink-0 flex items-center justify-center h-12 w-12 rounded-full bg-blue-100 sm:mx-0 sm:h-10 sm:w-10">
            <span class="text-2xl">🍪</span>
          </div>
          <div class="mt-3 text-center sm:mt-0 sm:ml-4 sm:text-left">
            <h3 class="text-lg leading-6 font-medium text-slate-900" id="modal-title">
              We Value Your Privacy
            </h3>
            <div class="mt-2">
              <p class="text-sm text-slate-500">
                We use cookies to enhance your browsing experience, serve personalized content, and analyze our traffic. By clicking 'Accept All', you consent to our use of cookies.
              </p>
            </div>
          </div>
        </div>
        <div class="mt-5 sm:mt-4 sm:flex sm:flex-row-reverse">
          <button type="button" 
                  id="cookie-accept"
                  @click="handleAcceptCookies"
                  class="w-full inline-flex justify-center rounded-md border border-transparent shadow-sm px-4 py-2 bg-blue-600 text-base font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 sm:ml-3 sm:w-auto sm:text-sm">
            Accept All
          </button>
          <!-- Optional Decline button if needed, but not in FSM for HOME -->
        </div>
      </div>
    </div>

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
    const signatureStore = useSignatureStore()
    const navDropdownOpen = ref(false)

    // ACT_HOME_ACCEPT_COOKIES
    const handleAcceptCookies = () => {
      signatureStore.cookie_consent_given = true
    }

    // ACT_HOME_GO_TO_TICKETS_DIRECT
    const handleGoToTicketsDirect = async () => {
      if (signatureStore.cookie_consent_given !== true) return
      signatureStore.home_nav_target = 'TICKETS_LIST'
      signatureStore.setCurrentPageId('TICKETS_LIST')
      await router.push({ name: 'TICKETS_LIST' })
    }

    // ACT_HOME_GO_TO_DASHBOARD_DIRECT
    const handleGoToDashboard = async () => {
      if (signatureStore.cookie_consent_given !== true) return
      signatureStore.home_nav_target = 'DASHBOARD'
      signatureStore.setCurrentPageId('DASHBOARD')
      await router.push({ name: 'DASHBOARD' })
    }

    // ACT_HOME_GO_TO_TICKETS_HOVER
    const handleGoToTicketsHover = async (value) => {
      // Logic from FSM: selector click on "#menu-support .tickets-link"
      // value arg is just for UI, FSM always goes to TICKETS_LIST
      if (signatureStore.cookie_consent_given !== true) return
      if (value === 'tickets') {
          signatureStore.home_nav_target = 'TICKETS_LIST'
          signatureStore.setCurrentPageId('TICKETS_LIST')
          await router.push({ name: 'TICKETS_LIST' })
      }
    }

    const toggleNavDropdown = () => {
      navDropdownOpen.value = !navDropdownOpen.value
    }

    // ACT_HOME_GO_TO_CONTACTS_MENU
    const handleGoToContactsMenu = async (value) => {
       if (signatureStore.cookie_consent_given !== true) return
       if (value === 'contacts') {
          signatureStore.home_nav_target = 'CONTACTS_LIST'
          signatureStore.setCurrentPageId('CONTACTS_LIST')
          await router.push({ name: 'CONTACTS_LIST' })
       }
       navDropdownOpen.value = false
    }

    return {
      signatureStore,
      navDropdownOpen,
      handleAcceptCookies,
      handleGoToTicketsDirect,
      handleGoToDashboard,
      handleGoToTicketsHover,
      toggleNavDropdown,
      handleGoToContactsMenu
    }
  }
}
</script>

<style scoped>
/* Custom animations if needed */
@keyframes fadeInUp {
  from {
    opacity: 0;
    transform: translate3d(0, 20px, 0);
  }
  to {
    opacity: 1;
    transform: none;
  }
}
.animate-fade-in-up {
  animation: fadeInUp 0.3s ease-out;
}
</style>