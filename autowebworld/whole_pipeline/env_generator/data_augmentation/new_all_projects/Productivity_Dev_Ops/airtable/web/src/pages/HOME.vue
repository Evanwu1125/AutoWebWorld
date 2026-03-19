<template>
  <div class="min-h-screen flex flex-col bg-white">
    <!-- Navbar -->
    <nav class="flex items-center justify-between px-6 py-4 bg-white border-b border-gray-100 z-50 relative">
      <div class="flex items-center gap-2">
        <div class="w-8 h-8 bg-blue-600 rounded-md flex items-center justify-center text-white font-bold text-xl">
          <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
            <path d="M5 3a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2V5a2 2 0 00-2-2H5zM5 11a2 2 0 00-2 2v2a2 2 0 002 2h2a2 2 0 002-2v-2a2 2 0 00-2-2H5zM11 5a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V5zM11 13a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
          </svg>
        </div>
        <span class="text-xl font-bold text-gray-900 tracking-tight">Airtable Clone</span>
      </div>

      <div class="hidden md:flex items-center gap-8">
        <button id="nav-bases-direct" @click="handleOpenBasesDirect" class="text-gray-600 hover:text-blue-600 font-medium transition-colors">
          Bases
        </button>
        
        <!-- Hover Menu -->
        <div 
          id="topbar-workspace-menu" 
          class="relative group py-2"
          @mouseenter="hoverMenuOpen = true"
          @mouseleave="hoverMenuOpen = false"
        >
          <button class="flex items-center gap-1 text-gray-600 hover:text-blue-600 font-medium">
            Workspaces
            <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          <div v-if="hoverMenuOpen" class="absolute top-full left-0 w-48 bg-white border border-gray-200 rounded-lg shadow-xl py-1 z-50">
             <div class="option-bases px-4 py-2 hover:bg-gray-50 cursor-pointer text-gray-700" @click="handleHoverMenuSelect('bases')">
               All Bases
             </div>
             <div class="option-workspaces px-4 py-2 hover:bg-gray-50 cursor-pointer text-gray-700" @click="handleHoverMenuSelect('workspaces')">
               My Workspace
             </div>
             <div class="option-templates px-4 py-2 hover:bg-gray-50 cursor-pointer text-gray-700" @click="handleHoverMenuSelect('templates')">
               Templates
             </div>
          </div>
        </div>

        <a href="#" class="text-gray-600 hover:text-blue-600 font-medium transition-colors">Pricing</a>
        <a href="#" class="text-gray-600 hover:text-blue-600 font-medium transition-colors">Enterprise</a>
      </div>

      <div class="flex items-center gap-4">
        <button class="text-gray-900 font-medium hover:underline">Log In</button>
        <button class="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-5 rounded-md transition-all shadow-md hover:shadow-lg transform hover:-translate-y-0.5">
          Sign Up for Free
        </button>
      </div>
    </nav>

    <!-- Main Content -->
    <div class="flex flex-1 relative overflow-hidden">
      <!-- Sidebar Navigation (Dropdown Variant) -->
      <aside class="w-64 border-r border-gray-100 bg-gray-50 p-6 hidden lg:block z-40">
        <div class="mb-8">
           <h3 class="text-xs font-bold text-gray-400 uppercase tracking-wider mb-4">Navigation</h3>
           
           <div class="relative">
             <button 
               id="sidebar-nav-dropdown" 
               @click="sidebarDropdownOpen = !sidebarDropdownOpen"
               class="w-full flex items-center justify-between px-3 py-2 bg-white border border-gray-200 rounded-md text-sm font-medium text-gray-700 hover:border-blue-400 transition-colors"
             >
               <span>Go to...</span>
               <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                 <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
               </svg>
             </button>

             <div v-if="sidebarDropdownOpen" class="absolute top-full left-0 w-full mt-1 bg-white border border-gray-200 rounded-md shadow-lg z-50">
               <div id="sidebar-nav-bases" class="px-3 py-2 hover:bg-blue-50 text-sm cursor-pointer text-gray-700" @click="handleSidebarSelect('bases')">
                 Bases Dashboard
               </div>
               <div id="sidebar-nav-automations" class="px-3 py-2 hover:bg-blue-50 text-sm cursor-pointer text-gray-700" @click="handleSidebarSelect('automations')">
                 Automations
               </div>
               <div id="sidebar-nav-forms" class="px-3 py-2 hover:bg-blue-50 text-sm cursor-pointer text-gray-700" @click="handleSidebarSelect('forms')">
                 Forms
               </div>
             </div>
           </div>
        </div>
      </aside>

      <!-- Hero Section -->
      <main class="flex-1 relative bg-blue-600 overflow-hidden flex flex-col justify-center items-center text-center p-12">
        <img src="/images/TeamCollaboration.jpg" alt="Team collaborating on project" class="absolute inset-0 w-full h-full object-cover opacity-20 mix-blend-multiply" />
        
        <div class="relative z-10 max-w-4xl mx-auto">
          <h1 class="text-5xl md:text-7xl font-extrabold text-white mb-6 tracking-tight leading-tight drop-shadow-lg">
            Connect Everything.<br/>Achieve Anything.
          </h1>
          <p class="text-xl text-blue-100 mb-10 max-w-2xl mx-auto leading-relaxed font-light">
            Airtable is a low-code platform for building collaborative apps. Customize your workflow, collaborate, and achieve your most ambitious goals.
          </p>
          <div class="flex flex-col sm:flex-row gap-4 justify-center">
             <button @click="handleOpenBasesDirect" class="bg-white text-blue-600 hover:bg-blue-50 font-bold py-4 px-8 rounded-lg text-lg shadow-xl hover:shadow-2xl transition-all transform hover:-translate-y-1">
               Get Started for Free
             </button>
             <button class="bg-blue-700 text-white hover:bg-blue-800 font-bold py-4 px-8 rounded-lg text-lg shadow-xl border border-blue-500 hover:shadow-2xl transition-all transform hover:-translate-y-1">
               Watch Demo
             </button>
          </div>
        </div>
      </main>
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
    const store = useSignatureStore()
    
    const hoverMenuOpen = ref(false)
    const sidebarDropdownOpen = ref(false)

    const handleOpenBasesDirect = async () => {
      // ACT_HOME_OPEN_BASES_DIRECT
      if (store.cookie_consent_given === true) {
        store.setCurrentPageId('BASES_DASHBOARD')
        await router.push({ name: 'BASES_DASHBOARD' })
      }
    }

    const handleHoverMenuSelect = async (value) => {
      // ACT_HOME_OPEN_BASES_HOVER
      if (store.cookie_consent_given === true && value === 'bases') {
        store.setCurrentPageId('BASES_DASHBOARD')
        await router.push({ name: 'BASES_DASHBOARD' })
      }
      hoverMenuOpen.value = false
    }

    const handleSidebarSelect = async (value) => {
      // ACT_HOME_OPEN_BASES_MENU
      if (store.cookie_consent_given === true && value === 'bases') {
        store.setCurrentPageId('BASES_DASHBOARD')
        await router.push({ name: 'BASES_DASHBOARD' })
      }
      sidebarDropdownOpen.value = false
    }

    return {
      hoverMenuOpen,
      sidebarDropdownOpen,
      handleOpenBasesDirect,
      handleHoverMenuSelect,
      handleSidebarSelect
    }
  }
}
</script>