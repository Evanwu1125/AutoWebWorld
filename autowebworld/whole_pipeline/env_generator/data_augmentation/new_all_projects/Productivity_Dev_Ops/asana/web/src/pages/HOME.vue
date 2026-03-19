<template>
  <div class="min-h-screen bg-gray-50 flex flex-col font-sans">
    <!-- Cookie Consent Interceptor -->
    <CookieConsentModal />

    <!-- Navigation Header -->
    <header class="bg-white border-b border-gray-200 sticky top-0 z-20">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between items-center h-16">
          <div class="flex items-center">
            <div class="flex-shrink-0 flex items-center gap-2">
              <div class="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                 <span class="text-white font-bold">A</span>
              </div>
              <span class="text-xl font-bold text-gray-900 tracking-tight">AsanaClone</span>
            </div>
            
            <!-- Desktop Navigation -->
            <nav class="hidden md:ml-10 md:flex md:space-x-8">
              <!-- Direct Link -->
              <a 
                id="nav-projects-direct" 
                @click="goToProjects"
                class="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium cursor-pointer"
              >
                Projects
              </a>

              <!-- Hover Menu -->
              <div 
                id="nav-work-menu"
                class="relative group inline-flex items-center px-1 pt-1 border-b-2 border-transparent text-sm font-medium text-gray-500 hover:text-gray-700 cursor-pointer"
              >
                <span>Work</span>
                <div class="absolute left-0 top-full mt-2 w-48 rounded-md shadow-lg bg-white ring-1 ring-black ring-opacity-5 opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-50">
                  <div class="py-1">
                    <a 
                      id="nav-work-menu-projects"
                      @click="goToProjects"
                      class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 cursor-pointer"
                    >
                      All Projects
                    </a>
                    <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Reporting</a>
                    <a href="#" class="block px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">Portfolios</a>
                  </div>
                </div>
              </div>

              <a 
                id="nav-my-tasks-direct"
                @click="goToMyTasks" 
                class="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium cursor-pointer"
              >
                My Tasks
              </a>
            </nav>
          </div>
          
          <!-- Mobile Menu Button -->
          <div class="flex items-center">
            <button 
              id="nav-menu-toggle"
              @click="toggleMobileMenu"
              class="inline-flex items-center justify-center p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-indigo-500"
            >
              <span class="sr-only">Open main menu</span>
              <svg class="block h-6 w-6" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" aria-hidden="true">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      <!-- Mobile Menu -->
      <div v-if="mobileMenuOpen" class="bg-white border-t border-gray-200">
        <div class="pt-2 pb-3 space-y-1">
          <a 
            id="nav-menu-projects"
            @click="goToProjects"
            class="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-500 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-700 cursor-pointer"
          >
            Projects
          </a>
          <a 
            @click="goToMyTasks"
            class="block pl-3 pr-4 py-2 border-l-4 border-transparent text-base font-medium text-gray-500 hover:bg-gray-50 hover:border-gray-300 hover:text-gray-700 cursor-pointer"
          >
            My Tasks
          </a>
        </div>
      </div>
    </header>

    <!-- Main Content (Hero Section) -->
    <main class="flex-grow">
      <div class="relative bg-white overflow-hidden">
        <div class="max-w-7xl mx-auto">
          <div class="relative z-10 pb-8 bg-white sm:pb-16 md:pb-20 lg:max-w-2xl lg:w-full lg:pb-28 xl:pb-32">
            <svg class="hidden lg:block absolute right-0 inset-y-0 h-full w-48 text-white transform translate-x-1/2" fill="currentColor" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
              <polygon points="50,0 100,0 50,100 0,100" />
            </svg>

            <div class="relative pt-6 px-4 sm:px-6 lg:px-8"></div>

            <main class="mt-10 mx-auto max-w-7xl px-4 sm:mt-12 sm:px-6 md:mt-16 lg:mt-20 lg:px-8 xl:mt-28">
              <div class="sm:text-center lg:text-left">
                <h1 class="text-4xl tracking-tight font-extrabold text-gray-900 sm:text-5xl md:text-6xl">
                  <span class="block xl:inline">Orchestrate your work,</span>
                  <span class="block text-indigo-600 xl:inline">from small details to big picture.</span>
                </h1>
                <p class="mt-3 text-base text-gray-500 sm:mt-5 sm:text-lg sm:max-w-xl sm:mx-auto md:mt-5 md:text-xl lg:mx-0">
                  Keep your team's work on track. Manage projects, tasks, and timelines with clarity and ease. The modern way to work together.
                </p>
                <div class="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
                  <div class="rounded-md shadow">
                    <a 
                      @click="goToProjects"
                      class="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 md:py-4 md:text-lg md:px-10 cursor-pointer"
                    >
                      Get Started
                    </a>
                  </div>
                  <div class="mt-3 sm:mt-0 sm:ml-3">
                    <a 
                      href="#" 
                      class="w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-indigo-700 bg-indigo-100 hover:bg-indigo-200 md:py-4 md:text-lg md:px-10"
                    >
                      See Demo
                    </a>
                  </div>
                </div>
              </div>
            </main>
          </div>
        </div>
        <div class="lg:absolute lg:inset-y-0 lg:right-0 lg:w-1/2">
          <img 
            class="h-56 w-full object-cover sm:h-72 md:h-96 lg:w-full lg:h-full" 
            src="/images/OfficeTeam.jpg" 
            alt="Team collaborating on project" 
          />
        </div>
      </div>
      
      <!-- Feature Section -->
      <div class="py-12 bg-gray-50">
        <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div class="lg:text-center">
            <h2 class="text-base text-indigo-600 font-semibold tracking-wide uppercase">Features</h2>
            <p class="mt-2 text-3xl leading-8 font-extrabold tracking-tight text-gray-900 sm:text-4xl">
              A better way to work
            </p>
            <p class="mt-4 max-w-2xl text-xl text-gray-500 lg:mx-auto">
              Everything you need to manage complex projects without the chaos.
            </p>
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
import CookieConsentModal from '../components/CookieConsentModal.vue'

export default {
  name: 'HOME',
  components: {
    CookieConsentModal
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    const mobileMenuOpen = ref(false)

    const toggleMobileMenu = () => {
      mobileMenuOpen.value = !mobileMenuOpen.value
    }

    const goToProjects = async () => {
      // Actions: HOME_GO_TO_PROJECTS_DIRECT, HOME_GO_TO_PROJECTS_HOVER, HOME_GO_TO_PROJECTS_MENU
      store.setCurrentPageId('PROJECTS_LIST')
      await router.push({ name: 'PROJECTS_LIST' })
    }

    const goToMyTasks = async () => {
      // Action: HOME_GO_TO_MY_TASKS_DIRECT
      store.setCurrentPageId('MY_TASKS_LIST')
      await router.push({ name: 'MY_TASKS_LIST' })
    }

    return {
      mobileMenuOpen,
      toggleMobileMenu,
      goToProjects,
      goToMyTasks
    }
  }
}
</script>