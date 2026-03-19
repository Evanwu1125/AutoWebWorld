<template>
  <div class="min-h-screen bg-white flex flex-col">
    <!-- Navigation -->
    <nav class="bg-white border-b border-gray-200 z-20 relative">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="flex justify-between h-16">
          <div class="flex">
            <div class="flex-shrink-0 flex items-center">
              <img class="h-8 w-auto" src="/images/Optimizely.jpg" alt="Optimizely" />
            </div>
            <div class="hidden sm:ml-6 sm:flex sm:space-x-8">
              <button id="nav-dashboard" @click="goToDashboard" class="border-transparent text-gray-500 hover:border-gray-300 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
                Dashboard
              </button>
              
              <!-- Hover Menu for Experiments -->
              <div class="relative group inline-flex items-center h-full">
                <button id="nav-main-experiments" class="border-transparent text-gray-500 group-hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
                  Experiments
                </button>
                <div class="absolute top-full left-0 w-48 bg-white shadow-lg rounded-b-lg border border-gray-100 hidden group-hover:block z-50">
                  <button id="nav-main-experiments-all" @click="goToExperiments" class="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    All Experiments
                  </button>
                </div>
              </div>

              <!-- More Menu for Audiences -->
              <div class="relative inline-flex items-center h-full">
                <button id="nav-more-toggle" @click="toggleMoreMenu" class="border-transparent text-gray-500 hover:text-gray-700 inline-flex items-center px-1 pt-1 border-b-2 text-sm font-medium">
                  More
                </button>
                <div v-if="moreMenuOpen" class="absolute top-full left-0 w-48 bg-white shadow-lg rounded-b-lg border border-gray-100 z-50">
                  <button id="nav-more-audiences" @click="goToAudiences" class="block w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100">
                    Audiences
                  </button>
                </div>
              </div>
            </div>
          </div>
          <div class="flex items-center">
            <button class="bg-blue-600 text-white px-4 py-2 rounded-md text-sm font-medium hover:bg-blue-700">
              Sign In
            </button>
          </div>
        </div>
      </div>
    </nav>

    <!-- Hero Section -->
    <div class="relative bg-gray-900 overflow-hidden">
      <div class="absolute inset-0">
        <img class="w-full h-full object-cover opacity-40" src="/images/photo1764924221.jpg" alt="Analytics Background" />
        <div class="absolute inset-0 bg-gradient-to-r from-blue-900/80 to-gray-900/80 mix-blend-multiply"></div>
      </div>
      <div class="relative max-w-7xl mx-auto py-24 px-4 sm:py-32 sm:px-6 lg:px-8">
        <h1 class="text-4xl font-extrabold tracking-tight text-white sm:text-5xl lg:text-6xl">
          Unlock Digital Potential
        </h1>
        <p class="mt-6 text-xl text-gray-300 max-w-3xl">
          The world's leading digital experience platform. Experiment, optimize, and deliver personalized experiences at scale.
        </p>
        <div class="mt-10 max-w-sm sm:flex sm:max-w-none">
          <div class="space-y-4 sm:space-y-0 sm:inline-grid sm:grid-cols-2 sm:gap-5">
            <button @click="goToDashboard" class="flex items-center justify-center px-4 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-blue-700 bg-white hover:bg-gray-50 sm:px-8">
              Get Started
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Feature Grid -->
    <div class="bg-gray-50 py-12">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div class="grid grid-cols-1 gap-8 sm:grid-cols-2 lg:grid-cols-3">
          <div class="bg-white overflow-hidden shadow rounded-lg">
            <div class="px-4 py-5 sm:p-6">
              <div class="flex items-center">
                <div class="flex-shrink-0 bg-blue-500 rounded-md p-3">
                  <svg class="h-6 w-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                  </svg>
                </div>
                <div class="ml-5 w-0 flex-1">
                  <dt class="text-sm font-medium text-gray-500 truncate">
                    A/B Testing
                  </dt>
                  <dd class="flex items-baseline">
                    <div class="text-2xl font-semibold text-gray-900">
                      Advanced
                    </div>
                  </dd>
                </div>
              </div>
            </div>
          </div>
          <!-- More cards... -->
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
    const moreMenuOpen = ref(false)

    function goToDashboard() {
      if (signatureStore.cookie_accepted) {
        signatureStore.setCurrentPageId('DASHBOARD')
        router.push({ name: 'DASHBOARD' })
      }
    }

    function goToExperiments() {
      if (signatureStore.cookie_accepted) {
        signatureStore.setCurrentPageId('EXPERIMENTS_LIST')
        router.push({ name: 'EXPERIMENTS_LIST' })
      }
    }

    function toggleMoreMenu() {
      moreMenuOpen.value = !moreMenuOpen.value
    }

    function goToAudiences() {
      if (signatureStore.cookie_accepted) {
        signatureStore.setCurrentPageId('AUDIENCES_LIST')
        router.push({ name: 'AUDIENCES_LIST' })
      }
    }

    return {
      moreMenuOpen,
      goToDashboard,
      goToExperiments,
      toggleMoreMenu,
      goToAudiences
    }
  }
}
</script>