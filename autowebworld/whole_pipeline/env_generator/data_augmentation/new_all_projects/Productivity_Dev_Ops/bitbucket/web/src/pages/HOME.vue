<template>
  <div class="min-h-screen flex flex-col bg-white">
    <!-- Top Navigation -->
    <header class="bg-[#0052CC] text-white shadow-md relative z-20">
      <div class="container mx-auto px-4 h-16 flex items-center justify-between">
        <!-- Logo & Brand -->
        <div class="flex items-center space-x-2 font-bold text-xl">
          <svg class="w-8 h-8" fill="currentColor" viewBox="0 0 24 24"><path d="M11.53,2.54,2.23,5.16a1,1,0,0,0-.7,1.11L5,20.21a1.09,1.09,0,0,0,1.2.91l15.53-2.62a1,1,0,0,0,.7-1.11L19,3.46A1.1,1.1,0,0,0,17.84,2.54ZM13.88,15.6l-5.32.88L7.4,9.15l5.32-.88Z"/></svg>
          <span>Bitbucket</span>
        </div>

        <!-- Desktop Nav (Direct & Hover) -->
        <nav class="hidden md:flex items-center space-x-6">
          <a href="#" id="nav-repositories-direct" @click.prevent="goToRepos" class="hover:text-blue-200 transition-colors">Repositories</a>
          <a href="#" id="nav-pull-requests-direct" @click.prevent="goToPRs" class="hover:text-blue-200 transition-colors">Pull Requests</a>
          <a href="#" id="nav-pipelines-direct" @click.prevent="goToPipelines" class="hover:text-blue-200 transition-colors">Pipelines</a>

          <!-- Hover Menu Variant -->
          <div id="topnav-workspace-menu" class="relative group py-4 cursor-pointer">
            <span class="flex items-center hover:text-blue-200 transition-colors">
              Workspace <svg class="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </span>
            <div class="absolute top-full left-0 w-48 bg-white rounded-md shadow-lg py-2 hidden group-hover:block text-gray-800 border border-gray-200">
              <div class="menu-repositories px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="goToRepos">Repositories</div>
              <div class="menu-pull-requests px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="goToPRs">Pull Requests</div>
              <div class="menu-pipelines px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="goToPipelines">Pipelines</div>
            </div>
          </div>
        </nav>

        <!-- Mobile / Side Menu Toggle -->
        <div class="flex items-center">
           <button id="side-menu-toggle" @click="toggleSideMenu" class="p-2 hover:bg-blue-700 rounded-md focus:outline-none">
             <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg>
           </button>
        </div>
      </div>
    </header>

    <!-- Side Menu (Dropdown Variant) -->
    <div v-if="isSideMenuOpen" class="fixed inset-0 z-50 flex">
      <div class="relative w-64 bg-white shadow-xl h-full flex flex-col" id="side-menu">
        <div class="p-4 border-b flex justify-between items-center">
          <span class="font-bold text-lg text-[#172B4D]">Menu</span>
          <button @click="toggleSideMenu" class="text-gray-500 hover:text-gray-700">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/></svg>
          </button>
        </div>
        <div class="p-4 space-y-2">
          <div id="side-menu-item-repositories" @click="goToRepos" class="px-4 py-2 hover:bg-gray-100 rounded-md cursor-pointer text-[#172B4D]">Repositories</div>
          <div id="side-menu-item-pull-requests" @click="goToPRs" class="px-4 py-2 hover:bg-gray-100 rounded-md cursor-pointer text-[#172B4D]">Pull Requests</div>
          <div id="side-menu-item-pipelines" @click="goToPipelines" class="px-4 py-2 hover:bg-gray-100 rounded-md cursor-pointer text-[#172B4D]">Pipelines</div>
          <div id="side-menu-item-workspace-members" @click="goToMembers" class="px-4 py-2 hover:bg-gray-100 rounded-md cursor-pointer text-[#172B4D]">Members</div>
        </div>
      </div>
      <div class="flex-1 bg-black/20 backdrop-blur-sm" @click="toggleSideMenu"></div>
    </div>

    <!-- Hero Section -->
    <main class="flex-grow">
      <div class="relative bg-[#172B4D] text-white py-20 px-4 overflow-hidden">
        <!-- Background Image -->
        <div class="absolute inset-0 z-0 opacity-30">
           <img src="/images/SoftwareTeamCollaboration.jpg" alt="software team collaboration" class="w-full h-full object-cover" />
        </div>
        
        <div class="container mx-auto relative z-10 text-center">
          <h1 class="text-5xl font-bold mb-6">Build the future, together.</h1>
          <p class="text-xl text-blue-100 mb-10 max-w-2xl mx-auto">
            Bitbucket is the Git solution for professional teams. Distributed version control system that makes it easy for you to collaborate with your team.
          </p>
          <div class="flex justify-center space-x-4">
            <button @click="goToRepos" class="px-8 py-3 bg-[#0052CC] hover:bg-blue-600 text-white font-bold rounded-md shadow-lg transition-transform transform hover:-translate-y-1">
              Get Started
            </button>
            <button class="px-8 py-3 bg-white text-[#0052CC] font-bold rounded-md shadow-lg hover:bg-gray-100 transition-transform transform hover:-translate-y-1">
              Learn More
            </button>
          </div>
        </div>
      </div>

      <!-- Feature Highlights -->
      <div class="container mx-auto py-16 px-4">
        <div class="grid md:grid-cols-3 gap-8">
          <div class="bg-white p-8 rounded-lg shadow-sm border border-gray-100 text-center hover:shadow-md transition-shadow">
            <div class="w-16 h-16 bg-blue-100 text-blue-600 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"/></svg>
            </div>
            <h3 class="text-xl font-bold mb-3">Repositories</h3>
            <p class="text-gray-600">Secure, scalable, and reliable code hosting for your projects.</p>
          </div>
          <div class="bg-white p-8 rounded-lg shadow-sm border border-gray-100 text-center hover:shadow-md transition-shadow">
            <div class="w-16 h-16 bg-green-100 text-green-600 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"/></svg>
            </div>
            <h3 class="text-xl font-bold mb-3">Pull Requests</h3>
            <p class="text-gray-600">Collaborate on code with inline comments and approval workflows.</p>
          </div>
          <div class="bg-white p-8 rounded-lg shadow-sm border border-gray-100 text-center hover:shadow-md transition-shadow">
            <div class="w-16 h-16 bg-purple-100 text-purple-600 rounded-full flex items-center justify-center mx-auto mb-6">
              <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
            </div>
            <h3 class="text-xl font-bold mb-3">Pipelines</h3>
            <p class="text-gray-600">Integrated CI/CD to build, test, and deploy with confidence.</p>
          </div>
        </div>
      </div>
    </main>

    <!-- Footer -->
    <footer class="bg-gray-50 border-t border-gray-200 py-12">
      <div class="container mx-auto px-4 text-center text-gray-500">
        <p>&copy; 2025 Bitbucket Clone. All rights reserved.</p>
      </div>
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
    const signatureStore = useSignatureStore()
    const isSideMenuOpen = ref(false)

    const toggleSideMenu = () => {
      isSideMenuOpen.value = !isSideMenuOpen.value
    }

    const goToRepos = async () => {
      signatureStore.currentPageId = 'REPO_LIST'
      await router.push({ name: 'REPO_LIST' })
    }

    const goToPRs = async () => {
      signatureStore.currentPageId = 'PR_LIST'
      await router.push({ name: 'PR_LIST' })
    }

    const goToPipelines = async () => {
      signatureStore.currentPageId = 'PIPELINE_LIST'
      await router.push({ name: 'PIPELINE_LIST' })
    }
    
    const goToMembers = async () => {
      signatureStore.currentPageId = 'WORKSPACE_MEMBERS'
      await router.push({ name: 'WORKSPACE_MEMBERS' })
    }

    return {
      isSideMenuOpen,
      toggleSideMenu,
      goToRepos,
      goToPRs,
      goToPipelines,
      goToMembers
    }
  }
}
</script>