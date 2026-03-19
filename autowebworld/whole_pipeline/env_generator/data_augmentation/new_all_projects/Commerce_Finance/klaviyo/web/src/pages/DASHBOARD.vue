<template>
  <div class="min-h-screen bg-slate-50">
    <!-- Top Bar -->
    <header class="bg-white border-b border-slate-200 sticky top-0 z-40">
      <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <div class="flex items-center cursor-pointer" id="logo-home" @click="handleBackHome">
          <div class="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center mr-2">
            <span class="text-white font-bold">K</span>
          </div>
          <span class="text-xl font-bold text-slate-900">Klaviyo</span>
        </div>
        <div class="flex items-center space-x-4">
          <div class="w-8 h-8 rounded-full bg-slate-200 overflow-hidden border border-slate-300">
            <img src="/images/UserAvatar.jpg" alt="User" class="w-full h-full object-cover" />
          </div>
        </div>
      </div>
    </header>

    <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div class="mb-8">
        <h1 class="text-3xl font-bold text-slate-900">Dashboard</h1>
        <p class="text-slate-500 mt-1">Welcome back. Here's what's happening today.</p>
      </div>

      <!-- Quick Actions / Cards Grid -->
      <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        
        <!-- Campaigns Card -->
        <div 
          id="card-campaigns"
          @click="goToCampaigns"
          class="bg-white rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 cursor-pointer overflow-hidden border border-slate-100 group"
        >
          <div class="h-32 overflow-hidden relative">
            <img src="/images/Campaign.jpg" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
            <div class="absolute inset-0 bg-blue-900/10 group-hover:bg-blue-900/0 transition-colors"></div>
          </div>
          <div class="p-6">
            <div class="flex items-center justify-between mb-2">
              <h3 class="text-lg font-bold text-slate-900">Campaigns</h3>
              <svg class="w-5 h-5 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"></path></svg>
            </div>
            <p class="text-slate-500 text-sm">Create and schedule email/SMS campaigns.</p>
          </div>
        </div>

        <!-- Flows Card -->
        <div 
          id="card-flows"
          @click="goToFlows"
          class="bg-white rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 cursor-pointer overflow-hidden border border-slate-100 group"
        >
          <div class="h-32 overflow-hidden relative">
            <img src="/images/CardFlows.jpg" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
            <div class="absolute inset-0 bg-emerald-900/10 group-hover:bg-emerald-900/0 transition-colors"></div>
          </div>
          <div class="p-6">
            <div class="flex items-center justify-between mb-2">
              <h3 class="text-lg font-bold text-slate-900">Flows</h3>
              <svg class="w-5 h-5 text-emerald-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
            </div>
            <p class="text-slate-500 text-sm">Automate your marketing with flows.</p>
          </div>
        </div>

        <!-- Lists & Segments Card -->
        <div 
          id="card-lists-segments"
          @click="goToLists"
          class="bg-white rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 cursor-pointer overflow-hidden border border-slate-100 group"
        >
          <div class="h-32 overflow-hidden relative">
            <img src="/images/Audience.jpg" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
            <div class="absolute inset-0 bg-purple-900/10 group-hover:bg-purple-900/0 transition-colors"></div>
          </div>
          <div class="p-6">
            <div class="flex items-center justify-between mb-2">
              <h3 class="text-lg font-bold text-slate-900">Audience</h3>
              <svg class="w-5 h-5 text-purple-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
            </div>
            <p class="text-slate-500 text-sm">Manage lists, segments, and profiles.</p>
          </div>
        </div>

        <!-- Signup Forms Card -->
        <div 
          id="card-signup-forms"
          @click="goToForms"
          class="bg-white rounded-xl shadow-sm hover:shadow-lg transition-all duration-300 cursor-pointer overflow-hidden border border-slate-100 group"
        >
          <div class="h-32 overflow-hidden relative">
            <img src="/images/CardForms.jpg" class="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500" />
            <div class="absolute inset-0 bg-orange-900/10 group-hover:bg-orange-900/0 transition-colors"></div>
          </div>
          <div class="p-6">
            <div class="flex items-center justify-between mb-2">
              <h3 class="text-lg font-bold text-slate-900">Signup Forms</h3>
              <svg class="w-5 h-5 text-orange-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path></svg>
            </div>
            <p class="text-slate-500 text-sm">Grow your audience with forms.</p>
          </div>
        </div>

      </div>

      <!-- Recent Activity Section (Decorative) -->
      <div class="mt-8 bg-white rounded-xl shadow-sm p-6 border border-slate-100">
        <h2 class="text-lg font-bold text-slate-900 mb-4">Performance Overview</h2>
        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
           <div class="p-4 bg-slate-50 rounded-lg">
             <div class="text-sm text-slate-500 mb-1">Total Revenue</div>
             <div class="text-2xl font-bold text-slate-900">$124,500</div>
             <div class="text-xs text-emerald-600 font-medium">+12% vs last month</div>
           </div>
           <div class="p-4 bg-slate-50 rounded-lg">
             <div class="text-sm text-slate-500 mb-1">Active Profiles</div>
             <div class="text-2xl font-bold text-slate-900">45,200</div>
             <div class="text-xs text-emerald-600 font-medium">+5% vs last month</div>
           </div>
           <div class="p-4 bg-slate-50 rounded-lg">
             <div class="text-sm text-slate-500 mb-1">Avg. Open Rate</div>
             <div class="text-2xl font-bold text-slate-900">24.5%</div>
             <div class="text-xs text-slate-400 font-medium">Flat vs last month</div>
           </div>
        </div>
      </div>

    </main>

    <PermissionModal 
      :is-visible="showLocationModal"
      @allow="handleGrantLocation" 
    />
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'
import PermissionModal from '../components/PermissionModal.vue'

export default {
  name: 'DASHBOARD',
  components: {
    PermissionModal
  },
  setup() {
    const router = useRouter()
    const store = useSignatureStore()
    
    // Only show if permission is null (not granted/denied yet)
    // FSM: preconditions: location_permission_granted eq null
    const showLocationModal = computed(() => store.location_permission_granted === null)

    function handleGrantLocation() {
      store.location_permission_granted = true
    }

    async function goToCampaigns() {
      store.setCurrentPageId('CAMPAIGNS_LIST')
      await router.push({ name: 'CAMPAIGNS_LIST' })
    }

    async function goToFlows() {
      store.setCurrentPageId('FLOWS_LIST')
      await router.push({ name: 'FLOWS_LIST' })
    }

    async function goToLists() {
      store.setCurrentPageId('LISTS_SEGMENTS')
      await router.push({ name: 'LISTS_SEGMENTS' })
    }

    async function goToForms() {
      store.setCurrentPageId('SIGNUP_FORMS_LIST')
      await router.push({ name: 'SIGNUP_FORMS_LIST' })
    }

    async function handleBackHome() {
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    return {
      showLocationModal,
      handleGrantLocation,
      goToCampaigns,
      goToFlows,
      goToLists,
      goToForms,
      handleBackHome
    }
  }
}
</script>