<template>
  <div class="min-h-screen bg-gray-50 flex flex-col">
    <header class="bg-white shadow-sm border-b border-gray-200">
      <div class="max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        <h1 class="text-xl font-bold text-gray-900">My Profile</h1>
        <button 
          id="profile-back-dashboard" 
          @click="goDashboard"
          class="text-blue-600 hover:text-blue-700 font-medium"
        >
          Dashboard
        </button>
      </div>
    </header>

    <main class="flex-grow max-w-5xl mx-auto px-4 sm:px-6 lg:px-8 py-10 w-full">
      <div class="bg-white shadow rounded-lg overflow-hidden">
        <!-- Profile Header -->
        <div class="px-6 py-8 border-b border-gray-200 flex flex-col sm:flex-row items-center gap-6">
          <div class="relative">
            <div class="h-24 w-24 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 text-3xl font-bold">
              {{ initials }}
            </div>
            <button 
              id="profile-settings-link" 
              @click="goToSettings"
              class="absolute bottom-0 right-0 bg-white border border-gray-300 rounded-full p-2 shadow-sm hover:bg-gray-50 text-gray-600"
            >
              <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
            </button>
          </div>
          
          <div class="text-center sm:text-left flex-grow">
            <h2 class="text-2xl font-bold text-gray-900">{{ store.display_name || 'John Doe' }}</h2>
            <p class="text-gray-500">{{ store.email }}</p>
            <p class="text-gray-400 text-sm mt-1">Account No: 1000294832</p>
          </div>
          
          <div class="flex flex-col gap-2 w-full sm:w-auto">
            <button 
              id="profile-rename-button"
              @click="goToRename"
              class="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50"
            >
              Edit Profile
            </button>
          </div>
        </div>

        <!-- Profile Details -->
        <div class="px-6 py-6 space-y-6">
          <div class="flex flex-col sm:flex-row justify-between py-4 border-b border-gray-100">
            <div class="sm:w-1/3">
              <span class="text-gray-500 font-medium">Sign-In Password</span>
            </div>
            <div class="sm:w-2/3 flex justify-between items-center mt-2 sm:mt-0">
              <span class="text-gray-900">**********</span>
              <button 
                id="profile-change-password-button"
                @click="goToChangePassword"
                class="text-blue-600 hover:text-blue-800 text-sm font-medium"
              >
                Change
              </button>
            </div>
          </div>

          <div class="flex flex-col sm:flex-row justify-between py-4 border-b border-gray-100">
            <div class="sm:w-1/3">
              <span class="text-gray-500 font-medium">Host Key</span>
            </div>
            <div class="sm:w-2/3 flex justify-between items-center mt-2 sm:mt-0">
              <span class="text-gray-900">******</span>
              <button class="text-blue-600 hover:text-blue-800 text-sm font-medium">Show</button>
            </div>
          </div>

          <div class="flex flex-col sm:flex-row justify-between py-4 border-b border-gray-100">
            <div class="sm:w-1/3">
              <span class="text-gray-500 font-medium">Personal Meeting ID</span>
            </div>
            <div class="sm:w-2/3 flex justify-between items-center mt-2 sm:mt-0">
              <span class="text-gray-900">312-456-7890</span>
              <button class="text-blue-600 hover:text-blue-800 text-sm font-medium">Edit</button>
            </div>
          </div>
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'PROFILE_OVERVIEW',
  setup() {
    const router = useRouter();
    const store = useSignatureStore();

    const initials = computed(() => {
      const name = store.display_name || 'John Doe';
      return name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase();
    });

    const goToSettings = async () => {
      if (store.handleAction('ACT_PROFILE_GO_TO_SETTINGS')) {
        await router.push({ name: 'SETTINGS_GENERAL' });
      }
    };

    const goToRename = async () => {
      if (store.handleAction('ACT_PROFILE_GO_TO_RENAME')) {
        await router.push({ name: 'PROFILE_RENAME_FORM' });
      }
    };

    const goToChangePassword = async () => {
      if (store.handleAction('ACT_PROFILE_GO_TO_CHANGE_PASSWORD')) {
        await router.push({ name: 'PROFILE_CHANGE_PASSWORD_FORM' });
      }
    };

    const goDashboard = async () => {
      if (store.handleAction('ACT_PROFILE_BACK_DASHBOARD')) {
        await router.push({ name: 'DASHBOARD' });
      }
    };

    return {
      store,
      initials,
      goToSettings,
      goToRename,
      goToChangePassword,
      goDashboard
    };
  }
}
</script>