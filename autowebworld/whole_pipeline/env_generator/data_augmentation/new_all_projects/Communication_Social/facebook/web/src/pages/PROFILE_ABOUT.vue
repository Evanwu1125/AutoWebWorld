<template>
  <div class="min-h-screen bg-gray-100 pb-10">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20 h-16 flex items-center px-4">
      <button 
        id="back-to-timeline"
        @click="goBack"
        class="flex items-center gap-2 text-gray-600 hover:bg-gray-100 px-3 py-2 rounded-lg transition-colors"
      >
        <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        <span class="font-bold text-lg">Back to Timeline</span>
      </button>
    </header>

    <div class="max-w-4xl mx-auto px-4 py-6">
      <div class="bg-white rounded-lg shadow-sm overflow-hidden">
        <div class="p-6 border-b border-gray-100">
          <h1 class="text-2xl font-bold text-gray-900">About</h1>
        </div>
        
        <div class="flex flex-col md:flex-row min-h-[400px]">
          <!-- Sidebar -->
          <div class="w-full md:w-1/3 border-r border-gray-100 p-2">
            <div class="space-y-1">
              <button class="w-full text-left px-4 py-3 rounded-lg bg-blue-50 text-blue-600 font-semibold">Overview</button>
              <button class="w-full text-left px-4 py-3 rounded-lg hover:bg-gray-50 text-gray-600 font-medium transition-colors">Work and Education</button>
              <button class="w-full text-left px-4 py-3 rounded-lg hover:bg-gray-50 text-gray-600 font-medium transition-colors">Places Lived</button>
              <button class="w-full text-left px-4 py-3 rounded-lg hover:bg-gray-50 text-gray-600 font-medium transition-colors">Contact and Basic Info</button>
              <button class="w-full text-left px-4 py-3 rounded-lg hover:bg-gray-50 text-gray-600 font-medium transition-colors">Family and Relationships</button>
            </div>
          </div>
          
          <!-- Content -->
          <div class="flex-1 p-6">
            <div class="space-y-6">
              <div class="flex items-start gap-4">
                <div class="mt-1 text-gray-400">
                  <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                  </svg>
                </div>
                <div>
                  <p class="text-gray-900 font-medium">Software Engineer at <span class="font-bold">Tech Corp</span></p>
                  <p class="text-sm text-gray-500">2020 - Present</p>
                </div>
              </div>
              
              <div class="flex items-start gap-4">
                <div class="mt-1 text-gray-400">
                  <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                  </svg>
                </div>
                <div>
                  <p class="text-gray-900 font-medium">Studied Computer Science at <span class="font-bold">University of Technology</span></p>
                </div>
              </div>

              <div class="flex items-start gap-4">
                <div class="mt-1 text-gray-400">
                  <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
                  </svg>
                </div>
                <div>
                  <p class="text-gray-900 font-medium">Lives in <span class="font-bold">San Francisco, California</span></p>
                </div>
              </div>
              
              <div class="flex items-start gap-4">
                <div class="mt-1 text-gray-400">
                  <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </div>
                <div>
                  <p class="text-gray-900 font-medium">From <span class="font-bold">New York, New York</span></p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'PROFILE_ABOUT',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();

    const goBack = async () => {
      signatureStore.currentPageId = 'PROFILE_TIMELINE';
      // Ensure we navigate back to the same profile
      const userId = route.params.id || signatureStore.selected_user_id;
      await router.push({ name: 'PROFILE_TIMELINE', params: { id: userId } });
    };

    return {
      goBack
    };
  }
}
</script>