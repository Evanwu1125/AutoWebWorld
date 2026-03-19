<template>
  <div class="min-h-screen bg-slate-50 flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
    <div class="max-w-md w-full text-center space-y-8">
      <div class="mx-auto flex items-center justify-center h-24 w-24 rounded-full bg-blue-100 mb-8">
        <svg class="h-12 w-12 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"></path></svg>
      </div>
      
      <h2 class="text-3xl font-extrabold text-slate-900">Segment Created!</h2>
      <p class="text-lg text-slate-600">Your new audience segment is ready.</p>

      <div class="flex flex-col space-y-4 pt-8">
        <button 
          id="btn-view-segments"
          @click="viewSegments"
          class="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none"
        >
          View Lists & Segments
        </button>
        <button 
          id="btn-go-home"
          @click="goHome"
          class="w-full flex justify-center py-3 px-4 border border-slate-300 rounded-md shadow-sm text-sm font-medium text-slate-700 bg-white hover:bg-slate-50 focus:outline-none"
        >
          Return Home
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SEGMENT_CREATED_SUCCESS',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    async function goHome() {
      store.success_message = "Segment created"
      store.setCurrentPageId('HOME')
      await router.push({ name: 'HOME' })
    }

    async function viewSegments() {
      store.setCurrentPageId('LISTS_SEGMENTS')
      await router.push({ name: 'LISTS_SEGMENTS' })
    }

    return {
      goHome,
      viewSegments
    }
  }
}
</script>