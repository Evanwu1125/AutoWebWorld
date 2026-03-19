<template>
  <div class="min-h-screen bg-gray-100 pb-10">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20 h-16 flex items-center px-4">
      <button 
        id="back-to-events-list"
        @click="goBack"
        class="flex items-center gap-2 text-gray-600 hover:bg-gray-100 px-3 py-2 rounded-lg transition-colors"
      >
        <svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
        </svg>
        <span class="font-bold text-lg">Back to Events</span>
      </button>
    </header>

    <div class="max-w-5xl mx-auto mt-6 bg-white rounded-xl shadow-sm overflow-hidden border border-gray-200">
      <!-- Cover Image -->
      <div class="h-64 md:h-96 w-full relative">
         <img :src="event?.image" class="w-full h-full object-cover" :alt="event?.name" />
         <!-- Date Badge -->
         <div class="absolute bottom-0 left-4 md:left-8 transform translate-y-1/2 bg-white rounded-xl shadow-md px-4 py-2 text-center border border-gray-100 hidden md:block">
            <span class="block text-sm font-bold text-red-500 uppercase">{{ event ? new Date(event.date).toLocaleString('default', { month: 'short' }) : '' }}</span>
            <span class="block text-2xl font-bold text-gray-900 leading-none">{{ event ? new Date(event.date).getDate() : '' }}</span>
         </div>
      </div>
      
      <div class="px-4 md:px-8 pt-8 md:pt-12 pb-8">
         <div class="flex flex-col md:flex-row justify-between items-start gap-4">
            <div>
               <div class="md:hidden text-red-500 font-bold uppercase text-sm mb-1">{{ event ? new Date(event.date).toDateString() : '' }}</div>
               <h1 class="text-3xl font-bold text-gray-900 mb-2">{{ event?.name }}</h1>
               <div class="text-gray-500 font-medium mb-4">{{ event?.location }}</div>
            </div>
            
            <div class="flex gap-3 w-full md:w-auto">
               <button class="flex-1 md:flex-none px-6 py-2 border border-gray-300 rounded-lg font-semibold text-gray-700 hover:bg-gray-50 transition-colors">Interested</button>
               <button class="flex-1 md:flex-none px-6 py-2 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors">Going</button>
            </div>
         </div>
         
         <hr class="my-6 border-gray-100" />
         
         <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div class="md:col-span-2 space-y-6">
               <h2 class="text-xl font-bold text-gray-900">Details</h2>
               <div class="space-y-4">
                  <div class="flex items-start gap-3">
                     <div class="mt-1 text-gray-400"><svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg></div>
                     <div>
                        <p class="font-medium text-gray-900">{{ event?.time || '10:00 AM' }}</p>
                        <p class="text-sm text-gray-500">{{ event ? new Date(event.date).toDateString() : '' }}</p>
                     </div>
                  </div>
                  <div class="flex items-start gap-3">
                     <div class="mt-1 text-gray-400"><svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg></div>
                     <div>
                        <p class="font-medium text-gray-900">{{ event?.location }}</p>
                     </div>
                  </div>
                  <div class="flex items-start gap-3">
                     <div class="mt-1 text-gray-400"><svg class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" /></svg></div>
                     <div>
                        <p class="font-medium text-gray-900">Event by <span class="font-bold">Alex Johnson</span></p>
                     </div>
                  </div>
               </div>
               
               <div class="prose max-w-none text-gray-600">
                  <p>Join us for an amazing event! There will be great people, good vibes, and memorable experiences. Don't miss out on this opportunity to connect and have fun.</p>
               </div>
            </div>
            
            <div class="bg-gray-50 rounded-xl p-4 border border-gray-100 h-fit">
               <div class="flex items-center justify-between mb-4">
                  <h3 class="font-bold text-gray-900">Guests</h3>
                  <a href="#" class="text-blue-600 text-sm font-medium hover:underline">See All</a>
               </div>
               <div class="flex items-center gap-4 mb-2">
                  <div class="flex -space-x-2 overflow-hidden">
                     <img class="inline-block h-8 w-8 rounded-full ring-2 ring-white" src="/images/photo1765161195.jpg" alt=""/>
                     <img class="inline-block h-8 w-8 rounded-full ring-2 ring-white" src="/images/photo1765161195.jpg" alt=""/>
                     <img class="inline-block h-8 w-8 rounded-full ring-2 ring-white" src="/images/EventAttendees.jpg" alt=""/>
                  </div>
                  <div class="text-sm text-gray-500">{{ event?.attending }} Going</div>
               </div>
            </div>
         </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'EVENT_DETAIL',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const event = computed(() => {
      const id = route.params.id || signatureStore.selected_event_id;
      return dataStore.events.find(e => e.id === id);
    });
    
    onMounted(() => {
        if (!event.value && route.params.id) {
            signatureStore.selected_event_id = route.params.id
        }
    })

    const goBack = async () => {
      signatureStore.currentPageId = 'EVENTS_LIST';
      await router.push({ name: 'EVENTS_LIST' });
    };

    return {
      event,
      goBack
    };
  }
}
</script>