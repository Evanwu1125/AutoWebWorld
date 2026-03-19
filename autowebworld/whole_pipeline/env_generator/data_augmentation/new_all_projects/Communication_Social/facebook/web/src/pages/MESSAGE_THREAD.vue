<template>
  <div class="min-h-screen bg-gray-100 flex flex-col">
    <!-- Header -->
    <header class="bg-white shadow-sm sticky top-0 z-20 h-16 flex items-center px-4 justify-between">
      <div class="flex items-center gap-3">
        <button 
          id="back-to-inbox"
          @click="goBack"
          class="p-2 hover:bg-gray-100 rounded-full transition-colors"
        >
          <svg class="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        
        <div class="flex items-center gap-3 cursor-pointer">
          <div class="relative">
             <img :src="thread?.avatar || '/images/photo1765161050.jpg'" class="h-10 w-10 rounded-full object-cover" alt="User" />
             <span v-if="thread?.unread" class="absolute bottom-0 right-0 h-2.5 w-2.5 rounded-full ring-2 ring-white bg-green-500"></span>
          </div>
          <div>
            <h1 class="text-base font-bold text-gray-900 leading-tight">{{ thread?.name || 'User' }}</h1>
            <p class="text-xs text-gray-500">Active now</p>
          </div>
        </div>
      </div>
      
      <div class="flex items-center gap-2 text-blue-600">
         <button class="p-2 hover:bg-gray-100 rounded-full"><svg class="h-6 w-6" fill="currentColor" viewBox="0 0 20 20"><path d="M2 3a1 1 0 011-1h2.153a1 1 0 01.986.836l.74 4.435a1 1 0 01-.54 1.06l-1.548.773a11.037 11.037 0 006.105 6.105l.774-1.548a1 1 0 011.059-.54l4.435.74a1 1 0 01.836.986V17a1 1 0 01-1 1h-2C7.82 18 2 12.18 2 5V3z"/></svg></button>
         <button class="p-2 hover:bg-gray-100 rounded-full"><svg class="h-6 w-6" fill="currentColor" viewBox="0 0 20 20"><path d="M2 6a2 2 0 012-2h6a2 2 0 012 2v8a2 2 0 01-2 2H4a2 2 0 01-2-2V6zM14.553 7.106A1 1 0 0014 8v4a1 1 0 00.553.894l2 1A1 1 0 0018 13V7a1 1 0 00-1.447-.894l-2 1z"/></svg></button>
         <button class="p-2 hover:bg-gray-100 rounded-full"><svg class="h-6 w-6" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/></svg></button>
      </div>
    </header>

    <!-- Chat Area -->
    <div class="flex-1 overflow-y-auto p-4 space-y-4">
      <div class="flex justify-center my-4">
         <span class="text-xs text-gray-500 bg-gray-200 px-2 py-1 rounded-full">{{ thread?.time }} ago</span>
      </div>
      
      <!-- Received Message -->
      <div class="flex items-end gap-2">
         <img :src="thread?.avatar || '/images/photo1765161050.jpg'" class="h-8 w-8 rounded-full mb-1" alt="User" />
         <div class="bg-white border border-gray-200 rounded-2xl rounded-bl-none px-4 py-2 max-w-[70%] shadow-sm">
           <p class="text-gray-900 text-sm">{{ thread?.last_message }}</p>
         </div>
      </div>

      <!-- Sent Message (Mock) -->
      <div class="flex items-end gap-2 justify-end">
         <div class="bg-blue-600 rounded-2xl rounded-br-none px-4 py-2 max-w-[70%] shadow-sm">
           <p class="text-white text-sm">Sounds good!</p>
         </div>
      </div>
    </div>

    <!-- Input Area -->
    <div class="bg-white p-3 border-t border-gray-200 flex items-end gap-2">
      <button class="p-2 text-blue-600 hover:bg-gray-100 rounded-full"><svg class="h-6 w-6" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zm-.464 5.535a1 1 0 10-1.415-1.414 3 3 0 01-4.242 0 1 1 0 00-1.415 1.414 5 5 0 007.072 0z" clip-rule="evenodd"/></svg></button>
      <div class="flex-1 bg-gray-100 rounded-full flex items-center px-4 py-2">
         <input type="text" placeholder="Aa" class="bg-transparent border-none focus:ring-0 w-full text-sm" />
      </div>
      <button class="p-2 text-blue-600 hover:bg-blue-50 rounded-full">
        <svg class="h-6 w-6 transform rotate-90" fill="currentColor" viewBox="0 0 20 20"><path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"/></svg>
      </button>
    </div>
  </div>
</template>

<script>
import { computed, onMounted } from 'vue';
import { useRouter, useRoute } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import { useDataStore } from '../stores/data';

export default {
  name: 'MESSAGE_THREAD',
  setup() {
    const router = useRouter();
    const route = useRoute();
    const signatureStore = useSignatureStore();
    const dataStore = useDataStore();

    const thread = computed(() => {
      const id = route.params.id || signatureStore.selected_thread_id;
      return dataStore.threads.find(t => t.id === id);
    });
    
    onMounted(() => {
        if (!thread.value && route.params.id) {
            signatureStore.selected_thread_id = route.params.id
        }
    })

    const goBack = async () => {
      signatureStore.currentPageId = 'MESSENGER_INBOX';
      await router.push({ name: 'MESSENGER_INBOX' });
    };

    return {
      thread,
      goBack
    };
  }
}
</script>