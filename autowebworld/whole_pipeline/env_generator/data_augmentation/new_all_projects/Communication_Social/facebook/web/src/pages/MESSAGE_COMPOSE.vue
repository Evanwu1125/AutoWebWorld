<template>
  <div class="min-h-screen bg-gray-100 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg w-full max-w-lg overflow-hidden h-[500px] flex flex-col">
      <!-- Header -->
      <div class="relative px-4 py-3 border-b border-gray-100 flex items-center justify-center bg-white z-10">
        <h2 class="text-lg font-bold text-gray-900">New Message</h2>
        <div 
          id="message-cancel" 
          @click="goBack"
          class="absolute right-4 top-1/2 transform -translate-y-1/2 p-1 text-gray-500 hover:bg-gray-100 rounded-full cursor-pointer transition-colors"
        >
          Cancel
        </div>
      </div>

      <!-- Recipient Selector -->
      <div class="px-4 py-3 border-b border-gray-100 flex items-center gap-2">
        <span class="text-gray-500 font-medium">To:</span>
        <div class="relative flex-1">
          <button 
            id="recipient-dropdown"
            @click="toggleDropdown"
            class="w-full text-left font-medium text-gray-900 focus:outline-none"
          >
            {{ selectedRecipientLabel || 'Type a name or group' }}
          </button>
          
          <div v-if="dropdownOpen" class="absolute top-full left-0 mt-2 w-full bg-white rounded-lg shadow-xl py-1 z-50 ring-1 ring-black ring-opacity-5 max-h-60 overflow-y-auto">
            <div class="px-3 py-2 text-xs font-bold text-gray-500 uppercase">Suggested</div>
            
            <div 
              id="recipient-option-friend-1"
              @click="selectRecipient('friend_1', 'Sarah Williams')"
              class="flex items-center gap-3 px-4 py-2 hover:bg-gray-100 cursor-pointer"
            >
              <img src="/images/Friend.jpg" class="h-8 w-8 rounded-full" alt="" />
              <span class="text-sm font-medium">Sarah Williams</span>
            </div>
            <div 
              id="recipient-option-friend-2"
              @click="selectRecipient('friend_2', 'Mike Chen')"
              class="flex items-center gap-3 px-4 py-2 hover:bg-gray-100 cursor-pointer"
            >
              <img src="/images/photo1765161058.jpg" class="h-8 w-8 rounded-full" alt="" />
              <span class="text-sm font-medium">Mike Chen</span>
            </div>
            <div 
              id="recipient-option-item-any"
              @click="selectRecipient('item_any', 'Random Friend')"
              class="flex items-center gap-3 px-4 py-2 hover:bg-gray-100 cursor-pointer"
            >
              <div class="h-8 w-8 rounded-full bg-gray-200 flex items-center justify-center">?</div>
              <span class="text-sm font-medium">Random Friend</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Message Body -->
      <div class="flex-1 p-4">
        <textarea 
          id="message-textarea"
          v-model="messageText"
          @input="handleInput"
          placeholder="Write a message..."
          class="w-full h-full resize-none border-none focus:ring-0 text-base placeholder-gray-400"
        ></textarea>
      </div>

      <!-- Footer -->
      <div class="p-4 border-t border-gray-100 bg-gray-50">
        <button 
          id="message-next-review"
          @click="goToReview"
          :disabled="!canProceed"
          class="w-full py-2 bg-blue-600 text-white font-semibold rounded-full shadow-sm hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
        >
          Review
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'MESSAGE_COMPOSE',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    const messageText = ref(signatureStore.message_text || '');
    const dropdownOpen = ref(false);
    const selectedRecipientLabel = ref(signatureStore.recipient_selected ? 'Selected User' : '');
    
    const canProceed = computed(() => {
      return messageText.value.length > 0 && signatureStore.recipient_selected === true;
    });

    const toggleDropdown = () => {
      dropdownOpen.value = !dropdownOpen.value;
    };

    const selectRecipient = (value, label) => {
      signatureStore.recipient_selected = true; // FSM Effect
      selectedRecipientLabel.value = label;
      dropdownOpen.value = false;
    };

    const handleInput = () => {
      signatureStore.message_text = messageText.value;
    };

    const goToReview = async () => {
      if (canProceed.value) {
        signatureStore.currentPageId = 'MESSAGE_REVIEW';
        await router.push({ name: 'MESSAGE_REVIEW' });
      }
    };

    const goBack = async () => {
      // Clear
      signatureStore.message_text = null;
      signatureStore.recipient_selected = null;
      signatureStore.currentPageId = 'MESSENGER_INBOX';
      await router.push({ name: 'MESSENGER_INBOX' });
    };

    return {
      messageText,
      dropdownOpen,
      selectedRecipientLabel,
      canProceed,
      toggleDropdown,
      selectRecipient,
      handleInput,
      goToReview,
      goBack
    };
  }
}
</script>