<template>
  <div class="flex flex-col min-h-screen bg-black text-white">
    <!-- Header -->
    <div class="sticky top-0 z-30 bg-black/80 backdrop-blur-md px-4 py-3 flex items-center justify-between border-b border-[#2F3336]">
      <div id="compose-back" @click="handleBack" class="p-2 -ml-2 rounded-full hover:bg-white/10 cursor-pointer transition-colors">
         <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M10.59 12L4.54 5.96l1.42-1.42L12 10.59l6.04-6.05 1.42 1.42L13.41 12l6.05 6.04-1.42 1.42L12 13.41l-6.04 6.05-1.42-1.42L10.59 12z"></path></g></svg>
      </div>
      <div class="flex gap-4 items-center">
         <div class="text-[#1D9BF0] font-bold text-sm cursor-pointer hidden sm:block">Drafts</div>
         <button id="tweet-submit-button" @click="handlePost" :disabled="!isValid" 
                 :class="isValid ? 'bg-[#1D9BF0] hover:bg-[#1A8CD8]' : 'bg-[#0F4E78] cursor-not-allowed opacity-50'"
                 class="text-white font-bold rounded-full px-4 py-1.5 transition-colors">
           Post
         </button>
      </div>
    </div>

    <div class="p-4 flex gap-3">
      <div class="w-10 h-10 rounded-full bg-gray-700 overflow-hidden">
         <img src="/images/photo1766328533.jpg" alt="avatar" class="w-full h-full object-cover">
      </div>
      <div class="flex-1 flex flex-col gap-4">
         <!-- Visibility Dropdown -->
         <div class="relative w-fit">
            <button id="tweet-visibility-dropdown" @click="showVisibility = !showVisibility" class="flex items-center gap-1 text-[#1D9BF0] border border-[#536471] rounded-full px-3 py-0.5 text-sm font-bold hover:bg-[#1D9BF0]/10 transition-colors">
               <span>{{ visibility === 'public' ? 'Everyone' : 'Twitter Circle' }}</span>
               <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4 w-4 fill-current"><g><path d="M3.543 8.96l1.414-1.42L12 14.59l7.043-7.05 1.414 1.42L12 17.41 3.543 8.96z"></path></g></svg>
            </button>
            <div v-if="showVisibility" class="absolute top-full left-0 mt-2 bg-black border border-[#2F3336] rounded-xl shadow-xl z-50 w-60 py-2">
               <div class="px-4 py-2 font-bold text-lg">Choose audience</div>
               <div id="tweet-visibility-public" @click="setVisibility('public')" class="px-4 py-3 hover:bg-white/10 cursor-pointer flex items-center justify-between">
                  <div class="flex items-center gap-3">
                     <div class="bg-[#1D9BF0] p-2 rounded-full text-white"><svg viewBox="0 0 24 24" class="h-5 w-5 fill-current"><g><path d="M12 1.75a8.25 8.25 0 00-8.25 8.25v2.887c0 2.225-1.077 4.192-2.81 5.378-.458.314-.492.969-.074 1.334.814.71 1.868 1.15 3.033 1.15H10.5a3.501 3.501 0 006.999 0h6.6c1.165 0 2.22-.44 3.034-1.15.418-.365.385-1.02-.073-1.334-1.734-1.186-2.811-3.153-2.811-5.378V10A8.25 8.25 0 0012 1.75zM14 22a1.5 1.5 0 11-3 0h3z"></path></g></svg></div>
                     <div class="font-bold">Everyone</div>
                  </div>
                  <svg v-if="visibility === 'public'" viewBox="0 0 24 24" class="h-5 w-5 fill-[#1D9BF0]"><g><path d="M9.64 18.952l-5.55-4.861 1.317-1.504 3.951 3.459 8.459-10.448 1.563 1.264-9.74 12.09z"></path></g></svg>
               </div>
               <div id="tweet-visibility-followers" @click="setVisibility('followers')" class="px-4 py-3 hover:bg-white/10 cursor-pointer flex items-center justify-between">
                  <div class="flex items-center gap-3">
                     <div class="bg-[#00BA7C] p-2 rounded-full text-white"><svg viewBox="0 0 24 24" class="h-5 w-5 fill-current"><g><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm5 11h-4v4h-2v-4H7v-2h4V7h2v4h4v2z"></path></g></svg></div> <!-- Circle icon placeholder -->
                     <div class="font-bold">Twitter Circle</div>
                  </div>
                  <svg v-if="visibility === 'followers'" viewBox="0 0 24 24" class="h-5 w-5 fill-[#1D9BF0]"><g><path d="M9.64 18.952l-5.55-4.861 1.317-1.504 3.951 3.459 8.459-10.448 1.563 1.264-9.74 12.09z"></path></g></svg>
               </div>
            </div>
         </div>

         <!-- Text Area -->
         <textarea 
            id="compose-tweet-textarea"
            v-model="tweetText"
            @input="handleInput"
            placeholder="What is happening?!" 
            class="w-full bg-transparent text-xl placeholder-gray-500 focus:outline-none resize-none min-h-[150px]"
         ></textarea>
         
         <div class="border-b border-[#2F3336] pb-3">
            <button id="tweet-replies-dropdown" @click="showReplies = !showReplies" class="text-[#1D9BF0] text-sm font-bold flex items-center gap-1 hover:bg-[#1D9BF0]/10 px-3 py-1 -ml-3 rounded-full w-fit transition-colors">
               <svg viewBox="0 0 24 24" aria-hidden="true" class="h-4 w-4 fill-current"><g><path d="M12 1.75a8.25 8.25 0 00-8.25 8.25v2.887c0 2.225-1.077 4.192-2.81 5.378-.458.314-.492.969-.074 1.334.814.71 1.868 1.15 3.033 1.15H10.5a3.501 3.501 0 006.999 0h6.6c1.165 0 2.22-.44 3.034-1.15.418-.365.385-1.02-.073-1.334-1.734-1.186-2.811-3.153-2.811-5.378V10A8.25 8.25 0 0012 1.75zM14 22a1.5 1.5 0 11-3 0h3z"></path></g></svg>
               <span>{{ allowReplies === 'everyone' ? 'Everyone can reply' : 'People you follow can reply' }}</span>
            </button>
             <div v-if="showReplies" class="absolute mt-1 bg-black border border-[#2F3336] rounded-xl shadow-xl z-50 w-60 py-2">
               <div class="px-4 py-2 font-bold text-lg">Who can reply?</div>
               <div id="tweet-replies-everyone" @click="setReplies('everyone')" class="px-4 py-3 hover:bg-white/10 cursor-pointer flex items-center justify-between">
                  <div class="flex items-center gap-3">
                     <div class="bg-[#1D9BF0] p-2 rounded-full text-white"><svg viewBox="0 0 24 24" class="h-5 w-5 fill-current"><g><path d="M12 1.75a8.25 8.25 0 00-8.25 8.25v2.887c0 2.225-1.077 4.192-2.81 5.378-.458.314-.492.969-.074 1.334.814.71 1.868 1.15 3.033 1.15H10.5a3.501 3.501 0 006.999 0h6.6c1.165 0 2.22-.44 3.034-1.15.418-.365.385-1.02-.073-1.334-1.734-1.186-2.811-3.153-2.811-5.378V10A8.25 8.25 0 0012 1.75zM14 22a1.5 1.5 0 11-3 0h3z"></path></g></svg></div>
                     <div class="font-bold">Everyone</div>
                  </div>
                  <svg v-if="allowReplies === 'everyone'" viewBox="0 0 24 24" class="h-5 w-5 fill-[#1D9BF0]"><g><path d="M9.64 18.952l-5.55-4.861 1.317-1.504 3.951 3.459 8.459-10.448 1.563 1.264-9.74 12.09z"></path></g></svg>
               </div>
               <div id="tweet-replies-following" @click="setReplies('following')" class="px-4 py-3 hover:bg-white/10 cursor-pointer flex items-center justify-between">
                  <div class="flex items-center gap-3">
                     <div class="bg-[#1D9BF0] p-2 rounded-full text-white"><svg viewBox="0 0 24 24" class="h-5 w-5 fill-current"><g><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm5 11h-4v4h-2v-4H7v-2h4V7h2v4h4v2z"></path></g></svg></div> 
                     <div class="font-bold">People you follow</div>
                  </div>
                  <svg v-if="allowReplies === 'following'" viewBox="0 0 24 24" class="h-5 w-5 fill-[#1D9BF0]"><g><path d="M9.64 18.952l-5.55-4.861 1.317-1.504 3.951 3.459 8.459-10.448 1.563 1.264-9.74 12.09z"></path></g></svg>
               </div>
            </div>
         </div>
         
         <!-- Toolbar -->
         <div class="flex justify-between items-center text-[#1D9BF0]">
            <div class="flex gap-4">
               <div class="p-2 -ml-2 rounded-full hover:bg-[#1D9BF0]/10 cursor-pointer">
                  <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M3 5.5C3 4.119 4.119 3 5.5 3h13C19.881 3 21 4.119 21 5.5v13c0 1.381-1.119 2.5-2.5 2.5h-13C4.119 21 3 19.881 3 18.5v-13zM5.5 5c-.276 0-.5.224-.5.5v9.086l3-3 3 3 5-5 3 3V5.5c0-.276-.224-.5-.5-.5h-13zM19 15.414l-3-3-5 5-3-3-3 3V18.5c0 .276.224.5.5.5h13c.276 0 .5-.224.5-.5v-3.086zM9.75 7C8.784 7 8 7.784 8 8.75s.784 1.75 1.75 1.75 1.75-.784 1.75-1.75S10.716 7 9.75 7z"></path></g></svg>
               </div>
               <div class="p-2 rounded-full hover:bg-[#1D9BF0]/10 cursor-pointer">
                  <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M3 5.5C3 4.119 4.119 3 5.5 3h13C19.881 3 21 4.119 21 5.5v13c0 1.381-1.119 2.5-2.5 2.5h-13C4.119 21 3 19.881 3 18.5v-13zM5.5 5c-.276 0-.5.224-.5.5v13.5c0 .276.224.5.5.5h13c.276 0 .5-.224.5-.5V5.5c0-.276-.224-.5-.5-.5h-13zM15 11.5l-3 3-3-3h6z"></path></g></svg>
               </div>
               <div class="p-2 rounded-full hover:bg-[#1D9BF0]/10 cursor-pointer">
                  <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M8 6h8v2H8V6zm-4 4h16v2H4v-2zm4 4h8v2H8v-2z"></path></g></svg>
               </div>
               <div class="p-2 rounded-full hover:bg-[#1D9BF0]/10 cursor-pointer">
                  <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M12 14c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zM7 14c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2zM17 14c-1.1 0-2 .9-2 2s.9 2 2 2 2-.9 2-2-.9-2-2-2z"></path></g></svg>
               </div>
               <!-- Schedule Button -->
               <div id="schedule-tweet-button" @click="showScheduler = !showScheduler" class="p-2 rounded-full hover:bg-[#1D9BF0]/10 cursor-pointer relative">
                  <svg viewBox="0 0 24 24" aria-hidden="true" class="h-5 w-5 fill-current"><g><path d="M6 3V2h2v1h6V2h2v1h1.5C18.88 3 20 4.12 20 5.5v13c0 1.38-1.12 2.5-2.5 2.5h-13C3.12 21 2 19.88 2 18.5v-13C2 4.12 3.12 3 4.5 3H6zm9.5 8h-2v2h2v-2zm-5 0h-2v2h2v-2zm-5 0h-2v2h2v-2zm10 4h-2v2h2v-2zm-5 0h-2v2h2v-2zm-5 0h-2v2h2v-2z"></path></g></svg>
                  
                  <!-- DatePicker Popup -->
                  <div v-if="showScheduler" class="absolute top-full left-0 mt-2 bg-black border border-[#2F3336] rounded-xl shadow-xl z-50 p-4 w-[320px]">
                      <h3 class="text-white font-bold mb-4">Schedule Post</h3>
                      <DateTimePicker 
                        id="date-picker"
                        @change="handleDateChange" 
                      />
                  </div>
               </div>
            </div>
         </div>
      </div>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import DateTimePicker from '../components/widgets/DateTimePicker.vue';

export default {
  name: 'COMPOSE_TWEET',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    const tweetText = ref('');
    const showVisibility = ref(false);
    const visibility = ref('public');
    const showReplies = ref(false);
    const allowReplies = ref('everyone');
    const showScheduler = ref(false);

    const isValid = computed(() => tweetText.value.length > 0);

    const handleInput = () => {
      signatureStore.draft_tweet_text = tweetText.value;
    };

    const setVisibility = (val) => {
      visibility.value = val;
      signatureStore.draft_tweet_visibility = val;
      showVisibility.value = false;
    };

    const setReplies = (val) => {
      allowReplies.value = val;
      signatureStore.draft_tweet_allow_replies = val;
      showReplies.value = false;
    };

    const handlePost = () => {
      if (!isValid.value) return;
      signatureStore.setCurrentPageId('TWEET_POST_SUCCESS');
      router.push({ name: 'TWEET_POST_SUCCESS' });
    };

    const handleDateChange = (date) => {
      // Logic for ACT_COMPOSE_SCHEDULE_TWEET_PICK_DATE
      // Store doesn't have a specific field for date in signature, 
      // but the action transitions to TWEET_SCHEDULE_SUCCESS.
      // We assume selecting date and pressing enter (handled by DateTimePicker usually or manual trigger) triggers the action
      // FSM says select op on date_picker then key_press Enter. 
      // Here we just simulate completion on valid selection if needed or wait for button.
      // Actually FSM has action ACT_COMPOSE_SCHEDULE_TWEET_PICK_DATE which IS the navigation action.
      // So selecting the date should trigger navigation.
      
      // Since DateTimePicker emits 'change', we can use that to trigger navigation
      // assuming the user is done. Or we can have a "Schedule" button in the popup.
      // FSM gui_procedure ends with "key_press Enter". 
      // Let's assume selecting all parts triggers it or we add a Confirm button?
      // FSM gui_procedure: click year, month, day, hour, minute, then ENTER.
      // So we should listen for ENTER on the component or just navigate after full selection.
      // The DatePicker usually emits change. We'll navigate on change for simplicity or add a listener.
      // Let's navigate on change to 'TWEET_SCHEDULE_SUCCESS'.
      
      signatureStore.setCurrentPageId('TWEET_SCHEDULE_SUCCESS');
      router.push({ name: 'TWEET_SCHEDULE_SUCCESS' });
    };

    const handleBack = () => {
      signatureStore.setCurrentPageId('HOME_TIMELINE');
      router.push({ name: 'HOME_TIMELINE' });
    };

    return {
      tweetText,
      isValid,
      handleInput,
      showVisibility,
      visibility,
      setVisibility,
      showReplies,
      allowReplies,
      setReplies,
      showScheduler,
      handlePost,
      handleDateChange,
      handleBack
    };
  }
}
</script>