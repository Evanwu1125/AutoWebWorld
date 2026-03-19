<template>
  <div class="flex flex-col items-center justify-center min-h-screen bg-black text-white relative">
    <!-- Hero Background (optional, but good for landing) -->
    <div class="absolute inset-0 z-0 opacity-20">
       <!-- Use ImageGetter path for background -->
       <img src="/images/SocialNetwork.jpg" alt="background" class="w-full h-full object-cover" />
    </div>

    <div class="z-10 flex flex-col items-center p-8 max-w-md w-full text-center">
      <svg viewBox="0 0 24 24" aria-hidden="true" class="h-16 w-16 fill-white mb-8"><g><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"></path></g></svg>
      
      <h1 class="text-5xl font-bold mb-12">Happening now</h1>
      <h2 class="text-3xl font-bold mb-8">Join today.</h2>

      <!-- Navigation Actions based on FSM -->
      <div id="primary-home-link" @click="handleGoHomeTimeline" class="w-full bg-white text-black font-bold rounded-full py-3 mb-4 cursor-pointer hover:bg-gray-200 transition-colors">
        Create account (Demo)
      </div>

      <div class="text-sm text-gray-500 mb-12">
        By signing up, you agree to the Terms of Service and Privacy Policy, including Cookie Use.
      </div>

      <div class="font-bold mb-4">Already have an account?</div>

      <div @click="handleGoHomeTimeline" class="w-full border border-gray-600 text-[#1D9BF0] font-bold rounded-full py-3 cursor-pointer hover:bg-white/10 transition-colors">
        Sign in
      </div>
      
      <!-- FSM Actions Hidden/Structural if needed, but primary nav is above -->
      
      <!-- Cookie Consent Trigger (Modal handled in App.vue, but action button ID needed for FSM mapping if modal is inside page. 
           However, FSM says ACT_HOME_ACCEPT_COOKIES gui_procedure selector is #cookie-accept. 
           Since modal is global, we need to ensure the ID exists in the DOM when modal is shown.) -->
      
      <!-- Hover Menu Trigger (Simulated for FSM compliance if needed on landing, though typically in nav) -->
      <!-- The Left Sidebar in App.vue handles the navigation actions like #nav-user-menu -->

    </div>
  </div>
</template>

<script>
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'HOME',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const handleGoHomeTimeline = async () => {
      // ACT_HOME_GO_HOME_TIMELINE_DIRECT
      // Check precondition: cookie_consent_given eq true
      if (signatureStore.cookie_consent_given !== true) {
        alert("Please accept cookies first.");
        return;
      }
      signatureStore.setCurrentPageId('HOME_TIMELINE');
      await router.push({ name: 'HOME_TIMELINE' });
    };

    return {
      handleGoHomeTimeline
    };
  }
}
</script>