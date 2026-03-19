<template>
  <div class="min-h-screen bg-black text-white font-sans flex flex-col items-center justify-start overflow-y-auto p-4 pt-4">
    <!-- Header Logo -->
    <div class="mb-4 text-center">
      <div class="inline-flex items-center space-x-2">
         <svg viewBox="0 0 167.5 167.5" class="w-10 h-10 fill-current text-white"><path d="M83.7 0C37.5 0 0 37.5 0 83.7c0 46.3 37.5 83.7 83.7 83.7 46.3 0 83.7-37.5 83.7-83.7S130 0 83.7 0zM122 120.8c-1.4 2.5-4.6 3.2-7.1 1.7-19.8-12.1-44.8-14.9-74.2-8.1-2.8.6-5.6-1.1-6.2-3.9-.6-2.8 1.1-5.6 3.9-6.2 32-7.3 59.6-4.2 81.9 9.3 2.5 1.5 3.4 4.7 1.7 7.2zm10.1-22.5c-1.8 3-5.6 3.9-8.5 2.1-22.8-14-57.6-18.1-84.5-9.9-3.3 1-6.9-1-7.9-4.3-1-3.3 1-6.9 4.3-7.9 30.3-9.2 69.2-4.6 94.6 11 3 1.8 3.9 5.6 2 8.5zm.4-23c-27.3-16.2-72.3-17.7-98.4-9.7-4.2 1.3-8.6-1-9.9-5.2-1.3-4.2 1-8.6 5.2-9.9 30.3-9.2 79.7-7.4 111 11.2 3.8 2.2 5 7.1 2.8 10.9-2.2 3.9-7.2 5.1-10.7 2.7z"/></svg>
         <span class="text-2xl font-bold tracking-tighter">Spotify</span>
      </div>
      <h1 class="text-2xl font-bold mt-3">Sign up for free to start listening.</h1>
    </div>

    <!-- Form Container -->
    <div class="w-full max-w-md bg-black md:bg-[#121212] p-0 md:p-4 rounded-lg md:border border-[#282828] space-y-3">
      <div id="back-home" @click="handleBackHome" class="flex items-center space-x-2 text-[#B3B3B3] hover:text-white cursor-pointer font-bold text-sm mb-4">
         <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 19l-7-7m0 0l7-7m-7 7h18"/></svg>
         <span>Back</span>
      </div>

      <!-- Email -->
      <div>
         <label class="block text-sm font-bold mb-2">What's your email?</label>
         <input 
           id="signup-email-input"
           v-model="form.email"
           @input="handleInputEmail"
           type="email" 
           placeholder="Enter your email." 
           class="w-full bg-[#121212] md:bg-[#282828] border border-[#727272] focus:border-white rounded-md px-4 py-3 text-white placeholder-[#B3B3B3] outline-none transition-colors"
         />
      </div>

      <!-- Password -->
      <div>
         <label class="block text-sm font-bold mb-2">Create a password</label>
         <input 
           id="signup-password-input"
           v-model="form.password"
           @input="handleInputPassword"
           type="password" 
           placeholder="Create a password." 
           class="w-full bg-[#121212] md:bg-[#282828] border border-[#727272] focus:border-white rounded-md px-4 py-3 text-white placeholder-[#B3B3B3] outline-none transition-colors"
         />
      </div>

      <!-- Username -->
      <div>
         <label class="block text-sm font-bold mb-2">What should we call you?</label>
         <input 
           id="signup-username-input"
           v-model="form.username"
           @input="handleInputUsername"
           type="text" 
           placeholder="Enter a profile name." 
           class="w-full bg-[#121212] md:bg-[#282828] border border-[#727272] focus:border-white rounded-md px-4 py-3 text-white placeholder-[#B3B3B3] outline-none transition-colors"
         />
         <p class="text-xs text-[#B3B3B3] mt-1">This appears on your profile.</p>
      </div>

      <!-- Birthdate Picker -->
      <div>
         <label class="block text-sm font-bold mb-2">What's your date of birth?</label>
         <DateTimePicker 
            id="date-picker1" 
            @change="handleDateChange"
            class="w-full"
         />
      </div>

      <!-- Plan Dropdown -->
      <div>
         <label class="block text-sm font-bold mb-2">Choose a plan</label>
         <div id="signup-plan-dropdown" class="relative group">
            <div class="w-full bg-[#121212] md:bg-[#282828] border border-[#727272] rounded-md px-4 py-3 text-white cursor-pointer flex justify-between items-center">
               <span>{{ selectedPlanLabel || 'Select a plan' }}</span>
               <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
            </div>
            <div class="hidden group-hover:block absolute w-full left-0 bottom-full mb-1 bg-[#282828] border border-[#3E3E3E] rounded-md shadow-xl z-50">
               <div id="signup-plan-free" class="px-4 py-3 hover:bg-[#3E3E3E] cursor-pointer" @click="handleSelectPlan('free')">Spotify Free</div>
               <div id="signup-plan-premium" class="px-4 py-3 hover:bg-[#3E3E3E] cursor-pointer" @click="handleSelectPlan('premium')">Premium Individual</div>
               <div id="signup-plan-family" class="px-4 py-3 hover:bg-[#3E3E3E] cursor-pointer" @click="handleSelectPlan('family')">Premium Family</div>
            </div>
         </div>
      </div>

      <!-- Submit -->
      <button
         id="signup-submit-button"
         @click="handleSubmit"
         class="w-full bg-[#1DB954] hover:bg-[#1ed760] text-black font-bold py-3 rounded-full uppercase tracking-widest hover:scale-105 transition-transform"
      >
         Sign Up
      </button>
      
      <p class="text-xs text-center text-[#B3B3B3]">
         By clicking on sign-up, you agree to Spotify's <a href="#" class="text-[#1DB954] underline">Terms and Conditions of Use</a>.
      </p>
    </div>
  </div>
</template>

<script>
import { ref, computed } from 'vue'
import { useSignatureStore } from '../stores/signature'
import { useRouter } from 'vue-router'
import DateTimePicker from '../components/widgets/DateTimePicker.vue'

export default {
  name: 'SIGNUP',
  components: {
    DateTimePicker
  },
  setup() {
    const store = useSignatureStore()
    const router = useRouter()

    const form = ref({
      email: '',
      password: '',
      username: ''
    })

    const selectedPlanLabel = computed(() => {
       const map = {
          'free': 'Spotify Free',
          'premium': 'Premium Individual',
          'family': 'Premium Family'
       }
       return map[store.signup_plan_selected] || ''
    })

    const handleBackHome = async () => {
       store.setCurrentPageId('HOME')
       await router.push({ name: 'HOME' })
    }

    const handleInputEmail = () => {
       store.signup_email = form.value.email
    }

    const handleInputPassword = () => {
       store.signup_password = form.value.password
    }

    const handleInputUsername = () => {
       store.signup_username = form.value.username
    }

    const handleDateChange = (val) => {
       // val is expected to be object { date, time? } or similar from DateTimePicker
       store.signup_birthdate = val
    }

    const handleSelectPlan = (plan) => {
       store.signup_plan_selected = plan
    }

    const handleSubmit = async () => {
       // Precondition check: all fields filled
       if (store.signup_email && store.signup_password && store.signup_username && store.signup_plan_selected) {
          store.setCurrentPageId('SIGNUP_SUCCESS')
          await router.push({ name: 'SIGNUP_SUCCESS' })
       }
    }

    return {
       form,
       selectedPlanLabel,
       handleBackHome,
       handleInputEmail,
       handleInputPassword,
       handleInputUsername,
       handleDateChange,
       handleSelectPlan,
       handleSubmit
    }
  }
}
</script>