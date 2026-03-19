<template>
  <div class="min-h-screen bg-slate-900 flex flex-col items-center justify-center p-4 relative overflow-hidden">
    <!-- Dynamic Background Elements -->
    <div class="absolute top-0 left-0 w-full h-full overflow-hidden z-0">
      <div class="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/20 rounded-full blur-[100px]"></div>
      <div class="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/20 rounded-full blur-[100px]"></div>
    </div>

    <!-- Main Card -->
    <div class="relative z-10 w-full max-w-md">
      <!-- Logo -->
      <div class="text-center mb-8">
        <h1 class="text-5xl font-bold text-white tracking-tighter">tumblr</h1>
      </div>

      <div class="bg-white/5 backdrop-blur-lg border border-white/10 p-8 rounded-lg shadow-2xl">
        <h2 class="text-xl text-white font-semibold mb-6 text-center">Sign up for your own slice of the internet.</h2>

        <div class="space-y-4">
          <!-- Email Input -->
          <div class="space-y-1">
            <input 
              id="signup-email"
              type="email"
              placeholder="Email"
              class="w-full px-4 py-3 bg-white/10 border border-transparent focus:border-white/30 rounded-md text-white placeholder-slate-400 outline-none transition-all"
              :value="store.signup_email"
              @input="handleEmailInput"
            />
          </div>

          <!-- Password Input -->
          <div class="space-y-1">
            <input 
              id="signup-password"
              type="password"
              placeholder="Password"
              class="w-full px-4 py-3 bg-white/10 border border-transparent focus:border-white/30 rounded-md text-white placeholder-slate-400 outline-none transition-all"
              :value="store.signup_password"
              @input="handlePasswordInput"
            />
          </div>

          <!-- Blogname Input -->
          <div class="space-y-1">
            <input 
              id="signup-blogname"
              type="text"
              placeholder="Blog name"
              class="w-full px-4 py-3 bg-white/10 border border-transparent focus:border-white/30 rounded-md text-white placeholder-slate-400 outline-none transition-all"
              :value="store.signup_blogname"
              @input="handleBlognameInput"
            />
          </div>

          <!-- Submit Button -->
          <button 
            id="signup-submit"
            @click="handleSubmit"
            :disabled="!isValid"
            :class="[
              'w-full py-3 px-4 rounded-md font-bold text-white transition-all transform',
              isValid ? 'bg-blue-500 hover:bg-blue-600 hover:scale-[1.02] shadow-lg shadow-blue-500/30' : 'bg-slate-700 cursor-not-allowed opacity-50'
            ]"
          >
            Sign up
          </button>
        </div>

        <div class="mt-6 text-center">
          <button 
            id="signup-back-home"
            @click="goBackHome"
            class="text-slate-400 hover:text-white text-sm transition-colors"
          >
            Nevermind, take me back home
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { useSignatureStore } from '../stores/signature'

export default {
  name: 'SIGNUP',
  setup() {
    const router = useRouter()
    const store = useSignatureStore()

    const handleEmailInput = (e) => {
      store.signup_email = e.target.value
    }

    const handlePasswordInput = (e) => {
      store.signup_password = e.target.value
    }

    const handleBlognameInput = (e) => {
      store.signup_blogname = e.target.value
    }

    // Precondition Logic: All fields length > 0
    const isValid = computed(() => {
      return (store.signup_email?.length > 0) && 
             (store.signup_password?.length > 0) && 
             (store.signup_blogname?.length > 0)
    })

    const handleSubmit = async () => {
      if (!isValid.value) return
      store.currentPageId = 'DASHBOARD_FEED'
      await router.push({ name: 'DASHBOARD_FEED' })
    }

    const goBackHome = async () => {
      store.currentPageId = 'HOME'
      await router.push({ name: 'HOME' })
    }

    return {
      store,
      handleEmailInput,
      handlePasswordInput,
      handleBlognameInput,
      handleSubmit,
      goBackHome,
      isValid
    }
  }
}
</script>