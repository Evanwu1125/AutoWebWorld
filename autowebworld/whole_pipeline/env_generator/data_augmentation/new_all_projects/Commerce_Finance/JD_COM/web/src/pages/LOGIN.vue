<template>
  <div class="min-h-screen bg-white flex flex-col">
    <!-- Simple Header -->
    <header class="container mx-auto px-4 py-6 flex justify-between items-center">
      <div class="text-[#E1251B] text-3xl font-black tracking-tighter">JD</div>
      <button id="back-home" @click="goHome" class="text-gray-500 hover:text-[#E1251B] text-sm">
        Back to Home
      </button>
    </header>

    <main class="flex-1 flex items-center justify-center bg-[#E93854] px-4 py-12">
      <div class="bg-white rounded-xl shadow-2xl w-full max-w-sm p-8 relative overflow-hidden">
        <div class="absolute top-0 left-0 w-full h-2 bg-[#E1251B]"></div>
        
        <h2 class="text-2xl font-bold text-gray-900 mb-6 text-center">Sign In</h2>

        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Username</label>
            <div class="relative">
              <span class="absolute left-3 top-3 text-gray-400">👤</span>
              <input 
                id="login-username"
                type="text"
                v-model="username"
                @input="handleUser"
                class="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-200 focus:border-[#E1251B] outline-none"
                placeholder="Username/Email"
              />
            </div>
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Password</label>
            <div class="relative">
              <span class="absolute left-3 top-3 text-gray-400">🔒</span>
              <input 
                id="login-password"
                type="password"
                v-model="password"
                @input="handlePass"
                class="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-200 focus:border-[#E1251B] outline-none"
                placeholder="Password"
              />
            </div>
          </div>

          <button 
            id="btn-login"
            @click="login"
            class="w-full bg-[#E1251B] text-white font-bold py-3 rounded-lg shadow-lg hover:bg-[#c91f16] transition-colors disabled:opacity-50 disabled:cursor-not-allowed mt-2"
            :disabled="!canLogin"
          >
            Log In
          </button>
        </div>

        <div class="mt-6 text-center text-sm text-gray-600">
          Don't have an account? 
          <button id="link-register" @click="goToRegister" class="text-[#E1251B] font-bold hover:underline">
            Register Now
          </button>
        </div>
      </div>
    </main>

    <footer class="py-6 text-center text-gray-500 text-xs">
      © 2004-2025 JD.com. All Rights Reserved.
    </footer>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'LOGIN',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const username = ref('');
    const password = ref('');

    const userEntered = computed(() => signatureStore.login_username_entered);
    const passEntered = computed(() => signatureStore.login_password_entered);
    const canLogin = computed(() => userEntered.value && passEntered.value);

    const handleUser = () => {
      if (username.value.length > 0) signatureStore.login_username_entered = true;
    };

    const handlePass = () => {
      if (password.value.length > 0) signatureStore.login_password_entered = true;
    };

    const login = async () => {
      // Set user session
      signatureStore.current_user_id = 'u1'; // Mock login
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    const goToRegister = async () => {
      signatureStore.currentPageId = 'REGISTER';
      await router.push({ name: 'REGISTER' });
    };

    const goHome = async () => {
      signatureStore.currentPageId = 'HOME';
      await router.push({ name: 'HOME' });
    };

    return {
      username,
      password,
      canLogin,
      handleUser,
      handlePass,
      login,
      goToRegister,
      goHome
    };
  }
}
</script>