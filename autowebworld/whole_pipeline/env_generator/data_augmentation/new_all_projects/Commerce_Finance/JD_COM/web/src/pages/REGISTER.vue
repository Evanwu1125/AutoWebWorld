<template>
  <div class="min-h-screen bg-white flex flex-col">
    <header class="container mx-auto px-4 py-6 flex justify-between items-center border-b shadow-sm">
      <div class="flex items-center gap-4">
        <div class="text-[#E1251B] text-3xl font-black tracking-tighter">JD</div>
        <h1 class="text-xl font-normal text-gray-700">Welcome Registration</h1>
      </div>
      <button id="back-login" @click="goBack" class="text-gray-600 hover:text-[#E1251B]">
        Have an account? <span class="text-[#E1251B] font-bold">Log in</span>
      </button>
    </header>

    <main class="flex-1 container mx-auto px-4 py-12 flex justify-center">
      <div class="w-full max-w-md space-y-6">
        <div class="space-y-4">
          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Username</label>
            <input 
              id="register-username"
              type="text"
              v-model="username"
              @input="handleUser"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-200 focus:border-[#E1251B] outline-none"
              placeholder="Your username"
            />
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Set Password</label>
            <input 
              id="register-password"
              type="password"
              v-model="password"
              @input="handlePass"
              class="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-red-200 focus:border-[#E1251B] outline-none"
              placeholder="Suggest using complex password"
            />
          </div>

          <div>
            <label class="block text-sm font-medium text-gray-700 mb-1">Phone Number</label>
            <div class="flex">
              <span class="inline-flex items-center px-3 rounded-l-lg border border-r-0 border-gray-300 bg-gray-50 text-gray-500 text-sm">
                +86
              </span>
              <input 
                id="register-phone"
                type="text"
                v-model="phone"
                @input="handlePhone"
                class="flex-1 px-4 py-3 border border-gray-300 rounded-r-lg focus:ring-2 focus:ring-red-200 focus:border-[#E1251B] outline-none"
                placeholder="Mobile number"
              />
            </div>
          </div>
        </div>

        <button 
          id="btn-register"
          @click="register"
          class="w-full bg-[#E1251B] text-white font-bold py-4 rounded-lg shadow-lg hover:bg-[#c91f16] transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          :disabled="!canRegister"
        >
          Complete Registration
        </button>

        <div class="text-xs text-gray-400 text-center mt-4">
          By registering, you agree to our <a href="#" class="text-blue-500">Terms of Service</a> and <a href="#" class="text-blue-500">Privacy Policy</a>.
        </div>
      </div>
    </main>
  </div>
</template>

<script>
import { ref, computed } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'REGISTER',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();

    const username = ref('');
    const password = ref('');
    const phone = ref('');

    const userEntered = computed(() => signatureStore.register_username_entered);
    const passEntered = computed(() => signatureStore.register_password_entered);
    const phoneEntered = computed(() => signatureStore.register_phone_entered);
    
    const canRegister = computed(() => userEntered.value && passEntered.value && phoneEntered.value);

    const handleUser = () => {
      if (username.value.length > 0) signatureStore.register_username_entered = true;
    };

    const handlePass = () => {
      if (password.value.length > 0) signatureStore.register_password_entered = true;
    };

    const handlePhone = () => {
      if (phone.value.length > 0) signatureStore.register_phone_entered = true;
    };

    const register = async () => {
      signatureStore.currentPageId = 'ACCOUNT_CREATED_SUCCESS';
      await router.push({ name: 'ACCOUNT_CREATED_SUCCESS' });
    };

    const goBack = async () => {
      signatureStore.currentPageId = 'LOGIN';
      await router.push({ name: 'LOGIN' });
    };

    return {
      username,
      password,
      phone,
      canRegister,
      handleUser,
      handlePass,
      handlePhone,
      register,
      goBack
    };
  }
}
</script>