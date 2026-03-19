<template>
  <div class="h-screen flex flex-col bg-gray-100">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md shrink-0 justify-between">
        <div class="flex items-center gap-4">
             <button id="settings-back-home" class="hover:bg-[#005A9E] p-1 rounded" @click="goHome">
                 <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
             </button>
             <span class="font-semibold">Settings</span>
        </div>
    </header>

    <div class="flex flex-1 overflow-hidden max-w-5xl mx-auto w-full mt-8 gap-8">
        <!-- Settings Nav -->
        <div class="w-64 bg-white rounded-lg shadow-sm h-fit overflow-hidden">
            <div class="p-4 font-bold border-b border-gray-100">General</div>
            <div class="p-3 bg-blue-50 text-[#0078D4] border-l-4 border-[#0078D4] cursor-pointer">Appearance</div>
            <div class="p-3 hover:bg-gray-50 cursor-pointer text-gray-700">Notifications</div>
            <div class="p-3 hover:bg-gray-50 cursor-pointer text-gray-700">Categories</div>
            <div class="p-3 hover:bg-gray-50 cursor-pointer text-gray-700">Accessibility</div>
            <div class="p-3 hover:bg-gray-50 cursor-pointer text-gray-700 border-t border-gray-100" id="settings-go-inbox" @click="goInbox">
                Return to Inbox
            </div>
        </div>
        
        <!-- Settings Content -->
        <div class="flex-1 bg-white rounded-lg shadow-sm p-8">
            <h2 class="text-2xl font-light mb-6 text-gray-800">Appearance</h2>
            
            <!-- Theme Selection -->
            <div class="mb-8">
                <label class="block text-sm font-semibold mb-2 text-gray-700">Theme</label>
                <div class="relative w-64">
                    <div id="theme-dropdown" class="border border-gray-300 rounded px-3 py-2 flex justify-between items-center cursor-pointer hover:border-[#0078D4]" @click="toggleThemeMenu">
                         <span>{{ currentTheme }}</span>
                         <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg>
                    </div>
                    <div v-if="showThemeMenu" class="absolute left-0 top-12 w-full bg-white shadow-lg border border-gray-200 rounded z-10">
                        <div id="theme-light" class="px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="selectTheme('Light')">Light</div>
                        <div id="theme-dark" class="px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="selectTheme('Dark')">Dark</div>
                        <div id="theme-contrast" class="px-4 py-2 hover:bg-gray-100 cursor-pointer" @click="selectTheme('HighContrast')">High Contrast</div>
                    </div>
                </div>
                <p class="text-xs text-gray-500 mt-2">Choose how Outlook looks to you.</p>
            </div>
            
            <!-- Signature -->
            <div class="mb-8">
                <label class="block text-sm font-semibold mb-2 text-gray-700">Email Signature</label>
                <div class="border border-gray-300 rounded overflow-hidden focus-within:ring-2 focus-within:ring-[#0078D4] focus-within:border-transparent">
                    <textarea id="settings-signature-input" v-model="signature" @input="handleSignatureInput" class="w-full h-32 p-3 outline-none resize-none" placeholder="Create a signature that will be automatically added to your email messages."></textarea>
                </div>
            </div>
        </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';

export default {
  name: 'MAIL_SETTINGS_GENERAL',
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    const showThemeMenu = ref(false);
    const currentTheme = ref('Light');
    const signature = ref('');

    const toggleThemeMenu = () => {
        showThemeMenu.value = !showThemeMenu.value;
    };

    const selectTheme = (theme) => {
        currentTheme.value = theme;
        showThemeMenu.value = false;
        signatureStore.handleAction('ACT_SETTINGS_SELECT_THEME', { widget: 'dropdown', theme });
    };

    const handleSignatureInput = () => {
        signatureStore.handleAction('ACT_SETTINGS_TYPE_SIGNATURE', { input_text: signature.value, field: 'signature' });
    };

    const goHome = async () => {
        await signatureStore.handleAction('ACT_SETTINGS_BACK_HOME');
        router.push({ name: 'HOME' });
    };

    const goInbox = async () => {
        await signatureStore.handleAction('ACT_SETTINGS_GO_INBOX');
        router.push({ name: 'MAIL_INBOX' });
    };

    return {
        showThemeMenu,
        currentTheme,
        signature,
        toggleThemeMenu,
        selectTheme,
        handleSignatureInput,
        goHome,
        goInbox
    };
  }
}
</script>