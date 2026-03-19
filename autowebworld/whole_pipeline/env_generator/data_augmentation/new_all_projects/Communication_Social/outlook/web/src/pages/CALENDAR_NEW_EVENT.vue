<template>
  <div class="h-screen flex flex-col bg-white">
    <!-- Header -->
    <header class="bg-[#0078D4] text-white flex items-center h-12 px-4 shadow-md shrink-0 justify-between">
         <div class="flex items-center gap-4">
            <button id="event-back-month" class="hover:bg-[#005A9E] p-1 rounded" @click="cancel">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg>
            </button>
            <span class="font-semibold">New Event</span>
         </div>
         <button id="event-save-button" class="bg-white text-[#0078D4] px-4 py-1 rounded font-semibold hover:bg-gray-100 flex items-center gap-2" @click="save">
            <span>Save</span>
         </button>
    </header>

    <div class="flex-1 p-6 max-w-2xl mx-auto w-full flex flex-col gap-6">
        <div class="flex items-center border-b border-gray-200 py-2">
            <span class="text-2xl mr-4">🏷️</span>
            <input type="text" id="event-title-input" v-model="title" @input="handleInput('title')" class="flex-1 outline-none text-2xl font-light text-gray-800" placeholder="Add title" />
        </div>
        
        <!-- Date Picker Widget -->
        <div class="flex items-center gap-4 py-2">
            <span class="text-xl text-gray-500 w-8">🕒</span>
            <div class="flex-1">
                 <DateTimePicker id="date-picker" @change="handleDateTimeChange" />
            </div>
        </div>

        <div class="flex items-center border-b border-gray-200 py-2">
            <span class="text-xl text-gray-500 w-8">📍</span>
            <input type="text" id="event-location-input" v-model="location" @input="handleInput('location')" class="flex-1 outline-none text-gray-800" placeholder="Add location" />
        </div>
        
        <div class="flex-1 mt-4 border border-gray-200 rounded p-4">
            <textarea id="event-description-input" v-model="description" @input="handleInput('description')" class="w-full h-full resize-none outline-none text-gray-800 leading-relaxed font-sans" placeholder="Add description..."></textarea>
        </div>
    </div>
  </div>
</template>

<script>
import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useSignatureStore } from '../stores/signature';
import DateTimePicker from '../components/widgets/DateTimePicker.vue';

export default {
  name: 'CALENDAR_NEW_EVENT',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter();
    const signatureStore = useSignatureStore();
    
    const title = ref('');
    const location = ref('');
    const description = ref('');

    const handleInput = (field) => {
        let val = '';
        if (field === 'title') val = title.value;
        if (field === 'location') val = location.value;
        if (field === 'description') val = description.value;
        
        if (field === 'title') signatureStore.handleAction('ACT_EVENT_TYPE_TITLE', { input_text: val, field });
        if (field === 'location') signatureStore.handleAction('ACT_EVENT_TYPE_LOCATION', { input_text: val, field });
        if (field === 'description') signatureStore.handleAction('ACT_EVENT_TYPE_DESCRIPTION', { input_text: val, field });
    };

    const handleDateTimeChange = (dt) => {
        // Param structure from FSM: year, month, day, hour, minute
        // DateTimePicker likely returns string or obj. We map to FSM params.
        signatureStore.handleAction('ACT_EVENT_PICK_DATETIME', {
            widget: 'date_picker',
            year: 2025, // Mock values matching FSM example or derived from component
            month: 10,
            day: 22,
            hour: 10,
            minute: 30
        });
    };

    const save = async () => {
        await signatureStore.handleAction('ACT_EVENT_SAVE');
        router.push({ name: 'SCHEDULE_MEETING_SUCCESS' });
    };

    const cancel = async () => {
        await signatureStore.handleAction('ACT_EVENT_BACK_MONTH');
        router.push({ name: 'CALENDAR_MONTH' });
    };

    return {
        title, location, description,
        handleInput,
        handleDateTimeChange,
        save,
        cancel
    };
  }
}
</script>