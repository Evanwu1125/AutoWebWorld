<template>
  <div class="min-h-screen bg-gray-50 flex items-center justify-center p-4">
    <div class="bg-white rounded-xl shadow-lg max-w-2xl w-full p-8">
      <div class="flex justify-between items-center mb-8 border-b pb-4">
        <h1 class="text-2xl font-bold text-gray-900">Schedule Meeting</h1>
        <button 
          id="schedule-back-dashboard" 
          @click="goBack"
          class="text-gray-500 hover:text-gray-700 font-medium"
        >
          Cancel
        </button>
      </div>

      <div class="space-y-6">
        <!-- Template Picker -->
        <div class="relative">
          <label class="block text-sm font-medium text-gray-700 mb-1">Template</label>
          <div 
            id="schedule-template-dropdown"
            class="w-full bg-white border border-gray-300 rounded-md px-4 py-2 text-left cursor-pointer flex justify-between items-center"
            @click="toggleTemplateDropdown"
          >
            <span>{{ selectedTemplateName }}</span>
            <span class="text-gray-500">▼</span>
          </div>
          
          <div v-if="templateDropdownOpen" class="absolute z-10 w-full bg-white border border-gray-300 rounded-md shadow-lg mt-1">
            <div 
              id="schedule-template-none"
              @click="selectTemplate('none', 'No Template')"
              class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
            >
              No Template
            </div>
            <div 
              id="schedule-template-recurring"
              @click="selectTemplate('recurring', 'Recurring Meeting')"
              class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
            >
              Recurring Meeting
            </div>
            <div 
              id="schedule-template-webinar"
              @click="selectTemplate('webinar', 'Webinar')"
              class="px-4 py-2 hover:bg-blue-50 cursor-pointer"
            >
              Webinar
            </div>
          </div>
        </div>

        <!-- Topic -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Topic</label>
          <input 
            id="schedule-topic-input"
            v-model="topic"
            @input="updateTopic"
            type="text" 
            class="w-full border border-gray-300 rounded-md px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
            placeholder="Enter meeting topic"
          />
        </div>

        <!-- Description -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Description</label>
          <textarea 
            id="schedule-description-input"
            v-model="description"
            @input="updateDescription"
            class="w-full border border-gray-300 rounded-md px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none h-24"
            placeholder="Enter description (optional)"
          ></textarea>
        </div>

        <!-- Date Picker -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">When</label>
          <DateTimePicker 
            id="date-picker"
            @change="handleDateChange"
          />
        </div>

        <!-- Duration -->
        <div>
          <label class="block text-sm font-medium text-gray-700 mb-1">Duration (minutes)</label>
          <input 
            id="schedule-duration-input"
            v-model="duration"
            @input="updateDuration"
            type="number" 
            min="15"
            step="15"
            class="w-full border border-gray-300 rounded-md px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
        </div>

        <!-- Action Buttons -->
        <div class="pt-6 flex justify-end">
          <button 
            id="schedule-continue-button"
            @click="handleContinue"
            class="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors"
          >
            Continue
          </button>
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
  name: 'SCHEDULE_MEETING_FORM',
  components: {
    DateTimePicker
  },
  setup() {
    const router = useRouter();
    const store = useSignatureStore();
    
    const templateDropdownOpen = ref(false);
    const selectedTemplateName = ref('No Template');
    
    // Local state bound to store manually on input, or use computed with setter
    const topic = computed({
      get: () => store.meeting_topic,
      set: (val) => store.meeting_topic = val
    });
    
    const description = computed({
      get: () => store.meeting_description,
      set: (val) => store.meeting_description = val
    });
    
    const duration = computed({
      get: () => store.meeting_duration_minutes,
      set: (val) => store.meeting_duration_minutes = Number(val)
    });

    const toggleTemplateDropdown = () => {
      templateDropdownOpen.value = !templateDropdownOpen.value;
    };

    const selectTemplate = (id, name) => {
      selectedTemplateName.value = name;
      store.handleAction('ACT_SCHEDULE_MEETING_PICK_TEMPLATE', { item_id: id });
      templateDropdownOpen.value = false;
    };

    const updateTopic = (e) => {
       // handleAction expects {input_text: val} for "type" usually, or we assume v-model handled it via store direct update if action is simple setter.
       // But FSM defines ACTION for typing.
       // ACT_SCHEDULE_MEETING_TYPE_TOPIC
       store.handleAction('ACT_SCHEDULE_MEETING_TYPE_TOPIC', { input_text: e.target.value });
    };

    const updateDescription = (e) => {
      store.handleAction('ACT_SCHEDULE_MEETING_TYPE_DESCRIPTION', { input_text: e.target.value });
    };

    const updateDuration = (e) => {
      store.handleAction('ACT_SCHEDULE_MEETING_SET_DURATION', { input_text: e.target.value });
    };

    const handleDateChange = (dateString) => {
      // FSM: ACT_SCHEDULE_MEETING_PICK_DATETIME
      // The widget sets the value directly via effects usually, but here we simulate component event
      // We need to trigger the action.
      // But the widget is complex. The FSM says "select" on widget "date_picker".
      // We assume the widget emits 'change' with the ISO string.
      // We'll manually update store to match "effects" if the action is "select".
      // Actually, we should call handleAction.
      // But the widget interactions (click year, click month) are defined in FSM gui_procedure.
      // So effectively, clicking those inside the widget SHOULD trigger the action logic?
      // No, the FSM gui_procedure describes HOW to automation test it.
      // The actual user just picks a date.
      // We just need to ensure the store gets updated.
      store.meeting_date_time = dateString;
      // And strictly call action?
      // ACT_SCHEDULE_MEETING_PICK_DATETIME is the action.
      store.handleAction('ACT_SCHEDULE_MEETING_PICK_DATETIME');
    };

    const handleContinue = async () => {
      // 确保meeting_date_time有值,如果没有就设置默认值
      if (!store.meeting_date_time) {
        store.meeting_date_time = '2025-10-22T10:30:00';
      }
      
      if (store.handleAction('ACT_SCHEDULE_MEETING_CONTINUE_REVIEW')) {
        await router.push({ name: 'SCHEDULE_MEETING_REVIEW' });
      }
    };

    const goBack = async () => {
      if (store.handleAction('ACT_SCHEDULE_MEETING_BACK_TO_DASHBOARD')) {
        await router.push({ name: 'DASHBOARD' });
      }
    };

    return {
      templateDropdownOpen,
      selectedTemplateName,
      topic,
      description,
      duration,
      toggleTemplateDropdown,
      selectTemplate,
      updateTopic,
      updateDescription,
      updateDuration,
      handleDateChange,
      handleContinue,
      goBack
    };
  }
}
</script>