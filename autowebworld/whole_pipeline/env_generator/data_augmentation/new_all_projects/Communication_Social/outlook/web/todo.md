# Development Checklist

## 1. Project Setup
- [x] Create style.txt
- [x] Create todo.md
- [ ] Update package.json (add pinia-plugin-persistedstate)
- [ ] Update src/main.js (configure pinia-plugin-persistedstate)
- [ ] Create src/fsm/index.js (FSM Runtime Engine)
- [ ] Create src/stores/signature.js (Signature Store)
- [ ] Create src/router/index.js (Router Configuration)
- [ ] Create src/stores/data.js (Mock Data Store - Last Step)

## 2. Page Implementation (22 Pages)
- [ ] src/pages/HOME.vue
- [ ] src/pages/MAIL_INBOX.vue
- [ ] src/pages/MAIL_MESSAGE_READ.vue
- [ ] src/pages/MAIL_COMPOSE.vue
- [ ] src/pages/MAIL_REPLY.vue
- [ ] src/pages/MAIL_FORWARD.vue
- [ ] src/pages/MAIL_MOVE.vue
- [ ] src/pages/MAIL_SENT.vue
- [ ] src/pages/MAIL_DRAFTS.vue
- [ ] src/pages/MAIL_TRASH.vue
- [ ] src/pages/CALENDAR_MONTH.vue
- [ ] src/pages/CALENDAR_DAY.vue
- [ ] src/pages/CALENDAR_NEW_EVENT.vue
- [ ] src/pages/CALENDAR_EVENT_DETAIL.vue
- [ ] src/pages/MAIL_SETTINGS_GENERAL.vue
- [ ] src/pages/PEOPLE_LIST.vue
- [ ] src/pages/PEOPLE_DETAIL.vue
- [ ] src/pages/SEND_EMAIL_SUCCESS.vue (Terminal)
- [ ] src/pages/REPLY_EMAIL_SUCCESS.vue (Terminal)
- [ ] src/pages/FORWARD_EMAIL_SUCCESS.vue (Terminal)
- [ ] src/pages/SCHEDULE_MEETING_SUCCESS.vue (Terminal)
- [ ] src/pages/MOVE_EMAIL_SUCCESS.vue (Terminal)

## 3. Components
- [ ] Ensure src/components/widgets/DateTimePicker.vue exists
- [ ] Implement Permission Modal (Location)
- [ ] Implement Cookie Consent Modal

## 4. Verification
- [ ] Verify all GUI selectors match FSM
- [ ] Verify FSM logic (preconditions/effects)
- [ ] Build and Lint