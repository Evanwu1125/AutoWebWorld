# Development Checklist

## 1. Project Setup
- [x] Add dependencies (pinia-plugin-persistedstate)
- [x] Create style.txt
- [ ] Initialize `src/stores/signature.js` (Pinia Store)
- [ ] Initialize `src/stores/data.js` (Mock Data Store)
- [ ] Create `src/fsm/FSMRuntime.js` (FSM Logic)
- [ ] Setup `src/router/index.js` (17 Routes)
- [ ] Update `src/main.js` (Plugins)
- [ ] Create `src/components/PermissionModal.vue`
- [ ] Create `src/components/CookieConsentModal.vue`

## 2. Page Implementation
- [ ] HOME (Cookie Consent, Navigation)
- [ ] TEAMS_LIST (List of teams, filters, search, create button)
- [ ] CHANNELS_LIST (Channels within team, filters, search)
- [ ] CHANNEL_POST_COMPOSE (Write post)
- [ ] CHANNEL_POST_SENT_SUCCESS (Success feedback)
- [ ] CREATE_TEAM (Form)
- [ ] TEAM_CREATED_SUCCESS (Success feedback)
- [ ] CALENDAR_VIEW (Calendar grid/list, meet now, new meeting)
- [ ] MEETING_DETAILS (New meeting form)
- [ ] MEETING_REVIEW (Add invitees, confirm)
- [ ] MEETING_SCHEDULED_SUCCESS (Success feedback)
- [ ] MEET_NOW_SETUP (Camera/Mic toggle)
- [ ] MEET_NOW_STARTED_SUCCESS (Success feedback)
- [ ] CHAT_LIST (List of chats, filters, search)
- [ ] CHAT_THREAD (Message history, input)
- [ ] CHAT_MESSAGE_SENT_SUCCESS (Success feedback)
- [ ] CALLS_HUB (Call history, filters)

## 3. Data Requirements (Mock Data)
- [ ] Users (Current user, contacts) - 15+ items
- [ ] Teams - 15-20 items
- [ ] Channels - 3-5 per team
- [ ] Chat Threads - 15-20 items
- [ ] Messages - 10+ per thread
- [ ] Calendar Events - 15-20 items
- [ ] Call History - 15-20 items
- [ ] Images for all avatars and thumbnails

## 4. Verification
- [ ] Verify all FSM pages exist
- [ ] Verify all selectors match FSM
- [ ] Lint check
- [ ] Build check