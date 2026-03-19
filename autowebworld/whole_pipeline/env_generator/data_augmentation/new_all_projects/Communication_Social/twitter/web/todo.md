# Development Checklist

## 1. Project Setup
- [ ] Update `package.json` with `pinia-plugin-persistedstate`
- [ ] Create `style.txt` (Visual Design Guidelines)
- [ ] Create `src/fsm/FSMRuntime.js` (FSM Engine)

## 2. State Management (Pinia)
- [ ] `src/stores/signature.js`
  - [ ] Implement all signature fields from FSM
  - [ ] Implement `setCurrentPageId`
  - [ ] Setup persistence
- [ ] `src/stores/data.js`
  - [ ] Users (10+ items)
  - [ ] Tweets (20+ items)
  - [ ] Messages/Threads (15+ items)
  - [ ] Notifications (15+ items)
  - [ ] Trends (10+ items)
  - [ ] Bookmarks (10+ items)
  - [ ] Integrate real images using paths

## 3. Router & App Structure
- [ ] `src/router/index.js` (Routes for all 22 pages)
- [ ] `src/App.vue`
  - [ ] Global layout (Sidebar/Bottom Nav)
  - [ ] Interceptors (Cookie Consent, Location Permission)
  - [ ] `router-view`
- [ ] `src/main.js` (Register Pinia + Persistence + Router)

## 4. Pages Implementation (22 Pages)
- [ ] `HOME.vue` (Landing/Feed entry)
- [ ] `HOME_TIMELINE.vue` (Main Feed)
- [ ] `TWEET_DETAIL.vue` (Single Tweet view)
- [ ] `COMPOSE_TWEET.vue` (New Tweet/Reply)
- [ ] `TWEET_POST_SUCCESS.vue` (Terminal)
- [ ] `TWEET_SCHEDULE_SUCCESS.vue` (Terminal)
- [ ] `PROFILE_OVERVIEW.vue` (Self Profile)
- [ ] `PROFILE_TWEETS.vue` (My Tweets List)
- [ ] `PROFILE_FOLLOWING_LIST.vue` (My Following)
- [ ] `USER_PROFILE_OVERVIEW.vue` (Other User Profile)
- [ ] `FOLLOW_USER_CONFIRM.vue` (Follow Action)
- [ ] `FOLLOW_USER_SUCCESS.vue` (Terminal)
- [ ] `MESSAGES_INBOX.vue` (DM List)
- [ ] `MESSAGES_THREAD.vue` (DM Chat)
- [ ] `MESSAGES_COMPOSE.vue` (New DM)
- [ ] `MESSAGE_SEND_SUCCESS.vue` (Terminal)
- [ ] `NOTIFICATIONS.vue` (Activity)
- [ ] `SETTINGS_PROFILE_EDIT.vue` (Edit Profile)
- [ ] `PROFILE_UPDATE_SUCCESS.vue` (Terminal)
- [ ] `TRENDS_EXPLORE.vue` (Search/Trends)
- [ ] `TOPIC_TWEET_LIST.vue` (Tweets for a Trend)
- [ ] `BOOKMARKS.vue` (Saved Tweets)

## 5. Components & Assets
- [ ] `src/components/widgets/DateTimePicker.vue` (Verify existence)
- [ ] `src/components/Navigation.vue` (Sidebar/Bottom Bar - Optional but recommended for layout)
- [ ] `src/components/CookieConsentModal.vue`
- [ ] `src/components/PermissionModal.vue`
- [ ] Download real images via ImageGetter

## 6. Validation
- [ ] Verify all FSM actions have UI elements
- [ ] Verify all selectors match `gui_procedure`
- [ ] Run lint & build