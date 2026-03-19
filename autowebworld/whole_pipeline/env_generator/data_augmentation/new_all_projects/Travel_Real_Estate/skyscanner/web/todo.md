# Development Checklist

## 1. Setup & Configuration
- [ ] Install `pinia-plugin-persistedstate`
- [ ] Configure `src/main.js` with Pinia and PersistedState
- [ ] Setup `src/router/index.js` with all 28 routes and navigation guards
- [ ] Implement `src/fsm/FSMRuntime.js` helper class
- [ ] Implement `src/stores/signature.js` (FSM State)
- [ ] Implement `src/stores/data.js` (Mock Data)

## 2. Global Components
- [ ] `src/components/CookieConsentModal.vue` (Interceptor)
- [ ] `src/components/PermissionModal.vue` (Interceptor)
- [ ] `src/components/NavBar.vue` (Optional, for consistent layout)

## 3. Pages Implementation (28 Pages)
### Core & Account
- [ ] `src/pages/HOME.vue`
- [ ] `src/pages/LOGIN.vue`
- [ ] `src/pages/ACCOUNT_OVERVIEW.vue`

### Flights Flow
- [ ] `src/pages/FLIGHTS_SEARCH.vue`
- [ ] `src/pages/MULTI_CITY_SEARCH.vue`
- [ ] `src/pages/FLIGHTS_RESULTS.vue`
- [ ] `src/pages/FLIGHT_DETAILS.vue`
- [ ] `src/pages/BOOKING_FORM_DIRECT.vue`
- [ ] `src/pages/BOOKING_REVIEW_DIRECT.vue`
- [ ] `src/pages/BOOKING_COMPLETE_DIRECT.vue`
- [ ] `src/pages/MULTI_CITY_RESULTS.vue`
- [ ] `src/pages/MULTI_CITY_INTRO.vue`
- [ ] `src/pages/BOOKING_FORM_MULTI.vue`
- [ ] `src/pages/BOOKING_REVIEW_MULTI.vue`
- [ ] `src/pages/BOOKING_COMPLETE_MULTI.vue`
- [ ] `src/pages/TRIP_SUMMARY.vue`

### Price Alerts
- [ ] `src/pages/PRICE_ALERT_FORM.vue`
- [ ] `src/pages/PRICE_ALERT_CREATED.vue`
- [ ] `src/pages/PRICE_ALERTS_LIST.vue`
- [ ] `src/pages/ALERT_DETAIL.vue`

### Hotels Flow
- [ ] `src/pages/HOTELS_SEARCH.vue`
- [ ] `src/pages/HOTELS_RESULTS.vue`
- [ ] `src/pages/HOTEL_DETAILS.vue`
- [ ] `src/pages/BOOKING_COMPLETE_HOTEL.vue`

### Cars Flow
- [ ] `src/pages/CARS_SEARCH.vue`
- [ ] `src/pages/CARS_RESULTS.vue`
- [ ] `src/pages/CAR_DETAILS.vue`
- [ ] `src/pages/BOOKING_COMPLETE_CAR.vue`

## 4. Final Steps
- [ ] Verify all FSM actions and selectors
- [ ] Run lint and build