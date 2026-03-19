import { defineStore } from 'pinia'
import { ref } from 'vue'
import { FSMRuntime } from '../fsm/FSMRuntime'
import fsmData from '../../fsm.json'

export const useSignatureStore = defineStore('signature', () => {
  const currentPageId = ref('HOME')

  // FSM State Fields
  const cookie_consent_given = ref(null)
  const location_permission_granted = ref(null)
  
  const login_email_entered = ref(null)
  const login_password_entered = ref(null)
  const login_form_valid = ref(null)
  const saved_trips = ref(null)
  
  const trip_type = ref(null)
  const origin_entered = ref(null)
  const destination_entered = ref(null)
  const dates_selected = ref(null)
  const cabin_selected = ref(null)
  const search_ready = ref(null)
  
  const leg1_filled = ref(null)
  const leg2_filled = ref(null)
  const multi_city_valid = ref(null)
  
  const flight_options = ref(null)
  const matched_item_id = ref(null)
  const selected_item_id = ref(null)
  const flights_results_has_searched = ref(null)
  const flights_results_viewport_anchor_id = ref(null)
  const flights_results_filters_applied = ref(null)
  const sort_option = ref(null)
  const stops_filter_nonstop = ref(null)
  const times_slider_used = ref(null)
  
  const selected_baggage_option = ref(null)
  const selected_seat_option = ref(null)
  const extras_form_valid = ref(null)
  
  const passenger_first_name_entered = ref(null)
  const passenger_last_name_entered = ref(null)
  const contact_email_entered = ref(null)
  const payment_card_entered = ref(null)
  const booking_form_valid = ref(null)
  
  const terms_checked = ref(null)
  const review_ready = ref(null)
  
  const confirmation_number = ref(null)
  
  const multi_flight_options = ref(null)
  const multi_results_viewport_anchor_id = ref(null)
  const multi_results_filters_applied = ref(null)
  const segments_summary_viewed = ref(null)
  
  const primary_passenger_entered = ref(null)
  const second_passenger_entered = ref(null)
  const multi_payment_entered = ref(null)
  const multi_booking_valid = ref(null)
  
  const multi_terms_checked = ref(null)
  const multi_review_ready = ref(null)
  
  const multi_confirmation_number = ref(null)
  
  const trip_items = ref(null)
  
  const alert_email_entered = ref(null)
  const alert_name_entered = ref(null)
  const alert_form_valid = ref(null)
  const alert_id = ref(null)
  const alerts = ref(null)
  const alerts_list_filters_applied = ref(null)
  const alerts_list_viewport_anchor_id = ref(null)
  const alert_selected_id = ref(null)
  
  const hotel_destination_entered = ref(null)
  const hotel_dates_selected = ref(null)
  const hotel_search_ready = ref(null)
  const hotel_options = ref(null)
  const hotels_results_filters_applied = ref(null)
  const hotels_results_viewport_anchor_id = ref(null)
  const hotels_sort_option = ref(null)
  const hotel_selected_id = ref(null)
  const room_type_selected = ref(null)
  const guest_name_entered = ref(null)
  const hotel_form_valid = ref(null)
  const hotel_confirmation_number = ref(null)
  
  const car_pickup_entered = ref(null)
  const car_dates_selected = ref(null)
  const car_search_ready = ref(null)
  const car_options = ref(null)
  const cars_results_filters_applied = ref(null)
  const cars_results_viewport_anchor_id = ref(null)
  const car_selected_id = ref(null)
  const driver_name_entered = ref(null)
  const car_form_valid = ref(null)
  const car_confirmation_number = ref(null)

  const fsmRuntime = new FSMRuntime(fsmData, {
    get currentPageId() { return currentPageId.value }
  })

  function setCurrentPageId(id) {
    currentPageId.value = id
  }

  return {
    currentPageId,
    fsmRuntime,
    setCurrentPageId,
    
    cookie_consent_given,
    location_permission_granted,
    
    login_email_entered,
    login_password_entered,
    login_form_valid,
    saved_trips,
    
    trip_type,
    origin_entered,
    destination_entered,
    dates_selected,
    cabin_selected,
    search_ready,
    
    leg1_filled,
    leg2_filled,
    multi_city_valid,
    
    flight_options,
    matched_item_id,
    selected_item_id,
    flights_results_has_searched,
    flights_results_viewport_anchor_id,
    flights_results_filters_applied,
    sort_option,
    stops_filter_nonstop,
    times_slider_used,
    
    selected_baggage_option,
    selected_seat_option,
    extras_form_valid,
    
    passenger_first_name_entered,
    passenger_last_name_entered,
    contact_email_entered,
    payment_card_entered,
    booking_form_valid,
    
    terms_checked,
    review_ready,
    
    confirmation_number,
    
    multi_flight_options,
    multi_results_viewport_anchor_id,
    multi_results_filters_applied,
    segments_summary_viewed,
    
    primary_passenger_entered,
    second_passenger_entered,
    multi_payment_entered,
    multi_booking_valid,
    
    multi_terms_checked,
    multi_review_ready,
    
    multi_confirmation_number,
    
    trip_items,
    
    alert_email_entered,
    alert_name_entered,
    alert_form_valid,
    alert_id,
    alerts,
    alerts_list_filters_applied,
    alerts_list_viewport_anchor_id,
    alert_selected_id,
    
    hotel_destination_entered,
    hotel_dates_selected,
    hotel_search_ready,
    hotel_options,
    hotels_results_filters_applied,
    hotels_results_viewport_anchor_id,
    hotels_sort_option,
    hotel_selected_id,
    room_type_selected,
    guest_name_entered,
    hotel_form_valid,
    hotel_confirmation_number,
    
    car_pickup_entered,
    car_dates_selected,
    car_search_ready,
    car_options,
    cars_results_filters_applied,
    cars_results_viewport_anchor_id,
    car_selected_id,
    driver_name_entered,
    car_form_valid,
    car_confirmation_number
  }
}, {
  persist: {
    storage: sessionStorage,
  },
})