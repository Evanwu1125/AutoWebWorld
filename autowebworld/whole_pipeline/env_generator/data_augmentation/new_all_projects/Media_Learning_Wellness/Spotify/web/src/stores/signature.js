import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useSignatureStore = defineStore('signature', () => {
  // Global Navigation State
  const currentPageId = ref('HOME')

  // HOME
  const current_user_id = ref(null)
  const cookie_consent_given = ref(null)

  // BROWSE
  const location_permission_granted = ref(null)
  const categories = ref(null)
  const browse_filters_applied = ref(null)
  const browse_viewport_anchor_id = ref(null)
  const browse_has_searched = ref(null)
  const matched_playlist_id = ref(null)
  const selected_playlist_id = ref(null)

  // GENRE_CATEGORY
  const genre_playlists = ref(null)
  const genre_category_filters_applied = ref(null)
  const genre_category_viewport_anchor_id = ref(null)
  const genre_category_has_searched = ref(null)

  // YOUR_LIBRARY
  const playlists = ref(null)
  const albums = ref(null)
  const artists = ref(null)
  const your_library_filters_applied = ref(null)
  const your_library_viewport_anchor_id = ref(null)
  const your_library_has_searched = ref(null)

  // PLAYLIST_DETAIL
  const tracks = ref(null)
  const playlist_detail_viewport_anchor_id = ref(null)
  const playlist_detail_has_searched = ref(null)
  const matched_track_id = ref(null)
  const selected_track_id = ref(null)
  const playlist_added_to_library = ref(null)

  // TRACK_DETAIL
  const track_liked = ref(null)

  // ALBUM_DETAIL
  const selected_album_id = ref(null)
  const album_download_ready = ref(null)

  // ALBUM_DOWNLOAD_CONFIRM
  const album_download_confirmed = ref(null)

  // ALBUM_DOWNLOAD_SUCCESS
  const success_message = ref(null)

  // ARTIST_DETAIL
  const selected_artist_id = ref(null)
  const artist_top_tracks_filters_applied = ref(null)
  const artist_top_tracks_viewport_anchor_id = ref(null)
  const artist_top_tracks_has_searched = ref(null)

  // SEARCH_PAGE
  const search_results_tracks = ref(null)
  const search_results_playlists = ref(null)
  const search_page_filters_applied = ref(null)
  const search_page_viewport_anchor_id = ref(null)
  const search_page_has_searched = ref(null)

  // SIGNUP
  const signup_email = ref(null)
  const signup_password = ref(null)
  const signup_username = ref(null)
  const signup_birthdate = ref(null)
  const signup_plan_selected = ref(null)

  // ACCOUNT_OVERVIEW
  const account_plan = ref(null)

  // PREMIUM_UPSELL
  const selected_premium_plan = ref(null)

  // PREMIUM_PAYMENT
  const card_number = ref(null)
  const card_expiry = ref(null)
  const card_cvc = ref(null)
  const billing_name = ref(null)

  // PLAYLIST_CREATE
  const playlist_name = ref(null)
  const playlist_description = ref(null)
  const playlist_visibility = ref(null)

  // PLAYLIST_SHARE
  const share_target = ref(null)
  const share_message = ref(null)

  // PAYMENT_METHODS
  const payment_methods = ref(null)
  const payment_methods_filters_applied = ref(null)
  const payment_methods_viewport_anchor_id = ref(null)

  // PAYMENT_METHOD_DETAIL
  const selected_payment_method_id = ref(null)

  // SETTINGS
  const selected_theme = ref(null)
  const explicit_content_filter_enabled = ref(null)

  function setCurrentPageId(id) {
    currentPageId.value = id
  }

  return {
    currentPageId,
    setCurrentPageId,
    
    current_user_id,
    cookie_consent_given,
    location_permission_granted,
    categories,
    browse_filters_applied,
    browse_viewport_anchor_id,
    browse_has_searched,
    matched_playlist_id,
    selected_playlist_id,
    genre_playlists,
    genre_category_filters_applied,
    genre_category_viewport_anchor_id,
    genre_category_has_searched,
    playlists,
    albums,
    artists,
    your_library_filters_applied,
    your_library_viewport_anchor_id,
    your_library_has_searched,
    tracks,
    playlist_detail_viewport_anchor_id,
    playlist_detail_has_searched,
    matched_track_id,
    selected_track_id,
    playlist_added_to_library,
    track_liked,
    selected_album_id,
    album_download_ready,
    album_download_confirmed,
    success_message,
    selected_artist_id,
    artist_top_tracks_filters_applied,
    artist_top_tracks_viewport_anchor_id,
    artist_top_tracks_has_searched,
    search_results_tracks,
    search_results_playlists,
    search_page_filters_applied,
    search_page_viewport_anchor_id,
    search_page_has_searched,
    signup_email,
    signup_password,
    signup_username,
    signup_birthdate,
    signup_plan_selected,
    account_plan,
    selected_premium_plan,
    card_number,
    card_expiry,
    card_cvc,
    billing_name,
    playlist_name,
    playlist_description,
    playlist_visibility,
    share_target,
    share_message,
    payment_methods,
    payment_methods_filters_applied,
    payment_methods_viewport_anchor_id,
    selected_payment_method_id,
    selected_theme,
    explicit_content_filter_enabled
  }
}, {
  persist: {
    storage: sessionStorage
  }
})