import { defineStore } from 'pinia'
import { ref } from 'vue'

export const useDataStore = defineStore('data', () => {
  // --- Flights Data ---
  const flights = ref([
    {
      id: 'flight_1',
      airline: 'British Airways',
      origin: 'LHR',
      destination: 'JFK',
      price: 450,
      duration: '7h 55m',
      departure_time: '10:00',
      arrival_time: '12:55',
      stops: 0,
      image: '/images/flight-ba-747.jpg'
    },
    {
      id: 'flight_2',
      airline: 'Virgin Atlantic',
      origin: 'LHR',
      destination: 'JFK',
      price: 420,
      duration: '8h 05m',
      departure_time: '14:30',
      arrival_time: '17:35',
      stops: 0,
      image: '/images/flight-virgin-a350.jpg'
    },
    {
      id: 'flight_3',
      airline: 'American Airlines',
      origin: 'LHR',
      destination: 'JFK',
      price: 380,
      duration: '8h 15m',
      departure_time: '08:45',
      arrival_time: '11:30',
      stops: 1,
      image: '/images/flight-aa-777.jpg'
    },
    {
      id: 'flight_4',
      airline: 'Delta',
      origin: 'LHR',
      destination: 'JFK',
      price: 460,
      duration: '8h 00m',
      departure_time: '11:15',
      arrival_time: '14:15',
      stops: 0,
      image: '/images/flight-delta-a330.jpg'
    },
    {
      id: 'flight_5',
      airline: 'United',
      origin: 'LHR',
      destination: 'JFK',
      price: 400,
      duration: '8h 20m',
      departure_time: '09:00',
      arrival_time: '12:20',
      stops: 1,
      image: '/images/flight-united-787.jpg'
    },
    {
      id: 'flight_6',
      airline: 'Lufthansa',
      origin: 'LHR',
      destination: 'JFK',
      price: 390,
      duration: '10h 30m',
      departure_time: '07:30',
      arrival_time: '13:00',
      stops: 1,
      image: '/images/flight-lufthansa-a380.jpg'
    },
    {
      id: 'flight_7',
      airline: 'Air France',
      origin: 'LHR',
      destination: 'JFK',
      price: 395,
      duration: '9h 45m',
      departure_time: '08:00',
      arrival_time: '12:45',
      stops: 1,
      image: '/images/flight-airfrance-777.jpg'
    },
    {
      id: 'flight_8',
      airline: 'KLM',
      origin: 'LHR',
      destination: 'JFK',
      price: 410,
      duration: '10h 00m',
      departure_time: '06:45',
      arrival_time: '11:45',
      stops: 1,
      image: '/images/flight-klm-787.jpg'
    },
    {
      id: 'flight_9',
      airline: 'Aer Lingus',
      origin: 'LHR',
      destination: 'JFK',
      price: 350,
      duration: '9h 15m',
      departure_time: '09:30',
      arrival_time: '13:45',
      stops: 1,
      image: '/images/flight-aerlingus-a330.jpg'
    },
    {
      id: 'flight_10',
      airline: 'Norse Atlantic',
      origin: 'LGW',
      destination: 'JFK',
      price: 280,
      duration: '8h 10m',
      departure_time: '13:00',
      arrival_time: '16:10',
      stops: 0,
      image: '/images/flight-norse-787.jpg'
    },
    {
      id: 'flight_11',
      airline: 'JetBlue',
      origin: 'LHR',
      destination: 'JFK',
      price: 430,
      duration: '8h 15m',
      departure_time: '10:45',
      arrival_time: '14:00',
      stops: 0,
      image: '/images/flight-jetblue-a321.jpg'
    },
    {
      id: 'flight_12',
      airline: 'Finnair',
      origin: 'LHR',
      destination: 'JFK',
      price: 440,
      duration: '11h 00m',
      departure_time: '07:00',
      arrival_time: '13:00',
      stops: 1,
      image: '/images/flight-finnair-a350.jpg'
    },
    {
      id: 'flight_13',
      airline: 'Iberia',
      origin: 'LHR',
      destination: 'JFK',
      price: 415,
      duration: '11h 30m',
      departure_time: '06:30',
      arrival_time: '13:00',
      stops: 1,
      image: '/images/flight-iberia-a350.jpg'
    },
    {
      id: 'flight_14',
      airline: 'Swiss',
      origin: 'LHR',
      destination: 'JFK',
      price: 425,
      duration: '10h 45m',
      departure_time: '08:30',
      arrival_time: '14:15',
      stops: 1,
      image: '/images/flight-swiss-777.jpg'
    },
    {
      id: 'flight_15',
      airline: 'TAP Air Portugal',
      origin: 'LHR',
      destination: 'JFK',
      price: 360,
      duration: '12h 00m',
      departure_time: '06:00',
      arrival_time: '13:00',
      stops: 1,
      image: '/images/flight-tap-a330.jpg'
    },
    {
      id: 'flight_16',
      airline: 'Emirates',
      origin: 'LHR',
      destination: 'DXB',
      price: 550,
      duration: '7h 00m',
      departure_time: '14:00',
      arrival_time: '00:00',
      stops: 0,
      image: '/images/flight-emirates-a380.jpg'
    },
    {
      id: 'flight_17',
      airline: 'Qatar Airways',
      origin: 'LHR',
      destination: 'DOH',
      price: 530,
      duration: '6h 45m',
      departure_time: '15:00',
      arrival_time: '00:45',
      stops: 0,
      image: '/images/flight-qatar-a350.jpg'
    }
  ])

  // --- Multi-City Flights Data ---
  const multiFlights = ref([
    {
      id: 'multi_1',
      legs: ['LHR -> JFK', 'JFK -> LAX'],
      airline: 'American Airlines',
      price: 650,
      total_duration: '16h 00m',
      stops: 1,
      image: '/images/flight-multi-aa.jpg'
    },
    {
      id: 'multi_2',
      legs: ['LHR -> JFK', 'JFK -> SFO'],
      airline: 'Delta',
      price: 680,
      total_duration: '17h 00m',
      stops: 1,
      image: '/images/flight-multi-delta.jpg'
    },
    {
      id: 'multi_3',
      legs: ['LHR -> DXB', 'DXB -> SYD'],
      airline: 'Emirates',
      price: 1200,
      total_duration: '22h 00m',
      stops: 1,
      image: '/images/flight-multi-emirates.jpg'
    },
    {
      id: 'multi_4',
      legs: ['LHR -> SIN', 'SIN -> SYD'],
      airline: 'Singapore Airlines',
      price: 1150,
      total_duration: '21h 30m',
      stops: 1,
      image: '/images/flight-multi-sq.jpg'
    },
    {
      id: 'multi_5',
      legs: ['LHR -> HND', 'HND -> CTS'],
      airline: 'JAL',
      price: 900,
      total_duration: '15h 00m',
      stops: 1,
      image: '/images/flight-multi-jal.jpg'
    }
  ])

  // --- Hotels Data ---
  const hotels = ref([
    {
      id: 'hotel_1',
      name: 'The Ritz London',
      location: 'London, UK',
      price: 850,
      rating: 5,
      stars: 5,
      free_cancellation: false,
      image: '/images/hotel-ritz-london.jpg'
    },
    {
      id: 'hotel_2',
      name: 'The Savoy',
      location: 'London, UK',
      price: 750,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotel-savoy-london.jpg'
    },
    {
      id: 'hotel_3',
      name: 'Shangri-La The Shard',
      location: 'London, UK',
      price: 900,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotel-shangrila-london.jpg'
    },
    {
      id: 'hotel_4',
      name: 'Claridge\'s',
      location: 'London, UK',
      price: 800,
      rating: 5,
      stars: 5,
      free_cancellation: false,
      image: '/images/hotel-claridges-london.jpg'
    },
    {
      id: 'hotel_5',
      name: 'The Langham',
      location: 'London, UK',
      price: 600,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotel-langham-london.jpg'
    },
    {
      id: 'hotel_6',
      name: 'Hilton London Metropole',
      location: 'London, UK',
      price: 250,
      rating: 4,
      stars: 4,
      free_cancellation: true,
      image: '/images/hotel-hilton-metropole.jpg'
    },
    {
      id: 'hotel_7',
      name: 'Marriott County Hall',
      location: 'London, UK',
      price: 450,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotel-marriott-county.jpg'
    },
    {
      id: 'hotel_8',
      name: 'Park Plaza Westminster Bridge',
      location: 'London, UK',
      price: 300,
      rating: 4,
      stars: 4,
      free_cancellation: true,
      image: '/images/hotel-park-plaza.jpg'
    },
    {
      id: 'hotel_9',
      name: 'Premier Inn London County Hall',
      location: 'London, UK',
      price: 150,
      rating: 3,
      stars: 3,
      free_cancellation: false,
      image: '/images/hotel-premier-inn.jpg'
    },
    {
      id: 'hotel_10',
      name: 'Travelodge London Central',
      location: 'London, UK',
      price: 120,
      rating: 3,
      stars: 3,
      free_cancellation: false,
      image: '/images/hotel-travelodge.jpg'
    },
    {
      id: 'hotel_11',
      name: 'Plaza Hotel',
      location: 'New York, USA',
      price: 950,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotel-plaza-nyc.jpg'
    },
    {
      id: 'hotel_12',
      name: 'The St. Regis New York',
      location: 'New York, USA',
      price: 1100,
      rating: 5,
      stars: 5,
      free_cancellation: false,
      image: '/images/hotel-stregis-nyc.jpg'
    },
    {
      id: 'hotel_13',
      name: 'Four Seasons Hotel New York',
      location: 'New York, USA',
      price: 1200,
      rating: 5,
      stars: 5,
      free_cancellation: true,
      image: '/images/hotel-fourseasons-nyc.jpg'
    },
    {
      id: 'hotel_14',
      name: 'Ace Hotel New York',
      location: 'New York, USA',
      price: 350,
      rating: 4,
      stars: 4,
      free_cancellation: true,
      image: '/images/hotel-ace-nyc.jpg'
    },
    {
      id: 'hotel_15',
      name: 'Pod 51 Hotel',
      location: 'New York, USA',
      price: 180,
      rating: 3,
      stars: 3,
      free_cancellation: false,
      image: '/images/hotel-pod51-nyc.jpg'
    }
  ])

  // --- Cars Data ---
  const cars = ref([
    {
      id: 'car_1',
      model: 'Ford Fiesta',
      type: 'Economy',
      price: 45,
      seats: 5,
      transmission: 'Manual',
      image: '/images/car-ford-fiesta.jpg'
    },
    {
      id: 'car_2',
      model: 'Volkswagen Golf',
      type: 'Compact',
      price: 55,
      seats: 5,
      transmission: 'Manual',
      image: '/images/car-vw-golf.jpg'
    },
    {
      id: 'car_3',
      model: 'Ford Focus',
      type: 'Compact',
      price: 60,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/car-ford-focus.jpg'
    },
    {
      id: 'car_4',
      model: 'BMW 3 Series',
      type: 'Premium',
      price: 120,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/car-bmw-3series.jpg'
    },
    {
      id: 'car_5',
      model: 'Mercedes C-Class',
      type: 'Premium',
      price: 130,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/car-merc-cclass.jpg'
    },
    {
      id: 'car_6',
      model: 'Toyota Corolla',
      type: 'Intermediate',
      price: 70,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/car-toyota-corolla.jpg'
    },
    {
      id: 'car_7',
      model: 'Nissan Qashqai',
      type: 'SUV',
      price: 85,
      seats: 5,
      transmission: 'Manual',
      image: '/images/car-nissan-qashqai.jpg'
    },
    {
      id: 'car_8',
      model: 'Range Rover Evoque',
      type: 'Luxury SUV',
      price: 180,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/car-rangerover-evoque.jpg'
    },
    {
      id: 'car_9',
      model: 'Fiat 500',
      type: 'Mini',
      price: 40,
      seats: 4,
      transmission: 'Manual',
      image: '/images/car-fiat-500.jpg'
    },
    {
      id: 'car_10',
      model: 'Audi A4',
      type: 'Standard',
      price: 110,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/car-audi-a4.jpg'
    },
    {
      id: 'car_11',
      model: 'Tesla Model 3',
      type: 'Electric',
      price: 140,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/car-tesla-model3.jpg'
    },
    {
      id: 'car_12',
      model: 'Vauxhall Corsa',
      type: 'Economy',
      price: 48,
      seats: 5,
      transmission: 'Manual',
      image: '/images/car-vauxhall-corsa.jpg'
    },
    {
      id: 'car_13',
      model: 'Jeep Wrangler',
      type: 'SUV',
      price: 160,
      seats: 5,
      transmission: 'Automatic',
      image: '/images/car-jeep-wrangler.jpg'
    },
    {
      id: 'car_14',
      model: 'Porsche 911',
      type: 'Sports',
      price: 350,
      seats: 2,
      transmission: 'Automatic',
      image: '/images/car-porsche-911.jpg'
    },
    {
      id: 'car_15',
      model: 'Volkswagen Transporter',
      type: 'Van',
      price: 100,
      seats: 9,
      transmission: 'Manual',
      image: '/images/car-vw-transporter.jpg'
    }
  ])

  // --- Price Alerts Data ---
  const priceAlerts = ref([
    {
      id: 'alert_1',
      origin: 'LHR',
      destination: 'JFK',
      target_price: 400,
      current_price: 380,
      active: true,
      image: '/images/alert-nyc.jpg'
    },
    {
      id: 'alert_2',
      origin: 'LHR',
      destination: 'DXB',
      target_price: 500,
      current_price: 550,
      active: true,
      image: '/images/alert-dubai.jpg'
    },
    {
      id: 'alert_3',
      origin: 'LHR',
      destination: 'SYD',
      target_price: 1000,
      current_price: 1100,
      active: false,
      image: '/images/alert-sydney.jpg'
    },
    {
      id: 'alert_4',
      origin: 'LHR',
      destination: 'TYO',
      target_price: 850,
      current_price: 900,
      active: true,
      image: '/images/alert-tokyo.jpg'
    },
    {
      id: 'alert_5',
      origin: 'LGW',
      destination: 'AGP',
      target_price: 150,
      current_price: 120,
      active: false,
      image: '/images/alert-malaga.jpg'
    }
  ])

  return {
    flights,
    multiFlights,
    hotels,
    cars,
    priceAlerts
  }
}, {
  persist: {
    storage: sessionStorage,
  },
})