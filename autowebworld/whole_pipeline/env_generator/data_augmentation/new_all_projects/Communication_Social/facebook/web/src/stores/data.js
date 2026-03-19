import { defineStore } from 'pinia';

export const useDataStore = defineStore('data', {
  state: () => ({
    currentUser: {
      id: 'user_1',
      name: 'Alex Johnson',
      avatar: '/images/UserAvatar.jpg',
      cover: '/images/UserCover.jpg'
    },
    
    // Posts (20 items)
    posts: [
      { id: 'post_1', user_id: 'user_2', author_name: 'Sarah Williams', author_avatar: '/images/Hiking.jpg', content: 'Just finished hiking the Grand Canyon! Amazing views.', image: '/images/friends_user_2.jpg', time: '2h', likes: 124, comments: 12 },
      { id: 'post_2', user_id: 'user_3', author_name: 'Mike Chen', author_avatar: '/images/MikeChen.jpg', content: 'Check out my new coding setup! #developer #setup', image: '/images/friends_user_3.jpg', time: '4h', likes: 89, comments: 5 },
      { id: 'post_3', user_id: 'user_4', author_name: 'Emma Davis', author_avatar: '/images/Coding.jpg', content: 'Delicious homemade pasta for dinner tonight.', image: '/images/friends_user_4.jpg', time: '6h', likes: 230, comments: 34 },
      { id: 'post_4', user_id: 'user_5', author_name: 'James Wilson', author_avatar: '/images/Book.jpg', content: 'Can anyone recommend a good book for summer reading?', image: '/images/friends_user_5.jpg', time: '8h', likes: 45, comments: 18 },
      { id: 'post_5', user_id: 'user_6', author_name: 'Linda Martinez', author_avatar: '/images/Birthday.jpg', content: 'Happy Birthday to my best friend!', image: '/images/friends_user_6.jpg', time: '10h', likes: 156, comments: 42 },
      { id: 'post_6', user_id: 'user_7', author_name: 'Robert Taylor', author_avatar: '/images/Birthday.jpg', content: 'Beautiful sunset at the beach today.', image: '/images/friends_user_7.jpg', time: '12h', likes: 312, comments: 28 },
      { id: 'post_7', user_id: 'user_8', author_name: 'Jennifer Anderson', author_avatar: '/images/JenniferAnderson.jpg', content: 'So excited to start my new job next week!', image: null, time: '1d', likes: 420, comments: 89 },
      { id: 'post_8', user_id: 'user_9', author_name: 'David Thomas', author_avatar: '/images/Basketball.jpg', content: 'Who wants to play basketball this weekend?', image: '/images/friends_user_9.jpg', time: '1d', likes: 23, comments: 7 },
      { id: 'post_9', user_id: 'user_10', author_name: 'Lisa Jackson', author_avatar: '/images/Cat.jpg', content: 'My cat is being so funny today.', image: '/images/friends_user_10.jpg', time: '1d', likes: 567, comments: 120 },
      { id: 'post_10', user_id: 'user_11', author_name: 'William White', author_avatar: '/images/Cat.jpg', content: 'Just bought a new car! Hard work pays off.', image: '/images/friends_user_11.jpg', time: '2d', likes: 210, comments: 45 },
      { id: 'post_11', user_id: 'user_12', author_name: 'Patricia Harris', author_avatar: '/images/Coffee.jpg', content: 'Enjoying a quiet morning with coffee.', image: '/images/friends_user_12.jpg', time: '2d', likes: 67, comments: 3 },
      { id: 'post_12', user_id: 'user_13', author_name: 'Richard Martin', author_avatar: '/images/Coffee.jpg', content: 'Traffic was terrible this morning. Ugh.', image: null, time: '2d', likes: 12, comments: 15 },
      { id: 'post_13', user_id: 'user_14', author_name: 'Susan Thompson', author_avatar: '/images/Paris.jpg', content: 'Throwback to our trip to Paris last year.', image: '/images/friends_user_14.jpg', time: '3d', likes: 450, comments: 67 },
      { id: 'post_14', user_id: 'user_15', author_name: 'Joseph Garcia', author_avatar: '/images/JosephGarcia.jpg', content: 'Working late on a new project.', image: '/images/friends_user_15.jpg', time: '3d', likes: 34, comments: 2 },
      { id: 'post_15', user_id: 'user_16', author_name: 'Karen Robinson', author_avatar: '/images/Plumber.jpg', content: 'Does anyone know a good plumber?', image: null, time: '4d', likes: 5, comments: 8 },
      { id: 'post_16', user_id: 'user_17', author_name: 'Thomas Clark', author_avatar: '/images/Graduation.jpg', content: 'Finally finished my degree! Graduation day.', image: '/images/suggestedFriends_user_17.jpg', time: '4d', likes: 890, comments: 150 },
      { id: 'post_17', user_id: 'user_18', author_name: 'Nancy Rodriguez', author_avatar: '/images/Graduation.jpg', content: 'Making pizza from scratch tonight.', image: '/images/suggestedFriends_user_18.jpg', time: '5d', likes: 78, comments: 12 },
      { id: 'post_18', user_id: 'user_19', author_name: 'Charles Lewis', author_avatar: '/images/Autumn.jpg', content: 'Autumn leaves are falling. My favorite season.', image: '/images/suggestedFriends_user_19.jpg', time: '5d', likes: 234, comments: 45 },
      { id: 'post_19', user_id: 'user_20', author_name: 'Margaret Lee', author_avatar: '/images/Family.jpg', content: 'Family reunion was a blast!', image: '/images/suggestedFriends_user_20.jpg', time: '6d', likes: 112, comments: 23 },
      { id: 'post_20', user_id: 'user_21', author_name: 'Daniel Walker', author_avatar: '/images/Haircut.jpg', content: 'New haircut! What do you think?', image: '/images/suggestedFriends_user_21.jpg', time: '1w', likes: 145, comments: 56 }
    ],
    
    // Friends (15 items)
    friends: [
      { id: 'user_2', name: 'Sarah Williams', mutual: 12, avatar: '/images/Hiking.jpg' },
      { id: 'user_3', name: 'Mike Chen', mutual: 5, avatar: '/images/MikeChen.jpg' },
      { id: 'user_4', name: 'Emma Davis', mutual: 23, avatar: '/images/Coding.jpg' },
      { id: 'user_5', name: 'James Wilson', mutual: 8, avatar: '/images/Book.jpg' },
      { id: 'user_6', name: 'Linda Martinez', mutual: 15, avatar: '/images/Birthday.jpg' },
      { id: 'user_7', name: 'Robert Taylor', mutual: 2, avatar: '/images/Birthday.jpg' },
      { id: 'user_8', name: 'Jennifer Anderson', mutual: 45, avatar: '/images/JenniferAnderson.jpg' },
      { id: 'user_9', name: 'David Thomas', mutual: 1, avatar: '/images/Basketball.jpg' },
      { id: 'user_10', name: 'Lisa Jackson', mutual: 34, avatar: '/images/Cat.jpg' },
      { id: 'user_11', name: 'William White', mutual: 9, avatar: '/images/Cat.jpg' },
      { id: 'user_12', name: 'Patricia Harris', mutual: 18, avatar: '/images/Coffee.jpg' },
      { id: 'user_13', name: 'Richard Martin', mutual: 7, avatar: '/images/Coffee.jpg' },
      { id: 'user_14', name: 'Susan Thompson', mutual: 21, avatar: '/images/Paris.jpg' },
      { id: 'user_15', name: 'Joseph Garcia', mutual: 3, avatar: '/images/JosephGarcia.jpg' },
      { id: 'user_16', name: 'Karen Robinson', mutual: 11, avatar: '/images/Plumber.jpg' }
    ],
    
    // Suggested Friends (10 items)
    suggestedFriends: [
      { id: 'user_17', name: 'Thomas Clark', mutual: 1, avatar: '/images/Graduation.jpg' },
      { id: 'user_18', name: 'Nancy Rodriguez', mutual: 2, avatar: '/images/Graduation.jpg' },
      { id: 'user_19', name: 'Charles Lewis', mutual: 0, avatar: '/images/Autumn.jpg' },
      { id: 'user_20', name: 'Margaret Lee', mutual: 3, avatar: '/images/Family.jpg' },
      { id: 'user_21', name: 'Daniel Walker', mutual: 1, avatar: '/images/Haircut.jpg' },
      { id: 'user_22', name: 'Paul Hall', mutual: 4, avatar: '/images/User.jpg' },
      { id: 'user_23', name: 'Dorothy Allen', mutual: 0, avatar: '/images/User.jpg' },
      { id: 'user_24', name: 'Mark Young', mutual: 2, avatar: '/images/User.jpg' },
      { id: 'user_25', name: 'Sandra Hernandez', mutual: 1, avatar: '/images/User.jpg' },
      { id: 'user_26', name: 'Donald King', mutual: 0, avatar: '/images/User.jpg' }
    ],
    
    // Messenger Threads (15 items)
    threads: [
      { id: 'thread_1', user_id: 'user_2', name: 'Sarah Williams', last_message: 'See you tomorrow!', time: '10m', unread: true, avatar: '/images/Hiking.jpg' },
      { id: 'thread_2', user_id: 'user_3', name: 'Mike Chen', last_message: 'Did you check the code?', time: '1h', unread: false, avatar: '/images/MikeChen.jpg' },
      { id: 'thread_3', user_id: 'user_4', name: 'Emma Davis', last_message: 'Thanks for the invite.', time: '3h', unread: true, avatar: '/images/Coding.jpg' },
      { id: 'thread_4', user_id: 'user_5', name: 'James Wilson', last_message: 'Ok, sounds good.', time: '5h', unread: false, avatar: '/images/Book.jpg' },
      { id: 'thread_5', user_id: 'user_6', name: 'Linda Martinez', last_message: 'Happy Birthday!', time: '1d', unread: false, avatar: '/images/Birthday.jpg' },
      { id: 'thread_6', user_id: 'user_7', name: 'Robert Taylor', last_message: 'Call me when you can.', time: '1d', unread: true, avatar: '/images/Birthday.jpg' },
      { id: 'thread_7', user_id: 'user_8', name: 'Jennifer Anderson', last_message: 'Can you send the file?', time: '2d', unread: false, avatar: '/images/JenniferAnderson.jpg' },
      { id: 'thread_8', user_id: 'user_9', name: 'David Thomas', last_message: 'Meeting at 5pm.', time: '2d', unread: false, avatar: '/images/Basketball.jpg' },
      { id: 'thread_9', user_id: 'user_10', name: 'Lisa Jackson', last_message: 'Lol that is funny.', time: '3d', unread: false, avatar: '/images/Cat.jpg' },
      { id: 'thread_10', user_id: 'user_11', name: 'William White', last_message: 'Sure thing.', time: '3d', unread: false, avatar: '/images/Cat.jpg' },
      { id: 'thread_11', user_id: 'user_12', name: 'Patricia Harris', last_message: 'Where are we going?', time: '4d', unread: false, avatar: '/images/Coffee.jpg' },
      { id: 'thread_12', user_id: 'user_13', name: 'Richard Martin', last_message: 'I will be there.', time: '4d', unread: false, avatar: '/images/Coffee.jpg' },
      { id: 'thread_13', user_id: 'user_14', name: 'Susan Thompson', last_message: 'Great photos!', time: '5d', unread: false, avatar: '/images/Paris.jpg' },
      { id: 'thread_14', user_id: 'user_15', name: 'Joseph Garcia', last_message: 'Talk soon.', time: '1w', unread: false, avatar: '/images/JosephGarcia.jpg' },
      { id: 'thread_15', user_id: 'user_16', name: 'Karen Robinson', last_message: 'Hello!', time: '1w', unread: false, avatar: '/images/Plumber.jpg' }
    ],
    
    // Events (15 items)
    events: [
      { id: 'event_1', name: 'Summer Music Festival', date: '2025-07-15', location: 'Central Park', image: '/images/events_event_1.jpg', attending: 1200 },
      { id: 'event_2', name: 'Tech Conference 2025', date: '2025-08-20', location: 'Convention Center', image: '/images/events_event_2.jpg', attending: 500 },
      { id: 'event_3', name: 'Art Gallery Opening', date: '2025-06-10', location: 'Downtown Gallery', image: '/images/events_event_3.jpg', attending: 150 },
      { id: 'event_4', name: 'Charity Run 5K', date: '2025-09-05', location: 'City Stadium', image: '/images/events_event_4.jpg', attending: 300 },
      { id: 'event_5', name: 'Food Truck Night', date: '2025-06-25', location: 'Main Street', image: '/images/events_event_5.jpg', attending: 800 },
      { id: 'event_6', name: 'Local Band Concert', date: '2025-06-30', location: 'The Blue Note', image: '/images/events_event_6.jpg', attending: 80 },
      { id: 'event_7', name: 'Coding Workshop', date: '2025-07-05', location: 'Tech Hub', image: '/images/events_event_7.jpg', attending: 40 },
      { id: 'event_8', name: 'Yoga in the Park', date: '2025-06-12', location: 'River Park', image: '/images/events_event_8.jpg', attending: 60 },
      { id: 'event_9', name: 'Book Club Meeting', date: '2025-06-18', location: 'Public Library', image: '/images/events_event_9.jpg', attending: 15 },
      { id: 'event_10', name: 'Movie Night', date: '2025-06-22', location: 'Community Center', image: '/images/events_event_10.jpg', attending: 100 },
      { id: 'event_11', name: 'Startup Pitch Night', date: '2025-07-10', location: 'Innovation Lab', image: '/images/events_event_11.jpg', attending: 200 },
      { id: 'event_12', name: 'Photography Walk', date: '2025-06-15', location: 'Botanic Garden', image: '/images/events_event_12.jpg', attending: 25 },
      { id: 'event_13', name: 'Farmers Market', date: '2025-06-17', location: 'Town Square', image: '/images/events_event_13.jpg', attending: 500 },
      { id: 'event_14', name: 'Gaming Tournament', date: '2025-08-01', location: 'Arcade Arena', image: '/images/events_event_14.jpg', attending: 120 },
      { id: 'event_15', name: 'Cooking Class', date: '2025-07-20', location: 'Culinary School', image: '/images/events_event_15.jpg', attending: 20 }
    ]
  }),
  persist: {
    storage: sessionStorage
  }
});