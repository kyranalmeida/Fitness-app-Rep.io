# Rep.io 🏋️‍♂️👁️

**Rep.io** is a cross-platform, mobile-first fitness tracking application that leverages real-time, on-device AI pose detection to automatically count exercise repetitions and monitor user form. 

By running Google's MediaPipe machine learning models directly in the browser via WebAssembly, Rep.io ensures zero latency and absolute privacy—your camera feed never leaves your device. 

## ✨ Features

- **On-Device AI Vision**: Processes 33 3D skeletal landmarks per frame locally using Google MediaPipe Pose Landmarker.
- **Form Validation & Smart Counting**: Uses custom kinematic algorithms to calculate joint angles on the fly. Built-in form validation (e.g., ensuring proper wrist-to-shoulder ratios during push-ups) prevents false positives from incidental movements.
- **Audio Feedback**: Utilizes the Web Audio API to provide immediate auditory tones upon completing a valid rep, so you never have to look at your screen mid-set.
- **Workout Logging & History**: Automatically logs completed sets to a secure cloud database. 
- **CSV Data Export**: Download your entire workout history locally for external tracking and analysis.
- **Cross-Platform Ready**: Responsive web interface designed for the gym environment, easily compilable to native iOS and Android apps using Capacitor.
- **Secure Authentication**: Complete user auth flow including sign-up, login, profile editing, and password recovery.

## 🛠️ Tech Stack

- **Frontend:** React, Tailwind CSS, Lucide React
- **AI / Computer Vision:** Google MediaPipe Tasks Vision (`@mediapipe/tasks-vision`)
- **Backend & Database:** Supabase (PostgreSQL, Row Level Security, Auth)
- **Mobile Compilation:** Capacitor

## 🚀 Getting Started

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- A [Supabase](https://supabase.com/) account and project

### Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/rep-io.git](https://github.com/yourusername/rep-io.git)
   cd rep-io
