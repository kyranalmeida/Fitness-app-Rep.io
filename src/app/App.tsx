import { useState, useEffect, useRef } from "react";
import type { User } from "@supabase/supabase-js";
import {
  Camera, Square, RotateCcw, ArrowLeft, User as UserIcon, ChevronRight, Settings,
  LogOut, Download, Volume2, VolumeX, SwitchCamera, ShieldCheck, History, BookCheck,
  ScanEye, Activity
} from "lucide-react";
import { PoseLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";
import { supabase } from "./supabase";

type ExerciseKey = "pushups" | "squats" | "deadlifts";
type Side = "left" | "right";

// ── AI Vision Helpers (moved up for type access) ──────────────────────────────
type Landmark = { x: number; y: number; z: number; visibility: number };

// FIX: per-exercise thresholds instead of one global 160/90 pair, which was
// tuned for nothing in particular and badly under-counted squats especially.
const EXERCISES: Record<
  ExerciseKey,
  {
    label: string;
    color: string;
    bgColor: string;
    emoji: string;
    joints: { left: [number, number, number]; right: [number, number, number] };
    upThreshold: number;
    downThreshold: number;
    minVisibility: number;
    validateForm?: (landmarks: Landmark[]) => boolean;
  }
> = {
  pushups: {
    label: "Push-ups",
    color: "#c8ff00",
    bgColor: "rgba(200,255,0,0.07)",
    emoji: "💪",
    joints: { left: [11, 13, 15], right: [12, 14, 16] }, // shoulder-elbow-wrist
    upThreshold: 155,
    downThreshold: 95,
    minVisibility: 0.6,
    // FIX: Ensure wrists stay below shoulders (y increases downwards)
    validateForm: (landmarks) => {
      const leftWrist = landmarks[15], leftShoulder = landmarks[11];
      const rightWrist = landmarks[16], rightShoulder = landmarks[12];
      
      const leftValid = leftWrist && leftShoulder && leftWrist.y > leftShoulder.y + 0.05;
      const rightValid = rightWrist && rightShoulder && rightWrist.y > rightShoulder.y + 0.05;
      
      return leftValid || rightValid;
    }
  },
  squats: {
    label: "Squats",
    color: "#00e5ff",
    bgColor: "rgba(0,229,255,0.07)",
    emoji: "🦵",
    joints: { left: [23, 25, 27], right: [24, 26, 28] }, // hip-knee-ankle
    upThreshold: 160,
    downThreshold: 110,
    minVisibility: 0.6,
  },
  deadlifts: {
    label: "Deadlifts",
    color: "#ff6b6b",
    bgColor: "rgba(255,107,107,0.07)",
    emoji: "🏋️",
    joints: { left: [11, 23, 25], right: [12, 24, 26] }, // shoulder-hip-knee
    upThreshold: 160,
    downThreshold: 120,
    minVisibility: 0.6,
  },
};

interface SupabaseLog {
  id: string;
  exercise: string;
  reps: number;
  created_at: string;
}

// FIX: typed instead of `any` — gets real autocomplete/safety at call sites.
interface AppSettings {
  audioFeedback: boolean;
  cameraFacing: "user" | "environment";
}

function formatTime(dateStr: string) {
  const d = new Date(dateStr);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}
function formatDate(dateStr: string) {
  const d = new Date(dateStr);
  return d.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" });
}

// FIX: simple mount-transition hook so panels "materialize" (per Apple's
// motion guidance) instead of hard-cutting in, while still respecting
// prefers-reduced-motion via the motion-reduce: Tailwind variant below.
function useEnterTransition() {
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    const id = requestAnimationFrame(() => setMounted(true));
    return () => cancelAnimationFrame(id);
  }, []);
  return mounted;
}
const enterTransitionClass =
  "transition-all duration-200 ease-out motion-reduce:transition-none motion-reduce:duration-0";

function calculateAngle(a: { x: number; y: number }, b: { x: number; y: number }, c: { x: number; y: number }) {
  const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180.0) / Math.PI);
  if (angle > 180.0) angle = 360 - angle;
  return angle;
}

// FIX: side selection now "sticks" to whichever side locked in for the
// current rep instead of recomputing every frame. Recomputing every frame
// let a visibility flicker mid-rep swap the joint triplet, which reads as a
// fake 20-30° angle jump and got miscounted as motion.
function getActiveJoints(landmarks: Landmark[], exerciseKey: ExerciseKey, lockedSide: Side | null) {
  const { left, right } = EXERCISES[exerciseKey].joints;

  const avgVisibility = (idxs: [number, number, number]) =>
    idxs.reduce((sum, i) => sum + (landmarks[i]?.visibility ?? 0), 0) / 3;

  const leftVis = avgVisibility(left);
  const rightVis = avgVisibility(right);

  let side: Side;
  if (lockedSide && ((lockedSide === "left" && leftVis > 0.4) || (lockedSide === "right" && rightVis > 0.4))) {
    side = lockedSide;
  } else {
    side = leftVis >= rightVis ? "left" : "right";
  }

  const idxs = side === "left" ? left : right;
  return {
    a: landmarks[idxs[0]],
    b: landmarks[idxs[1]],
    c: landmarks[idxs[2]],
    side,
    visibility: side === "left" ? leftVis : rightVis,
  };
}

// FIX: settings.audioFeedback existed as a toggle in Settings but nothing
// ever read it. This makes it real.
let audioCtx: AudioContext | null = null;
function playRepTone() {
  try {
    audioCtx = audioCtx ?? new (window.AudioContext || (window as any).webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.0001, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.15, audioCtx.currentTime + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.12);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.13);
  } catch {
    // Blocked by autoplay policy or unsupported — it's just a beep, fail silently.
  }
}

// FIX: load the pose model once per app session instead of once per
// CameraScreen mount. Previously, backing out and picking a different
// exercise reloaded the WASM runtime + model from scratch every time.
let landmarkerPromise: Promise<PoseLandmarker> | null = null;
function getPoseLandmarker(): Promise<PoseLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm");
      const landmarker = await PoseLandmarker.createFromModelPath(
        vision,
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
      );
      await landmarker.setOptions({ runningMode: "VIDEO" });
      return landmarker;
    })().catch((err) => {
      landmarkerPromise = null; // allow retry on next mount
      throw err;
    });
  }
  return landmarkerPromise;
}

// ── Settings Modal ────────────────────────────────────────────────────────────
function SettingsModal({
  onBack,
  settings,
  setSettings,
}: {
  onBack: () => void;
  settings: AppSettings;
  setSettings: React.Dispatch<React.SetStateAction<AppSettings>>;
}) {
  const mounted = useEnterTransition();
  return (
    <div className={`flex flex-col h-full w-full bg-[#0a0a0a] ${enterTransitionClass} ${mounted ? "opacity-100" : "opacity-0"}`}>
      <div className="flex items-center px-4 py-3 shrink-0 border-b border-white/10">
        <button onClick={onBack} aria-label="Back to account" className="p-2 -ml-2 text-[#888] active:opacity-60 transition-opacity">
          <ArrowLeft size={20} />
        </button>
        <span className="font-black uppercase ml-2 text-[#f0f0f0] text-xl tracking-wider" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
          App Settings
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-6">
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between p-4 rounded-sm bg-[#141414] border border-[#222]">
            <label htmlFor="audio-feedback-toggle" className="flex items-center gap-3 cursor-pointer">
              {settings.audioFeedback ? <Volume2 size={20} color="#c8ff00" /> : <VolumeX size={20} color="#555" />}
              <div>
                <div className="font-bold text-sm text-[#f0f0f0]">Rep Sound Feedback</div>
                <div className="text-xs text-[#888]" style={{ fontFamily: "'DM Mono', monospace" }}>Play audio tone on completed rep</div>
              </div>
            </label>
            <input
              id="audio-feedback-toggle"
              type="checkbox"
              checked={settings.audioFeedback}
              onChange={(e) => setSettings({ ...settings, audioFeedback: e.target.checked })}
              className="w-5 h-5 accent-[#c8ff00]"
            />
          </div>

          <div className="flex items-center justify-between p-4 rounded-sm bg-[#141414] border border-[#222]">
            <div className="flex items-center gap-3">
              <SwitchCamera size={20} color="#00e5ff" />
              <div>
                <div className="font-bold text-sm text-[#f0f0f0]">Camera Mode</div>
                <div className="text-xs text-[#888]" style={{ fontFamily: "'DM Mono', monospace" }}>Preferred camera angle</div>
              </div>
            </div>
            <button
              onClick={() => setSettings({ ...settings, cameraFacing: settings.cameraFacing === "user" ? "environment" : "user" })}
              aria-label={`Switch to ${settings.cameraFacing === "user" ? "back" : "front"} camera`}
              className="px-3 py-1.5 rounded-sm font-bold text-xs uppercase bg-[#222] text-[#c8ff00] active:scale-95 transition-transform"
              style={{ fontFamily: "'DM Mono', monospace" }}
            >
              {settings.cameraFacing === "user" ? "Front" : "Back"}
            </button>
          </div>

          <div className="flex items-center justify-between p-4 rounded-sm bg-[#141414] border border-[#222]">
            <div className="flex items-center gap-3">
              <ShieldCheck size={20} color="#ff6b6b" />
              <div>
                <div className="font-bold text-sm text-[#f0f0f0]">On-Device AI Processing</div>
                <div className="text-xs text-[#888]" style={{ fontFamily: "'DM Mono', monospace" }}>Camera feeds are never stored</div>
              </div>
            </div>
            <span className="text-xs font-bold text-[#c8ff00]" style={{ fontFamily: "'DM Mono', monospace" }}>ACTIVE</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Account Screen ────────────────────────────────────────────────────────────
function AccountScreen({
  onBack,
  onLogout,
  user,
  logs,
  settings,
  setSettings,
}: {
  onBack: () => void;
  onLogout: () => void;
  user: User | null;
  logs: SupabaseLog[];
  settings: AppSettings;
  setSettings: React.Dispatch<React.SetStateAction<AppSettings>>;
}) {
  const [view, setView] = useState<"account" | "settings" | "profile">("account");
  const mounted = useEnterTransition();
  const [editName, setEditName] = useState(user?.user_metadata?.full_name || "");
  const [savingProfile, setSavingProfile] = useState(false);

  // FIX: quote + escape every field so exercise/user data containing commas
  // or quotes can't corrupt the CSV (not exploitable today with fixed
  // exercise labels, but a landmine if custom exercise names are ever added).
  const csvField = (value: string | number) => `"${String(value).replace(/"/g, '""')}"`;

  const exportCSV = () => {
    if (logs.length === 0) {
      alert("No workout logs available to export.");
      return;
    }
    const headers = ["ID", "Date", "Time", "Exercise", "Reps"].map(csvField).join(",") + "\n";
    const rows = logs
      .map((l) => [l.id, formatDate(l.created_at), formatTime(l.created_at), l.exercise, l.reps].map(csvField).join(","))
      .join("\n");
    const blob = new Blob([headers + rows], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `rep_io_workout_history_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleUpdateProfile = async () => {
    setSavingProfile(true);
    const { error } = await supabase.auth.updateUser({
      data: { full_name: editName }
    });
    setSavingProfile(false);
    if (!error) setView("account");
    else alert("Failed to update profile.");
  };

  if (view === "settings") {
    return <SettingsModal onBack={() => setView("account")} settings={settings} setSettings={setSettings} />;
  }

  if (view === "profile") {
    return (
      <div className={`flex flex-col h-full w-full ${enterTransitionClass} ${mounted ? "opacity-100" : "opacity-0"}`}>
        <div className="flex items-center px-4 py-3 shrink-0 border-b border-white/10">
          <button onClick={() => setView("account")} aria-label="Back" className="p-2 -ml-2 text-[#888] active:opacity-60 transition-opacity">
            <ArrowLeft size={20} />
          </button>
          <span className="font-black uppercase ml-2 text-[#f0f0f0] text-xl tracking-wider" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>Edit Profile</span>
        </div>
        <div className="p-6 flex flex-col gap-4">
          <label className="text-sm font-bold text-[#888]" style={{ fontFamily: "'DM Mono', monospace" }}>Full Name</label>
          <input
            type="text"
            value={editName}
            onChange={(e) => setEditName(e.target.value)}
            className="w-full px-4 py-3.5 rounded-sm bg-[#141414] border border-[#333] text-[#f0f0f0] focus:outline-none focus:border-[#c8ff00] transition-colors"
          />
          <button
            onClick={handleUpdateProfile}
            disabled={savingProfile}
            className="w-full mt-4 py-4 rounded-sm font-black uppercase tracking-widest text-base active:scale-95 transition-transform"
            style={{ fontFamily: "'Barlow Condensed', sans-serif", backgroundColor: "#c8ff00", color: "#0a0a0a" }}
          >
            {savingProfile ? "Saving..." : "Save Changes"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-full w-full ${enterTransitionClass} ${mounted ? "opacity-100" : "opacity-0"}`}>
      <div className="flex items-center px-4 py-3 shrink-0 border-b border-white/10">
        <button onClick={onBack} aria-label="Back" className="p-2 -ml-2 text-[#888] active:opacity-60 transition-opacity">
          <ArrowLeft size={20} />
        </button>
        <span className="font-black uppercase ml-2 text-[#f0f0f0] text-xl tracking-wider" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
          Your Account
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-8">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-full flex items-center justify-center shrink-0 bg-[#141414] border border-[#333]">
            <UserIcon size={28} color="#c8ff00" />
          </div>
          <div className="overflow-hidden flex-1">
            <h2 className="font-bold text-lg truncate text-[#f0f0f0] tracking-wide" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
              {user?.user_metadata?.full_name || user?.email?.split("@")[0] || "Athlete"}
            </h2>
            <p className="truncate text-xs text-[#888]" style={{ fontFamily: "'DM Mono', monospace" }}>{user?.email}</p>
          </div>
          <button 
            onClick={() => setView("profile")}
            className="text-xs uppercase font-bold text-[#c8ff00] px-3 py-1.5 rounded-sm bg-[#141414] border border-[#333]"
            style={{ fontFamily: "'DM Mono', monospace" }}
          >
            Edit
          </button>
        </div>

        <div className="flex flex-col gap-3">
          <button
            onClick={() => setView("settings")}
            className="w-full flex items-center justify-between px-4 py-4 rounded-sm bg-[#141414] border border-[#222] active:bg-[#1a1a1a] transition-colors"
          >
            <div className="flex items-center gap-3 text-[#f0f0f0] text-sm" style={{ fontFamily: "'DM Mono', monospace" }}>
              <Settings size={18} color="#888" />
              App Settings
            </div>
            <ChevronRight size={16} color="#888" />
          </button>

          <button
            onClick={exportCSV}
            className="w-full flex items-center justify-between px-4 py-4 rounded-sm bg-[#141414] border border-[#222] active:bg-[#1a1a1a] transition-colors"
          >
            <div className="flex items-center gap-3 text-[#f0f0f0] text-sm" style={{ fontFamily: "'DM Mono', monospace" }}>
              <Download size={18} color="#888" />
              Export Session Data (.CSV)
            </div>
            <ChevronRight size={16} color="#888" />
          </button>
        </div>

        <div className="mt-auto pt-8 pb-10">
          <button
            onClick={onLogout}
            className="w-full flex items-center justify-center gap-2 px-4 py-4 rounded-sm bg-[#ff3b3b]/10 border border-[#ff3b3b]/20 text-[#ff3b3b] font-bold uppercase tracking-wider text-base active:opacity-60 transition-opacity"
            style={{ fontFamily: "'Barlow Condensed', sans-serif" }}
          >
            <LogOut size={16} />
            Log Out
          </button>
        </div>
      </div>
    </div>
  );
}

// ── Logged Sessions Screen ────────────────────────────────────────────────────
function SessionsScreen({ logs, onBack, onRefresh }: { logs: SupabaseLog[]; onBack: () => void; onRefresh: () => void }) {
  const mounted = useEnterTransition();
  return (
    <div className={`flex flex-col h-full w-full ${enterTransitionClass} ${mounted ? "opacity-100" : "opacity-0"}`}>
      <div className="flex items-center justify-between px-4 py-3 shrink-0 border-b border-white/10">
        <div className="flex items-center">
          <button onClick={onBack} aria-label="Back" className="p-2 -ml-2 text-[#888] active:opacity-60 transition-opacity">
            <ArrowLeft size={20} />
          </button>
          <span className="font-black uppercase ml-2 text-[#f0f0f0] text-xl tracking-wider" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
            Logged Sessions
          </span>
        </div>
        <button onClick={onRefresh} aria-label="Refresh sessions" className="p-2 text-[#888] active:scale-90 transition-transform">
          <RotateCcw size={18} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {logs.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full opacity-50 mt-20">
            <BookCheck size={48} color="#555" className="mb-4" />
            <p className="text-center text-[#f0f0f0] text-sm" style={{ fontFamily: "'DM Sans', sans-serif" }}>
              No workouts logged yet.<br />Use the camera to record your sets!
            </p>
          </div>
        ) : (
          <div className="flex flex-col gap-3 pb-20">
            {logs.map((log, i) => (
              <div key={log.id || i} className="p-5 rounded-sm bg-[#141414] border border-white/10 flex items-center justify-between">
                <div>
                  <div className="font-bold text-lg uppercase text-[#f0f0f0] tracking-wide" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
                    {log.exercise}
                  </div>
                  <div className="text-xs text-[#888] mt-1" style={{ fontFamily: "'DM Mono', monospace" }}>
                    {formatDate(log.created_at)} · {formatTime(log.created_at)}
                  </div>
                </div>
                <div className="flex flex-col items-end">
                  <span className="font-black text-4xl text-[#c8ff00] leading-none" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
                    {log.reps}
                  </span>
                  <span className="uppercase text-[10px] font-bold text-[#888] tracking-widest mt-1" style={{ fontFamily: "'DM Mono', monospace" }}>
                    reps
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ── AI Camera Screen ──────────────────────────────────────────────────────────
export function CameraScreen({
  exerciseKey,
  onClose,
  onSave,
  cameraFacing,
  audioFeedback,
}: {
  exerciseKey: ExerciseKey;
  onClose: () => void;
  onSave: (reps: number) => void;
  cameraFacing: "user" | "environment";
  audioFeedback: boolean;
}) {
  const ex = EXERCISES[exerciseKey];
  const [reps, setReps] = useState(0);
  const [active, setActive] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const poseLandmarkerRef = useRef<PoseLandmarker | null>(null);

  const repState = useRef<"up" | "down">("up");
  const lockedSideRef = useRef<Side | null>(null);
  const smoothedAngleRef = useRef<number | null>(null);
  const confirmFramesRef = useRef(0);
  const CONFIRM_FRAMES = 3;
  const lastRepTimeRef = useRef(0);
  const REP_COOLDOWN_MS = 400;

  useEffect(() => {
    let cancelled = false;
    getPoseLandmarker()
      .then((landmarker) => {
        if (cancelled) return;
        poseLandmarkerRef.current = landmarker;
        setIsLoaded(true);
      })
      .catch((err) => {
        console.error("MediaPipe init error:", err);
        if (!cancelled) setCameraError("Couldn't load the AI model. Check your connection and try again.");
      });
    return () => {
      cancelled = true;
      // NOTE: intentionally not closing the shared landmarker here — it's a
      // module-level singleton reused across camera sessions.
    };
  }, []);

  useEffect(() => {
    if (!active || !isLoaded || !videoRef.current) return;
    let animationFrameId: number;
    const video = videoRef.current;

    repState.current = "up";
    lockedSideRef.current = null;
    smoothedAngleRef.current = null;
    confirmFramesRef.current = 0;
    lastRepTimeRef.current = 0;

    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: cameraFacing, width: { ideal: 960 }, height: { ideal: 720 } },
        });
        video.srcObject = stream;
        await video.play();
        setCameraError(null);
      } catch (err) {
        console.error("Camera access error:", err);
        setCameraError("Couldn't access the camera. Check your browser permissions.");
        setActive(false);
      }
    }

    async function processVideoFrame() {
      if (video.videoWidth > 0 && poseLandmarkerRef.current && canvasRef.current) {
        if (canvasRef.current.width !== video.videoWidth || canvasRef.current.height !== video.videoHeight) {
          canvasRef.current.width = video.videoWidth;
          canvasRef.current.height = video.videoHeight;
        }

        const startTimeMs = performance.now();
        const results = poseLandmarkerRef.current.detectForVideo(video, startTimeMs);
        const canvasCtx = canvasRef.current.getContext("2d");

        if (canvasCtx && results.landmarks) {
          canvasCtx.save();
          canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
          const drawingUtils = new DrawingUtils(canvasCtx);

          for (const landmark of results.landmarks) {
            drawingUtils.drawConnectors(landmark, PoseLandmarker.POSE_CONNECTIONS, { color: ex.color, lineWidth: 5 });
            drawingUtils.drawLandmarks(landmark, { color: "#ffffff", radius: 6 });
          }
          canvasCtx.restore();

          if (results.landmarks[0]) {
            const landmarks = results.landmarks[0] as Landmark[];
            const joints = getActiveJoints(landmarks, exerciseKey, lockedSideRef.current);

            if (joints.visibility > ex.minVisibility) {
              const isValidForm = ex.validateForm ? ex.validateForm(landmarks) : true;

              if (isValidForm) {
                const rawAngle = calculateAngle(joints.a, joints.b, joints.c);

                const alpha = 0.35;
                smoothedAngleRef.current =
                  smoothedAngleRef.current == null
                    ? rawAngle
                    : smoothedAngleRef.current + alpha * (rawAngle - smoothedAngleRef.current);
                const angle = smoothedAngleRef.current;

                const now = performance.now();
                let stateChanged = false;

                if (repState.current === "up" && angle < ex.downThreshold) {
                  confirmFramesRef.current++;
                  if (confirmFramesRef.current >= CONFIRM_FRAMES) {
                    repState.current = "down";
                    lockedSideRef.current = joints.side;
                    confirmFramesRef.current = 0;
                    stateChanged = true;
                  }
                } else if (repState.current === "down" && angle > ex.upThreshold) {
                  confirmFramesRef.current++;
                  if (confirmFramesRef.current >= CONFIRM_FRAMES) {
                    if (now - lastRepTimeRef.current > REP_COOLDOWN_MS) {
                      setReps((prev) => prev + 1);
                      lastRepTimeRef.current = now;
                      if (audioFeedback) playRepTone();
                    }
                    repState.current = "up";
                    lockedSideRef.current = null;
                    confirmFramesRef.current = 0;
                    stateChanged = true;
                  }
                }

                if (!stateChanged) {
                  const movingTowardDown = repState.current === "up" && angle >= ex.downThreshold;
                  const movingTowardUp = repState.current === "down" && angle <= ex.upThreshold;
                  if (movingTowardDown || movingTowardUp) confirmFramesRef.current = 0;
                }
              }
            }
          }
        }
      }
      animationFrameId = requestAnimationFrame(processVideoFrame);
    }

    video.addEventListener("loadeddata", processVideoFrame);
    startCamera();

    return () => {
      cancelAnimationFrame(animationFrameId);
      video.removeEventListener("loadeddata", processVideoFrame);
      if (video.srcObject) {
        (video.srcObject as MediaStream).getTracks().forEach((track) => track.stop());
      }
    };
  }, [active, isLoaded, exerciseKey, ex.color, ex.downThreshold, ex.upThreshold, ex.minVisibility, ex.validateForm, cameraFacing, audioFeedback]);

  const mirrorStyle = cameraFacing === "user" ? { transform: "scaleX(-1)" } : undefined;

  return (
    <div className="flex-1 flex flex-col w-full h-full bg-[#0a0a0a]">
      <div className="flex items-center justify-between px-4 py-3 shrink-0">
        <button
          onClick={() => {
            setActive(false);
            onClose();
          }}
          aria-label="Back"
          className="flex items-center gap-1.5 p-1 text-[#888] active:opacity-60 transition-opacity text-sm"
          style={{ fontFamily: "'DM Sans', sans-serif" }}
        >
          <ArrowLeft size={16} /> Back
        </button>
        <span className="font-black uppercase text-lg tracking-wider" style={{ fontFamily: "'Barlow Condensed', sans-serif", color: ex.color }}>
          {ex.label}
        </span>
        {reps > 0 ? (
          <button
            onClick={() => {
              setActive(false);
              onSave(reps);
            }}
            className="px-3 py-1.5 rounded-sm font-bold uppercase text-xs text-[#0a0a0a] active:scale-95 transition-transform tracking-wider"
            style={{ fontFamily: "'Barlow Condensed', sans-serif", backgroundColor: ex.color }}
          >
            Save
          </button>
        ) : (
          <div className="w-16" />
        )}
      </div>

      <div className="relative flex-1 mx-3 rounded-sm overflow-hidden bg-[#0d0d0d]" style={{ border: `1px solid ${ex.color}30` }}>
        <video ref={videoRef} className="absolute inset-0 w-full h-full object-cover" style={mirrorStyle} playsInline muted autoPlay />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-cover z-10" style={mirrorStyle} />

        {!active && reps === 0 && !cameraError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 z-20 bg-[#0d0d0d]">
            <Camera size={40} color={`${ex.color}30`} />
            <p className="text-xs text-center px-12 text-[#666]" style={{ fontFamily: "'DM Sans', sans-serif" }}>
              Position yourself in frame, then tap Start
            </p>
          </div>
        )}

        {cameraError && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 z-20 bg-[#0d0d0d] px-8" role="alert">
            <p className="text-sm text-center text-[#ff6b6b]" style={{ fontFamily: "'DM Sans', sans-serif" }}>
              {cameraError}
            </p>
            <button
              onClick={() => setCameraError(null)}
              className="px-4 py-2 rounded-sm text-xs font-bold uppercase text-[#0a0a0a]"
              style={{ backgroundColor: ex.color }}
            >
              Try Again
            </button>
          </div>
        )}

        {active && (
          <div className="absolute top-3 right-3 flex items-center gap-1.5 px-2.5 py-1 rounded-sm z-20 bg-[#ff3b3b]/90">
            <span className="w-1.5 h-1.5 rounded-full bg-white motion-safe:animate-pulse" />
            <span className="text-white font-bold text-[11px]" style={{ fontFamily: "'DM Mono', monospace" }}>
              LIVE
            </span>
          </div>
        )}

        {/* Updated: Moved to bottom-left corner so it doesn't obstruct the user */}
        <div className="absolute bottom-5 left-5 flex flex-col items-center px-6 py-3 rounded-sm z-20 bg-black/70 backdrop-blur-md">
          <span
            className="font-black leading-none tabular-nums text-[88px]"
            style={{ fontFamily: "'Barlow Condensed', sans-serif", color: ex.color }}
            aria-live="polite"
          >
            {reps}
          </span>
          <span className="uppercase tracking-widest text-[10px] text-[#888]" style={{ fontFamily: "'DM Mono', monospace" }}>
            reps
          </span>
        </div>
      </div>

      <div className="shrink-0 px-4 pt-4 pb-5 flex flex-col gap-2.5">
        <button
          disabled={!isLoaded}
          onClick={() => setActive((a) => !a)}
          className="w-full h-14 rounded-sm flex items-center justify-center gap-2.5 font-black uppercase tracking-wider text-base active:scale-95 transition-transform disabled:opacity-50"
          style={{ fontFamily: "'Barlow Condensed', sans-serif", backgroundColor: active ? "#ff3b3b" : ex.color, color: active ? "#fff" : "#0a0a0a" }}
        >
          {active ? (
            <>
              <Square size={17} fill="currentColor" /> Stop
            </>
          ) : (
            <>
              <Camera size={17} /> {isLoaded ? "Start Counting" : "Loading AI..."}
            </>
          )}
        </button>
        {reps > 0 && (
          <button
            onClick={() => {
              setActive(false);
              setReps(0);
            }}
            className="flex items-center justify-center gap-1.5 py-2 text-[#888] text-sm active:opacity-50 transition-opacity"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            <RotateCcw size={12} /> Reset
          </button>
        )}
      </div>
    </div>
  );
}

// ── Exercise Card Component ───────────────────────────────────────────────────
function ExerciseCard({ exerciseKey, count, onOpen, onReset }: { exerciseKey: ExerciseKey; count: number; onOpen: () => void; onReset: () => void }) {
  const ex = EXERCISES[exerciseKey];
  return (
    <div className="rounded-sm border flex flex-col gap-4 p-5 transition-all" style={{ backgroundColor: ex.bgColor, borderColor: `${ex.color}22` }}>
      <div className="flex items-start justify-between">
        <div>
          <div className="text-xl mb-1">{ex.emoji}</div>
          <h2 className="uppercase font-extrabold leading-none text-[26px] tracking-wide" style={{ fontFamily: "'Barlow Condensed', sans-serif", color: ex.color }}>{ex.label}</h2>
        </div>
        <button onClick={onReset} aria-label={`Reset ${ex.label} count`} className="p-1 text-[#888] active:scale-90 transition-transform">
          <RotateCcw size={13} />
        </button>
      </div>
      <div className="flex flex-col items-center py-1">
        <span className="font-black leading-none tabular-nums text-[80px]" style={{ fontFamily: "'Barlow Condensed', sans-serif", color: ex.color }}>{count}</span>
        <span className="uppercase tracking-widest text-[10px] text-[#888] mt-1" style={{ fontFamily: "'DM Mono', monospace" }}>reps</span>
      </div>
      <button onClick={onOpen} className="w-full h-12 rounded-sm flex items-center justify-center gap-2 font-black uppercase tracking-wider text-sm text-[#0a0a0a] active:scale-95 transition-transform" style={{ fontFamily: "'Barlow Condensed', sans-serif", backgroundColor: ex.color }}>
        <Camera size={15} /> Use Camera
      </button>
    </div>
  );
}

// ── Authentication Screen ─────────────────────────────────────────────────────
const ONBOARDING_SLIDES = [
  {
    title: "TRACK EVERY REP",
    description: "Advanced on-device AI vision counts your reps automatically.",
    color: "#c8ff00",
    icon: <ScanEye size={64} color="#c8ff00" />,
  },
  {
    title: "PERFECT YOUR FORM",
    description: "Real-time skeleton tracking ensures you hit the right angles.",
    color: "#00e5ff",
    icon: <Activity size={64} color="#00e5ff" />,
  },
  {
    title: "LOG YOUR PROGRESS",
    description: "Save your session history and export your data anytime.",
    color: "#ff6b6b",
    icon: <BookCheck size={64} color="#ff6b6b" />,
  },
];

// FIX: map raw Supabase auth errors to friendlier copy instead of showing
// SDK error strings verbatim.
function friendlyAuthError(message: string): string {
  const m = message.toLowerCase();
  if (m.includes("invalid login credentials")) return "That email or password isn't right. Try again.";
  if (m.includes("email not confirmed")) return "Please confirm your email before logging in — check your inbox.";
  if (m.includes("user already registered")) return "An account with that email already exists. Try logging in instead.";
  if (m.includes("password should be at least")) return "Password is too short — use at least 6 characters.";
  if (m.includes("rate limit")) return "Too many attempts. Please wait a moment and try again.";
  return message;
}

function AuthScreen() {
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [isSignUp, setIsSignUp] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [signUpSuccessMessage, setSignUpSuccessMessage] = useState<string | null>(null);

  const [currentSlide, setCurrentSlide] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % ONBOARDING_SLIDES.length);
    }, 3500);
    return () => clearInterval(timer);
  }, []);

  const handleAuth = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setSignUpSuccessMessage(null);
    try {
      if (isSignUp) {
        const { data, error } = await supabase.auth.signUp({
          email,
          password,
          options: { data: { full_name: fullName } },
        });
        if (error) throw error;
        // FIX: don't assume email confirmation is off — Supabase returns no
        // session when confirmation is required, so branch on that instead
        // of always claiming "you can now log in."
        if (!data.session) {
          setSignUpSuccessMessage("Account created! Check your email to confirm before logging in.");
        } else {
          setSignUpSuccessMessage("Account created — you're all set.");
        }
        setIsSignUp(false);
      } else {
        const { error } = await supabase.auth.signInWithPassword({ email, password });
        if (error) throw error;
      }
    } catch (err: any) {
      setError(friendlyAuthError(err.message ?? "Something went wrong. Please try again."));
    } finally {
      setLoading(false);
    }
  };

  const handleForgotPassword = async () => {
    if (!email) {
      setError("Please enter your email address first.");
      return;
    }
    setLoading(true);
    setError(null);
    const { error } = await supabase.auth.resetPasswordForEmail(email);
    setLoading(false);
    
    if (error) setError(error.message);
    else setSignUpSuccessMessage("Password reset email sent! Check your inbox.");
  };

  const slide = ONBOARDING_SLIDES[currentSlide];

  return (
    <div className="flex flex-col items-center justify-between h-full w-full bg-[#0a0a0a] px-6 py-10">
      <div className="flex items-center gap-3 shrink-0">
        <div className="w-10 h-10 rounded-sm flex items-center justify-center font-black bg-[#c8ff00] text-[#0a0a0a] text-xl" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>R</div>
        <span className="font-black uppercase text-3xl tracking-wider text-[#f0f0f0]" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
          Rep<span className="text-[#c8ff00]">.io</span>
        </span>
      </div>

      <div className="flex-1 flex flex-col items-center justify-center w-full max-w-sm mt-4">
        <div className={`h-40 flex items-center justify-center ${enterTransitionClass} duration-500`}>
          {slide.icon}
        </div>

        <h2 className={`font-black text-2xl tracking-wide uppercase mt-6 ${enterTransitionClass} duration-500`} style={{ fontFamily: "'Barlow Condensed', sans-serif", color: slide.color }}>
          {slide.title}
        </h2>
        <p className="text-center text-sm text-[#888] mt-3 px-4 h-12" style={{ fontFamily: "'DM Sans', sans-serif" }}>
          {slide.description}
        </p>

        <div className="flex items-center gap-2 mt-6" role="tablist" aria-label="Onboarding slides">
          {ONBOARDING_SLIDES.map((s, index) => (
            <button
              key={index}
              role="tab"
              aria-selected={currentSlide === index}
              aria-label={`Slide ${index + 1}: ${s.title}`}
              onClick={() => setCurrentSlide(index)}
              className="h-1.5 rounded-full transition-all duration-300"
              style={{
                width: currentSlide === index ? 24 : 6,
                backgroundColor: currentSlide === index ? slide.color : "#333",
              }}
            />
          ))}
        </div>
      </div>

      <form onSubmit={handleAuth} className="w-full max-w-sm flex flex-col gap-4 shrink-0 mt-8">
        <div className="flex flex-col gap-3">
          {isSignUp && (
            <input
              type="text"
              placeholder="Full Name"
              aria-label="Full Name"
              value={fullName}
              onChange={(e) => setFullName(e.target.value)}
              required={isSignUp}
              className="w-full px-4 py-3.5 rounded-sm bg-[#141414] border border-[#333] text-[#f0f0f0] placeholder-[#666] focus:outline-none focus:border-[#c8ff00] transition-colors"
              style={{ fontFamily: "'DM Mono', monospace", fontSize: 14 }}
            />
          )}
          <input
            type="email"
            placeholder="Email Address"
            aria-label="Email Address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="w-full px-4 py-3.5 rounded-sm bg-[#141414] border border-[#333] text-[#f0f0f0] placeholder-[#666] focus:outline-none focus:border-[#c8ff00] transition-colors"
            style={{ fontFamily: "'DM Mono', monospace", fontSize: 14 }}
          />
          <input
            type="password"
            placeholder="Password"
            aria-label="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="w-full px-4 py-3.5 rounded-sm bg-[#141414] border border-[#333] text-[#f0f0f0] placeholder-[#666] focus:outline-none focus:border-[#c8ff00] transition-colors"
            style={{ fontFamily: "'DM Mono', monospace", fontSize: 14 }}
          />
        </div>
        
        {!isSignUp && (
          <div className="flex justify-end w-full">
            <button
              type="button"
              onClick={handleForgotPassword}
              className="text-xs text-[#888] active:text-[#f0f0f0] transition-colors"
              style={{ fontFamily: "'DM Sans', sans-serif" }}
            >
              Forgot Password?
            </button>
          </div>
        )}

        {error && (
          <p className="text-[#ff3b3b] text-xs text-center font-bold tracking-wide" role="alert" style={{ fontFamily: "'DM Mono', monospace" }}>
            {error}
          </p>
        )}
        {signUpSuccessMessage && (
          <p className="text-[#c8ff00] text-xs text-center font-bold tracking-wide" role="status" style={{ fontFamily: "'DM Mono', monospace" }}>
            {signUpSuccessMessage}
          </p>
        )}

        <button
          type="submit"
          disabled={loading}
          className="w-full mt-2 py-4 rounded-sm font-black uppercase tracking-widest text-base disabled:opacity-50 active:scale-95 transition-all"
          style={{ fontFamily: "'Barlow Condensed', sans-serif", backgroundColor: "#c8ff00", color: "#0a0a0a" }}
        >
          {loading ? "Processing..." : isSignUp ? "Create Account" : "Log In"}
        </button>

        <button
          type="button"
          onClick={() => {
            setIsSignUp(!isSignUp);
            setError(null);
            setSignUpSuccessMessage(null);
          }}
          className="text-[#888] text-sm mt-3 pb-2 active:text-[#f0f0f0] transition-colors"
          style={{ fontFamily: "'DM Sans', sans-serif" }}
        >
          {isSignUp ? "Already have an account? Log in" : "Need an account? Sign up"}
        </button>
      </form>
    </div>
  );
}

// ── Main App Controller ───────────────────────────────────────────────────────
export default function App() {
  const [currentView, setCurrentView] = useState<"app" | "sessions" | "account">("app");
  const [user, setUser] = useState<User | null>(null);
  const [logs, setLogs] = useState<SupabaseLog[]>([]);
  const [camera, setCamera] = useState<ExerciseKey | null>(null);
  const [counts, setCounts] = useState<Record<ExerciseKey, number>>({ pushups: 0, squats: 0, deadlifts: 0 });
  const [saveError, setSaveError] = useState<string | null>(null); // FIX: surfaced instead of console-only

  const [settings, setSettings] = useState<AppSettings>({
    audioFeedback: true,
    cameraFacing: "user",
  });

  const fetchLogs = async (userId: string) => {
    try {
      const { data, error } = await supabase
        .from("workout_logs")
        .select("*")
        .eq("user_id", userId)
        .order("created_at", { ascending: false });

      if (error) console.error("Error fetching logs:", error);
      else if (data) setLogs(data);
    } catch (err) {
      console.error("Fetch logs failed:", err);
    }
  };

  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session?.user) {
        setUser(session.user);
        fetchLogs(session.user.id);
      }
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null);
      if (session?.user) fetchLogs(session.user.id);
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleCameraSave = async (key: ExerciseKey, reps: number) => {
    setCounts((c) => ({ ...c, [key]: c[key] + reps }));
    setSaveError(null);

    if (user) {
      try {
        const { data, error } = await supabase
          .from("workout_logs")
          .insert([{ user_id: user.id, exercise: EXERCISES[key].label, reps }])
          .select();

        if (error) {
          console.error("Supabase insert error:", error);
          setSaveError("Your reps were counted, but saving to your history failed. Check your connection.");
        } else if (data) {
          setLogs((prev) => [data[0], ...prev]);
        }
      } catch (err) {
        console.error("Failed to insert log:", err);
        setSaveError("Your reps were counted, but saving to your history failed. Check your connection.");
      }
    }
    setCamera(null);
  };

  const handleLogout = async () => {
    await supabase.auth.signOut();
    setUser(null);
    setLogs([]);
    setCounts({ pushups: 0, squats: 0, deadlifts: 0 });
    setCurrentView("app");
  };

  // ── Determine Active View ───────────────────────────────────────────────────

  if (!user) {
    return (
      <div
        className="flex flex-col w-full overflow-hidden bg-[#0a0a0a] text-[#f0f0f0]"
        style={{
          height: "100dvh",
          paddingTop: "max(env(safe-area-inset-top), 24px)",
          paddingBottom: "env(safe-area-inset-bottom)",
        }}
      >
        <AuthScreen />
      </div>
    );
  }

  let activeContent;
  if (camera) {
    activeContent = (
      <CameraScreen
        exerciseKey={camera}
        onClose={() => setCamera(null)}
        onSave={(reps) => handleCameraSave(camera, reps)}
        cameraFacing={settings.cameraFacing}
        audioFeedback={settings.audioFeedback}
      />
    );
  } else if (currentView === "account") {
    activeContent = <AccountScreen user={user} logs={logs} settings={settings} setSettings={setSettings} onBack={() => setCurrentView("app")} onLogout={handleLogout} />;
  } else if (currentView === "sessions") {
    activeContent = <SessionsScreen logs={logs} onBack={() => setCurrentView("app")} onRefresh={() => user && fetchLogs(user.id)} />;
  } else {
    activeContent = (
      <div className="flex flex-col w-full h-full">
        <div className="flex items-center justify-between px-5 py-3 shrink-0 border-b border-white/10">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 rounded-sm flex items-center justify-center font-black bg-[#c8ff00] text-[#0a0a0a] text-[15px]" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>R</div>
            <span className="font-black uppercase text-xl tracking-wider text-[#f0f0f0]" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
              Rep<span className="text-[#c8ff00]">.io</span>
            </span>
          </div>
          <div className="flex items-center gap-3 text-[#f0f0f0]">
            <button onClick={() => setCurrentView("sessions")} aria-label="View logged sessions" className="p-2 -mr-1 rounded-sm active:bg-[#1a1a1a] transition-colors flex items-center justify-center">
              <History size={18} />
            </button>
            <button onClick={() => setCurrentView("account")} aria-label="View account" className="p-2 -mr-1 rounded-sm active:bg-[#1a1a1a] transition-colors flex items-center justify-center">
              <UserIcon size={18} />
            </button>
          </div>
        </div>

        {saveError && (
          <div role="alert" className="mx-4 mt-3 px-4 py-3 rounded-sm bg-[#ff3b3b]/10 border border-[#ff3b3b]/20 text-[#ff6b6b] text-xs flex items-center justify-between gap-3" style={{ fontFamily: "'DM Mono', monospace" }}>
            <span>{saveError}</span>
            <button onClick={() => setSaveError(null)} aria-label="Dismiss" className="text-[#ff6b6b] font-bold px-1">×</button>
          </div>
        )}

        <div className="flex-1 overflow-y-auto px-4 pt-5 pb-20 flex flex-col gap-6">
          {logs.length > 0 && (
            <div
              onClick={() => setCurrentView("sessions")}
              className="relative p-5 rounded-sm bg-[#141414] border border-[#c8ff00]/20 flex items-center justify-between active:scale-95 transition-transform cursor-pointer min-h-[85px]"
            >
              <div className="absolute top-0 left-0 w-1.5 h-full bg-[#c8ff00] rounded-l-sm" />
              <div className="pl-1 flex flex-col justify-center">
                <div className="text-[10px] font-bold text-[#c8ff00] uppercase tracking-widest mb-1.5 leading-none" style={{ fontFamily: "'DM Mono', monospace" }}>Recent Session</div>
                <div className="font-bold text-xl uppercase text-[#f0f0f0] tracking-wide leading-none" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
                  {logs[0].exercise}
                </div>
              </div>
              <div className="flex flex-col items-end justify-center">
                <span className="font-black text-3xl text-[#f0f0f0] leading-none mb-0.5" style={{ fontFamily: "'Barlow Condensed', sans-serif" }}>
                  {logs[0].reps}
                </span>
                <span className="uppercase text-[10px] font-bold text-[#888] tracking-widest leading-none" style={{ fontFamily: "'DM Mono', monospace" }}>
                  reps
                </span>
              </div>
            </div>
          )}

          <div className="flex flex-col gap-4">
            {(Object.keys(EXERCISES) as ExerciseKey[]).map((key) => (
              <ExerciseCard key={key} exerciseKey={key} count={counts[key]} onOpen={() => setCamera(key)} onReset={() => setCounts((c) => ({ ...c, [key]: 0 }))} />
            ))}
          </div>
          <p className="text-center pb-8 text-[10px] text-[#444]" style={{ fontFamily: "'DM Mono', monospace" }}>rep.io — track every rep</p>
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex flex-col w-full overflow-hidden bg-[#0a0a0a] text-[#f0f0f0]"
      style={{
        height: "100dvh",
        paddingTop: "max(env(safe-area-inset-top), 24px)",
        paddingBottom: "env(safe-area-inset-bottom)",
      }}
    >
      {activeContent}
    </div>
  );
}