/*
23092026 - KHAI - Create context for persistent result while running pipeline
*/
import { createContext, useContext, useEffect, useState } from 'react';
import type { DetectionObject } from '../components/DetectionResultImage';

export type InputMode = 'folder' | 'camera';
export type TaskMode = 'inspection' | 'detection';

export interface InferenceMetrics {
  latency_ms?: number;
  total_objects?: number;
  counts_by_class?: Record<string, number>;
  ng_count?: number;
  max_score?: number;
}

export interface InferenceResult {
  status?: string;
  message?: string;
  detail?: string;
  timestamp?: string;
  task?: 'inspection' | 'detection';
  task_type?: 'inspection' | 'detection';
  camera_id?: string;
  ng_detected?: boolean;
  original_image_url?: string;
  image_width?: number;
  image_height?: number;
  metrics?: InferenceMetrics;
  objects?: DetectionObject[];
  vis_urls?: {
    heatmap?: string;
    crops?: string;
    overall?: string;
  };
}

interface InspectionContextValue {
  inputMode: InputMode;
  setInputMode: (value: InputMode) => void;
  taskMode: TaskMode;
  setTaskMode: (value: TaskMode) => void;
  isInspecting: boolean;
  setIsInspecting: (value: boolean) => void;
  inputPreviewUrl: string | null;
  setInputPreviewUrl: (value: string | null) => void;
  latestResult: InferenceResult | null;
  setLatestResult: (value: InferenceResult | null) => void;
}

const InspectionContext = createContext<InspectionContextValue | null>(null);

const STORAGE_KEY = 'smartic-inspection-state';

interface PersistedInspectionState {
  inputMode: InputMode;
  taskMode: TaskMode;
  inputPreviewUrl: string | null;
  latestResult: InferenceResult | null;
}

function loadPersistedState(): Partial<PersistedInspectionState> {
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : {};
  } catch {
    return {};
  }
}

export function InspectionProvider({ children }: { children: React.ReactNode }) {
  const [persistedState] = useState(loadPersistedState);
  const [inputMode, setInputMode] = useState<InputMode>(persistedState.inputMode || 'folder');
  const [taskMode, setTaskMode] = useState<TaskMode>(persistedState.taskMode || 'detection');
  const [isInspecting, setIsInspecting] = useState(false);
  const [inputPreviewUrl, setInputPreviewUrl] = useState<string | null>(
    persistedState.inputPreviewUrl || null,
  );
  const [latestResult, setLatestResult] = useState<InferenceResult | null>(
    persistedState.latestResult || null,
  );

  useEffect(() => {
    try {
      const state: PersistedInspectionState = {
        inputMode,
        taskMode,
        inputPreviewUrl,
        latestResult,
      };
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // Persistence is best effort; the in-memory state remains authoritative.
    }
  }, [inputMode, taskMode, inputPreviewUrl, latestResult]);

  return (
    <InspectionContext.Provider value={{
      inputMode,
      setInputMode,
      taskMode,
      setTaskMode,
      isInspecting,
      setIsInspecting,
      inputPreviewUrl,
      setInputPreviewUrl,
      latestResult,
      setLatestResult,
    }}>
      {children}
    </InspectionContext.Provider>
  );
}

export function useInspection() {
  const context = useContext(InspectionContext);
  if (!context) {
    throw new Error('useInspection must be used inside InspectionProvider');
  }
  return context;
}
