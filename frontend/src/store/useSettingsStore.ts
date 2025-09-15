// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { useHealthStatus } from "../api/useHealthApi";

/**
 * Interface defining the shape of the settings state.
 * Contains RAG configuration, feature toggles, model settings, and endpoints.
 * Most fields are optional to avoid sending unnecessary defaults in API calls.
 */
interface SettingsState {
  // RAG Configuration - All optional, only sent if user has configured them
  temperature?: number;
  topP?: number;
  maxTokens?: number;
  vdbTopK?: number;
  rerankerTopK?: number;
  confidenceScoreThreshold?: number;
  
  // Feature Toggles - Only enableReranker and includeCitations default to true
  enableQueryRewriting?: boolean;
  enableReranker: boolean;
  useGuardrails?: boolean;
  includeCitations: boolean;
  enableVlmInference?: boolean;
  enableFilterGenerator?: boolean;
  
  // Models - All optional, populated from health endpoint or user input
  model?: string;
  embeddingModel?: string;
  rerankerModel?: string;
  vlmModel?: string;
  
  // Endpoints - All optional, populated from health endpoint or user input
  llmEndpoint?: string;
  embeddingEndpoint?: string;
  rerankerEndpoint?: string;
  vlmEndpoint?: string;
  vdbEndpoint?: string;
  
  // Theme - Required
  theme: 'light' | 'dark';
  
  // Other - Optional
  stopTokens?: string[];
  useLocalStorage?: boolean;
  
  set: (values: Partial<SettingsState>) => void;
}

/**
 * Zustand store for application settings with persistence.
 * 
 * Manages RAG configuration, feature toggles, model settings, and API endpoints.
 * Settings are automatically persisted to localStorage.
 * 
 * @returns Settings store with state and setter function
 * 
 * @example
 * ```tsx
 * const { temperature, enableReranker, set } = useSettingsStore();
 * set({ temperature: 0.7, enableReranker: true });
 * ```
 */
export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      // RAG Configuration - All start undefined (empty)
      temperature: undefined,
      topP: undefined,
      maxTokens: undefined,
      vdbTopK: undefined,
      rerankerTopK: undefined,
      confidenceScoreThreshold: undefined,
      
      // Feature Toggles - Only enableReranker and includeCitations default to true
      enableQueryRewriting: undefined,
      enableReranker: true,
      useGuardrails: undefined,
      includeCitations: true,
      enableVlmInference: undefined,
      enableFilterGenerator: undefined,
      
      // Models - All start undefined, will be populated from health endpoint
      model: undefined,
      embeddingModel: undefined,
      rerankerModel: undefined,
      vlmModel: undefined,
      
      // Endpoints - All start undefined, will be populated from health endpoint
      llmEndpoint: undefined,
      embeddingEndpoint: undefined,
      rerankerEndpoint: undefined,
      vlmEndpoint: undefined,
      vdbEndpoint: undefined,
      
      // Theme - Required field
      theme: 'dark' as const,
      
      // Other - Optional
      stopTokens: undefined,
      useLocalStorage: undefined,
      
      set: (values) => set(values),
    }),
    {
      name: "rag-settings",
      storage: {
        getItem: (name) => {
          const str = localStorage.getItem(name);
          if (!str) return null;
          return JSON.parse(str);
        },
        setItem: (name, value) => {
          // Only save if useLocalStorage is explicitly enabled
          const state = JSON.parse(JSON.stringify(value.state));
          if (state?.useLocalStorage === true) {
            localStorage.setItem(name, JSON.stringify(value));
          } else {
            // Clear localStorage if disabled or undefined
            localStorage.removeItem(name);
          }
        },
        removeItem: (name) => localStorage.removeItem(name),
      },
    }
  )
);


/**
 * Hook to get application health status.
 * Used for monitoring service health without performing any settings initialization.
 * Models and endpoints remain empty until explicitly set by users.
 */
export function useAppHealthStatus() {
  const { data: health, isLoading, error } = useHealthStatus();

  // Returns health status data for application monitoring
  return { health, isLoading, error };
}

/**
 * Hook to check if health-dependent features should be disabled.
 * Returns true if health checks are loading or have failed.
 */
export function useHealthDependentFeatures() {
  const { data: health, isLoading, error } = useHealthStatus();
  
  const isHealthLoading = isLoading;
  const hasHealthError = !!error;
  const hasHealthData = !!health;
  
  // Disable features if health is loading or has error
  const shouldDisableHealthFeatures = isHealthLoading || hasHealthError || !hasHealthData;
  
  return {
    isHealthLoading,
    hasHealthError, 
    hasHealthData,
    shouldDisableHealthFeatures,
    health,
    error
  };
}

