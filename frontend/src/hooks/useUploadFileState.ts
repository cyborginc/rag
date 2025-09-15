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

import { useState, useCallback } from "react";
import { useFileValidation } from "./useFileValidation";
import { useFileUtils } from "./useFileUtils";

/**
 * Interface representing a file in the upload state.
 */
export interface UploadFile {
  id: string;
  file: File;
  status: 'uploading' | 'uploaded' | 'error';
  progress: number;
  errorMessage?: string;
}

/**
 * Configuration options for the upload file state hook.
 */
interface UploadFileStateOptions {
  acceptedTypes: string[];
  maxFileSize: number;
  maxFiles: number;
  audioFileMaxSize?: number; // in MB, for .mp3 and .wav files
  onFilesChange?: (files: File[]) => void;
}

/**
 * Custom hook for managing file upload state with validation and progress tracking.
 * 
 * @param options - Configuration options for file uploads
 * @returns Object with upload files, validation methods, and file management functions
 * 
 * @example
 * ```tsx
 * const { uploadFiles, addFiles, removeFile, clearFiles } = useUploadFileState({
 *   acceptedTypes: ['.pdf', '.txt'],
 *   maxFileSize: 10 * 1024 * 1024, // 10MB
 *   maxFiles: 5
 * });
 * ```
 */
export const useUploadFileState = ({ 
  acceptedTypes, 
  maxFileSize, 
  maxFiles, 
  audioFileMaxSize,
  onFilesChange 
}: UploadFileStateOptions) => {
  const [uploadFiles, setUploadFiles] = useState<UploadFile[]>([]);
  const { validateFile } = useFileValidation({ acceptedTypes, maxFileSize, audioFileMaxSize });
  const { generateFileId } = useFileUtils();

  const addFiles = useCallback((files: FileList | File[]) => {
    const newFiles: UploadFile[] = [];
    const validFiles: File[] = [];

    Array.from(files).forEach((file) => {
      if (uploadFiles.length + newFiles.length >= maxFiles) {
        return;
      }

      const error = validateFile(file);
      const uploadFile: UploadFile = {
        id: generateFileId(),
        file,
        status: error ? 'error' : 'uploaded',
        progress: error ? 0 : 100,
        errorMessage: error || undefined,
      };

      newFiles.push(uploadFile);
      if (!error) {
        validFiles.push(file);
      }
    });

    setUploadFiles(prev => [...prev, ...newFiles]);
    if (validFiles.length > 0) {
      onFilesChange?.(validFiles);
    }
  }, [uploadFiles.length, maxFiles, validateFile, onFilesChange, generateFileId]);

  const removeFile = useCallback((id: string) => {
    setUploadFiles(prev => prev.filter(f => f.id !== id));
    
    const remainingFiles = uploadFiles
      .filter(f => f.id !== id && f.status !== 'error')
      .map(f => f.file);
    
    onFilesChange?.(remainingFiles);
  }, [uploadFiles, onFilesChange]);

  const clearAllFiles = useCallback(() => {
    setUploadFiles([]);
    onFilesChange?.([]);
  }, [onFilesChange]);

  const getValidFiles = useCallback(() => {
    return uploadFiles
      .filter(f => f.status !== 'error')
      .map(f => f.file);
  }, [uploadFiles]);

  return {
    uploadFiles,
    addFiles,
    removeFile,
    clearAllFiles,
    getValidFiles,
  };
}; 