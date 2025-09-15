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

import React, { useState, useRef, useEffect, useMemo } from "react";
import type { Filter } from "../../types/chat";
import type { MetadataFieldType, Collection, APIMetadataField } from "../../types/collections";
import { useCollections } from "../../api/useCollectionsApi";
import { useMetadataValues } from "../../api/useMetadataValues";
import { useCollectionsStore } from "../../store/useCollectionsStore";

interface Props {
  filters: Filter[];
  setFilters: (filters: Filter[]) => void;
}

const OPS: Record<MetadataFieldType, string[]> = {
  string: ["=", "!=", "contains", "like", "in", "not in"],
  integer: ["=", "!=", ">", "<", ">=", "<=", "between", "in", "not in"],
  float: ["=", "!=", ">", "<", ">=", "<=", "between", "in", "not in"],
  number: ["=", "!=", ">", "<", ">=", "<=", "between", "in", "not in"],
  boolean: ["=", "!="],
  datetime: ["=", "!=", ">", "<", ">=", "<=", "between", "before", "after", "in", "not in"],
  array: ["in", "contains", "array_contains", "array_contains_all", "array_contains_any", "array_length"],
};

export default function FilterBar({ filters, setFilters }: Props) {
  const [input, setInput] = useState("");
  const [stage, setStage] = useState<"field" | "op" | "value">("field");
  const [draft, setDraft] = useState<{ field?: string; op?: string; value?: string }>({});
  const [activeIdx, setActiveIdx] = useState(0);
  const [show, setShow] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);

  const { data: collections = [] } = useCollections();
  const { selectedCollections } = useCollectionsStore();

  // Build metadata field map
  const fieldMeta = useMemo(() => {
    const map: Record<string, MetadataFieldType> = {};
  const source = selectedCollections.length
    ? collections.filter((c: Collection) => selectedCollections.includes(c.collection_name))
    : collections;
  source.forEach((col: Collection) => {
    (col.metadata_schema || []).forEach((f: APIMetadataField) => {
        map[f.name] = f.type || "string";
      });
    });
    return map;
  }, [collections, selectedCollections]);

  const availableFields = useMemo(() => Object.keys(fieldMeta).filter((f) => f !== "filename" && !filters.some((fi) => fi.field === f)), [fieldMeta, filters]);

  // Fetch distinct metadata values for current field across selected collections
  const { data: valueOptions = [], isLoading: valuesLoading, error: valuesError } = useMetadataValues(
    selectedCollections.length ? selectedCollections : collections.map((c: Collection) => c.collection_name),
    draft.field || ""
  );

  // Suggestions based on stage
  const suggestions: string[] = useMemo(() => {
    if (stage === "field") {
      const term = input.trim();
      return term ? availableFields.filter((f) => f.startsWith(term)) : availableFields;
    } else if (stage === "op" && draft.field) {
      return OPS[fieldMeta[draft.field]];
    } else if (stage === "value" && draft.field && draft.op) {
      // suggestions fetched asynchronously
      return valueOptions.filter((v) => v.startsWith(input));
    }
    return [];
  }, [stage, input, availableFields, draft, fieldMeta, valueOptions]);

  // Debug logging
  console.log('FilterBar Debug:', {
    stage,
    draftField: draft.field,
    draftOp: draft.op,
    valueOptions,
    valuesLoading,
    valuesError,
    input,
    suggestions: suggestions.length,
    show
  });

  // Convert string input value to appropriate type based on field metadata
  const convertValueToType = (value: string, fieldType: MetadataFieldType): string | number | boolean => {
    const trimmedValue = value.trim();
    
    switch (fieldType) {
      case "integer": {
        const intValue = parseInt(trimmedValue, 10);
        return isNaN(intValue) ? trimmedValue : intValue;
      }
      
      case "float":
      case "number": {
        const numValue = parseFloat(trimmedValue);
        return isNaN(numValue) ? trimmedValue : numValue;
      }
      
      case "boolean": {
        const lowerValue = trimmedValue.toLowerCase();
        if (lowerValue === "true") return true;
        if (lowerValue === "false") return false;
        return trimmedValue; // Keep as string if not a valid boolean
      }
      
      case "string":
      case "datetime":
      case "array":
      default:
        return trimmedValue;
    }
  };

  const commitFilter = () => {
    if (draft.field && draft.op && draft.value?.trim()) {
      const fieldType = fieldMeta[draft.field] || "string";
      const convertedValue = convertValueToType(draft.value, fieldType);
      
      const newFilter: Filter = {
        field: draft.field, 
        operator: draft.op as Filter['operator'], 
        value: convertedValue,
        // Set default logical operator to OR for filters after the first one
        ...(filters.length > 0 && { logicalOperator: "OR" }),
      };
      
      setFilters([...filters, newFilter]);
      setDraft({});
      setStage("field");
      setInput("");
    }
  };

  const updateFilterLogicalOperator = (index: number, logicalOperator: "AND" | "OR") => {
    const updatedFilters = [...filters];
    updatedFilters[index] = { ...updatedFilters[index], logicalOperator };
    setFilters(updatedFilters);
  };

  const chooseSuggestion = (idx: number) => {
    const choice = suggestions[idx];
    if (!choice) return;
    if (stage === "field") {
      setDraft({ field: choice });
      setStage("op");
      setInput("");
      // Immediately show operator suggestions
      setTimeout(() => {
        setShow(true);
        setActiveIdx(0);
      }, 0);
    } else if (stage === "op") {
      setDraft((d) => ({ ...d, op: choice }));
      setStage("value");
      setInput("");
      setShow(false);
    } else if (stage === "value") {
      setDraft((d) => ({ ...d, value: choice }));
      setInput(choice);
      setShow(false);
      // Auto-commit the filter when value is selected
      setTimeout(() => {
        if (draft.field && draft.op && choice.trim()) {
          const fieldType = fieldMeta[draft.field] || "string";
          const convertedValue = convertValueToType(choice, fieldType);
          
          setFilters([...filters, { 
            field: draft.field, 
            operator: draft.op as Filter['operator'], 
            value: convertedValue 
          }]);
          setDraft({});
          setStage("field");
          setInput("");
        }
      }, 0);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (stage !== "value" && show && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      e.preventDefault();
      setActiveIdx((prev) => {
        const next = e.key === "ArrowDown" ? prev + 1 : prev - 1;
        return (next + suggestions.length) % suggestions.length;
      });
    } else if (e.key === "Enter") {
      if (stage === "value") {
        commitFilter();
      } else if (show) {
        e.preventDefault();
        chooseSuggestion(activeIdx);
      }
    } else if (e.key === "Backspace" && !input) {
      if (stage === "value") {
        setStage("op");
        setDraft((d) => ({ field: d.field }));
      } else if (stage === "op") {
        setStage("field");
        setDraft({});
      } else if (stage === "field" && filters.length) {
        // pop last filter
        const updated = [...filters];
        const last = updated.pop()!;
        setFilters(updated);
        setDraft({ field: last.field, op: last.operator, value: String(last.value) });
        setStage("value");
        setInput(String(last.value));
      }
    }
  };

  // show suggestions control
  useEffect(() => {
    const inputFocused = document.activeElement === inputRef.current;
    if (stage === "value") {
      setShow(inputFocused && (valueOptions.length > 0 || valuesLoading));
    } else if (stage === "op" && draft.field) {
      // Always show operators when we have a field selected
      setShow(inputFocused && suggestions.length > 0);
    } else {
      setShow(inputFocused && suggestions.length > 0);
    }
    setActiveIdx(0);
  }, [stage, suggestions, valueOptions, valuesLoading, draft.field]);

  // close suggestions on outside click
  useEffect(() => {
    if (!show) return;
    const handler = (e: MouseEvent) => {
      if (!wrapperRef.current?.contains(e.target as Node)) {
        setShow(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [show]);

  return (
    <div ref={wrapperRef} className="relative rounded-xl border border-neutral-700 bg-neutral-800/80 backdrop-blur-sm">
      <div className="flex flex-wrap items-center gap-2 p-2.5 min-h-[44px]">
        {/* Filter icon and label */}
        <div className="flex items-center gap-1.5 text-xs text-gray-400 font-medium">
          <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c2.755 0 5.455.232 8.083.678.533.09.917.556.917 1.096v1.044a2.25 2.25 0 01-.659 1.591l-5.432 5.432a2.25 2.25 0 00-.659 1.591v2.927a2.25 2.25 0 01-1.244 2.013L9.75 21v-6.568a2.25 2.25 0 00-.659-1.591L3.659 7.409A2.25 2.25 0 013 5.818V4.774c0-.54.384-1.006.917-1.096A48.32 48.32 0 0112 3z" />
          </svg>
          Filters:
        </div>

        {filters.map((f, i) => (
          <React.Fragment key={`${f.field}-${i}`}>
            {/* Show logical operator selector between filters */}
            {i > 0 && (
              <div className="flex items-center gap-0">
                <button
                  onClick={() => updateFilterLogicalOperator(i, "AND")}
                  className={`px-1.5 py-1 text-xs font-medium rounded-l-md border transition-colors ${
                    f.logicalOperator === "AND" 
                      ? "bg-[var(--nv-green)] text-black border-[var(--nv-green)]" 
                      : "bg-neutral-700 text-gray-300 border-neutral-600 hover:bg-neutral-600"
                  }`}
                >
                  AND
                </button>
                <button
                  onClick={() => updateFilterLogicalOperator(i, "OR")}
                  className={`px-1.5 py-1 text-xs font-medium rounded-r-md border-l-0 border transition-colors ${
                    f.logicalOperator === "OR" || !f.logicalOperator
                      ? "bg-[var(--nv-green)] text-black border-[var(--nv-green)]" 
                      : "bg-neutral-700 text-gray-300 border-neutral-600 hover:bg-neutral-600"
                  }`}
                >
                  OR
                </button>
              </div>
            )}
            
            {/* Filter tag */}
            <div className="flex items-center gap-1.5 bg-[var(--nv-green)]/20 border border-[var(--nv-green)]/40 text-[var(--nv-green)] px-2.5 py-1 rounded-md text-xs font-medium">
              <span className="font-semibold">{f.field}</span>
              <span className="text-xs opacity-80">{f.operator}</span>
              <span className="font-medium">
                {typeof f.value === 'string' ? `"${f.value}"` : String(f.value)}
              </span>
              <button
                onClick={() => {
                  const updated = filters.filter((_, idx) => idx !== i);
                  setFilters(updated);
                }}
                className="ml-1 hover:text-red-400 transition-colors"
                aria-label="Remove filter"
                title="Remove filter"
              >
                <svg className="w-3 h-3" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </React.Fragment>
        ))}

        {/* Draft chip */}
        {draft.field && (
          <div className="flex items-center gap-1.5 bg-neutral-600/50 border border-neutral-600 text-gray-300 px-2.5 py-1 rounded-md text-xs opacity-70">
            <span className="font-semibold">{draft.field}</span>
            {draft.op && <span className="text-xs opacity-80">{draft.op}</span>}
            {stage === "value" && <span className="text-xs opacity-60">...</span>}
          </div>
        )}

        <div className="relative flex-1 min-w-[120px]">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => {
              setInput(e.target.value);
              if (stage === "value") {
                setDraft((d) => ({ ...d, value: e.target.value }));
              }
            }}
            onBlur={() => {
              if (stage === "value") commitFilter();
            }}
            onKeyDown={handleKeyDown}
            placeholder={
              stage === "field"
                ? "Add metadata filter..."
                : stage === "op"
                ? "Choose operator (=, !=, >, <, >=, <=)"
                : valuesLoading
                ? "Loading values..."
                : valueOptions.length === 0
                ? "No values found"
                : "Enter or select value"
            }
                      onFocus={() => {
            if (stage === "value") {
              setShow(valueOptions.length > 0 || valuesLoading);
            } else {
              setShow(suggestions.length > 0);
              // For operator stage, ensure we show immediately
              if (stage === "op" && draft.field) {
                setTimeout(() => setShow(true), 0);
              }
            }
          }}
            className="w-full bg-transparent text-white text-xs focus:outline-none placeholder-gray-500"
          />
          
          {stage === "field" && availableFields.length === 0 && (
            <div className="absolute right-0 top-1/2 -translate-y-1/2 text-xs text-gray-500">
              No metadata fields available
            </div>
          )}
          
          {stage === "value" && valuesLoading && (
            <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-2 text-xs text-[var(--nv-green)] bg-neutral-700/80 px-2 py-1 rounded-md">
              <div className="w-3 h-3 animate-spin rounded-full border-2 border-[var(--nv-green)] border-t-transparent"></div>
              Loading values...
            </div>
          )}
        </div>
      </div>

      {show && (
        <ul className="absolute left-0 bottom-full mb-1 max-h-40 w-full overflow-y-auto rounded-lg border border-neutral-600 bg-neutral-900 shadow-xl z-50 backdrop-blur-sm">
          {stage === "value" && valuesLoading ? (
            <li className="px-4 py-3 text-sm text-[var(--nv-green)] flex items-center justify-center gap-2 bg-neutral-800">
              <div className="w-4 h-4 animate-spin rounded-full border-2 border-[var(--nv-green)] border-t-transparent"></div>
              Loading available values...
            </li>
          ) : suggestions.length > 0 ? (
            suggestions.map((s, idx) => (
            <li
              key={s}
              className={`px-3 py-1.5 cursor-pointer text-xs transition-colors ${
                idx === activeIdx 
                  ? "bg-[var(--nv-green)]/20 text-[var(--nv-green)]" 
                  : "hover:bg-neutral-700 text-white"
              }`}
              onMouseDown={(e) => {
                e.preventDefault();
                chooseSuggestion(idx);
              }}
            >
              {stage === "field" && (
                <div className="flex items-center gap-2">
                  <svg className="w-3 h-3 text-gray-400" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span>{s}</span>
                  <span className="text-xs text-gray-500 ml-auto">{fieldMeta[s]}</span>
                </div>
              )}
              {stage === "op" && <span className="font-mono text-xs">{s}</span>}
              {stage === "value" && <span>"{s}"</span>}
            </li>
            ))
          ) : (
            <li className="px-3 py-2 text-xs text-gray-400">
              No suggestions available
            </li>
          )}
        </ul>
      )}
    </div>
  );
} 