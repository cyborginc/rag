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
import { useCollectionsStore } from "../../store/useCollectionsStore";
import { 
  Block, 
  Flex, 
  Text, 
  TextInput, 
  Tag,
  Popover,
  Button
} from "@kui/react";

interface Props {
  filters: Filter[];
  setFilters: (filters: Filter[]) => void;
}

const ALL_OPS = ["=", "!=", ">", "<", ">=", "<=", "contains", "like", "in", "not in", "between", "before", "after", "array_contains", "array_contains_all", "array_contains_any", "array_length"];

export default function SimpleFilterBar({ filters, setFilters }: Props) {
  const [input, setInput] = useState("");
  const [stage, setStage] = useState<"field" | "op" | "value">("field");
  const [draft, setDraft] = useState<{ field?: string; op?: string; value?: string }>({});
  const [activeIdx, setActiveIdx] = useState(0);
  const [showOperators, setShowOperators] = useState(false);
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

  // Always show all operator suggestions when in operator stage
  const operatorSuggestions: string[] = useMemo(() => {
    if (stage === "op" && draft.field) {
      return ALL_OPS;
    }
    return [];
  }, [stage, draft.field]);

  // Convert string input value to appropriate type based on field metadata
  const convertValueToType = (value: string, fieldType: MetadataFieldType = "string"): string | number | boolean => {
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

  const chooseOperator = (idx: number) => {
    const choice = operatorSuggestions[idx];
    if (!choice) return;
    setDraft((d) => ({ ...d, op: choice }));
    setStage("value");
    setInput("");
    setShowOperators(false);
    // Focus input for value entry
    setTimeout(() => {
      inputRef.current?.focus();
    }, 0);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (stage === "op" && showOperators && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      e.preventDefault();
      setActiveIdx((prev) => {
        const next = e.key === "ArrowDown" ? prev + 1 : prev - 1;
        return (next + operatorSuggestions.length) % operatorSuggestions.length;
      });
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (stage === "field" && input.trim()) {
        // Move to operator stage and show operators immediately
        const fieldName = input.trim();
        setDraft({ field: fieldName });
        setStage("op");
        setInput("");
        // Use setTimeout to ensure state updates properly
        setTimeout(() => {
          setShowOperators(true);
          setActiveIdx(0);
          inputRef.current?.focus();
        }, 0);
      } else if (stage === "op") {
        if (showOperators && operatorSuggestions.length > 0) {
          // Choose the currently highlighted operator
          chooseOperator(activeIdx);
        } else if (!showOperators && draft.field) {
          // Show operators if they're not visible
          setShowOperators(true);
          setActiveIdx(0);
        }
      } else if (stage === "value" && input.trim()) {
        commitFilter();
      }
    } else if (e.key === "Tab") {
      if (stage === "op" && showOperators && operatorSuggestions.length > 0) {
        e.preventDefault();
        chooseOperator(activeIdx);
      }
    } else if (e.key === "Escape") {
      if (showOperators) {
        setShowOperators(false);
      } else if (stage !== "field") {
        // Reset to field stage
        setStage("field");
        setDraft({});
        setInput("");
        setShowOperators(false);
      }
    } else if (e.key === "Backspace" && !input) {
      if (stage === "value") {
        setStage("op");
        setDraft((d) => ({ field: d.field }));
        setTimeout(() => {
          setShowOperators(true);
          setActiveIdx(0);
        }, 0);
      } else if (stage === "op") {
        setStage("field");
        setDraft({});
        setShowOperators(false);
      } else if (stage === "field" && filters.length) {
        // Pop last filter
        const updated = [...filters];
        const last = updated.pop()!;
        setFilters(updated);
        setDraft({ field: last.field, op: last.operator, value: String(last.value) });
        setStage("value");
        setInput(String(last.value));
      }
    }
  };

  // Show operator suggestions when in op stage
  useEffect(() => {
    const inputFocused = document.activeElement === inputRef.current;
    if (stage === "op" && draft.field) {
      if (inputFocused) {
        setShowOperators(true);
        setActiveIdx(0);
      }
    } else {
      setShowOperators(false);
    }
  }, [stage, draft.field]);

  // Auto-focus input when stage changes
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.focus();
    }
  }, [stage]);

  // Close suggestions on outside click
  useEffect(() => {
    if (!showOperators) return;
    const handler = (e: MouseEvent) => {
      if (!wrapperRef.current?.contains(e.target as Node)) {
        setShowOperators(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [showOperators]);

  return (
    <>
      <Block 
        ref={wrapperRef} 
        style={{ 
          position: 'relative',
          backgroundColor: 'var(--background-color-surface)',
          border: '1px solid var(--border-color-subtle)',
          borderRadius: '8px',
          minHeight: '48px'
        }}
      >
        <Flex 
          align="center" 
          gap="density-sm" 
          style={{ 
            flexWrap: 'wrap',
            minHeight: '48px'
          }}
        >
          {/* Filter icon and label */}
          <Flex align="center" gap="density-xs">
            <svg 
              style={{ 
                width: '12px', 
                height: '12px', 
                color: 'var(--text-color-subtle)' 
              }} 
              fill="none" 
              stroke="currentColor" 
              strokeWidth="2" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 3c2.755 0 5.455.232 8.083.678.533.09.917.556.917 1.096v1.044a2.25 2.25 0 01-.659 1.591l-5.432 5.432a2.25 2.25 0 00-.659 1.591v2.927a2.25 2.25 0 01-1.244 2.013L9.75 21v-6.568a2.25 2.25 0 00-.659-1.591L3.659 7.409A2.25 2.25 0 013 5.818V4.774c0-.54.384-1.006.917-1.096A48.32 48.32 0 0112 3z" />
            </svg>
            <Text kind="label/regular/md" style={{ color: 'var(--text-color-subtle)' }}>
              Filters:
            </Text>
          </Flex>

          {filters.map((f, i) => (
            <React.Fragment key={`${f.field}-${i}`}>
              {/* Show logical operator selector between filters */}
              {i > 0 && (
                <Flex align="center" gap="density-xs">
                  <Button
                    kind={f.logicalOperator === "AND" ? "primary" : "tertiary"}
                    color={f.logicalOperator === "AND" ? "brand" : "neutral"}
                    size="small"
                    onClick={() => updateFilterLogicalOperator(i, "AND")}
                    style={{
                      fontSize: '10px',
                      padding: '2px 6px',
                      minWidth: '32px',
                      borderRadius: '4px 0 0 4px'
                    }}
                  >
                    AND
                  </Button>
                  <Button
                    kind={f.logicalOperator === "OR" ? "primary" : "tertiary"}
                    color={f.logicalOperator === "OR" ? "brand" : "neutral"}
                    size="small"
                    onClick={() => updateFilterLogicalOperator(i, "OR")}
                    style={{
                      fontSize: '10px',
                      padding: '2px 6px',
                      minWidth: '32px',
                      borderRadius: '0 4px 4px 0'
                    }}
                  >
                    OR
                  </Button>
                </Flex>
              )}
              
              {/* Filter tag */}
            <Tag
              color="green"
              kind="outline"
              density="compact"
              onClick={() => {
                const updated = filters.filter((_, idx) => idx !== i);
                setFilters(updated);
              }}
            >
              <Flex align="center" gap="density-xs">
                <Text kind="body/bold/xs">{f.field}</Text>
                <Text kind="body/regular/xs" style={{ opacity: 0.8 }}>{f.operator}</Text>
                <Text kind="body/regular/xs">
                  {typeof f.value === 'string' ? `"${f.value}"` : String(f.value)}
                </Text>
                <svg 
                  style={{ width: '12px', height: '12px' }} 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="2" 
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </Flex>
            </Tag>
            </React.Fragment>
          ))}

          {/* Draft chip */}
          {draft.field && (
            <Tag
              color="gray"
              kind="outline"
              density="compact"
              style={{ opacity: 0.7 }}
              readOnly
            >
              <Flex align="center" gap="density-xs">
                <Text kind="body/bold/xs">{draft.field}</Text>
                {draft.op && <Text kind="body/regular/xs" style={{ opacity: 0.8 }}>{draft.op}</Text>}
                {stage === "value" && <Text kind="body/regular/xs" style={{ opacity: 0.6 }}>...</Text>}
              </Flex>
            </Tag>
          )}

          <Popover
            open={showOperators && stage === "op"}
            onOpenChange={setShowOperators}
            side="bottom"
            align="start"
            slotContent={
              <Block style={{ 
                maxHeight: '160px', 
                overflowY: 'auto',
              }}>
                {operatorSuggestions.map((op, idx) => (
                  <Button
                    key={op}
                    kind={idx === activeIdx ? "primary" : "tertiary"}
                    color={idx === activeIdx ? "brand" : "neutral"}
                    size="small"
                    onClick={() => chooseOperator(idx)}
                    onMouseEnter={() => setActiveIdx(idx)}
                    style={{
                      width: '100%',
                      justifyContent: 'flex-start',
                      fontFamily: 'monospace',
                      margin: '2px 0'
                    }}
                  >
                    {op}
                  </Button>
                ))}
              </Block>
            }
          >
            <Block style={{ position: 'relative', flex: 1, minWidth: '120px' }}>
              <TextInput
                ref={inputRef}
                value={input}
                onChange={(e) => {
                  const target = e.target as HTMLInputElement;
                  setInput(target.value);
                  if (stage === "value") {
                    setDraft((d) => ({ ...d, value: target.value }));
                  }
                }}
                onBlur={() => {
                  if (stage === "value") commitFilter();
                }}
                onKeyDown={handleKeyDown}
                onFocus={() => {
                  if (stage === "op" && draft.field) {
                    setShowOperators(true);
                    setActiveIdx(0);
                  }
                }}
                placeholder={
                  stage === "field"
                    ? "Type field name and press Enter..."
                    : stage === "op"
                    ? "Press Enter to see operators or use ↑↓ keys"
                    : "Type value and press Enter..."
                }
                style={{
                  width: '100%',
                  border: 'none',
                  backgroundColor: 'transparent',
                  color: 'var(--text-color-primary)',
                  fontSize: '14px',
                  lineHeight: '20px',
                  padding: '0',
                  outline: 'none'
                }}
              />
            </Block>
          </Popover>
        </Flex>
      </Block>
    </>
  );
} 