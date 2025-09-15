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

import { useCallback, useMemo, useState } from "react";
import { useNewCollectionStore } from "../../store/useNewCollectionStore";
import { useCollectionDrawerStore } from "../../store/useCollectionDrawerStore";
import { useCollectionActions } from "../../hooks/useCollectionActions";
import { DrawerActions } from "../drawer/DrawerActions";
import { ConfirmationModal } from "../modals/ConfirmationModal";
import { Block, Notification, SidePanel } from "@kui/react";
import { DocumentsList } from "../tasks/DocumentsList";
import { UploaderSection } from "../drawer/UploaderSection";

// Export all drawer components for external uYou arese
export { LoadingState } from "../ui/LoadingState";
export { ErrorState } from "../ui/ErrorState";
export { EmptyState } from "../ui/EmptyState";
export { DocumentItem } from "../tasks/DocumentItem";
export { DocumentsList } from "../tasks/DocumentsList";
export { UploaderSection } from "../drawer/UploaderSection";
export { DrawerActions } from "../drawer/DrawerActions";

export default function CollectionDrawer() {
  const { activeCollection, closeDrawer, toggleUploader, deleteError, showUploader } = useCollectionDrawerStore();
  const { setMetadataSchema } = useNewCollectionStore();
  const { deleteCollectionWithoutConfirm, isDeleting } = useCollectionActions();
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const title = useMemo(() => 
    activeCollection?.collection_name || "Collection", 
    [activeCollection]
  );

  const handleClose = useCallback(() => {
    useNewCollectionStore.getState().reset();
    closeDrawer();
  }, [closeDrawer]);

  const handleAddSource = useCallback(() => {
    setMetadataSchema(activeCollection?.metadata_schema || []);
    toggleUploader(true);
  }, [activeCollection, setMetadataSchema, toggleUploader]);

  const handleDeleteClick = useCallback(() => {
    if (activeCollection?.collection_name) {
      setShowDeleteModal(true);
    }
  }, [activeCollection?.collection_name]);

  const handleConfirmDelete = useCallback(() => {
    if (activeCollection?.collection_name) {
      deleteCollectionWithoutConfirm(activeCollection.collection_name);
    }
  }, [activeCollection?.collection_name, deleteCollectionWithoutConfirm]);

  return (
    <SidePanel
      style={{
        "--side-panel-width": "50vw",
        background: 'var(--background-color-interaction-inverse)',
        color: 'var(--text-color-inverse)'
      }}
      modal
      open={!!activeCollection}
      onOpenChange={(open) => {
        if (!open) {
          handleClose();
        }
      }}
      side="right"
      slotHeading={<span style={{ color: 'var(--text-color-inverse)' }}>{title}</span>}
      slotFooter={
        <DrawerActions 
          onDelete={handleDeleteClick}
          onAddSource={handleAddSource}
          isDeleting={isDeleting}
        />
      }
      closeOnClickOutside
    >
      <Block 
          style={{ 
            overflowY: 'auto',
            flex: 1,
            height: '100%',
            padding: '16px'
          }}
        >
          <>
            <DocumentsList />
            
            {deleteError && (
              <div style={{ color: 'var(--text-color-inverse)' }}>
                <Notification
                  status="error"
                  slotHeading="Delete Error"
                  slotSubheading={deleteError}
                />
              </div>
            )}
            
            {showUploader && <UploaderSection />}
          </>
        </Block>
        
        <ConfirmationModal
          isOpen={showDeleteModal}
          onClose={() => setShowDeleteModal(false)}
          onConfirm={handleConfirmDelete}
          title="Delete Collection"
          message={`Are you sure you want to delete the collection "${activeCollection?.collection_name}"? This action will permanently delete all documents and metadata. This cannot be undone.`}
          confirmText="Delete Collection"
          confirmColor="danger"
        />
    </SidePanel>
  );
}
