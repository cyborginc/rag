import { useCallback } from "react";
import { useNewCollectionStore } from "../../store/useNewCollectionStore";
import { MetadataField } from "./MetadataField";
import type { UIMetadataField } from "../../types/collections";

interface FileCardProps {
  file: File;
  index: number;
}

const RemoveButton = ({ onClick }: { onClick: () => void }) => (
  <button
    onClick={onClick}
    className="text-red-400 hover:text-red-300 text-sm transition-colors"
  >
    Remove
  </button>
);

export const FileCard = ({ file, index }: FileCardProps) => {
  const { metadataSchema, fileMetadata, removeFile, updateMetadataField } = useNewCollectionStore();

  const handleRemove = useCallback(() => {
    removeFile(index);
  }, [index, removeFile]);

  const handleMetadataChange = useCallback((fieldName: string, value: unknown) => {
    updateMetadataField(file.name, fieldName, value);
  }, [file.name, updateMetadataField]);

  return (
    <div className="p-3 bg-neutral-800 rounded-md">
      <div className="flex justify-between items-center">
        <span className="text-sm text-white truncate flex-1 mr-2">{file.name}</span>
        <RemoveButton onClick={handleRemove} />
      </div>
      
      {metadataSchema.length > 0 && (
        <div className="mt-2 space-y-4">
          {metadataSchema
            .filter((field: UIMetadataField) => field.name !== 'filename') // Filter out filename field
            .map((field: UIMetadataField) => (
            <MetadataField
              key={field.name}
              fileName={file.name}
              field={field}
              value={(() => {
                const existingValue = fileMetadata[file.name]?.[field.name];
                if (existingValue !== undefined) return existingValue;
                switch (field.type) {
                  case "boolean": return false;
                  case "array": return [];
                  case "integer":
                  case "float":
                  case "number": return null;
                  default: return "";
                }
              })()}
              onChange={handleMetadataChange}
            />
          ))}
        </div>
      )}
    </div>
  );
}; 