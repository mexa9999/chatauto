import { useEffect, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

type StoredValue<T> = {
  value: T;
  error?: string;
};

export function useStoredState<T>(
  key: string,
  fallback: T,
  isValid: (value: unknown) => value is T,
  onError: (message: string) => void,
): [T, Dispatch<SetStateAction<T>>] {
  const [stored] = useState<StoredValue<T>>(() => {
    try {
      const raw = window.localStorage.getItem(key);
      if (raw === null) return { value: fallback };
      const value: unknown = JSON.parse(raw);
      if (isValid(value)) return { value };
      return { value: fallback, error: "Сохранённые настройки не соответствуют текущей версии; применены значения по умолчанию." };
    } catch (reason) {
      return {
        value: fallback,
        error: `Не удалось прочитать локальные настройки: ${reason instanceof Error ? reason.message : String(reason)}`,
      };
    }
  });
  const [value, setValue] = useState(stored.value);
  const writeErrorReported = useRef(false);

  useEffect(() => {
    if (stored.error) onError(stored.error);
  }, [onError, stored.error]);

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
      writeErrorReported.current = false;
    } catch (reason) {
      if (!writeErrorReported.current) {
        writeErrorReported.current = true;
        onError(`Не удалось сохранить локальные настройки: ${reason instanceof Error ? reason.message : String(reason)}`);
      }
    }
  }, [key, onError, value]);

  return [value, setValue];
}
