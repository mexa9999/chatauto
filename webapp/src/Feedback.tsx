import { createContext, useCallback, useContext, useEffect, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";

type ToastKind = "success" | "error" | "info" | "warning";
type ToastMessage = {
  id: number;
  kind: ToastKind;
  message: string;
  action?: { label: string; onSelect: () => void };
};
type ConfirmOptions = {
  title: string;
  message: string;
  confirmLabel?: string;
  danger?: boolean;
};
type PromptOptions = {
  title: string;
  message: string;
  label: string;
  initialValue?: string;
  confirmLabel?: string;
};
type DialogState =
  | { kind: "confirm"; options: ConfirmOptions }
  | { kind: "prompt"; options: PromptOptions };
type DialogResult = boolean | string | null;
type FeedbackContextValue = {
  notify: (message: Omit<ToastMessage, "id">) => void;
  confirm: (options: ConfirmOptions) => Promise<boolean>;
  prompt: (options: PromptOptions) => Promise<string | null>;
};

const FeedbackContext = createContext<FeedbackContextValue | null>(null);

export function FeedbackProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<ToastMessage | null>(null);
  const [dialog, setDialog] = useState<DialogState | null>(null);
  const [dialogValue, setDialogValue] = useState("");
  const dialogRef = useRef<HTMLDialogElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const resolver = useRef<((result: DialogResult) => void) | null>(null);
  const toastTimer = useRef<number | null>(null);
  const toastSequence = useRef(0);

  const dismissToast = useCallback(() => {
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
    toastTimer.current = null;
    setToast(null);
  }, []);

  const notify = useCallback((message: Omit<ToastMessage, "id">) => {
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
    setToast({ ...message, id: ++toastSequence.current });
    toastTimer.current = window.setTimeout(
      () => setToast(null),
      message.kind === "error" ? 10_000 : 6_000,
    );
  }, []);

  const settleDialog = useCallback((result: DialogResult) => {
    const resolve = resolver.current;
    resolver.current = null;
    setDialog(null);
    resolve?.(result);
  }, []);

  const confirm = useCallback((options: ConfirmOptions) => new Promise<boolean>((resolve) => {
    if (resolver.current) {
      resolve(false);
      return;
    }
    resolver.current = (result) => resolve(result === true);
    setDialog({ kind: "confirm", options });
  }), []);

  const prompt = useCallback((options: PromptOptions) => new Promise<string | null>((resolve) => {
    if (resolver.current) {
      resolve(null);
      return;
    }
    resolver.current = (result) => resolve(typeof result === "string" ? result : null);
    setDialogValue(options.initialValue ?? "");
    setDialog({ kind: "prompt", options });
  }), []);

  useEffect(() => {
    const element = dialogRef.current;
    if (!element) return;
    if (dialog && !element.open) {
      element.showModal();
      if (dialog.kind === "prompt") {
        inputRef.current?.focus();
        inputRef.current?.select();
      }
    } else if (!dialog && element.open) {
      element.close();
    }
  }, [dialog]);

  useEffect(() => () => {
    if (toastTimer.current !== null) window.clearTimeout(toastTimer.current);
    resolver.current?.(null);
    resolver.current = null;
  }, []);

  const submitDialog = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (dialog?.kind === "prompt") {
      settleDialog(dialogValue.trim());
    } else {
      settleDialog(true);
    }
  };

  return (
    <FeedbackContext.Provider value={{ notify, confirm, prompt }}>
      {children}
      <div className="toast-region" aria-live="polite" aria-relevant="additions text">
        {toast && (
          <div
            key={toast.id}
            className={`toast toast-${toast.kind}`}
            role={toast.kind === "error" ? "alert" : "status"}
          >
            <span>{toast.message}</span>
            <div className="toast-actions">
              {toast.action && (
                <button
                  className="toast-action"
                  onClick={() => {
                    const action = toast.action?.onSelect;
                    dismissToast();
                    action?.();
                  }}
                >
                  {toast.action.label}
                </button>
              )}
              <button className="toast-close" aria-label="Закрыть уведомление" onClick={dismissToast}>×</button>
            </div>
          </div>
        )}
      </div>
      <dialog
        ref={dialogRef}
        className="confirm-dialog"
        aria-labelledby="feedback-dialog-title"
        onCancel={(event) => {
          event.preventDefault();
          settleDialog(dialog?.kind === "prompt" ? null : false);
        }}
        onClick={(event) => {
          if (event.target === dialogRef.current) {
            settleDialog(dialog?.kind === "prompt" ? null : false);
          }
        }}
      >
        {dialog && (
          <form onSubmit={submitDialog}>
            <p className="eyebrow">{dialog.kind === "prompt" ? "ВВОД ДАННЫХ" : "ПОДТВЕРЖДЕНИЕ"}</p>
            <h2 id="feedback-dialog-title">{dialog.options.title}</h2>
            <p className="confirm-dialog-message">{dialog.options.message}</p>
            {dialog.kind === "prompt" && (
              <label className="field confirm-dialog-field">
                <span>{dialog.options.label}</span>
                <input
                  ref={inputRef}
                  value={dialogValue}
                  maxLength={100}
                  onChange={(event) => setDialogValue(event.target.value)}
                />
              </label>
            )}
            <div className="button-row confirm-dialog-actions">
              <button type="button" className="button secondary" onClick={() => settleDialog(dialog.kind === "prompt" ? null : false)}>
                Отмена
              </button>
              <button
                type="submit"
                className={`button ${dialog.kind === "confirm" && dialog.options.danger ? "danger" : "primary"}`}
                disabled={dialog.kind === "prompt" && !dialogValue.trim()}
              >
                {dialog.kind === "prompt" ? dialog.options.confirmLabel ?? "Сохранить" : dialog.options.confirmLabel ?? "Подтвердить"}
              </button>
            </div>
          </form>
        )}
      </dialog>
    </FeedbackContext.Provider>
  );
}

export function useFeedback() {
  const context = useContext(FeedbackContext);
  if (!context) throw new Error("useFeedback must be used within FeedbackProvider");
  return context;
}
