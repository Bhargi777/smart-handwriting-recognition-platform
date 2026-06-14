"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";

type PredictionResponse = {
  confidence?: number;
  confidence_scores?: Record<string, number>;
  error?: string;
  predicted_digit?: number;
  success?: boolean;
};

const CANVAS_SIZE = 280;
const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const isDrawingRef = useRef(false);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isPredicting, setIsPredicting] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Checking backend health...");
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    context.lineWidth = 16;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = "#000000";
  }, []);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const payload = await response.json();
        if (response.ok) {
          setStatusMessage(`Backend: ${payload.status} (${payload.model_status})`);
          return;
        }
        setStatusMessage("Backend health check failed.");
      } catch {
        setStatusMessage("Backend unavailable. Start FastAPI on port 8000.");
      }
    };

    void checkHealth();
  }, []);

  const startDrawing = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    isDrawingRef.current = true;
    const rect = canvas.getBoundingClientRect();
    context.beginPath();
    context.moveTo(event.clientX - rect.left, event.clientY - rect.top);
  };

  const draw = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!isDrawingRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }

    const rect = canvas.getBoundingClientRect();
    context.lineTo(event.clientX - rect.left, event.clientY - rect.top);
    context.stroke();
  };

  const stopDrawing = () => {
    isDrawingRef.current = false;
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }
    const context = canvas.getContext("2d");
    if (!context) {
      return;
    }
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
    setPrediction(null);
  };

  const requestPrediction = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/predict`, {
      body: formData,
      method: "POST",
    });

    const payload = (await response.json()) as PredictionResponse;
    if (!response.ok || payload.success === false) {
      throw new Error(payload.error ?? "Prediction failed.");
    }
    return payload;
  };

  const predictCanvas = async () => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return;
    }

    setIsPredicting(true);
    setPrediction(null);

    try {
      const imageBlob = await (await fetch(canvas.toDataURL("image/png"))).blob();
      const payload = await requestPrediction(
        new File([imageBlob], "drawing.png", { type: "image/png" }),
      );
      setPrediction(payload);
    } catch (error) {
      setPrediction({ error: error instanceof Error ? error.message : "Prediction failed." });
    } finally {
      setIsPredicting(false);
    }
  };

  const onFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    setPrediction(null);
    const nextFile = event.target.files?.[0] ?? null;
    setSelectedFile(nextFile);
  };

  const predictUpload = async () => {
    if (!selectedFile) {
      setPrediction({ error: "Choose a file first." });
      return;
    }

    setIsPredicting(true);
    setPrediction(null);
    try {
      const payload = await requestPrediction(selectedFile);
      setPrediction(payload);
    } catch (error) {
      setPrediction({ error: error instanceof Error ? error.message : "Prediction failed." });
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-8 p-6 md:p-10">
      <header className="space-y-2">
        <h1 className="text-3xl font-bold tracking-tight">Smart Handwriting Recognition</h1>
        <p className="text-zinc-600 dark:text-zinc-300">
          Draw a digit or upload an image and run inference against the FastAPI model endpoint.
        </p>
        <p className="text-sm font-medium text-zinc-500 dark:text-zinc-400">{statusMessage}</p>
      </header>

      <section className="grid gap-6 lg:grid-cols-2">
        <article className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          <h2 className="mb-3 text-xl font-semibold">Draw a digit</h2>
          <canvas
            ref={canvasRef}
            width={CANVAS_SIZE}
            height={CANVAS_SIZE}
            className="w-full max-w-[280px] touch-none rounded-lg border border-zinc-300 bg-white"
            onPointerDown={startDrawing}
            onPointerMove={draw}
            onPointerUp={stopDrawing}
            onPointerLeave={stopDrawing}
          />
          <div className="mt-4 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={clearCanvas}
              className="rounded-md border border-zinc-300 px-4 py-2 text-sm font-medium"
            >
              Clear
            </button>
            <button
              type="button"
              onClick={predictCanvas}
              disabled={isPredicting}
              className="rounded-md bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-60 dark:bg-white dark:text-black"
            >
              {isPredicting ? "Predicting..." : "Predict drawing"}
            </button>
          </div>
        </article>

        <article className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
          <h2 className="mb-3 text-xl font-semibold">Upload image</h2>
          <input
            type="file"
            accept="image/*"
            onChange={onFileSelect}
            className="block w-full text-sm file:mr-3 file:rounded-md file:border-0 file:bg-zinc-100 file:px-3 file:py-2 file:font-medium dark:file:bg-zinc-800"
          />
          <button
            type="button"
            onClick={predictUpload}
            disabled={isPredicting || !selectedFile}
            className="mt-4 rounded-md bg-black px-4 py-2 text-sm font-medium text-white disabled:opacity-60 dark:bg-white dark:text-black"
          >
            {isPredicting ? "Predicting..." : "Predict uploaded image"}
          </button>
        </article>
      </section>

      <section className="rounded-xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-950">
        <h2 className="mb-3 text-xl font-semibold">Prediction output</h2>
        {!prediction && <p className="text-zinc-600 dark:text-zinc-300">No prediction yet.</p>}
        {prediction?.error && (
          <p className="font-medium text-red-600 dark:text-red-400">{prediction.error}</p>
        )}
        {typeof prediction?.predicted_digit === "number" && (
          <div className="space-y-4">
            <p className="text-lg">
              Predicted digit:{" "}
              <span className="font-bold text-2xl">{prediction.predicted_digit}</span>
            </p>
            <p className="text-zinc-600 dark:text-zinc-300">
              Confidence: {((prediction.confidence ?? 0) * 100).toFixed(2)}%
            </p>
            <div className="space-y-2">
              {Object.entries(prediction.confidence_scores ?? {})
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([digit, score]) => (
                  <div key={digit} className="flex items-center gap-3 text-sm">
                    <span className="w-4 font-medium">{digit}</span>
                    <div className="h-2 flex-1 rounded bg-zinc-200 dark:bg-zinc-800">
                      <div
                        className="h-2 rounded bg-zinc-900 dark:bg-zinc-100"
                        style={{ width: `${Math.max(1, score * 100)}%` }}
                      />
                    </div>
                    <span className="w-14 text-right">{(score * 100).toFixed(2)}%</span>
                  </div>
                ))}
            </div>
          </div>
        )}
      </section>
    </main>
  );
}
