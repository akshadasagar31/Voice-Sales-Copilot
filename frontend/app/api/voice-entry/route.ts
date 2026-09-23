// ============================================================================
// NEXT.JS API GATEWAY: SPEECH-TO-TEXT VOICE ENTRY (/api/voice-entry)
// ============================================================================
// This API route accepts raw recorded audio from the browser microphone
// (encoded as WAV or WebM) and forwards it to the FastAPI backend service
// for transcription via Deepgram Speech-to-Text.
//
// DATA FLOW:
// 1. Browser records user voice using Web Audio API ScriptProcessorNode.
// 2. Audio is encoded into standard 16-bit linear PCM WAV or WebM blob.
// 3. Browser sends multipart/form-data with the audio file to this route.
// 4. This route forwards the bytes to FastAPI: POST /api/voice-entry.
// 5. FastAPI calls Deepgram Nova-2 model and returns clean transcribed text.
// ============================================================================

import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
  try {
    // Extract the multipart/form-data containing the recorded audio file
    const formData = await req.formData();
    const file = (formData.get("file") || formData.get("audio_file")) as File | null;

    if (!file) {
      return NextResponse.json(
        { error: "No audio file was provided in the voice entry request." },
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    console.log(
      `[voice-entry API] Received upload: filename='${file.name}', size=${file.size} bytes, type='${file.type}'`
    );

    if (file.size === 0) {
      return NextResponse.json(
        { error: "Uploaded audio file is empty (0 bytes)." },
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }

    const fastApiUrl =
      process.env.FASTAPI_BACKEND_URL ||
      process.env.NEXT_PUBLIC_FASTAPI_URL ||
      "http://127.0.0.1:8001";

    const targetUrl = `${fastApiUrl.replace(/\/+$/, "")}/api/voice-entry`;


    // Reconstruct FormData for FastAPI upload, preserving exact audio blob bytes and MIME type
    const backendFormData = new FormData();
    const resolvedType =
      file.type ||
      (file.name.endsWith(".wav")
        ? "audio/wav"
        : file.name.endsWith(".ogg")
        ? "audio/ogg"
        : file.name.endsWith(".mp4")
        ? "audio/mp4"
        : "audio/webm");

    const audioBytes = await file.arrayBuffer();
    const audioBlob = new Blob([audioBytes], { type: resolvedType });
    backendFormData.append("file", audioBlob, file.name || "voice_note.webm");

    // Forward all non-file fields (language, module, existing_lead, lead_id, etc.)
    for (const [key, value] of formData.entries()) {
      if (key !== "file" && typeof value === "string") {
        backendFormData.append(key, value);
      }
    }

    console.log(
      `[voice-entry API] Forwarding to FastAPI ${targetUrl}: ` +
      `bytes=${audioBlob.size}, type='${resolvedType}', filename='${file.name || "voice_note.webm"}'`
    );

    let backendRes: Response | null = null;
    let data: any = null;
    const maxAttempts = 2;

    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        backendRes = await fetch(targetUrl, {
          method: "POST",
          body: backendFormData,
        });

        const contentType = backendRes.headers.get("content-type") || "";
        if (contentType.includes("application/json")) {
          try {
            data = await backendRes.json();
          } catch {
            data = null;
          }
        }

        if (!data) {
          const rawText = await backendRes.text().catch(() => "");
          try {
            data = JSON.parse(rawText);
          } catch {
            data = {
              error: rawText || `FastAPI returned status ${backendRes.status}`,
              status: backendRes.status,
            };
          }
        }

        // Retry if 503 / 408 / 504 / transient error from Deepgram or FastAPI
        const isTransient =
          backendRes.status === 503 ||
          backendRes.status === 408 ||
          backendRes.status === 504 ||
          (backendRes.status === 502 && (
            JSON.stringify(data || {}).toLowerCase().includes("timeout") ||
            JSON.stringify(data || {}).toLowerCase().includes("unavailable") ||
            JSON.stringify(data || {}).toLowerCase().includes("503") ||
            JSON.stringify(data || {}).toLowerCase().includes("408")
          ));

        if (isTransient && attempt < maxAttempts) {
          console.warn(`FastAPI voice-entry transient error (HTTP ${backendRes.status}) on attempt ${attempt}. Retrying in 500ms...`);
          await new Promise((resolve) => setTimeout(resolve, 500));
          continue;
        }

        break;
      } catch (fetchErr: any) {
        if (attempt < maxAttempts) {
          console.warn(`FastAPI voice-entry fetch error on attempt ${attempt}. Retrying in 500ms...`, fetchErr);
          await new Promise((resolve) => setTimeout(resolve, 500));
          continue;
        }
        throw fetchErr;
      }
    }

    if (!backendRes || !backendRes.ok) {
      const errorMsg =
        data?.detail ||
        data?.error ||
        `FastAPI voice processing returned an error (HTTP ${backendRes?.status || 500}).`;

      return NextResponse.json(
        {
          error: errorMsg,
          status: backendRes?.status || 500,
        },
        {
          status: backendRes?.status || 500,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    return NextResponse.json(data, {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  } catch (error: any) {
    console.error("Error proxying request to FastAPI /api/voice-entry:", error);
    return NextResponse.json(
      {
        error:
          "Unable to connect to FastAPI voice entry service. Please ensure the backend is running at http://127.0.0.1:8001.",
        details: error?.message || String(error),
      },
      {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}
