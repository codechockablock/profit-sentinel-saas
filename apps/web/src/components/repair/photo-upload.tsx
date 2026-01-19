'use client'

import { useRef, useState, useCallback } from 'react'

interface PhotoUploadProps {
  onCapture: (base64: string) => void
  imageBase64: string | null
  onClear: () => void
}

// Max image size: 1024px, quality 0.8, strips EXIF by default via canvas
const MAX_SIZE = 1024
const QUALITY = 0.8

function resizeImage(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()

    reader.onload = (e) => {
      const img = new Image()

      img.onload = () => {
        // Create canvas for resizing (also strips EXIF)
        const canvas = document.createElement('canvas')
        let { width, height } = img

        // Calculate new dimensions
        if (width > MAX_SIZE || height > MAX_SIZE) {
          if (width > height) {
            height = (height / width) * MAX_SIZE
            width = MAX_SIZE
          } else {
            width = (width / height) * MAX_SIZE
            height = MAX_SIZE
          }
        }

        canvas.width = width
        canvas.height = height

        const ctx = canvas.getContext('2d')
        if (!ctx) {
          reject(new Error('Failed to get canvas context'))
          return
        }

        // Draw resized image (strips EXIF)
        ctx.drawImage(img, 0, 0, width, height)

        // Convert to base64 JPEG
        const base64 = canvas.toDataURL('image/jpeg', QUALITY).split(',')[1]
        resolve(base64)
      }

      img.onerror = () => reject(new Error('Failed to load image'))
      img.src = e.target?.result as string
    }

    reader.onerror = () => reject(new Error('Failed to read file'))
    reader.readAsDataURL(file)
  })
}

export function PhotoUpload({ onCapture, imageBase64, onClear }: PhotoUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isCapturing, setIsCapturing] = useState(false)
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file')
      return
    }

    // Validate file size (5MB max before resize)
    if (file.size > 5 * 1024 * 1024) {
      setError('Image too large. Max 5MB.')
      return
    }

    try {
      setError(null)
      const base64 = await resizeImage(file)
      onCapture(base64)
    } catch (err) {
      console.error('Failed to process image:', err)
      setError('Failed to process image')
    }
  }, [onCapture])

  const startCamera = useCallback(async () => {
    try {
      setError(null)
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }, // Prefer back camera
      })
      setStream(mediaStream)
      setIsCapturing(true)

      // Wait for video element to be ready
      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream
        }
      }, 100)
    } catch (err) {
      console.error('Camera access failed:', err)
      setError('Could not access camera. Try uploading a photo instead.')
    }
  }, [])

  const capturePhoto = useCallback(() => {
    if (!videoRef.current) return

    const video = videoRef.current
    const canvas = document.createElement('canvas')

    // Use video dimensions
    let { videoWidth: width, videoHeight: height } = video

    // Resize if needed
    if (width > MAX_SIZE || height > MAX_SIZE) {
      if (width > height) {
        height = (height / width) * MAX_SIZE
        width = MAX_SIZE
      } else {
        width = (width / height) * MAX_SIZE
        height = MAX_SIZE
      }
    }

    canvas.width = width
    canvas.height = height

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.drawImage(video, 0, 0, width, height)
    const base64 = canvas.toDataURL('image/jpeg', QUALITY).split(',')[1]

    // Stop camera
    stopCamera()
    onCapture(base64)
  }, [onCapture])

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop())
      setStream(null)
    }
    setIsCapturing(false)
  }, [stream])

  // Show preview if image captured
  if (imageBase64) {
    return (
      <div className="relative">
        <img
          src={`data:image/jpeg;base64,${imageBase64}`}
          alt="Captured"
          className="w-full rounded-xl object-cover max-h-64"
        />
        <button
          onClick={() => {
            onClear()
            stopCamera()
          }}
          className="absolute top-2 right-2 p-2 bg-red-500 text-white rounded-full hover:bg-red-600 transition"
          title="Remove photo"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
            <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
          </svg>
        </button>
        <div className="absolute bottom-2 left-2 px-2 py-1 bg-emerald-500/80 text-white text-xs rounded">
          Photo ready
        </div>
      </div>
    )
  }

  // Show camera view if capturing
  if (isCapturing) {
    return (
      <div className="relative">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full rounded-xl bg-black"
        />
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-4">
          <button
            onClick={stopCamera}
            className="p-3 bg-slate-700 text-white rounded-full hover:bg-slate-600 transition"
            title="Cancel"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-6 h-6">
              <path d="M6.28 5.22a.75.75 0 00-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 101.06 1.06L10 11.06l3.72 3.72a.75.75 0 101.06-1.06L11.06 10l3.72-3.72a.75.75 0 00-1.06-1.06L10 8.94 6.28 5.22z" />
            </svg>
          </button>
          <button
            onClick={capturePhoto}
            className="p-4 bg-white text-slate-900 rounded-full hover:bg-slate-200 transition ring-4 ring-white/30"
            title="Capture"
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-8 h-8">
              <path fillRule="evenodd" d="M1 8a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 018.07 3h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0016.07 6H17a2 2 0 012 2v7a2 2 0 01-2 2H3a2 2 0 01-2-2V8zm13.5 3a4.5 4.5 0 11-9 0 4.5 4.5 0 019 0zM10 14a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
      </div>
    )
  }

  // Default: show upload options
  return (
    <div className="space-y-4">
      {error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {error}
        </div>
      )}

      <div className="flex gap-4">
        {/* Camera Button */}
        <button
          onClick={startCamera}
          className="flex-1 flex flex-col items-center gap-2 p-6 bg-slate-900/50 border-2 border-dashed border-slate-600 rounded-xl hover:border-blue-500 hover:bg-blue-500/5 transition"
        >
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-10 h-10 text-blue-400">
            <path d="M12 9a3.75 3.75 0 100 7.5A3.75 3.75 0 0012 9z" />
            <path fillRule="evenodd" d="M9.344 3.071a49.52 49.52 0 015.312 0c.967.052 1.83.585 2.332 1.39l.821 1.317c.24.383.645.643 1.11.71.386.054.77.113 1.152.177 1.432.239 2.429 1.493 2.429 2.909V18a3 3 0 01-3 3H4.5a3 3 0 01-3-3V9.574c0-1.416.997-2.67 2.429-2.909.382-.064.766-.123 1.151-.178a1.56 1.56 0 001.11-.71l.822-1.315a2.942 2.942 0 012.332-1.39zM6.75 12.75a5.25 5.25 0 1110.5 0 5.25 5.25 0 01-10.5 0zm12-1.5a.75.75 0 100-1.5.75.75 0 000 1.5z" clipRule="evenodd" />
          </svg>
          <span className="text-slate-300 font-medium">Use Camera</span>
        </button>

        {/* Upload Button */}
        <label className="flex-1 flex flex-col items-center gap-2 p-6 bg-slate-900/50 border-2 border-dashed border-slate-600 rounded-xl hover:border-emerald-500 hover:bg-emerald-500/5 transition cursor-pointer">
          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-10 h-10 text-emerald-400">
            <path fillRule="evenodd" d="M11.47 2.47a.75.75 0 011.06 0l4.5 4.5a.75.75 0 01-1.06 1.06l-3.22-3.22V16.5a.75.75 0 01-1.5 0V4.81L8.03 8.03a.75.75 0 01-1.06-1.06l4.5-4.5zM3 15.75a.75.75 0 01.75.75v2.25a1.5 1.5 0 001.5 1.5h13.5a1.5 1.5 0 001.5-1.5V16.5a.75.75 0 011.5 0v2.25a3 3 0 01-3 3H5.25a3 3 0 01-3-3V16.5a.75.75 0 01.75-.75z" clipRule="evenodd" />
          </svg>
          <span className="text-slate-300 font-medium">Upload Photo</span>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            className="hidden"
          />
        </label>
      </div>

      <p className="text-slate-500 text-xs text-center">
        Photos are resized locally and never stored
      </p>
    </div>
  )
}
