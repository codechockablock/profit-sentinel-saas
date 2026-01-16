// src/app/page.tsx
import ChannelView from '@/components/channel-view'

export default function Home() {
  return (
    <div className="flex h-screen bg-gray-100 text-gray-900">
      <ChannelView />
    </div>
  )
}