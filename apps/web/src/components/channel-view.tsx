// src/components/channel-view.tsx
import ChannelMessages from './channel-messages'
import MessageInput from './message-input'

export default function ChannelView() {
  return (
    <div className="flex-1 flex flex-col bg-gray-50">
      <div className="border-b p-6 bg-white">
        <h2 className="text-2xl font-bold"># general</h2>
        <p className="text-sm text-gray-600">This is the beginning of the #general channel.</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        <ChannelMessages channelId="general" />
      </div>

      <MessageInput channelId="general" />
    </div>
  )
}