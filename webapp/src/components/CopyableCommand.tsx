import { useEffect, useState } from 'react'

/** A one-line terminal command with a copy button. */
export default function CopyableCommand({ label, command }: { label: string; command: string }) {
  const [copied, setCopied] = useState(false)
  useEffect(() => {
    if (!copied) return
    const t = setTimeout(() => setCopied(false), 1500)
    return () => clearTimeout(t)
  }, [copied])
  return (
    <div className="command-row">
      <span className="command-label">{label}</span>
      <code>{command}</code>
      <button
        onClick={() => {
          navigator.clipboard.writeText(command).then(() => setCopied(true))
        }}
      >
        {copied ? 'copied' : 'copy'}
      </button>
    </div>
  )
}
