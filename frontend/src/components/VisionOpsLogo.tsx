interface VisionOpsLogoProps {
  size?: number
  className?: string
}

export function VisionOpsLogo({ size = 24, className = '' }: VisionOpsLogoProps) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={`visionops-logo-svg ${className}`}
      aria-hidden="true"
      style={{ display: 'inline-block', verticalAlign: 'middle', flexShrink: 0 }}
    >
      <defs>
        <linearGradient id="vo-laser-beam" x1="0" y1="32" x2="64" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#06b6d4" stopOpacity="0" />
          <stop offset="0.25" stopColor="#22d3ee" />
          <stop offset="0.5" stopColor="#ffffff" />
          <stop offset="0.75" stopColor="#22d3ee" />
          <stop stopColor="#06b6d4" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="vo-wafer-ring" x1="8" y1="8" x2="56" y2="56" gradientUnits="userSpaceOnUse">
          <stop stopColor="#38bdf8" />
          <stop offset="1" stopColor="#0284c7" />
        </linearGradient>
      </defs>

      {/* 硅晶圆圆盘轮廓 (Wafer Outer Ring) */}
      <circle cx="32" cy="32" r="26" stroke="url(#vo-wafer-ring)" strokeWidth="2.5" strokeLinecap="round" />
      <circle cx="32" cy="32" r="23" stroke="#0284c7" strokeWidth="1" strokeDasharray="3 2" opacity="0.6" />

      {/* 晶圆 Die 划片阵列网格 (Silicon Wafer Die Grid) */}
      <path d="M16 22H48M16 32H48M16 42H48M22 16V48M32 16V48M42 16V48" stroke="#38bdf8" strokeWidth="1.2" strokeOpacity="0.55" />

      {/* 核心微芯片与引脚 (Microchip Core & Pins) */}
      <rect x="25" y="25" width="14" height="14" rx="3" fill="#0b111e" stroke="#22d3ee" strokeWidth="2" />
      <circle cx="32" cy="32" r="2.5" fill="#38bdf8" />
      <path d="M29 25V21M35 25V21M29 39V43M35 39V43M25 29H21M25 35H21M39 29H43M39 35H43" stroke="#22d3ee" strokeWidth="1.5" strokeLinecap="round" />

      {/* AOI 纳米级贯穿激光扫描光束 (Optical Inspection Laser Beam) */}
      <line x1="2" y1="32" x2="62" y2="32" stroke="url(#vo-laser-beam)" strokeWidth="2.5" strokeLinecap="round" />
      <polygon points="32,28 35,32 32,36 29,32" fill="#ffffff" />
    </svg>
  )
}
