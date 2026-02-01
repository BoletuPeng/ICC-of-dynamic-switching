import {
  Brain,
  Filter,
  Activity,
  GitBranch,
  Layers,
  GitMerge,
  Share2,
  BarChart2,
  Grid3X3,
  Target,
  Repeat,
  Sliders,
  Settings,
  Trash2,
  X,
  ChevronDown,
  ChevronUp,
  Play,
  Zap,
  Box,
  type LucideIcon,
} from 'lucide-react';

const iconMap: Record<string, LucideIcon> = {
  brain: Brain,
  filter: Filter,
  activity: Activity,
  'git-branch': GitBranch,
  layers: Layers,
  'git-merge': GitMerge,
  'share-2': Share2,
  'bar-chart-2': BarChart2,
  grid: Grid3X3,
  target: Target,
  repeat: Repeat,
  sliders: Sliders,
  settings: Settings,
  trash: Trash2,
  x: X,
  'chevron-down': ChevronDown,
  'chevron-up': ChevronUp,
  play: Play,
  zap: Zap,
  box: Box,
};

interface IconProps {
  name: string;
  className?: string;
  size?: number;
  style?: React.CSSProperties;
}

export function Icon({ name, className = '', size = 20, style }: IconProps) {
  const IconComponent = iconMap[name] || Box;
  return <IconComponent className={className} size={size} style={style} />;
}

export {
  Brain,
  Filter,
  Activity,
  GitBranch,
  Layers,
  GitMerge,
  Share2,
  BarChart2,
  Grid3X3,
  Target,
  Repeat,
  Sliders,
  Settings,
  Trash2,
  X,
  ChevronDown,
  ChevronUp,
  Play,
  Zap,
  Box,
};
